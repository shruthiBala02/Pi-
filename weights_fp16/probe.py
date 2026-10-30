"""Probe of this machine before the run (from MLSys-QPINNs code/pi_ablations/probe.py, trimmed to what these rows use).

    python probe.py

It rounds and trains nothing. Each test is independent: a failure is printed, never hidden, and the next test runs.
  fp16     float16 matmul runs, and is within fp16_probe_rtol x the field max of the float32 matmul of the same float16
           values, on every layer shape a full-grid pass of an FP16-weight row runs (the hidden layers after the first)
  memory   the peak memory can be reset (/proc/self/clear_refs) and glibc takes the allocator settings: the inference
           memory reading needs both
  pi       temperature, clock and the throttle word (vcgencmd) can be read
The result goes to probe_result.json; the printed table names every row's value matmul path. run_one.py takes the
paths from that file only, and refuses to run when the memory or Pi test failed.
Rows with INT8 / INT16 / INT32 weights and FP16 values have no native path (no integer x float16 kernel): their weights
are decoded to float32 one layer at a time (decode_per_layer), whatever the probe finds.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import torch
import yaml

import net as N
import sysmon as SM

HERE = Path(__file__).resolve().parent


def say(name: str, status: str, detail: str = "") -> None:
    print(f"[{status:5}] {name}" + (f": {detail}" if detail else ""))


def load_cfg(root: Path = HERE) -> dict:
    return yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))


def networks_dir(root: Path, cfg: dict) -> Path:
    d = (root / cfg["networks_dir"]).resolve()
    if not d.is_dir():
        raise FileNotFoundError(f"{d} is missing: put exp25's Pi folder (ablations, with its equation folders) next "
                                "to this one, or set networks_dir in config.yaml")
    return d


def choose_path(row: dict, fp16_ok: bool) -> str:
    """The row's value matmul path (net.choose_path; no integer path fits FP16 values, so only fp16 is asked)."""
    if row["method"] == "autograd":
        return "fp32"
    return N.choose_path(row["weights"], row["value"], {"int": False, "int16": False, "fp16": fp16_ok})


def shapes(root: Path, cfg: dict) -> dict[str, list[tuple[int, int, int]]]:
    """{equation: [(grid points, in, out), ...]} of the hidden layers after the first (the ones a native path runs)."""
    nd = networks_dir(root, cfg)
    out = {}
    for eq in cfg["equations"]:
        d = nd / eq
        layers = N.load_settings(d)["layers"]
        n = int(np.load(d / "data.npz")["x"].shape[0])
        hidden = layers[1:-1]
        out[eq] = sorted({(n, a, b) for a, b in zip(hidden[:-1], hidden[1:])})
    return out


def probe_fp16(found: dict, gen: torch.Generator, rtol: float) -> tuple[bool, str]:
    worst, bad = 0.0, []
    for eq, v in found.items():
        for m, k, n in v:
            x = torch.randn(m, k, generator=gen).half()
            w = torch.randn(n, k, generator=gen).half()
            got = (x @ w.T).float()
            ref = x.float() @ w.float().T
            gap = float((got - ref).abs().max()) / max(float(ref.abs().max()), 1e-30)
            worst = max(worst, gap)
            if gap > rtol:
                bad.append((eq, m, k, n))
    return not bad, f"largest gap {worst:.2e} of the field max (allowed {rtol:.2e})" + (f"; over it at {bad[:5]}" if bad else "")


def probe_pi() -> tuple[bool, str]:
    t, c, th = SM.read_temp_c(), SM.read_clock_hz(), SM.read_throttled()
    return True, f"{t:.1f} C, clock {c / 1e9:.2f} GHz, throttle word {th}" + (
        " (events since boot: reboot before the run so the record starts clean)" if int(th, 16) & SM.HISTORY_BITS else "")


def main() -> None:
    cfg = load_cfg()
    torch.set_num_threads(int(cfg["threads"]))
    print(f"machine {platform.machine()}, {platform.platform()}, python {platform.python_version()}, "
          f"torch {torch.__version__}, threads {torch.get_num_threads()}")
    result: dict = {"torch": torch.__version__, "threads": torch.get_num_threads()}
    try:
        env = SM.env_info()
        result["cpu"] = env
        say("cpu", "info", f"{env['cpu_model']}; asimdhp (fp16 vector)={env['asimdhp']}; fphp (fp16 scalar)={env['fphp']}")
    except RuntimeError as e:
        result["cpu"] = {"error": str(e)}
        say("cpu", "FAIL", str(e))
    found = shapes(HERE, cfg)
    result["equations_checked"] = {eq: [list(s) for s in v] for eq, v in found.items()}
    tests = {"fp16 matmul": lambda: probe_fp16(found, torch.Generator().manual_seed(int(cfg["probe_seed"])),
                                               cfg["fp16_probe_rtol"]),
             "memory reading": SM.peak_reset_works, "pi readings": probe_pi}
    ok = {}
    for name, test in tests.items():
        try:
            ok[name], detail = test()
        except Exception as e:  # noqa: BLE001  any failure is the answer
            ok[name], detail = False, f"{type(e).__name__}: {str(e)[:200]}"
        say(name, "ok" if ok[name] else "FAIL", detail)
        result[name.replace(" ", "_") + "_detail"] = detail
    result["native_fp16"] = ok["fp16 matmul"]
    result["memory_ok"] = ok["memory reading"]
    result["pi_ok"] = ok["pi readings"]
    (HERE / cfg["probe_file"]).write_text(json.dumps(result, indent=1), encoding="utf-8")
    bar = "=" * 72
    print(f"\n{bar}\n  fp16 matmul={str(result['native_fp16']).upper()}   memory reading={str(result['memory_ok']).upper()}"
          f"   pi readings={str(result['pi_ok']).upper()}")
    for row in cfg["rows"]:
        print(f"  {row['id']:14} {row['method']:8} weights {row['weights']:6} value {row['value']:5} "
              f"tracks {row['tracks']:5} -> {choose_path(row, result['native_fp16'])}")
    if not (result["memory_ok"] and result["pi_ok"]):
        print("  run_one.py will refuse to run: fix the failed test above first")
    print(f"  written to {HERE / cfg['probe_file']}\n{bar}")


if __name__ == "__main__":
    main()
