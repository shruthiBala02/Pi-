"""exp25 probe: which native maths this machine has for the formats, and whether the power chip can be read (written for
the Raspberry Pi 5; from code/pi/probe_int8.py).

    python probe.py        (after the equation folders are copied in: the verdict depends on the layer shapes that run)

It rounds and trains nothing. Each test is independent: a failure is printed, never hidden, and the next test runs.
  int     INT8 x INT8 torch._int_mm equal to an int64 reference on every shipped layer shape
  int16   int64 torch.matmul of INT16-range integers equal to a float64 reference (its sums stay below 2^53) on every shape
  fp16    float16 matmul runs, and is within fp16_probe_rtol x the field max of the float32 matmul of the same float16
          values, on every shape
  BF16    measured the same way and reported; never a path
  power   `vcgencmd pmic_read_adc` readable, its supply lines (both V and A), readings per second
The result, with the raw power-chip text, goes to probe_result.json; the printed table names every row's matmul path.
run_one.py takes the paths from that file only.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import net as N
import sysmon as SM

HERE = Path(__file__).resolve().parent


def say(name: str, status: str, detail: str = "") -> None:
    print(f"[{status:5}] {name}" + (f": {detail}" if detail else ""))


def timed_ms(fn, repeats: int) -> float:
    fn()
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return statistics.median(t) * 1e3


def load_cfg(root: Path = HERE) -> dict:
    return yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))


def shipped_shapes(root: Path, cfg: dict) -> dict[str, list[tuple[int, int, int]]]:
    """{equation: [(rows, in, out), ...]} of the layers a native path runs (the hidden layers after the first: jets.py)
    of every equation folder with a rows.json: the row counts are the batch sizes, the accuracy chunk and the last
    (shorter) chunk of the grid. Stops when no folder is there."""
    chunk = int(cfg["accuracy_chunk"])
    found = {}
    for rows in sorted(root.glob("*/rows.json")):
        d = rows.parent
        layers = N.load_settings(d)["layers"]
        n = int(np.load(d / "data.npz")["x"].shape[0])
        counts = set(cfg["batch_sizes"]) | {min(n, chunk)} | ({n % chunk} if n > chunk and n % chunk else set())
        hidden = layers[1:-1]                                  # widths of the hidden layers
        found[d.name] = sorted({(m, a, b) for a, b in zip(hidden[:-1], hidden[1:]) for m in counts})
    if not found:
        raise RuntimeError(f"no equation folder with rows.json in {root}: copy the export folders here first")
    return found


def _check_shapes(shapes: dict, test) -> list:
    return [(eq, *s) for eq, v in shapes.items() for s in v if not test(*s)]


def probe_int8(shapes: dict, gen: torch.Generator) -> tuple[bool, str]:
    def ok(m, k, n):
        x = torch.randint(-127, 128, (m, k), generator=gen, dtype=torch.int8)
        w = torch.randint(-127, 128, (n, k), generator=gen, dtype=torch.int8)
        o = torch._int_mm(x, w.T.contiguous())
        return o.dtype == torch.int32 and torch.equal(o.long(), x.long() @ w.long().T)
    bad = _check_shapes(shapes, ok)
    return not bad, (f"NOT exact for (equation, rows, in, out) {bad}" if bad else "exact on every shipped shape")


def probe_int16(shapes: dict, gen: torch.Generator) -> tuple[bool, str]:
    def ok(m, k, n):
        x = torch.randint(-32767, 32768, (m, k), generator=gen, dtype=torch.int64)
        w = torch.randint(-32767, 32768, (n, k), generator=gen, dtype=torch.int64)
        return torch.equal(x @ w.T, (x.double() @ w.double().T).long())
    bad = _check_shapes(shapes, ok)
    return not bad, (f"NOT exact for (equation, rows, in, out) {bad}" if bad else "exact on every shipped shape")


def probe_half(dtype: torch.dtype, shapes: dict, gen: torch.Generator, rtol: float) -> tuple[bool, str]:
    worst = 0.0

    def ok(m, k, n):
        nonlocal worst
        x = torch.randn(m, k, generator=gen).to(dtype)
        w = torch.randn(n, k, generator=gen).to(dtype)
        got = (x @ w.T).float()
        ref = x.float() @ w.float().T
        gap = float((got - ref).abs().max()) / max(float(ref.abs().max()), 1e-30)
        worst = max(worst, gap)
        return gap <= rtol
    bad = _check_shapes(shapes, ok)
    return not bad, f"largest gap {worst:.2e} of the field max (allowed {rtol:.2e})" + (f"; over it at {bad[:5]}" if bad else "")


def probe_power(reads: int, reader=None) -> dict:
    read = reader or SM.read_pmic
    t0 = time.perf_counter()
    texts = [read() for _ in range(reads)]
    secs = time.perf_counter() - t0
    rails = SM.parse_pmic(texts[0])
    watts = [SM.total_watts(SM.parse_pmic(t)) for t in texts]
    return {"readable": True, "rails": list(rails), "rate_hz": reads / secs, "watts_median": statistics.median(watts),
            "raw_sample": texts[0]}


def main() -> None:
    cfg = load_cfg()
    torch.set_num_threads(int(cfg["threads"]))
    print(f"machine {platform.machine()}, {platform.platform()}, python {platform.python_version()}, "
          f"torch {torch.__version__}, threads {torch.get_num_threads()}")
    result: dict = {"torch": torch.__version__, "threads": torch.get_num_threads()}
    try:
        env = SM.env_info()
        result["cpu"] = env
        say("cpu", "info", f"{env['cpu_model']}; asimddp (int8 dot)={env['asimddp']}; i8mm (int8 matrix)={env['i8mm']}; "
                           f"asimdhp (fp16 vector)={env['asimdhp']}; fphp (fp16 scalar)={env['fphp']}; bf16={env['bf16']}")
    except RuntimeError as e:
        result["cpu"] = {"error": str(e)}
        say("cpu", "FAIL", str(e))
    shapes = shipped_shapes(HERE, cfg)
    result["equations_checked"] = {eq: [list(s) for s in v] for eq, v in shapes.items()}
    gen = torch.Generator().manual_seed(int(cfg["probe_seed"]))
    native = {}
    tests = {"int": lambda: probe_int8(shapes, gen), "int16": lambda: probe_int16(shapes, gen),
             "fp16": lambda: probe_half(torch.float16, shapes, gen, cfg["fp16_probe_rtol"]),
             "bf16": lambda: probe_half(torch.bfloat16, shapes, gen, cfg["fp16_probe_rtol"])}
    speed = {"int": lambda x, w: torch._int_mm(x.to(torch.int8), w.to(torch.int8).T.contiguous()),
             "int16": lambda x, w: x.long() @ w.long().T, "fp16": lambda x, w: x.half() @ w.half().T,
             "bf16": lambda x, w: x.bfloat16() @ w.bfloat16().T, "fp32": lambda x, w: x.float() @ w.float().T}
    m, k, n = cfg["probe_speed_shape"]
    xs = torch.randint(-127, 128, (m, k), generator=gen, dtype=torch.int8)
    ws = torch.randint(-127, 128, (n, k), generator=gen, dtype=torch.int8)
    result["speed_shape"], result["speed_ms"] = [m, k, n], {}
    for name, fn in speed.items():
        try:
            result["speed_ms"][name] = timed_ms(lambda: fn(xs, ws), int(cfg["probe_timing_repeats"]))
        except Exception as e:  # noqa: BLE001  a missing kernel is the answer
            result["speed_ms"][name] = f"{type(e).__name__}: {str(e)[:160]}"
    for name, test in tests.items():
        try:
            ok, detail = test()
        except Exception as e:  # noqa: BLE001  any failure means the path is not used
            ok, detail = False, f"{type(e).__name__}: {str(e)[:200]}"
        say(f"{name} matmul", "ok" if ok else "FAIL", f"{detail}; {result['speed_ms'][name]} ms ({m}x{k}x{n})")
        native[name] = ok
        result[f"{name}_detail"] = detail
    result["native"] = {k: native[k] for k in ("int", "int16", "fp16")}
    result["bf16_ok"] = native["bf16"]
    try:
        result["power"] = probe_power(int(cfg["probe_power_reads"]))
        p = result["power"]
        say("power chip", "ok", f"{len(p['rails'])} lines with V and A, {p['rate_hz']:.1f} readings/s, "
                                f"{p['watts_median']:.2f} W now")
    except Exception as e:  # noqa: BLE001  run_one refuses to run without power readings
        result["power"] = {"readable": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        say("power chip", "FAIL", result["power"]["error"])
    (HERE / cfg["probe_file"]).write_text(json.dumps(result, indent=1), encoding="utf-8")
    bar = "=" * 72
    print(f"\n{bar}\n  native paths: " + ", ".join(f"{k}={str(v).upper()}" for k, v in result["native"].items())
          + f"   power chip readable={str(result['power']['readable']).upper()}")
    for rows in sorted(HERE.glob("*/rows.json")):
        listing = json.loads(rows.read_text(encoding="utf-8"))
        print(f"  {listing['equation']}:")
        for r in listing["rows"]:
            path = N.choose_path(r["weight_format"], r["act_format"], result["native"])
            print(f"    {r['part']:5} {r['format']:9} weights {r['weight_format']:9} values {r['act_format']:9} "
                  f"tracks {r['track_format']:9} -> {path}")
    print(f"  written to {HERE / cfg['probe_file']}\n{bar}")


if __name__ == "__main__":
    main()
