"""Probe: which integer matmul paths work on this machine (written for a Raspberry Pi 5, runs anywhere).

Run on the device:  python probe_int8.py            (needs only torch and numpy)
                    python probe_int8.py --rows 10000 --width 64 --repeats 20
It trains and rounds nothing. Every test is independent: a failure is printed, never hidden, and the next test still runs.
Paste the whole output back. A test is "exact" when its integer result equals an int64 reference bit for bit.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

import numpy as np
import torch


def say(name: str, status: str, detail: str = "") -> None:
    print(f"[{status:5}] {name}" + (f": {detail}" if detail else ""))


def timed(fn, repeats: int) -> float:
    fn()   # warm-up
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return statistics.median(t) * 1e3   # ms


HERE = os.path.dirname(os.path.abspath(__file__))


def pi_config() -> dict:
    import yaml
    with open(os.path.join(HERE, "config.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def args_out() -> str:
    """The probe result file named in config.yaml, next to this file (run_arm.py and summarize.py read it from there)."""
    return os.path.join(HERE, pi_config()["probe_file"])


def shipped_shapes() -> dict[str, list[tuple[int, int, int]]]:
    """{equation: [(rows, in, out), ...]} for every layer of every shipped network: torch._int_mm may use a different kernel for
    different shapes, so exactness is checked for all of them. The row counts are the batch sizes of config.yaml, the accuracy
    chunk and the last (shorter) accuracy chunk of the equation's grid. Stops when no equation folder is shipped."""
    cfg = pi_config()
    chunk = int(cfg["accuracy_chunk"])
    base = os.path.join(HERE, cfg["checkpoint_dir"])
    found: dict[str, list[tuple[int, int, int]]] = {}
    if os.path.isdir(base):
        for eq in sorted(os.listdir(base)):
            path = os.path.join(base, eq, "settings" + ".json")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as fh:
                layers = json.load(fh)["layers"]
            n = int(np.load(os.path.join(base, eq, "data.npz"))["x"].shape[0])
            rows = set(cfg["batch_sizes"]) | {min(n, chunk)} | ({n % chunk} if n > chunk and n % chunk else set())
            found[eq] = sorted({(m, a, b) for a, b in zip(layers[:-1], layers[1:]) for m in rows})
    if not found:
        raise RuntimeError(f"no equation folder with its settings file in {base}: copy the export folders there first; the verdict "
                           "depends on the layer shapes that will run, so it is not given without them")
    return found


def cpu_flags() -> None:
    try:
        text = open("/proc/cpuinfo").read()
        feats = next((ln.split(":", 1)[1].split() for ln in text.splitlines() if ln.startswith("Features")), None)
        model = next((ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("Model")), "unknown")
        if feats is None:
            say("cpu features", "info", f"model={model}; no Features line in /proc/cpuinfo: unknown")
        else:
            say("cpu features", "info", f"model={model}; asimddp (int8 dot product)={'asimddp' in feats}; "
                                        f"asimdhp (fp16 vector)={'asimdhp' in feats}; fphp (fp16 scalar)={'fphp' in feats}")
    except OSError as e:
        say("cpu features", "skip", f"/proc/cpuinfo not readable ({e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10000, help="points in the batch")
    ap.add_argument("--width", type=int, default=64, help="layer width (in = out)")
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    M, K, N = a.rows, a.width, a.width

    print(f"machine {platform.machine()}, {platform.platform()}, python {platform.python_version()}")
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, "
          f"quantized engine {torch.backends.quantized.engine!r}, supported {torch.backends.quantized.supported_engines}")
    cpu_flags()
    for mod in ("executorch", "torchao"):
        try:
            __import__(mod)
            say(f"import {mod}", "info", "available")
        except Exception as e:   # noqa: BLE001  any import failure is the answer
            say(f"import {mod}", "info", f"not available ({type(e).__name__})")

    g = torch.Generator().manual_seed(0)
    xi = torch.randint(-127, 128, (M, K), generator=g, dtype=torch.int8)
    wi = torch.randint(-127, 128, (N, K), generator=g, dtype=torch.int8)
    ref = (xi.to(torch.int64) @ wi.to(torch.int64).T)   # exact integer answer, [M, N]

    # 1. storage: an int8 tensor is 1 byte per value
    say("int8 storage", "ok" if wi.element_size() == 1 else "FAIL", f"{wi.numel()} values, {wi.element_size()} byte each")

    # 2. float32 matmul, the baseline every other time is compared with
    xf, wf = xi.float(), wi.float()
    ms = timed(lambda: xf @ wf.T, a.repeats)
    say("fp32 matmul (baseline)", "ok", f"{ms:.3f} ms; exact={bool(torch.equal((xf @ wf.T).long(), ref))} (fp32 is exact here: sums < 2^24)")

    # 3. dequantize one layer, then fp32 matmul (the always-available path)
    sc = torch.full((N, 1), 0.01)
    def deq():
        return xf @ (wi.to(torch.float32) * sc).T
    say("dequantize per call + fp32 matmul", "ok", f"{timed(deq, a.repeats):.3f} ms")

    # 4. torch._int_mm: int8 x int8 -> int32. This test decides arm E (integer matmul) or arm D.
    equations_checked = shipped_shapes()         # stops (no verdict) when no equation folder is shipped
    int_mm_exact, int_mm_detail = False, ""
    try:
        out = torch._int_mm(xi, wi.T.contiguous())
        ms = timed(lambda: torch._int_mm(xi, wi.T.contiguous()), a.repeats)
        int_mm_exact = bool(out.dtype == torch.int32 and torch.equal(out.long(), ref))
        bad = []
        for eq, eq_shapes in equations_checked.items():
            for (m_, k_, n_) in eq_shapes:
                xs = torch.randint(-127, 128, (m_, k_), generator=g, dtype=torch.int8)
                ws = torch.randint(-127, 128, (n_, k_), generator=g, dtype=torch.int8)
                o = torch._int_mm(xs, ws.T.contiguous())
                if not (o.dtype == torch.int32 and torch.equal(o.long(), xs.long() @ ws.long().T)):
                    bad.append((eq, m_, k_, n_))
        int_mm_exact = int_mm_exact and not bad
        n_shapes = sum(len(v) for v in equations_checked.values())
        int_mm_detail = (f"{ms:.3f} ms; dtype={out.dtype}; exact on the probe shape and on {n_shapes} shipped layer shapes of "
                         f"{len(equations_checked)} equations={int_mm_exact}" + (f"; NOT exact for (equation, rows, in, out) {bad}" if bad else ""))
        say("torch._int_mm", "ok" if int_mm_exact else "FAIL", int_mm_detail)
    except Exception as e:   # noqa: BLE001  any failure of the integer matmul means arm D, never E
        int_mm_exact = False
        int_mm_detail = f"{type(e).__name__}: {str(e)[:200]}"
        say("torch._int_mm", "FAIL", int_mm_detail)

    # 5. integer matmul through torch.matmul on int32 (slow but integer-exact; shows a floor)
    try:
        x32, w32 = xi.to(torch.int32), wi.to(torch.int32)
        out = x32 @ w32.T
        ms = timed(lambda: x32 @ w32.T, a.repeats)
        say("torch.matmul on int32", "ok", f"{ms:.3f} ms; exact={bool(torch.equal(out.long(), ref))}")
    except Exception as e:   # noqa: BLE001
        say("torch.matmul on int32", "FAIL", f"{type(e).__name__}: {str(e)[:160]}")

    # 6. torch.matmul directly on int8
    try:
        out = xi @ wi.T
        say("torch.matmul on int8", "ok", f"result dtype={out.dtype} (int8 output would overflow; not usable if int8)")
    except Exception as e:   # noqa: BLE001
        say("torch.matmul on int8", "FAIL", f"{type(e).__name__}: {str(e)[:120]}")

    # 7. the quantized linear operator, once per engine this build supports
    old = torch.backends.quantized.engine
    for eng in torch.backends.quantized.supported_engines:
        if eng == "none":
            continue
        try:
            torch.backends.quantized.engine = eng
            xs, ws = 0.02, 0.01
            qx = torch.quantize_per_tensor(xi.float() * xs, xs, 0, torch.qint8)
            qw = torch.quantize_per_tensor(wi.float() * ws, ws, 0, torch.qint8)
            packed = torch.ops.quantized.linear_prepack(qw, None)
            out_scale = float(xs * ws * K ** 0.5 * 40)
            out = torch.ops.quantized.linear(qx, packed, out_scale, 0)
            ms = timed(lambda: torch.ops.quantized.linear(qx, packed, out_scale, 0), a.repeats)
            deq_out = out.dequantize()
            fp = (xi.float() * xs) @ (wi.float() * ws).T
            err = float((deq_out - fp).abs().max())
            say(f"quantized.linear, engine {eng}", "ok",
                f"{ms:.3f} ms; max |difference| vs fp32 = {err:.3g} (one output step = {out_scale:.3g}; output is re-rounded to int8)")
        except Exception as e:   # noqa: BLE001
            say(f"quantized.linear, engine {eng}", "FAIL", f"{type(e).__name__}: {str(e)[:160]}")
    if old in torch.backends.quantized.supported_engines:     # the default can be 'none', which cannot be set back
        torch.backends.quantized.engine = old

    print("\nDone. 'exact=True' means the integer result equals the int64 reference.")
    arm = "E" if int_mm_exact else "D"
    result = {"arm": arm, "int_mm_exact": int_mm_exact, "int_mm_detail": int_mm_detail, "torch": torch.__version__,
              "rows": M, "width": K, "equations_checked": {eq: [list(s) for s in v] for eq, v in equations_checked.items()}}
    with open(args_out(), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    bar = "=" * 64
    print(f"\n{bar}\n  ARM E (integer matmul) = {str(int_mm_exact).upper()}   ->  run arm {arm} "
          f"({'integer matmul on the value path' if int_mm_exact else 'INT8 storage, one layer dequantized at a time'})\n"
          f"  written to {args_out()}\n{bar}")


if __name__ == "__main__":
    main()
