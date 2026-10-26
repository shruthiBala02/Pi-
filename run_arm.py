"""One fresh process = one (equation, seed, arm): load the shipped INT8 network, compute u and every scored derivative on the
equation's grid, time it, read memory and temperature, write the files into <equation>_results/.

    python run_arm.py --equation beam_eb_swimpde --seed 18 --arm B      # B C D E (D or E: the one probe_result.json names)

Nothing is rounded, calibrated or trained here, and nothing is substituted: a missing file, probe result, reference or
temperature source stops the run with a message.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import arms
import metrics
import net as N
import sysmon as SM

MATMUL_PATH = {"B": "fp32", "C": "fp32", "D": "dequant_per_layer", "E": "int"}


def load_cfg(root: Path) -> dict:
    return yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))


def check_probe(root: Path, cfg: dict, arm: str, equation: str) -> None:
    if arm not in ("D", "E"):
        return
    p = root / cfg["probe_file"]
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: run `python probe_int8.py` on this machine first (it decides D or E)")
    probe = json.loads(p.read_text(encoding="utf-8"))
    if probe["arm"] != arm:
        raise RuntimeError(f"the probe chose arm {probe['arm']}, not {arm}; run arm {probe['arm']}")
    if arm == "E" and equation not in probe["equations_checked"]:
        raise RuntimeError(f"the probe did not check the layer shapes of {equation} (it checked {sorted(probe['equations_checked'])}); "
                           "copy the equation folder in and run `python probe_int8.py` again")


def timed(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    return t


def accuracy_pass(arm_fn, net, x: torch.Tensor, cols: list[dict], chunk: int) -> dict:
    parts: dict[str, list] = {c["name"]: [] for c in cols}
    for i in range(0, x.shape[0], chunk):
        f = arm_fn(net, x[i:i + chunk], cols)
        for k, v in f.items():
            parts[k].append(v.detach().cpu().numpy())
    return {k: np.concatenate(v) for k, v in parts.items()}


def run(equation: str, seed: int, arm: str, root: Path, cfg: dict, sysmon=SM) -> dict:
    torch.set_num_threads(int(cfg["threads"]))
    check_probe(root, cfg, arm, equation)
    eq_dir = root / cfg["checkpoint_dir"] / equation
    data = np.load(eq_dir / "data.npz")
    settings = N.load_settings(eq_dir)
    cols = settings["columns"]
    net = N.load(eq_dir, seed, arms.ROUNDING[arm], arms.STORAGE[arm], cfg["int8_max"])
    arm_fn = arms.ARMS[arm]
    x = torch.from_numpy(np.ascontiguousarray(data["x"])).float()
    rss_after_load = sysmon.rss_mb()
    log = sysmon.ThermalLog(cfg["thermal_interval_s"])
    log.start()
    fields = accuracy_pass(arm_fn, net, x, cols, int(cfg["accuracy_chunk"]))
    errors = metrics.column_errors(fields, data, cols)
    timing = []
    for bs in cfg["batch_sizes"]:
        if bs > x.shape[0]:
            raise ValueError(f"batch size {bs} is larger than the grid ({x.shape[0]} points)")
        xb = x[:bs]
        ms = timed(lambda: arm_fn(net, xb, cols), int(cfg["warmup_runs"]), int(cfg["timed_runs"]))
        q = statistics.quantiles(ms, n=4, method="inclusive")
        med = statistics.median(ms)
        timing.append({"batch_size": bs, "latency_ms_median": med, "latency_ms_q1": q[0], "latency_ms_q3": q[2],
                       "points_per_s": bs / (med / 1e3)})
    thermal = log.stop()
    peak = sysmon.peak_rss_mb()
    ckpt = N.ckpt_path(eq_dir, seed, arms.ROUNDING[arm])
    out = {"equation": equation, "seed": seed, "arm": arm, "rounding": arms.ROUNDING[arm], "matmul_path": MATMUL_PATH[arm],
           "n_points": int(x.shape[0]), "rss_after_load_mb": rss_after_load, "peak_rss_mb": peak,
           "working_mb": peak - rss_after_load, "weights_bytes_in_ram": net.held_weight_bytes(),
           "file_bytes": os.path.getsize(ckpt), "fp32_equivalent_bytes": net.fp32_equivalent_bytes(),
           "int8_network_bytes": net.int8_network_bytes(), "errors": errors, "timing": timing, "env": sysmon.env_info(),
           "threads": int(cfg["threads"])}
    for e in errors:
        out[("rel_l2_error_" if e["order"] == 0 else "deriv_err_") + e["column"]] = e["error"]
    out.update(thermal)
    res_dir = root / (equation + cfg["results_suffix"])
    res_dir.mkdir(exist_ok=True)
    stem = res_dir / f"{equation}__{seed:02d}__{arm}"
    stem.with_suffix(".json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    np.savez(stem.with_suffix(".npz"), x=data["x"], **{"f__" + k: v.astype(np.float32) for k, v in fields.items()})
    with open(str(stem) + "__thermal.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "temp_c", "arm_clock_hz"])
        w.writerows(log.rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equation", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", required=True, choices=sorted(arms.ARMS))
    a = ap.parse_args()
    root = Path(__file__).resolve().parent
    out = run(a.equation, a.seed, a.arm, root, load_cfg(root))
    print(f"{a.equation} seed {a.seed} arm {a.arm}: peak {out['peak_rss_mb']:.1f} MB, "
          f"throttled_any={out['throttled_any']}, wrote {a.equation}_results/")


if __name__ == "__main__":
    main()
