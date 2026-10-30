"""One fresh process = one row (equation, part, format): load the row's file, read the idle power, compute u and every
scored derivative on the equation's grid, time it, measure its energy, read memory and temperature, write the files
into results/<equation>/ (from code/pi/run_arm.py).

    python run_one.py --equation beam_eb_swimpde --part a1 --format INT4      (part fp32: --format FP32)
    python run_one.py --equation beam_eb_swimpde --todo                       (the rows with no result yet, base row first)

The row's value matmul path comes from probe_result.json only (net.choose_path); the FP32 row runs unrounded jets.
Latency: `warmup_runs` + `timed_runs` passes per batch size with the power sampler paused (as exp17, which had none).
Energy, per batch size (MLPerf Tiny rule): `energy_repeats` windows of at least `energy_window_s` seconds and
`energy_min_passes` passes, the median window reported; the idle power is read once, `energy_idle_s` seconds with the
power log running and no pass, before anything else. Nothing is rounded, calibrated or trained here, and nothing is
substituted: a missing file, probe result, power reading, reference or temperature source stops the run.
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

import jets as J
import metrics
import net as N
import sysmon as SM

HERE = Path(__file__).resolve().parent


def load_cfg(root: Path) -> dict:
    return yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))


def load_probe(root: Path, cfg: dict) -> dict:
    p = root / cfg["probe_file"]
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: run `python probe.py` on this machine first (it names the native paths)")
    probe = json.loads(p.read_text(encoding="utf-8"))
    if probe["torch"] != torch.__version__:
        raise RuntimeError(f"the probe ran with torch {probe['torch']}, this is {torch.__version__}: run `python probe.py` again")
    if not probe["power"]["readable"]:
        raise RuntimeError(f"the probe could not read the power chip ({probe['power'].get('error')}): no energy, no run")
    return probe


def rows_of(eq_dir: Path) -> dict:
    p = eq_dir / "rows.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: copy the export's folder for this equation here")
    return json.loads(p.read_text(encoding="utf-8"))


def result_stem(root: Path, cfg: dict, equation: str, part: str, fmt: str) -> Path:
    return root / cfg["results_dir"] / equation / f"{equation}__{part}__{fmt}"


def todo(root: Path, cfg: dict, equation: str) -> list[tuple[str, str]]:
    return [(r["part"], r["format"]) for r in rows_of(root / equation)["rows"]
            if not result_stem(root, cfg, equation, r["part"], r["format"]).with_suffix(".json").exists()]


def path_for(row: dict, probe: dict, equation: str) -> str:
    path = N.choose_path(row["weight_format"], row["act_format"], probe["native"])
    if path not in ("fp32", "decode_per_layer") and equation not in probe["equations_checked"]:
        raise RuntimeError(f"the probe did not check the layer shapes of {equation}: copy the folder in and run "
                           "`python probe.py` again")
    return path


def timed(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    return t


def quartiles(v: list[float]) -> tuple[float, float, float]:
    q = statistics.quantiles(v, n=4, method="inclusive")
    return q[0], statistics.median(v), q[2]


def run(equation: str, part: str, fmt: str, root: Path, cfg: dict, sysmon=SM) -> dict:
    torch.set_num_threads(int(cfg["threads"]))
    probe = load_probe(root, cfg)
    eq_dir = root / equation
    listing = rows_of(eq_dir)
    row = next((r for r in listing["rows"] if (r["part"], r["format"]) == (part, fmt)), None)
    if row is None:
        raise ValueError(f"{equation} has no row {part} {fmt} in rows.json")
    path = path_for(row, probe, equation)
    mode = "fp32" if path == "fp32" else "real"
    net = N.load(eq_dir, part, fmt, mode, listing["quant"], None if mode == "fp32" else path)
    data = np.load(eq_dir / "data.npz")
    cols = N.load_settings(eq_dir)["columns"]
    x = torch.from_numpy(np.ascontiguousarray(data["x"])).float()
    rss_after_load = sysmon.rss_mb()
    tlog = sysmon.ThermalLog(cfg["thermal_interval_s"])
    plog = sysmon.PowerLog(cfg["power_interval_s"])
    tlog.start()
    plog.start()
    plog.mark("idle")
    t_idle = time.perf_counter()
    time.sleep(float(cfg["energy_idle_s"]))
    idle_secs = time.perf_counter() - t_idle
    idle = plog.watts("idle")
    if not idle:
        raise RuntimeError("no power reading during the idle window")
    idle_w = statistics.median(idle)
    plog.mark("accuracy")
    stats: dict = {}
    chunk = int(cfg["accuracy_chunk"])
    with torch.no_grad():
        parts = [J.score(net, x[i:i + chunk], cols, stats) for i in range(0, x.shape[0], chunk)]
    fields = {c["name"]: torch.cat([p[c["name"]] for p in parts]).numpy() for c in cols}
    errors = metrics.column_errors(fields, data, cols)
    timing = []
    for bs in cfg["batch_sizes"]:
        if bs > x.shape[0]:
            raise ValueError(f"batch size {bs} is larger than the grid ({x.shape[0]} points)")
        xb = x[:bs]
        fn = lambda: J.score(net, xb, cols)          # noqa: E731
        plog.mark(f"timed_{bs}")
        plog.pause()                                    # latency timed without the power sampler (exp17's conditions)
        ms = timed(fn, int(cfg["warmup_runs"]), int(cfg["timed_runs"]))
        plog.resume()
        q1, med, q3 = quartiles(ms)
        windows = []
        for r in range(int(cfg["energy_repeats"])):
            phase = f"energy_{bs}_{r}"
            plog.mark(phase)
            t0, n = time.perf_counter(), 0
            while n < int(cfg["energy_min_passes"]) or time.perf_counter() - t0 < float(cfg["energy_window_s"]):
                fn()
                n += 1
            secs = time.perf_counter() - t0
            plog.mark(f"after_{phase}")
            windows.append({**SM.window_energy(plog.watts(phase), secs, n, idle_w), "passes": n, "seconds": secs})
        run_w = [w for r in range(int(cfg["energy_repeats"])) for w in plog.watts(f"energy_{bs}_{r}")]
        p1, pm, p3 = quartiles(run_w)
        e = statistics.median(w["energy_j_per_pass"] for w in windows)
        timing.append({"batch_size": bs, "latency_ms_median": med, "latency_ms_q1": q1, "latency_ms_q3": q3,
                       "points_per_s": bs / (med / 1e3), "power_w_run_median": pm, "power_w_run_q1": p1,
                       "power_w_run_q3": p3, "energy_j_per_pass": e,
                       "energy_j_per_pass_total": statistics.median(w["energy_j_per_pass_total"] for w in windows),
                       "energy_uj_per_point": e / bs * 1e6, "energy_windows": windows})
    plog.mark("end")
    plog.stop()
    thermal = tlog.stop()
    peak = sysmon.peak_rss_mb()
    read_secs = idle_secs + sum(w["seconds"] for t in timing for w in t["energy_windows"])      # the sampler ran
    n_read = len(idle) + sum(w["readings"] for t in timing for w in t["energy_windows"])       # (not paused) here
    out = {"equation": equation, "part": part, "format": fmt, "weight_format": row["weight_format"],
           "act_format": row["act_format"], "track_format": row["track_format"], "matmul_path": path,
           "probe_native": probe["native"], "probe_torch": probe["torch"], "n_points": int(x.shape[0]),
           "rss_after_load_mb": rss_after_load, "peak_rss_mb": peak, "working_mb": peak - rss_after_load,
           "weights_bytes": net.weights_bytes(), "weights_bytes_in_ram": net.held_weight_bytes(),
           "track_bytes_peak": stats["track_bytes_peak"], "act_bytes_peak": stats["act_bytes_peak"],
           "file_bytes": os.path.getsize(N.row_path(eq_dir, part, fmt)), "fp32_equivalent_bytes": net.fp32_equivalent_bytes(),
           "errors": errors, "timing": timing, "power_w_idle_median": idle_w, "power_samples": len(plog.rows),
           "power_sample_hz": n_read / read_secs, "power_rails": plog.rails,
           "env": sysmon.env_info(), "threads": int(cfg["threads"])}
    for e in errors:
        out[("rel_l2_error_" if e["order"] == 0 else "deriv_err_") + e["column"]] = e["error"]
    out.update(thermal)
    stem = result_stem(root, cfg, equation, part, fmt)
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stem.with_suffix(".npz"), x=data["x"], **{"f__" + k: v.astype(np.float32) for k, v in fields.items()})
    with open(str(stem) + "__thermal.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "temp_c", "arm_clock_hz"])
        w.writerows(tlog.rows)
    with open(str(stem) + "__power.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "phase"] + [f"{r}_{u}" for r in plog.rails for u in ("V", "A")] + ["total_w"])
        for t, ph, rails, watts in plog.rows:
            w.writerow([t, ph] + [v for r in plog.rails for v in rails[r]] + [watts])
    stem.with_suffix(".json").write_text(json.dumps(out, indent=1), encoding="utf-8")       # last: its presence = done
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equation", required=True)
    ap.add_argument("--part")
    ap.add_argument("--format")
    ap.add_argument("--todo", action="store_true", help="print the rows with no result yet, one 'part format' per line")
    a = ap.parse_args()
    cfg = load_cfg(HERE)
    if a.todo:
        for part, fmt in todo(HERE, cfg, a.equation):
            print(part, fmt)
        return
    if not (a.part and a.format):
        ap.error("--part and --format are needed (or --todo)")
    out = run(a.equation, a.part, a.format, HERE, cfg)
    e = {t["batch_size"]: t["energy_j_per_pass"] for t in out["timing"]}
    print(f"{a.equation} {a.part} {a.format}: path {out['matmul_path']}, peak {out['peak_rss_mb']:.1f} MB, "
          f"energy J/pass {e}, throttled_any={out['throttled_any']}")


if __name__ == "__main__":
    main()
