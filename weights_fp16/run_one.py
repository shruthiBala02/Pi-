"""One fresh process = one row (equation, row id of config.yaml): load the row's network from networks_dir, compute u
and every scored derivative on the equation's whole grid in one inference, time it, read the memory it adds, write the
files into results/<equation>/ (from MLSys-QPINNs code/pi_ablations/run_one.py).

    python run_one.py --equation burgers_own --row jets_INT8
    python run_one.py --equation burgers_own --todo          (the row ids with no result yet, baseline first)

Rows: the baseline is FP32 autograd on the unrounded FP32 network (autograd.py); the others are jets (jets.py, real
mode) on the row file's rounded weights, held packed in their format and decoded one layer at a time, with the value and
the derivative tracks held in FP16 between layers (the row file's own value and track formats are replaced; only its
weights and biases are used). The value matmul path comes from probe_result.json only.

Order inside the process:
  1. wait until the CPU is at most `cool_c` degrees (sysmon.cool_down)
  2. `warmup_runs` passes (PyTorch's one-time set-up is done before anything is read)
  3. inference memory, `memory_repeats` times: the peak memory one pass adds (sysmon.inference_peak_bytes)
  4. runtime: `timing_blocks` blocks of `passes_per_block` passes, `block_pause_s` apart, with the temperature log on
  5. one more pass: its outputs are saved and scored against the reference
Nothing is rounded, calibrated or trained here, and nothing is substituted: a missing file or probe result stops the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

import sysmon as SM

SM.setup_allocator()                    # before torch or numpy make any array: large blocks are returned on free

import numpy as np                      # noqa: E402
import torch                            # noqa: E402

import autograd as AG                   # noqa: E402
import jets as J                        # noqa: E402
import metrics                          # noqa: E402
import net as N                         # noqa: E402
import probe as P                       # noqa: E402

HERE = Path(__file__).resolve().parent


def load_probe(root: Path, cfg: dict, equation: str) -> dict:
    p = root / cfg["probe_file"]
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: run `python probe.py` on this machine first")
    probe = json.loads(p.read_text(encoding="utf-8"))
    if probe["torch"] != torch.__version__:
        raise RuntimeError(f"the probe ran with torch {probe['torch']}, this is {torch.__version__}: run `python probe.py` again")
    if not probe["memory_ok"]:
        raise RuntimeError(f"the probe found the memory reading broken ({probe['memory_reading_detail']}): no run")
    if not probe["pi_ok"]:
        raise RuntimeError(f"the probe could not read the Pi's temperature or throttle word ({probe['pi_readings_detail']})")
    if equation not in probe["equations_checked"]:
        raise RuntimeError(f"the probe did not check {equation}: run `python probe.py` again")
    return probe


def row_of(cfg: dict, row_id: str) -> dict:
    row = next((r for r in cfg["rows"] if r["id"] == row_id), None)
    if row is None:
        raise ValueError(f"no row {row_id!r} in config.yaml (rows: {[r['id'] for r in cfg['rows']]})")
    return row


def load_net(eq_dir: Path, row: dict, path: str) -> N.Net:
    """The row's network: the FP32 row file as it is (baseline), or a rounded row file with its value and track formats
    replaced by the row's."""
    p = N.row_path(eq_dir, row["file_part"], row["file_format"])
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: networks_dir must be exp25's Pi folder with its equation folders")
    files = torch.load(p, map_location="cpu", weights_only=True)
    settings = N.load_settings(eq_dir)
    quant = json.loads((eq_dir / "rows.json").read_text(encoding="utf-8"))["quant"]
    meta = files["meta"]
    if (meta["equation"], meta["part"], meta["format"], meta["weight_format"]) != (
            settings["equation"], row["file_part"], row["file_format"], row["weights"]):
        raise ValueError(f"{p}: the file holds {meta['equation']} {meta['part']} {meta['format']} with "
                         f"{meta['weight_format']} weights, the row asks for {row}")
    if row["method"] == "autograd":
        if (row["weights"], row["value"], row["tracks"]) != ("FP32", "FP32", "FP32"):
            raise ValueError(f"the autograd baseline runs on the FP32 row only, got {row}")
        return N.Net(files, settings, "fp32", quant)
    files = {**files, "meta": {**meta, "act_format": row["value"], "track_format": row["tracks"]}}
    return N.Net(files, settings, "real", quant, path)


def result_stem(root: Path, cfg: dict, equation: str, row_id: str) -> Path:
    return root / cfg["results_dir"] / equation / f"{equation}__{row_id}"


def todo(root: Path, cfg: dict, equation: str) -> list[str]:
    return [r["id"] for r in cfg["rows"] if not result_stem(root, cfg, equation, r["id"]).with_suffix(".json").exists()]


def quartiles(v: list[float]) -> tuple[float, float, float]:
    q = statistics.quantiles(v, n=4, method="inclusive")
    return q[0], statistics.median(v), q[2]


def run(equation: str, row_id: str, root: Path, cfg: dict) -> dict:
    torch.set_num_threads(int(cfg["threads"]))
    if equation not in cfg["equations"]:
        raise ValueError(f"{equation} is not in config.yaml's equations {cfg['equations']}")
    probe = load_probe(root, cfg, equation)
    row = row_of(cfg, row_id)
    path = P.choose_path(row, probe["native_fp16"])
    eq_dir = P.networks_dir(root, cfg) / equation
    net = load_net(eq_dir, row, path)
    data = np.load(eq_dir / "data.npz")
    cols = N.load_settings(eq_dir)["columns"]
    x = torch.from_numpy(np.ascontiguousarray(data["x"])).float()
    if row["method"] == "autograd":
        f = AG.net_forward(net)
        fn = lambda: AG.columns(f, x, cols)          # noqa: E731
    else:
        fn = lambda: J.score(net, x, cols)           # noqa: E731

    waited, temp_before = SM.cool_down(float(cfg["cool_c"]), float(cfg["cool_max_wait_s"]))
    load_avg = os.getloadavg()[0]
    for _ in range(int(cfg["warmup_runs"])):
        fn()
    mem = [SM.inference_peak_bytes(fn) for _ in range(int(cfg["memory_repeats"]))]

    tlog = SM.ThermalLog(cfg["thermal_interval_s"])
    tlog.start()
    blocks = []
    for b in range(int(cfg["timing_blocks"])):
        if b:
            time.sleep(float(cfg["block_pause_s"]))
        t = []
        for _ in range(int(cfg["passes_per_block"])):
            t0 = time.perf_counter()
            fn()
            t.append((time.perf_counter() - t0) * 1e3)
        blocks.append(t)
    thermal = tlog.stop()

    stats: dict = {}
    with torch.no_grad():
        res = AG.columns(f, x, cols) if row["method"] == "autograd" else J.score(net, x, cols, stats)
    fields = {c["name"]: res[c["name"]].numpy() for c in cols}
    errors = metrics.column_errors(fields, data, cols)
    ms = [v for t in blocks for v in t]
    q1, med, q3 = quartiles(ms)
    model_bytes = net.held_weight_bytes() + sum(b.numel() * 4 for b in net.bias)
    out = {"equation": equation, "row": row_id, "method": row["method"], "weight_format": row["weights"],
           "value_format": row["value"], "track_format": row["tracks"], "matmul_path": path,
           "row_file": N.row_path(eq_dir, row["file_part"], row["file_format"]).name, "n_points": int(x.shape[0]),
           "outputs": [c["name"] for c in cols if c["order"] == 0],
           "runtime_ms_median": med, "runtime_ms_q1": q1, "runtime_ms_q3": q3,
           "runtime_ms_block_medians": [statistics.median(t) for t in blocks], "runtime_ms_all": ms,
           "inference_peak_bytes": int(statistics.median(mem)), "inference_peak_bytes_all": mem,
           "weights_bytes": net.weights_bytes(), "model_bytes_in_ram": model_bytes,
           "memory_bytes": model_bytes + int(statistics.median(mem)),
           "held_value_bytes_peak": stats.get("act_bytes_peak"), "held_track_bytes_peak": stats.get("track_bytes_peak"),
           "errors": errors, "cool_wait_s": waited, "temp_c_before": temp_before, "load_avg_1min": load_avg,
           "env": SM.env_info(), "threads": int(cfg["threads"]), "probe_native_fp16": probe["native_fp16"],
           "config": {k: cfg[k] for k in ("warmup_runs", "timing_blocks", "passes_per_block", "block_pause_s",
                                          "memory_repeats", "cool_c")}}
    for e in errors:
        out[("rel_l2_error_" if e["order"] == 0 else "deriv_err_") + e["column"]] = e["error"]
    out.update(thermal)
    stem = result_stem(root, cfg, equation, row_id)
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stem.with_suffix(".npz"), x=data["x"], **{"f__" + k: v.astype(np.float32) for k, v in fields.items()})
    with open(str(stem) + "__thermal.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "temp_c", "arm_clock_hz"])
        w.writerows(tlog.rows)
    stem.with_suffix(".json").write_text(json.dumps(out, indent=1), encoding="utf-8")       # last: its presence = done
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equation", required=True)
    ap.add_argument("--row")
    ap.add_argument("--todo", action="store_true", help="print the row ids with no result yet, one per line")
    a = ap.parse_args()
    cfg = P.load_cfg(HERE)
    if a.todo:
        for r in todo(HERE, cfg, a.equation):
            print(r)
        return
    if not a.row:
        ap.error("--row is needed (or --todo)")
    out = run(a.equation, a.row, HERE, cfg)
    print(f"{a.equation} {a.row}: path {out['matmul_path']}, runtime {out['runtime_ms_median']:.2f} ms "
          f"(q1 {out['runtime_ms_q1']:.2f}, q3 {out['runtime_ms_q3']:.2f}), inference memory "
          f"{out['inference_peak_bytes'] / 1e6:.2f} MB, max temp {out['temp_c_max']:.1f} C, "
          f"throttled_any={out['throttled_any']}")


if __name__ == "__main__":
    main()
