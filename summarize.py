"""Build the three CSV tables of one equation from the per-process files in <equation>_results/ (re-runnable).

    python summarize.py --equation beam_eb_swimpde

Expects, for every seed listed in the equation's settings file, arms B, C and the arm probe_result.json names (D or E). Any
missing file is listed and the script stops: nothing is filled in.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml

import net as N

FIXED = ["equation", "seed", "arm", "rounding", "matmul_path", "n_points", "rss_after_load_mb", "peak_rss_mb", "working_mb",
         "weights_bytes_in_ram", "file_bytes", "fp32_equivalent_bytes", "int8_network_bytes", "temp_c_start", "temp_c_max",
         "temp_c_end", "throttled", "throttled_start", "throttled_end", "throttled_any", "arm_clock_hz", "arm_clock_hz_start",
         "arm_clock_hz_end", "arm_clock_hz_min", "threads"]


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    """Strict: every row must hold every header key (a missing one stops; nothing is written as an empty cell)."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i, r in enumerate(rows):
            missing = [k for k in header if k not in r]
            if missing:
                raise KeyError(f"{path.name}: row {i} ({r.get('arm')}, seed {r.get('seed')}) has no value for {missing}")
            w.writerow([r[k] for k in header])


def summarize(root: Path, cfg: dict, eq: str) -> int:
    res = root / (eq + cfg["results_suffix"])
    settings = N.load_settings(root / cfg["checkpoint_dir"] / eq)
    probe_path = root / cfg["probe_file"]
    if not probe_path.exists():
        raise FileNotFoundError(f"{probe_path} is missing: run probe_int8.py on this machine first")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    arms_expected = ["B", "C", probe["arm"]]
    missing = [f"{eq}__{s:02d}__{arm}.json" for s in settings["seeds"] for arm in arms_expected
               if not (res / f"{eq}__{s:02d}__{arm}.json").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} result file(s) missing in {res}: " + ", ".join(missing))
    runs = [json.loads((res / f"{eq}__{s:02d}__{arm}.json").read_text(encoding="utf-8"))
            for s in settings["seeds"] for arm in arms_expected]
    metric_keys = sorted({k for r in runs for k in r if k.startswith(("rel_l2_error_", "deriv_err_"))},
                         key=lambda k: (k.startswith("deriv_err_"), k))
    write_csv(res / f"{eq}_summary.csv", FIXED + metric_keys, runs)
    write_csv(res / f"{eq}_errors_by_column.csv", ["equation", "seed", "arm", "column", "order", "error", "n_points", "ref_rms"],
              [{"equation": eq, "seed": r["seed"], "arm": r["arm"], **e} for r in runs for e in r["errors"]])
    write_csv(res / f"{eq}_timing.csv", ["equation", "seed", "arm", "batch_size", "latency_ms_median", "latency_ms_q1",
                                         "latency_ms_q3", "points_per_s"],
              [{"equation": eq, "seed": r["seed"], "arm": r["arm"], **t} for r in runs for t in r["timing"]])
    shutil.copy(probe_path, res / cfg["probe_file"])
    (res / "environment.json").write_text(json.dumps(runs[0]["env"], indent=1), encoding="utf-8")
    return len(runs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equation", required=True)
    a = ap.parse_args()
    root = Path(__file__).resolve().parent
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    n = summarize(root, cfg, a.equation)
    print(f"wrote {a.equation}_summary.csv, {a.equation}_errors_by_column.csv, {a.equation}_timing.csv ({n} runs)")


if __name__ == "__main__":
    main()
