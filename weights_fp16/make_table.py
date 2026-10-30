"""The final table from results/: one line per (equation, row), formatted, written to results/final_table.csv and printed.

    python make_table.py          (after run_all.sh; it also runs at its end)

  Runtime (ms)                   median of every timed full-grid inference of the row
  Memory (bytes)                 the model as held in RAM (weights in their format, scales, float32 biases) + the peak
                                 memory one full-grid inference adds (sysmon.inference_peak_bytes)
  ... change vs baseline (%)     (row - baseline) / baseline x 100; + is more than the baseline
  Error vs reference (%)         relative L2 error of the solution output against the equation's reference (metrics.py)
  Change vs baseline (%)         relative L2 distance of the solution output from the FP32 autograd baseline's,
                                 ||row - baseline|| / ||baseline|| on every grid point
One group of the two error columns per solution output (Kovasznay: u, v, p). A missing result stops the table.
It reads results/ only (numpy and pyyaml, no torch): it also runs on a copy of results/ on another machine.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
NAMES = {"burgers_own": "Burgers", "kovasznay_hayford": "Kovasznay (Navier–Stokes)", "beam_eb_swimpde": "Beam",
         "black_scholes_bpfree": "Black–Scholes", "reaction_rho5_rathore": "Reaction ρ=5"}
MAX_OUTPUTS = 3


def pct(v: float) -> str:
    return f"{v * 100:.3g}%"


def change(v: float, base: float) -> str:
    return f"{(v - base) / base * 100:+.1f}%"


def distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def load(res: Path, eq: str, row_id: str) -> tuple[dict, np.lib.npyio.NpzFile]:
    stem = res / eq / f"{eq}__{row_id}"
    if not stem.with_suffix(".json").exists():
        raise FileNotFoundError(f"{stem}.json is missing: run bash run_all.sh (it resumes) before the table")
    return json.loads(stem.with_suffix(".json").read_text(encoding="utf-8")), np.load(stem.with_suffix(".npz"))


def build(root: Path, cfg: dict) -> tuple[list[str], list[list[str]]]:
    res = root / cfg["results_dir"]
    base_id = next(r["id"] for r in cfg["rows"] if r["method"] == "autograd")
    header = ["Equation", "Method", "Weight format", "Value format", "Track format", "Matmul path",
              "Weights storage (bytes)", "Runtime (ms)", "Runtime change vs baseline (%)", "Memory (bytes)",
              "Memory change vs baseline (%)", "Solution outputs"]
    for i in range(1, MAX_OUTPUTS + 1):
        header += [f"Error vs reference, output {i} (%)", f"Change vs baseline, output {i} (%)"]
    header += ["Throttled during timing"]
    lines = []
    for eq in cfg["equations"]:
        b, bf = load(res, eq, base_id)
        outputs = b["outputs"]
        if len(outputs) > MAX_OUTPUTS:
            raise ValueError(f"{eq} has {len(outputs)} solution outputs, the table has room for {MAX_OUTPUTS}")
        for row in cfg["rows"]:
            r, rf = load(res, eq, row["id"])
            is_base = row["id"] == base_id
            ref_err = {e["column"]: e["error"] for e in r["errors"]}
            line = [NAMES.get(eq, eq), "autograd (baseline)" if is_base else row["method"], r["weight_format"],
                    r["value_format"], r["track_format"], r["matmul_path"], f"{r['weights_bytes']:,}",
                    f"{r['runtime_ms_median']:,.3f}",
                    "baseline" if is_base else change(r["runtime_ms_median"], b["runtime_ms_median"]),
                    f"{r['memory_bytes']:,}", "baseline" if is_base else change(r["memory_bytes"], b["memory_bytes"]),
                    ", ".join(outputs)]
            for i in range(MAX_OUTPUTS):
                if i < len(outputs):
                    o = outputs[i]
                    line += [pct(ref_err[o]), "baseline" if is_base else pct(distance(rf["f__" + o], bf["f__" + o]))]
                else:
                    line += ["", ""]
            line += ["yes" if r["throttled_any"] else "no"]
            lines.append(line)
    return header, lines


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    header, lines = build(HERE, cfg)
    out = HERE / cfg["results_dir"] / cfg["table_file"]
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:          # BOM: Excel reads the dashes and ρ right
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(lines)
    widths = [max(len(h), *(len(l[i]) for l in lines)) for i, h in enumerate(header[:12])]
    for l in [header[:12]] + [x[:12] for x in lines]:
        print("  ".join(v.ljust(n) for v, n in zip(l, widths)))
    print(f"\nwritten to {out} ({len(lines)} lines; the error columns are in the file)")


if __name__ == "__main__":
    main()
