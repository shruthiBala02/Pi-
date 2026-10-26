"""Frozen evaluation, copied from pinnbench/exp02/collect.py `rel_l2`: the points where the reference is finite (and, where a
reference file marks some points unusable, only the usable ones), ||pred - ref|| / ||ref|| in float64; NaN when ||ref|| is 0.
A NaN stays NaN."""

from __future__ import annotations

import numpy as np


def rel_l2(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> tuple[float, int, float]:
    """(relative L2 error, number of points used, reference rms) on the points where `mask` holds and ref is finite."""
    m = mask & np.isfinite(ref)
    n = int(m.sum())
    nb = float(np.linalg.norm(ref[m].astype(np.float64)))
    rms = nb / np.sqrt(n) if n else float("nan")
    err = float(np.linalg.norm(pred[m].astype(np.float64) - ref[m].astype(np.float64)) / nb) if nb > 0 else float("nan")
    return err, n, rms


def column_errors(fields: dict, data, cols: list[dict]) -> list[dict]:
    """One row per scored column: name, order, error, points used, reference rms. A column without a reference in the data
    file stops with a message (the export guarantees one for every scored column)."""
    rows = []
    for c in cols:
        name = c["name"]
        if f"ref__{name}" not in data.files:
            raise KeyError(f"column {name} has no reference in data.npz: the equation folder is broken or mixed up")
        err, n, rms = rel_l2(np.asarray(fields[name], dtype=np.float64), data[f"ref__{name}"], data[f"mask__{name}"])
        rows.append({"column": name, "order": c["order"], "error": err, "n_points": n, "ref_rms": rms})
    return rows
