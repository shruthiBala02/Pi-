"""One fresh process = one (arm, stage, repeat): load an exp20 checkpoint (weights only), run the stage, read memory and
temperature, write the files into results/.

    python run_stage.py --arm J --stage accuracy --repeat 1       # arm J | G | none; stage empty | accuracy | b<size>

    empty     torch and the checkpoint loaded, no pass (the measured floor; arm none, `empty_checkpoint`)
    accuracy  all S1 points in chunks of `accuracy_chunk`, then the S3 points; writes the raw outputs
    b<size>   `warmup_runs` + `timed_runs` passes on the first <size> S1 points

Nothing is rounded, calibrated or trained here, and nothing is substituted: a missing file, a checksum that differs from
manifest.json or a missing temperature source stops the process with a message.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import psinn as P
import psinn_arms
import sysmon as SM

COLUMNS = ("u", "u_t", "u_x", "u_xx")


def load_cfg(root: Path) -> dict:
    return yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_hashes(ck_dir: Path, names: list[str], manifest: dict) -> dict[str, str]:
    """sha256 of each file; stops if one differs from manifest.json (the files collected from exp20)."""
    out = {}
    for n in names:
        h = sha256(ck_dir / n)
        if manifest["files"].get(n) != h:
            raise ValueError(f"{ck_dir / n}: sha256 {h} differs from manifest.json ({manifest['files'].get(n)}); "
                             "copy the collected files again")
        out[n] = h
    return out


def stages(cfg: dict) -> list[str]:
    return ["empty", "accuracy"] + [f"b{n}" for n in cfg["batch_sizes"]]


def run(arm: str, stage: str, repeat: int, root: Path, cfg: dict, sysmon=SM) -> dict:
    if stage not in stages(cfg):
        raise ValueError(f"unknown stage {stage!r}; stages: {stages(cfg)}")
    if (arm == "none") != (stage == "empty"):
        raise ValueError("the empty stage runs with arm none, every other stage with arm J or G")
    torch.set_num_threads(int(cfg["threads"]))
    ck_dir = root / cfg["checkpoint_dir"]
    manifest = json.loads((ck_dir / "manifest.json").read_text(encoding="utf-8"))
    ckpt = cfg["empty_checkpoint"] if arm == "none" else cfg["checkpoints"][arm]
    hashes = checked_hashes(ck_dir, [ckpt, "data.npz", "settings.json"], manifest)
    settings = P.load_settings(ck_dir)
    if settings["track_cast"] != cfg["track_cast"] or settings["orders"] != cfg["orders"]:
        raise ValueError(f"settings.json (track_cast {settings['track_cast']}, orders {settings['orders']}) differs from "
                         f"config.yaml ({cfg['track_cast']}, {cfg['orders']})")
    data = np.load(ck_dir / "data.npz")
    net = P.load(ck_dir / ckpt, settings)
    rss_after_load = sysmon.rss_mb()
    log = sysmon.ThermalLog(float(cfg["thermal_interval_s"]))
    log.start()
    out: dict = {"arm": arm, "stage": stage, "repeat": repeat, "checkpoint": ckpt, "quant": net.meta["quant"],
                 "rss_after_load_mb": rss_after_load}
    res_dir = root / cfg["results_dir"]
    res_dir.mkdir(exist_ok=True)
    stem = f"psinn__{arm}__{stage}__r{repeat}"
    s1 = torch.from_numpy(np.ascontiguousarray(data["s1_tx"])).float()
    if stage != "empty":
        fn = psinn_arms.ARMS[arm]
        orders = cfg["orders"]
    if stage == "accuracy":
        chunk = int(cfg["accuracy_chunk"])
        parts: dict[str, list] = {c: [] for c in COLUMNS}
        for i in range(0, s1.shape[0], chunk):
            f = fn(net, s1[i:i + chunk], orders)
            for c in COLUMNS:
                parts[c].append(f[c].detach().cpu().numpy())
        s3 = torch.from_numpy(np.ascontiguousarray(data["s3_tx"])).float()
        u3 = fn(net, s3, orders)["u"].detach().cpu().numpy()
        out["n_points"] = int(s1.shape[0])
        np.savez(res_dir / f"psinn__{arm}__s1__r{repeat}.npz", points=data["s1_tx"],
                 **{c: np.concatenate(parts[c]).astype(np.float32) for c in COLUMNS})
        np.savez(res_dir / f"psinn__{arm}__s3__r{repeat}.npz", points=data["s3_tx"], u=u3.astype(np.float32))
    elif stage.startswith("b"):
        n = int(stage[1:])
        if n > s1.shape[0]:
            raise ValueError(f"batch size {n} is larger than the S1 grid ({s1.shape[0]} points)")
        xb = s1[:n]
        for _ in range(int(cfg["warmup_runs"])):
            fn(net, xb, orders)
        out["rss_after_warmup_mb"] = sysmon.rss_mb()
        ms = []
        for _ in range(int(cfg["timed_runs"])):
            t0 = time.perf_counter()
            fn(net, xb, orders)
            ms.append((time.perf_counter() - t0) * 1e3)
        q = statistics.quantiles(ms, n=4)
        med = statistics.median(ms)
        out.update({"n_points": n, "latency_ms_all": ms, "latency_ms_median": med, "latency_ms_q1": q[0],
                    "latency_ms_q3": q[2], "points_per_s": n / (med / 1e3)})
    thermal = log.stop()
    peak = sysmon.peak_rss_mb()
    out.update({"peak_rss_mb": peak, "working_mb": peak - rss_after_load, "weights_bytes_in_ram": net.held_weight_bytes(),
                "file_bytes": os.path.getsize(ck_dir / ckpt), "int8_network_bytes": net.int8_network_bytes(),
                "fp32_equivalent_bytes": net.fp32_equivalent_bytes(), "threads_in_effect": torch.get_num_threads(),
                "sha256": hashes, "pi_code_commit": manifest["git_commit"], "env": sysmon.env_info()})
    out.update(thermal)
    out["clock_drop"] = thermal["arm_clock_hz_min"] < thermal["arm_clock_hz_start"]
    (res_dir / f"{stem}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    with open(res_dir / f"{stem}__thermal.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "temp_c", "arm_clock_hz"])
        w.writerows(log.rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["J", "G", "none"])
    ap.add_argument("--stage", required=True)
    ap.add_argument("--repeat", type=int, required=True)
    a = ap.parse_args()
    root = Path(__file__).resolve().parent
    out = run(a.arm, a.stage, a.repeat, root, load_cfg(root))
    print(f"arm {a.arm} {a.stage} r{a.repeat}: peak {out['peak_rss_mb']:.1f} MB, throttled_any={out['throttled_any']}, "
          f"clock_drop={out['clock_drop']}")


if __name__ == "__main__":
    main()
