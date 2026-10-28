"""All processes of exp21, one after another, each a fresh Python (run_stage.py), in config.yaml's order, `repeats` times.
Stops at the first process that fails.

    python run_all.py        (run.sh)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import run_stage as RS


def plan(cfg: dict) -> list[tuple[str, str, int]]:
    """(arm, stage, repeat) in run order. Every stage named in `order` must be one run_stage knows."""
    known = RS.stages(cfg)
    out = []
    for r in range(1, int(cfg["repeats"]) + 1):
        for item in cfg["order"]:
            for st in item["stages"]:
                if st not in known:
                    raise ValueError(f"config order names stage {st!r}; stages: {known}")
                out.append((item["arm"], st, r))
    return out


def main() -> None:
    root = Path(__file__).resolve().parent
    steps = plan(RS.load_cfg(root))
    for i, (arm, st, r) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] arm {arm} {st} r{r}", flush=True)
        subprocess.run([sys.executable, str(root / "run_stage.py"), "--arm", arm, "--stage", st, "--repeat", str(r)],
                       cwd=root, check=True)
    print(f"done: {len(steps)} processes, files in {root / RS.load_cfg(root)['results_dir']}")


if __name__ == "__main__":
    main()
