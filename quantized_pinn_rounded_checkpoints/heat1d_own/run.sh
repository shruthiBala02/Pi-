#!/usr/bin/env bash
# exp17: all runs of heat1d_own on this machine, one fresh process per (seed, arm). Run from anywhere: bash quantized_pinn_rounded_checkpoints/heat1d_own/run.sh
# Needs probe_result.json (python probe_int8.py first): it names arm D or E. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/../.."
LAST=$(python -c "import json; print(json.load(open('probe_result.json'))['arm'])")
for seed in 0 9 13 14 20; do
  for arm in B C "$LAST"; do
    python run_arm.py --equation heat1d_own --seed "$seed" --arm "$arm"
  done
done
python summarize.py --equation heat1d_own
