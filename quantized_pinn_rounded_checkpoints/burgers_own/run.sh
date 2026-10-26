#!/usr/bin/env bash
# exp17: all runs of burgers_own on this machine, one fresh process per (seed, arm). Run from anywhere: bash quantized_pinn_rounded_checkpoints/burgers_own/run.sh
# Needs probe_result.json (python probe_int8.py first): it names arm D or E. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/../.."
LAST=$(python -c "import json; print(json.load(open('probe_result.json'))['arm'])")
for seed in 2 5 8 14 17; do
  for arm in B C "$LAST"; do
    python run_arm.py --equation burgers_own --seed "$seed" --arm "$arm"
  done
done
python summarize.py --equation burgers_own
