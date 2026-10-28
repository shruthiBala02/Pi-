#!/usr/bin/env bash
# exp17: all runs of cavity_re100_pinnacle on this machine, one fresh process per (seed, arm). Run from anywhere: bash quantized_pinn_rounded_checkpoints/cavity_re100_pinnacle/run.sh
# Needs probe_result.json (python probe_int8.py first): it names arm D or E. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/../.."
LAST=$(python -c "import json; print(json.load(open('probe_result.json'))['arm'])")
for seed in 5 8 11 18 24; do
  for arm in B C "$LAST"; do
    python run_arm.py --equation cavity_re100_pinnacle --seed "$seed" --arm "$arm"
  done
done
python summarize.py --equation cavity_re100_pinnacle
