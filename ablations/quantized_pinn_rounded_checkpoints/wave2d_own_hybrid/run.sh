#!/usr/bin/env bash
# exp17: all runs of wave2d_own_hybrid on this machine, one fresh process per (seed, arm). Run from anywhere: bash quantized_pinn_rounded_checkpoints/wave2d_own_hybrid/run.sh
# Needs probe_result.json (python probe_int8.py first): it names arm D or E. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/../.."
LAST=$(python -c "import json; print(json.load(open('probe_result.json'))['arm'])")
for seed in 0 12 18 19 23; do
  for arm in B C "$LAST"; do
    python run_arm.py --equation wave2d_own_hybrid --seed "$seed" --arm "$arm"
  done
done
python summarize.py --equation wave2d_own_hybrid
