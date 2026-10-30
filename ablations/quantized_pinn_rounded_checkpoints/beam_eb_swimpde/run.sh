#!/usr/bin/env bash
# exp17: all runs of beam_eb_swimpde on this machine, one fresh process per (seed, arm). Run from anywhere: bash quantized_pinn_rounded_checkpoints/beam_eb_swimpde/run.sh
# Needs probe_result.json (python probe_int8.py first): it names arm D or E. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/../.."
LAST=$(python -c "import json; print(json.load(open('probe_result.json'))['arm'])")
for seed in 4 16 17 18 27; do
  for arm in B C "$LAST"; do
    python run_arm.py --equation beam_eb_swimpde --seed "$seed" --arm "$arm"
  done
done
python summarize.py --equation beam_eb_swimpde
