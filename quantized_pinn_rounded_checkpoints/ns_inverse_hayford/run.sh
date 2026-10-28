#!/usr/bin/env bash
# exp17: all runs of ns_inverse_hayford on this machine, one fresh process per (seed, arm). Run from anywhere: bash quantized_pinn_rounded_checkpoints/ns_inverse_hayford/run.sh
# Needs probe_result.json (python probe_int8.py first): it names arm D or E. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/../.."
LAST=$(python -c "import json; print(json.load(open('probe_result.json'))['arm'])")
for seed in 2 6 8 23 27; do
  for arm in B C "$LAST"; do
    python run_arm.py --equation ns_inverse_hayford --seed "$seed" --arm "$arm"
  done
done
python summarize.py --equation ns_inverse_hayford
