#!/usr/bin/env bash
# exp21: Jurj's Ψ-NN (exp20's N1) on this machine, all 27 processes of pi_psinn/run_all.py. Run from anywhere: bash quantized_pinn_rounded_checkpoints/jurej/run.sh
# Uses its own code in pi_psinn/ (not run_arm.py), which reads its own copy of this folder's files from pi_psinn/checkpoints/.
# Results go to pi_psinn/results/. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/../.."
python pi_psinn/run_all.py
