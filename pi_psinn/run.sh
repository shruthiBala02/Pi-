#!/usr/bin/env bash
# exp21: all 27 processes on the Raspberry Pi 5 (see run_all.py). Run from anywhere: bash run.sh
set -euo pipefail
cd "$(dirname "$0")"
python run_all.py
