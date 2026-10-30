#!/usr/bin/env bash
# The Pi run: every row of every equation in config.yaml, one fresh process each (python run_one.py), baseline first,
# then the table (python make_table.py -> results/final_table.csv). Run from anywhere: bash run_all.sh
# Needs probe_result.json: python probe.py first.
# A row whose results/<equation>/<equation>__<row>.json exists is skipped, so a re-run resumes where it stopped.
# A throttled row is run again, at most `throttle_repeats` times (config.yaml); the last run's files are kept and the
# rows still throttled are listed at the end. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")"
cfg() { python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['$1'])"; }
PROBE=$(cfg probe_file)
[ -f "$PROBE" ] || { echo "$PROBE is missing: run python probe.py first"; exit 1; }
REPEATS=$(cfg throttle_repeats)
RESULTS=$(cfg results_dir)
EQUATIONS=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('config.yaml'))['equations']))")
still_throttled=()
for eq in $EQUATIONS; do
  out=$(python run_one.py --equation "$eq" --todo)        # a plain assignment: set -e stops here if --todo fails
  todo=()
  if [ -n "$out" ]; then mapfile -t todo <<< "$out"; fi
  echo "== $eq: ${#todo[@]} rows to run ($(date +%H:%M))"
  for row in "${todo[@]}"; do
    tries=0
    while true; do
      python run_one.py --equation "$eq" --row "$row"
      json="$RESULTS/$eq/${eq}__${row}.json"
      throttled=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['throttled_any'])" "$json")
      if [ "$throttled" != "True" ]; then break; fi
      if [ "$tries" -ge "$REPEATS" ]; then still_throttled+=("$eq $row"); break; fi
      tries=$((tries + 1))
      echo "   $eq $row was throttled: run again ($tries of $REPEATS)"
      rm -f "$json"
    done
  done
done
if [ ${#still_throttled[@]} -eq 0 ]; then
  echo "all rows done, none throttled"
else
  echo "all rows done; still throttled after the repeats (${#still_throttled[@]}):"
  printf '  %s\n' "${still_throttled[@]}"
fi
python make_table.py
