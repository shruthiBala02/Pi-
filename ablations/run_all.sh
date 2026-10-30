#!/usr/bin/env bash
# exp25 on the Pi: every row of every equation folder here (<equation>/rows.json), one fresh process each
# (python run_one.py), in rows.json's order (the base row first). Run from anywhere: bash run_all.sh
# Needs probe_result.json: python probe.py first (it names the native paths and checks the power chip).
# A row whose results/<equation>/<equation>__<part>__<format>.json exists is skipped, so a re-run resumes where it stopped.
# A throttled row is run again, at most `throttle_repeats` times (config.yaml); the last run's files are kept and the
# rows still throttled are listed at the end. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")"
PROBE=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['probe_file'])")
[ -f "$PROBE" ] || { echo "$PROBE is missing: run python probe.py first"; exit 1; }
REPEATS=$(python -c "import yaml; print(int(yaml.safe_load(open('config.yaml'))['throttle_repeats']))")
RESULTS=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['results_dir'])")
still_throttled=()
for rows in */rows.json; do
  eq=$(dirname "$rows")
  out=$(python run_one.py --equation "$eq" --todo)        # a plain assignment: set -e stops here if --todo fails
  todo=()
  if [ -n "$out" ]; then mapfile -t todo <<< "$out"; fi
  echo "== $eq: ${#todo[@]} rows to run ($(date +%H:%M))"
  for line in "${todo[@]}"; do
    read -r part fmt <<< "$line"
    tries=0
    while true; do
      python run_one.py --equation "$eq" --part "$part" --format "$fmt"
      json="$RESULTS/$eq/${eq}__${part}__${fmt}.json"
      throttled=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['throttled_any'])" "$json")
      if [ "$throttled" != "True" ]; then break; fi
      if [ "$tries" -ge "$REPEATS" ]; then still_throttled+=("$eq $part $fmt"); break; fi
      tries=$((tries + 1))
      echo "   $eq $part $fmt was throttled: run again ($tries of $REPEATS)"
    done
  done
done
if [ ${#still_throttled[@]} -eq 0 ]; then
  echo "all rows done, none throttled"
else
  echo "all rows done; still throttled after the repeats (${#still_throttled[@]}):"
  printf '  %s\n' "${still_throttled[@]}"
fi
