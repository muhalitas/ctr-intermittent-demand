#!/bin/bash
# Additional benchmark models of Section 3.2 (TSB, ADIDA, IMAPA, LightGBM, XGBoost, two-stage, GRU).
#   bash M5/run/run_rivals.sh m5     -> M5 catalogue (after steps 2-3 of "Reproducing the M5 experiment")
#   bash M5/run/run_rivals.sh btm    -> primary catalogue (needs the proprietary Veri/filtered_data.csv)
# Environment variables: WORKERS (default 8), PY_CLASSICAL, PY_ML (Python interpreters, default python3).
# The classical part needs statsforecast and the ml part needs TensorFlow; on macOS TensorFlow and pyarrow
# (a dependency of statsforecast) cannot be loaded in one process, so the two parts can be given separate
# environments through PY_CLASSICAL and PY_ML. An interrupted run resumes when the same command is repeated.
set -e
cd "$(dirname "$0")"
DS="${1:-m5}"
WORKERS="${WORKERS:-8}"
PY_CLASSICAL="${PY_CLASSICAL:-python3}"
PY_ML="${PY_ML:-python3}"
OUT="../rivals_$DS"; mkdir -p "$OUT"
if command -v caffeinate >/dev/null 2>&1; then KEEP="caffeinate -dims"; else KEEP=""; fi

run_part() {   # part python
  local part=$1 py=$2 pids="" w
  echo "==> $part: $WORKERS workers -> $OUT"
  for w in $(seq 0 $((WORKERS-1))); do
    $KEEP "$py" -u -W ignore rivals.py --dataset "$DS" --part "$part" --out "$OUT" --workers "$WORKERS" --worker-id "$w" > "$OUT/log_${part}_w$w.txt" 2>&1 &
    pids="$pids $!"
  done
  trap "kill $pids 2>/dev/null; echo 'stopped - run the same command again to resume'; exit 1" INT TERM
  wait $pids
}

run_part classical "$PY_CLASSICAL"
run_part ml "$PY_ML"
"$PY_ML" -W ignore merge_rivals.py "$OUT"
