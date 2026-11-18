#!/bin/bash
# Run the full M5 experiment.
#   bash M5/run/run.sh            -> run (resumable: re-run the same command after an interruption)
#   bash M5/run/run.sh merge      -> merge the result parts and print the summary tables
# Environment variables: WORKERS (default 8), OUT (default M5/results_m5)
set -e
cd "$(dirname "$0")"
WORKERS="${WORKERS:-8}"
OUT="${OUT:-../results_m5}"

if [ "$1" = "merge" ]; then
  exec python3 merge_results.py "$OUT" --dataset m5
fi

python3 -c "import tensorflow as tf, sklearn, statsmodels; \
print('TF', tf.__version__, '| sklearn', sklearn.__version__, '| statsmodels', statsmodels.__version__)"

echo "==> starting: $WORKERS workers  ->  $OUT"
# keep a macOS machine awake during the run if caffeinate is available
if command -v caffeinate >/dev/null 2>&1; then KEEP="caffeinate -dims"; else KEEP=""; fi
$KEEP python3 ctr_experiment.py --dataset m5 --out "$OUT" --workers "$WORKERS" 2>&1 | tee -a ../m5_run.log
