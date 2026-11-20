#!/bin/bash
# CTR ablation study - Mac mini M4.
#   cd M5/run
#   bash run_ablation.sh
#
# If interrupted (Ctrl-C), run the same command again: it resumes where it stopped.
set -e
cd "$(dirname "$0")"
VENV="${VENV:-$HOME/.m5venv}"
WORKERS="${WORKERS:-8}"

if [ ! -d "$VENV" ]; then
  echo "==> creating virtual environment: $VENV (first time 3-5 min)"
  python3 -m venv "$VENV"
  "$VENV"/bin/pip install --upgrade pip -q
  "$VENV"/bin/pip install -q tensorflow numpy pandas scikit-learn statsmodels scipy
fi
"$VENV"/bin/python -c "import tensorflow as tf; print('TF', tf.__version__)"

# keep a macOS machine awake during the run if caffeinate is available
if command -v caffeinate >/dev/null 2>&1; then KEEP="caffeinate -dims"; else KEEP=""; fi

# The primary catalogue is proprietary; its ablation results are released in ../results_ablation/primary/.
echo "==> M5 (random 5,000 series, seed 42); requires M5/prep outputs (run M5/prep/m5_prepare.py first)"
$KEEP "$VENV"/bin/python ctr_ablation.py --dataset m5 --out ../ablation_m5 --workers "$WORKERS" --sample 5000

echo "==> done. Results: ../ablation_m5/ablation_summary.csv"
