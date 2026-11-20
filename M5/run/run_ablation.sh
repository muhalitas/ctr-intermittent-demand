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

echo "==> 1/2 primary catalogue (4,256 SKUs)"
caffeinate -dims "$VENV"/bin/python ctr_ablation.py --dataset primary --out ../ablation_primary --workers "$WORKERS"

echo "==> 2/2 M5 (random 5,000 series, seed 42)"
caffeinate -dims "$VENV"/bin/python ctr_ablation.py --dataset m5 --out ../ablation_m5 --workers "$WORKERS" --sample 5000

echo "==> done. Results: ../ablation_primary/ablation_summary.csv and ../ablation_m5/ablation_summary.csv"
