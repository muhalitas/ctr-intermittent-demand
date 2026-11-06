#!/bin/bash
# M5 CTR deneyi - Mac mini M4.
#   cd ~/Drive\'ım/doktora/M5/run
#   bash run_mac.sh
#
# Yarida kesersen (Ctrl-C) ayni komutu tekrar calistir: kaldigi yerden devam eder.
# Sonuclari birlestirmek icin:  bash run_mac.sh merge
set -e
cd "$(dirname "$0")"
# .venv Drive DISINDA tutulur - yoksa 1.5 GB paket her seferinde buluta senkronlanir
VENV="${VENV:-$HOME/.m5venv}"
WORKERS="${WORKERS:-8}"      # M4 icin 8; makineyi hic kullanmayacaksan 10
OUT="${OUT:-../sonuc_m5}"

if [ ! -d "$VENV" ]; then
  echo "==> sanal ortam kuruluyor: $VENV  (ilk seferde 3-5 dk)"
  python3 -m venv "$VENV"
  "$VENV"/bin/pip install --upgrade pip -q
  "$VENV"/bin/pip install -q tensorflow numpy pandas scikit-learn statsmodels scipy
fi

if [ "$1" = "merge" ]; then
  exec "$VENV"/bin/python merge_results.py "$OUT" --dataset m5
fi

"$VENV"/bin/python -c "import tensorflow as tf, sklearn, statsmodels; \
print('TF', tf.__version__, '| sklearn', sklearn.__version__, '| statsmodels', statsmodels.__version__)"

echo "==> baslıyor: $WORKERS isci  ->  $OUT"
caffeinate -dims "$VENV"/bin/python ctr_experiment.py \
    --dataset m5 --out "$OUT" --workers "$WORKERS" 2>&1 | tee -a ../m5_run.log
