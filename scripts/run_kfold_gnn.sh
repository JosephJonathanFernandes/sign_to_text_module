#!/usr/bin/env bash
set -euo pipefail
# run_kfold_gnn.sh
# Usage: ./scripts/run_kfold_gnn.sh [folds] [epochs] [lr] [outdir]

FOLDS=${1:-5}
EPOCHS=${2:-8}
LR=${3:-1e-4}
OUTDIR=${4:-logs/$(date +%Y%m%d_%H%M%S)_kfold}

echo "Run settings: folds=${FOLDS}, epochs=${EPOCHS}, lr=${LR}"
mkdir -p "${OUTDIR}"
echo "Started at: $(date --iso-8601=seconds)" > "${OUTDIR}/run.log"

# Prefer local venv if present
if [ -f "venv/bin/activate" ]; then
  echo "Activating venv"
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

pip install -r requirements.txt

python train.py --kfold ${FOLDS} --epochs ${EPOCHS} --lr ${LR} 2>&1 | tee "${OUTDIR}/train.log"

echo "Finished at: $(date --iso-8601=seconds)" >> "${OUTDIR}/run.log"
