#!/bin/bash
# Precompute ParaSurf surface features (256-d) for the MIPE splits (5-fold train_val + test).
set -eo pipefail
cd "$(dirname "$0")/.."                                   # repo root
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate antisite

WEIGHTS=weights/MIPE_fold_1_best.pth
mkdir -p logs
for SPLIT in TRAIN_VAL TEST; do
  echo "=== MIPE $SPLIT ===" | tee -a logs/mipe.log
  python -m antisite.parasurf.precompute \
    --pdb-dir test_data/pdbs/MIPE/$SPLIT \
    --weights $WEIGHTS \
    --out-dir cache/MIPE/$SPLIT \
    --pattern '*receptor*.pdb' 2>&1 | tee -a logs/mipe.log
done
echo "=== MIPE DONE ===" | tee -a logs/mipe.log
