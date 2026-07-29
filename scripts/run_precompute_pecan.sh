#!/bin/bash
# Precompute ParaSurf surface features (256-d) for the PECAN splits.
set -eo pipefail
cd "$(dirname "$0")/.."                                   # repo root
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate antisite

WEIGHTS=weights/PECAN_best.pth
mkdir -p logs
for SPLIT in TRAIN VAL TEST; do
  echo "=== PECAN $SPLIT ===" | tee -a logs/pecan.log
  python -m antisite.parasurf.precompute \
    --pdb-dir test_data/pdbs/PECAN/$SPLIT \
    --weights $WEIGHTS \
    --out-dir cache/PECAN/$SPLIT \
    --pattern '*receptor*.pdb' 2>&1 | tee -a logs/pecan.log
done
echo "=== PECAN DONE ===" | tee -a logs/pecan.log
