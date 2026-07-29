#!/bin/bash
# Precompute ParaSurf surface features (256-d) for the Paragraph (expanded) splits.
set -eo pipefail
cd "$(dirname "$0")/.."                                   # repo root
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate antisite

WEIGHTS=weights/Paragraph_expanded_entire_dataset_best.pth
mkdir -p logs
for SPLIT in TRAIN VAL TEST; do
  echo "=== Paragraph $SPLIT ===" | tee -a logs/paragraph.log
  python -m antisite.parasurf.precompute \
    --pdb-dir test_data/pdbs/Expanded_dataset_Paragraph/Entire_antibody_experiment/$SPLIT \
    --weights $WEIGHTS \
    --out-dir cache/Paragraph_expanded/$SPLIT \
    --pattern '*receptor*.pdb' 2>&1 | tee -a logs/paragraph.log
done
echo "=== Paragraph DONE ===" | tee -a logs/paragraph.log
