#!/bin/bash
set -e
cd /home/panagiotis/PycharmProjects/Antisite
source /home/panagiotis/anaconda3/etc/profile.d/conda.sh
conda activate antisite
WEIGHTS=weights/Paragraph_expanded_entire_dataset_best.pth
for SPLIT in TRAIN VAL TEST; do
  echo "=== Paragraph $SPLIT ===" | tee -a logs/paragraph.log
  python -m antisite.teacher.precompute \
    --pdb-dir test_data/pdbs/Expanded_dataset_Paragraph/Entire_antibody_experiment/$SPLIT \
    --weights $WEIGHTS \
    --out-dir cache/Paragraph_expanded/$SPLIT \
    --pattern '*receptor*.pdb' 2>&1 | tee -a logs/paragraph.log
done
echo "=== Paragraph DONE ===" | tee -a logs/paragraph.log
