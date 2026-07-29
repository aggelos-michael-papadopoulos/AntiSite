#!/bin/bash
set -e
cd /home/panagiotis/PycharmProjects/Antisite
source /home/panagiotis/anaconda3/etc/profile.d/conda.sh
conda activate antisite
WEIGHTS=weights/PECAN_best.pth
for SPLIT in TRAIN VAL TEST; do
  echo "=== PECAN $SPLIT ===" | tee -a logs/pecan.log
  python -m antisite.teacher.precompute \
    --pdb-dir test_data/pdbs/PECAN/$SPLIT \
    --weights $WEIGHTS \
    --out-dir cache/PECAN/$SPLIT \
    --pattern '*receptor*.pdb' 2>&1 | tee -a logs/pecan.log
done
echo "=== PECAN DONE ===" | tee -a logs/pecan.log
