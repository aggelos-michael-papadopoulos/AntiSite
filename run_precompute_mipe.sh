#!/bin/bash
set -e
cd /home/panagiotis/PycharmProjects/Antisite
source /home/panagiotis/anaconda3/etc/profile.d/conda.sh
conda activate antisite
WEIGHTS=weights/MIPE_fold_1_best.pth
for SPLIT in TRAIN_VAL TEST; do
  echo "=== MIPE $SPLIT ===" | tee -a logs/mipe.log
  python -m antisite.teacher.precompute \
    --pdb-dir test_data/pdbs/MIPE/$SPLIT \
    --weights $WEIGHTS \
    --out-dir cache/MIPE/$SPLIT \
    --pattern '*receptor*.pdb' 2>&1 | tee -a logs/mipe.log
done
echo "=== MIPE DONE ===" | tee -a logs/mipe.log
