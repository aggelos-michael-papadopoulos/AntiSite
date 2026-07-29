#!/bin/bash
# Precompute 6-PLM per-residue embeddings for every antibody example across all splits.
set -eo pipefail
cd /home/panagiotis/PycharmProjects/Antisite
source /home/panagiotis/anaconda3/etc/profile.d/conda.sh
conda activate antisite

BS=${1:-16}
LOG=logs/embeddings.log
mkdir -p logs

run() {
  local name="$1"; local ex="$2"; local out="$3"
  echo "=== $name ===" | tee -a $LOG
  python -m antisite.data.build_embeddings \
      --examples-dir "$ex" --out-dir "$out" --batch-size $BS --gpu 0 2>&1 | tee -a $LOG
}

for split in TEST VAL TRAIN; do
  run "PECAN $split"     "examples/PECAN/$split"     "embeddings/PECAN/$split"
done
for split in TEST VAL TRAIN; do
  run "Paragraph $split" "examples/Paragraph_expanded/$split" "embeddings/Paragraph_expanded/$split"
done
run "MIPE TEST"      "examples/MIPE/TEST"      "embeddings/MIPE/TEST"
run "MIPE TRAIN_VAL" "examples/MIPE/TRAIN_VAL" "embeddings/MIPE/TRAIN_VAL"

echo "=== ALL EMBEDDINGS DONE ===" | tee -a $LOG
