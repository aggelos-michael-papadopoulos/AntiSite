#!/bin/bash
# Run ParaSurf (teacher) and Paraplume baseline evaluations on all 3 dataset test sets.
# Usage: bash scripts/run_baselines.sh [parasurf|paraplume|both]
set -eo pipefail
cd "$(dirname "$0")/.."                                   # repo root
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate antisite

MODE=${1:-both}
LOG=logs/baselines.log
mkdir -p logs

run_parasurf() {
  echo "=== ParaSurf: $1 ===" | tee -a $LOG
  python -m antisite.eval.eval_parasurf \
    --dataset "$1" \
    --pdb-dir  "$2" \
    --meta     "$3" \
    --cache    "$4" 2>&1 | tee -a $LOG
}

run_paraplume() {
  echo "=== Paraplume: $1 ===" | tee -a $LOG
  python -m antisite.eval.eval_paraplume \
    --dataset "$1" \
    --pdb-dir  "$2" \
    --meta     "$3" \
    --gpu 0 2>&1 | tee -a $LOG
}

DATA=test_data/pdbs

if [[ "$MODE" == "parasurf" || "$MODE" == "both" ]]; then
  run_parasurf "PECAN TEST"       $DATA/PECAN/TEST       $DATA/PECAN/preprocess/test_set.csv          cache/PECAN/TEST
  run_parasurf "Paragraph TEST"   $DATA/Expanded_dataset_Paragraph/Entire_antibody_experiment/TEST   $DATA/Expanded_dataset_Paragraph/Preprocess/test_set.csv   cache/Paragraph_expanded/TEST
  run_parasurf "MIPE TEST"        $DATA/MIPE/TEST        $DATA/MIPE/Preprocess/test_chains_info_reformatted.csv   cache/MIPE/TEST
fi

if [[ "$MODE" == "paraplume" || "$MODE" == "both" ]]; then
  run_paraplume "PECAN TEST"       $DATA/PECAN/TEST       $DATA/PECAN/preprocess/test_set.csv
  run_paraplume "Paragraph TEST"   $DATA/Expanded_dataset_Paragraph/Entire_antibody_experiment/TEST   $DATA/Expanded_dataset_Paragraph/Preprocess/test_set.csv
  run_paraplume "MIPE TEST"        $DATA/MIPE/TEST        $DATA/MIPE/Preprocess/test_chains_info_reformatted.csv
fi

echo "" | tee -a $LOG
echo "=== ALL DONE ===" | tee -a $LOG
