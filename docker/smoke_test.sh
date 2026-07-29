#!/usr/bin/env bash
# Fully offline smoke test — exercises BOTH inference modes on baked-in weights and
# the bundled sample structure. No network access required.
set -euo pipefail

CKPT=release/checkpoints/antisite_paragraph.pt
PARASURF=${PARASURF_WEIGHTS:-ParaSurf/Paragraph_expanded_entire_dataset_best.pth}

echo "=========================================================="
echo " AntiSite smoke test  (sequence-only, then 3D)"
echo "=========================================================="

echo
echo ">>> [1/2] Sequence-only mode (no structure)"
python antisite.py --weights "$CKPT" \
  -vh DVKLVQSGPGLVAPSQSLSITCTVSGFSLTTYGVSWVRQPPGKGLEWLGVIWGDGNTTYHSALISRLSISKDNSRSQVFLKLNSLHTDDTATYYCAGNYYGMDYWGQGTSVTVSS \
  -vl DIAMTQTTSSLSASLGQKVTISCRASQDIGNYLNWYQQKPDGTVRLLIYYTSRLHSGVPSRFSGSGSGTDYSLTISNLESEDIATYFCQNGGTNPWTFGGGTKLEVKR

echo
echo ">>> [2/2] 3D mode (bundled sample structure + ParaSurf)"
python antisite.py --weights "$CKPT" \
  --antibody sample/1BVK_receptor.pdb \
  --parasurf-weights "$PARASURF"

echo
echo "=========================================================="
echo " Smoke test OK — both modes ran with zero downloads."
echo "=========================================================="
