#!/usr/bin/env bash
set -euo pipefail

cd /home/ray/Desktop/centeredSquare/cdm_grom_code_20260803
dataset=/home/ray/Desktop/centeredSquare/cdm_grom_hopf_r64_v1
output="$dataset/evaluation/heldout"
operator="$dataset/cdm_operator_tailref/centeredsquare_hopf_cdm_operator_r11_u53.npz"
frozen="$dataset/frozen/FROZEN_METHOD.json"
test -f "$operator"
test -f "$frozen"
test ! -e "$output"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 evaluate_cdm_grom.py \
  --role heldout \
  --dataset-root "$dataset" \
  --operator-asset "$operator" \
  --canonical-cases /home/ray/Desktop/centeredSquare/three_regime_overlap_v1/config/canonical_cases.json \
  --output-dir "$output" \
  --frozen-method "$frozen" \
  --warmup-intervals 2 \
  --max-internal-step 0.1 \
  --consistency-step 0.2 \
  > heldout.log 2>&1
cat "$output/HELDOUT_METRICS.json"
