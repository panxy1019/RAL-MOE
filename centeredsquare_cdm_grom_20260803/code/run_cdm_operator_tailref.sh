#!/usr/bin/env bash
set -euo pipefail

cd /home/ray/Desktop/centeredSquare/cdm_grom_code_20260803
dataset=/home/ray/Desktop/centeredSquare/cdm_grom_hopf_r64_v1
tensor_dir="$dataset/pod/rom/full64_ru64_rp64"
output="$dataset/cdm_operator_tailref"
test -f "$tensor_dir/manifest.json"
test ! -e "$output"

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
python3 build_cdm_operator_assets.py \
  --dataset-root "$dataset" \
  --tensor-dir "$tensor_dir" \
  --output-dir "$output" \
  --resolved-rank 11 \
  --stability-margin 0.02 \
  --tail-snapshots 32 \
  --reference-policy train_tail_mean \
  > build_cdm_operator_tailref.log 2>&1
cat "$output/CDM_OPERATOR_MANIFEST.json"
