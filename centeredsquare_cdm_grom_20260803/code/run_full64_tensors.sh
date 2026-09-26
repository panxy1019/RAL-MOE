#!/usr/bin/env bash
set -euo pipefail

cd /home/ray/Desktop/centeredSquare/cdm_grom_code_20260803
dataset=/home/ray/Desktop/centeredSquare/cdm_grom_hopf_r64_v1
output="$dataset/pod/rom"
test -f "$dataset/POD_BUILD_MANIFEST.json"
test ! -e "$output"

nohup env \
  OPENBLAS_NUM_THREADS=4 \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 \
  python3 build_centered_square_rom_tensors.py \
    --dataset "$dataset" \
    --output-root "$output" \
    --rank full64:64:64 \
    --derivative-backend vtk \
    --vtk-center-atol 5e-6 \
    --blas-threads 4 \
    --skip-bundle \
  > build_full64_tensors.log 2>&1 &
echo $! > build_full64_tensors.pid
cat build_full64_tensors.pid
