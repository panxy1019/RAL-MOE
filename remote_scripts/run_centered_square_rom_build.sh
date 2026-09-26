#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-/home/ray/Desktop/centeredSquare/dataset_Re50_150_N100_npz}"
SCRIPT="${SCRIPT:-$DATASET/pod/build_centered_square_rom_tensors.py}"
THREADS="${THREADS:-4}"
NEIGHBORS="${NEIGHBORS:-32}"

if [ ! -f "$SCRIPT" ]; then
  echo "Missing ROM builder: $SCRIPT" >&2
  exit 1
fi

python3 "$SCRIPT" \
  --dataset "$DATASET" \
  --rank rank99:12:6 \
  --rank rank999:26:13 \
  --derivative-backend vtk \
  --neighbors "$NEIGHBORS" \
  --blas-threads "$THREADS"
