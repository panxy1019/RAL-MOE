#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE
SOURCE_ROOT="${PROJECT_ROOT}/Hopf/migrated_h4_expanded"
EXPERIMENT_ROOT="${PROJECT_ROOT}/CTDM_GALERKIN_ROM_R32_20260730"
PYTHON=/root/miniconda3/envs/pt_env/bin/python

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "GPU is occupied; refusing to start CTDM benchmarks." >&2
  exit 3
fi

COMMON=(
  --variant b4
  --baseline-trainer "${SOURCE_ROOT}/code/train_hopf_moe_expanded.py"
  --h4-trainer "${SOURCE_ROOT}/code/train_h4_expanded.py"
  --coefficient-view "${SOURCE_ROOT}/assets_r32/expanded_h4_trainval_r32.npz"
  --galerkin-path "${SOURCE_ROOT}/assets_r32/expanded_h4_trainonly_galerkin_r32.npz"
  --pressure-path "${SOURCE_ROOT}/assets_r32/expanded_h4_trainonly_pressure_r32.npz"
  --asset-manifest "${SOURCE_ROOT}/assets_r32/TRAINING_ASSET_MANIFEST.json"
  --fluctuation-contract "${SOURCE_ROOT}/trainonly_contract/trainonly_fluctuation_contract.npz"
  --output-root "${EXPERIMENT_ROOT}/benchmark"
  --seed 1248
  --grad-accum 1
  --benchmark-steps 8
  --benchmark-horizon 56
  --gpu-memory-fraction 0.42
  --swanlab-mode disabled
)

for micro_batch in 4 8 16; do
  if nice -n 10 "${PYTHON}" "${EXPERIMENT_ROOT}/code/train_ctdm_comparison.py" \
      --experiment-name "B4_K56_benchmark_mb${micro_batch}" \
      --micro-batch "${micro_batch}" \
      "${COMMON[@]}"; then
    return_code=0
  else
    return_code=$?
  fi
  printf '%s\n' "${return_code}" \
    > "${EXPERIMENT_ROOT}/benchmark/B4_K56_benchmark_mb${micro_batch}.rc"
  if [[ "${micro_batch}" == 4 && "${return_code}" != 0 ]]; then
    echo "Minimum benchmark batch failed; aborting preflight." >&2
    exit "${return_code}"
  fi
done
