#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE
SOURCE_ROOT="${PROJECT_ROOT}/Hopf/migrated_h4_expanded"
EXPERIMENT_ROOT="${PROJECT_ROOT}/CTDM_GALERKIN_ROM_R32_20260730"
PYTHON=/root/miniconda3/envs/pt_env/bin/python

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "GPU is occupied; refusing to start CTDM smoke tests." >&2
  exit 3
fi

COMMON=(
  --baseline-trainer "${SOURCE_ROOT}/code/train_hopf_moe_expanded.py"
  --h4-trainer "${SOURCE_ROOT}/code/train_h4_expanded.py"
  --coefficient-view "${SOURCE_ROOT}/assets_r32/expanded_h4_trainval_r32.npz"
  --galerkin-path "${SOURCE_ROOT}/assets_r32/expanded_h4_trainonly_galerkin_r32.npz"
  --pressure-path "${SOURCE_ROOT}/assets_r32/expanded_h4_trainonly_pressure_r32.npz"
  --asset-manifest "${SOURCE_ROOT}/assets_r32/TRAINING_ASSET_MANIFEST.json"
  --fluctuation-contract "${SOURCE_ROOT}/trainonly_contract/trainonly_fluctuation_contract.npz"
  --output-root "${EXPERIMENT_ROOT}/smoke"
  --smoke-only
  --gpu-memory-fraction 0.42
  --swanlab-mode disabled
)

for variant in b1 b2 b3 b4; do
  nice -n 10 "${PYTHON}" "${EXPERIMENT_ROOT}/code/train_ctdm_comparison.py" \
    --variant "${variant}" "${COMMON[@]}"
done
