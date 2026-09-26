#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE
SOURCE_ROOT="${PROJECT_ROOT}/Hopf/migrated_h4_expanded"
EXPERIMENT_ROOT="${PROJECT_ROOT}/CTDM_GALERKIN_ROM_R32_20260730"
PYTHON=/root/miniconda3/envs/pt_env/bin/python

read -r selected_micro_batch selected_grad_accum < <(
  "${PYTHON}" "${EXPERIMENT_ROOT}/code/select_benchmark_batch.py" \
    --benchmark-root "${EXPERIMENT_ROOT}/benchmark" \
    --output "${EXPERIMENT_ROOT}/benchmark/BATCH_SELECTION.json"
)
export CTDM_MICRO_BATCH="${selected_micro_batch}"
export CTDM_GRAD_ACCUM="${selected_grad_accum}"

bash "${EXPERIMENT_ROOT}/code/run_single_seed_all.sh"

for model_dir in \
  B1_Deep_FNN_H3_seed1248 \
  B2_Deep_FNN_current_seed1248 \
  B3_Discrete_KDA_FNN_seed1248 \
  B4_CTDM_Galerkin_ROM_seed1248
do
  run_dir="${EXPERIMENT_ROOT}/single_seed/${model_dir}"
  if [[ -f "${run_dir}/best_validation.pt" ]]; then
    checkpoint="${run_dir}/best_validation.pt"
  else
    checkpoint="${run_dir}/final_training.pt"
  fi
  nice -n 10 "${PYTHON}" "${EXPERIMENT_ROOT}/code/analyze_ctdm_checkpoint.py" \
    --split validation \
    --checkpoint "${checkpoint}" \
    --trainer "${EXPERIMENT_ROOT}/code/train_ctdm_comparison.py" \
    --baseline-trainer "${SOURCE_ROOT}/code/train_hopf_moe_expanded.py" \
    --h4-trainer "${SOURCE_ROOT}/code/train_h4_expanded.py" \
    --coefficient-view "${SOURCE_ROOT}/assets_r32/expanded_h4_trainval_r32.npz" \
    --galerkin-path "${SOURCE_ROOT}/assets_r32/expanded_h4_trainonly_galerkin_r32.npz" \
    --pressure-path "${SOURCE_ROOT}/assets_r32/expanded_h4_trainonly_pressure_r32.npz" \
    --asset-manifest "${SOURCE_ROOT}/assets_r32/TRAINING_ASSET_MANIFEST.json" \
    --fluctuation-contract "${SOURCE_ROOT}/trainonly_contract/trainonly_fluctuation_contract.npz" \
    --output-dir "${EXPERIMENT_ROOT}/validation/${model_dir}" \
    --windows-per-re 16 \
    --gpu-memory-fraction 0.42
done
