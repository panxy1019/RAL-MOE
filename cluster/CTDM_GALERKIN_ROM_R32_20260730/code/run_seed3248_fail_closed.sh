#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE
SOURCE_ROOT="${PROJECT_ROOT}/Hopf/migrated_h4_expanded"
EXPERIMENT_ROOT="${PROJECT_ROOT}/CTDM_GALERKIN_ROM_R32_20260730"
PYTHON=/root/miniconda3/envs/pt_env/bin/python
SEED=3248

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "GPU is occupied; refusing to start seed-3248 pipeline." >&2
  exit 3
fi

read -r MICRO_BATCH GRAD_ACCUM < <(
  "${PYTHON}" "${EXPERIMENT_ROOT}/code/select_benchmark_batch.py" \
    --benchmark-root "${EXPERIMENT_ROOT}/benchmark" \
    --output "${EXPERIMENT_ROOT}/benchmark/BATCH_SELECTION.json"
)

declare -A MODEL_DIR=(
  [b1]=B1_Deep_FNN_H3
  [b2]=B2_Deep_FNN_current
  [b3]=B3_Discrete_KDA_FNN
  [b4]=B4_CTDM_Galerkin_ROM
)

COMMON=(
  --baseline-trainer "${SOURCE_ROOT}/code/train_hopf_moe_expanded.py"
  --h4-trainer "${SOURCE_ROOT}/code/train_h4_expanded.py"
  --coefficient-view "${SOURCE_ROOT}/assets_r32/expanded_h4_trainval_r32.npz"
  --galerkin-path "${SOURCE_ROOT}/assets_r32/expanded_h4_trainonly_galerkin_r32.npz"
  --pressure-path "${SOURCE_ROOT}/assets_r32/expanded_h4_trainonly_pressure_r32.npz"
  --asset-manifest "${SOURCE_ROOT}/assets_r32/TRAINING_ASSET_MANIFEST.json"
  --fluctuation-contract "${SOURCE_ROOT}/trainonly_contract/trainonly_fluctuation_contract.npz"
)

for variant in b1 b2 b3 b4; do
  model_dir="${MODEL_DIR[$variant]}_seed${SEED}"
  run_dir="${EXPERIMENT_ROOT}/multi_seed/${model_dir}"
  failure="${EXPERIMENT_ROOT}/training_failures/${model_dir}.json"
  log="${EXPERIMENT_ROOT}/logs/${model_dir}.log"
  if [[ ! -f "${run_dir}/final_training.pt" && ! -f "${failure}" ]]; then
    set +e
    nice -n 10 "${PYTHON}" "${EXPERIMENT_ROOT}/code/train_ctdm_comparison.py" \
      --variant "${variant}" \
      "${COMMON[@]}" \
      --output-root "${EXPERIMENT_ROOT}/multi_seed" \
      --seed "${SEED}" \
      --max-steps 8000 \
      --micro-batch "${MICRO_BATCH}" \
      --grad-accum "${GRAD_ACCUM}" \
      --gpu-memory-fraction 0.42 \
      --swanlab-mode online \
      --swanlab-group B1_B4_three_seed >"${log}" 2>&1
    exit_code=$?
    set -e
    if (( exit_code != 0 )); then
      "${PYTHON}" "${EXPERIMENT_ROOT}/code/record_training_failure.py" \
        --variant "${variant}" --seed "${SEED}" --run-dir "${run_dir}" \
        --log "${log}" --exit-code "${exit_code}" --output "${failure}"
      continue
    fi
  fi
  if [[ -f "${run_dir}/final_training.pt" ]]; then
    analysis_dir="${EXPERIMENT_ROOT}/validation/${model_dir}"
    if [[ ! -f "${analysis_dir}/VALIDATION_RESULTS.json" ]]; then
      checkpoint="${run_dir}/best_validation.pt"
      [[ -f "${checkpoint}" ]] || checkpoint="${run_dir}/final_training.pt"
      nice -n 10 "${PYTHON}" "${EXPERIMENT_ROOT}/code/analyze_ctdm_checkpoint.py" \
        --split validation --checkpoint "${checkpoint}" \
        --trainer "${EXPERIMENT_ROOT}/code/train_ctdm_comparison.py" \
        "${COMMON[@]}" --output-dir "${analysis_dir}" \
        --windows-per-re 16 --gpu-memory-fraction 0.42
    fi
  fi
done

bash "${EXPERIMENT_ROOT}/code/finalize_evaluation.sh"
