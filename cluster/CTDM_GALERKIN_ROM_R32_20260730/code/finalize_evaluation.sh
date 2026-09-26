#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE
SOURCE_ROOT="${PROJECT_ROOT}/Hopf/migrated_h4_expanded"
HELDOUT_ROOT="${PROJECT_ROOT}/Hopf/artifacts/hopf"
EXPERIMENT_ROOT="${PROJECT_ROOT}/CTDM_GALERKIN_ROM_R32_20260730"
PYTHON=/root/miniconda3/envs/pt_env/bin/python
FREEZE_DIR="${EXPERIMENT_ROOT}/validation_freeze"
FINAL_DIR="${EXPERIMENT_ROOT}/final_report"

exec 9>"${EXPERIMENT_ROOT}/logs/finalize_evaluation.lock"
if ! flock -n 9; then
  echo "Another final evaluation process already holds the lock; exiting."
  exit 0
fi

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

analysis_args=()
checkpoint_args=()
failure_args=()
for seed in 1248 2248 3248; do
  for variant in b1 b2 b3 b4; do
    model_dir="${MODEL_DIR[$variant]}_seed${seed}"
    analysis="${EXPERIMENT_ROOT}/validation/${model_dir}/VALIDATION_RESULTS.json"
    failure="${EXPERIMENT_ROOT}/training_failures/${model_dir}.json"
    if [[ -f "${analysis}" ]]; then
      checkpoint="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "${analysis}")"
      [[ -f "${checkpoint}" ]] || { echo "missing ${checkpoint}" >&2; exit 4; }
      analysis_args+=(--analysis "${analysis}")
      checkpoint_args+=(--checkpoint "${checkpoint}")
    elif [[ -f "${failure}" ]]; then
      failure_args+=(--failure "${failure}")
    else
      echo "missing analysis or failure record for ${model_dir}" >&2
      exit 4
    fi
  done
done

b0_analysis="${EXPERIMENT_ROOT}/validation/B0_seed1248/VALIDATION_RESULTS.json"
b0_checkpoint="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "${b0_analysis}")"
checkpoint_args+=(--checkpoint "${b0_checkpoint}")

"${PYTHON}" "${EXPERIMENT_ROOT}/code/freeze_validation_decision.py" \
  "${analysis_args[@]}" \
  "${checkpoint_args[@]}" \
  "${failure_args[@]}" \
  --candidate-variant b1 \
  --candidate-variant b2 \
  --candidate-variant b3 \
  --candidate-variant b4 \
  --output-dir "${FREEZE_DIR}"

gate="${FREEZE_DIR}/VALIDATION_FREEZE_MANIFEST.json"
authorized="$(${PYTHON} -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["test_access_authorized"]).lower())' "${gate}")"
if [[ "${authorized}" == true ]]; then
  for seed in 1248 2248 3248; do
    for variant in b1 b2 b3 b4; do
      model_dir="${MODEL_DIR[$variant]}_seed${seed}"
      validation="${EXPERIMENT_ROOT}/validation/${model_dir}/VALIDATION_RESULTS.json"
      checkpoint="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "${validation}")"
      output_dir="${EXPERIMENT_ROOT}/test/${model_dir}"
      if [[ ! -f "${output_dir}/TEST_RESULTS.json" ]]; then
        nice -n 10 "${PYTHON}" "${EXPERIMENT_ROOT}/code/analyze_ctdm_checkpoint.py" \
          --split test \
          --checkpoint "${checkpoint}" \
          --trainer "${EXPERIMENT_ROOT}/code/train_ctdm_comparison.py" \
          "${COMMON[@]}" \
          --heldout-source-root "${HELDOUT_ROOT}" \
          --gate-manifest "${gate}" \
          --output-dir "${output_dir}" \
          --windows-per-re 16 \
          --gpu-memory-fraction 0.42
      fi
    done
  done
  if [[ ! -f "${EXPERIMENT_ROOT}/test/B0_seed1248/TEST_RESULTS.json" ]]; then
    nice -n 10 "${PYTHON}" "${EXPERIMENT_ROOT}/code/analyze_b0_checkpoint.py" \
      --split test \
      --checkpoint "${b0_checkpoint}" \
      --baseline-trainer "${SOURCE_ROOT}/code/train_hopf_moe_expanded.py" \
      --analysis-helper "${EXPERIMENT_ROOT}/code/analyze_ctdm_checkpoint.py" \
      --coefficient-view "${SOURCE_ROOT}/assets_r32/expanded_h4_trainval_r32.npz" \
      --galerkin-path "${SOURCE_ROOT}/assets_r32/expanded_h4_trainonly_galerkin_r32.npz" \
      --pressure-path "${SOURCE_ROOT}/assets_r32/expanded_h4_trainonly_pressure_r32.npz" \
      --heldout-source-root "${HELDOUT_ROOT}" \
      --gate-manifest "${gate}" \
      --output-dir "${EXPERIMENT_ROOT}/test/B0_seed1248" \
      --windows-per-re 16 \
      --gpu-memory-fraction 0.42
  fi
else
  echo "Validation gate denied held-out access; no test tensor was opened."
fi

"${PYTHON}" "${EXPERIMENT_ROOT}/code/collect_environment.py" \
  --output "${EXPERIMENT_ROOT}/environment_manifest.json"
"${PYTHON}" "${EXPERIMENT_ROOT}/code/make_final_report.py" \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --output-dir "${FINAL_DIR}"
