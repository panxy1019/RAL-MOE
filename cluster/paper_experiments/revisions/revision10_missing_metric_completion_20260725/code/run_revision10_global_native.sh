#!/usr/bin/env bash
# Sequential evaluation-only Global rollout exports for revision 10.
set -euo pipefail

ROOT=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE
REV="$ROOT/paper_experiments/revisions/revision10_missing_metric_completion_20260725"
PY="$ROOT/.runtime/pt_env/bin/python"
SCRIPT="$REV/code/global_eval_revision10.py"
CHECKPOINT="$ROOT/paper_experiments/revisions/revision5_supplemental_evaluation_20260725/recovery/global_k56_valid_history_windows_v5/V16_1_SteadyPressureAnchor32_ru32_rp32_Re_24p630436_checkpoint.pt"
DATA="$ROOT/V16_1_SteadyPressureAnchor32/assets/common_global_data/Global_POD_AreaWeighted_L2"
TENSOR="$ROOT/V16_1_SteadyPressureAnchor32/assets/common_global_data/semi_intrusive_galerkin_tensors_allRe100_areaWeightedL2_ru80_rp80_compact.npz"
PRESSURE="$ROOT/V16_1_SteadyPressureAnchor32/assets/common_global_data/pressure_poisson_surrogate_tensors_allRe100_areaWeightedL2_ru80_rp80.npz"

run_eval() {
  local name="$1" horizon="$2" values="$3"
  local out="$REV/evaluations/$name"
  "$PY" "$SCRIPT" \
    --output-dir "$out" --experiment-name "$name" --experiment-tag V16_1_SteadyPressureAnchor32 \
    --data-root "$DATA" --tensor-path "$TENSOR" --pressure-surrogate-path "$PRESSURE" \
    --r-u 32 --r-p 32 --num-blocks 3 --num-regime-groups 3 --experts-per-group 6 --num-experts 6 \
    --num-shared-experts 1 --top-k 2 --group-top-k 1 --hidden-dim 224 --expert-hidden 768 \
    --expert-blocks 3 --quadratic-rank 4 --quadratic-scale 0.05 --dropout 0.04 --temperature 0.95 \
    --gate-floor 0 --group-temperature 0.9 --group-gate-floor 0 --shared-scale 1 --routed-scale 0.85 \
    --rhs-target residual --pressure-target closure --pressure-input-mode pressure_only --closure-mode adaptive_gate \
    --pressure-base-mode static --attractor-balanced-sampling --test-re-selection regime_default \
    --rollout-steps "$horizon" --eval-re-values $values --eval-only-checkpoint "$CHECKPOINT" \
    --save-rollout-arrays --swanlab-mode offline --allow-tf32
}

run_eval GLOBAL_R10_STEADY_K56 56 "24.630436 32.740068 39.685479 45.142703"
run_eval GLOBAL_R10_HOPF_K56 56 "47.081356 49.022357 51.786450"
