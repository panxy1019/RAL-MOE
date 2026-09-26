#!/usr/bin/env bash
set -euo pipefail

case_name="${1:?usage: run_v16_1_one.sh V16_1_HopfOnsetGrowthLoss32|V16_1_SteadyPressureAnchor32|V16_1_TrainStableCombined32}"

ru=32
rp=32
extra_args=()

lambda_hopf_radius="0.02"
lambda_hopf_overshoot="0.02"
lambda_hopf_onset="0.01"
lambda_v16_1_hopf_growth="0.0"
lambda_v16_1_hopf_false_growth="0.0"
lambda_v16_1_hopf_floor_rel="0.0"
lambda_v16_1_steady_p_state="0.0"
lambda_v16_1_steady_p_mean="0.0"
lambda_v16_1_steady_p_delta="0.0"
lambda_v16_1_steady_residual_damp="0.0"
lambda_v16_1_steady_p_energy="0.0"
v16_1_warmup="0"

case "${case_name}" in
  V16_1_HopfOnsetGrowthLoss32)
    lambda_hopf_radius="0.01"
    lambda_v16_1_hopf_growth="0.03"
    lambda_v16_1_hopf_false_growth="0.03"
    lambda_v16_1_hopf_floor_rel="0.02"
    ;;
  V16_1_SteadyPressureAnchor32)
    lambda_v16_1_steady_p_state="0.02"
    lambda_v16_1_steady_p_mean="0.02"
    lambda_v16_1_steady_p_delta="0.01"
    lambda_v16_1_steady_residual_damp="0.01"
    lambda_v16_1_steady_p_energy="0.01"
    ;;
  V16_1_TrainStableCombined32)
    lambda_hopf_radius="0.01"
    lambda_v16_1_hopf_growth="0.03"
    lambda_v16_1_hopf_false_growth="0.03"
    lambda_v16_1_hopf_floor_rel="0.02"
    lambda_v16_1_steady_p_state="0.02"
    lambda_v16_1_steady_p_mean="0.02"
    lambda_v16_1_steady_p_delta="0.01"
    lambda_v16_1_steady_residual_damp="0.01"
    lambda_v16_1_steady_p_energy="0.01"
    v16_1_warmup="30"
    ;;
  *)
    echo "unknown V16_1 case: ${case_name}" >&2
    exit 2
    ;;
esac

if [[ -n "${V16_1_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${V16_1_EXTRA_ARGS})
fi

export LD_LIBRARY_PATH=/root/miniconda3/envs/pt_env/lib:${LD_LIBRARY_PATH:-}
PY=/root/miniconda3/envs/pt_env/bin/python
ROOT="/root/moe/V16_1_TrainStableAttractorMoE/test_results_v16_1"
SCRIPT="${ROOT}/train_v16_1_train_stable_attractor_moe.py"
name="${case_name}_ru${ru}_rp${rp}"
out="${ROOT}/results/${case_name}/${name}"
TRACKING_MODE="${SWANLAB_TRACKING_MODE:-online}"
TRACKING_PROJECT="${SWANLAB_TRACKING_PROJECT:-V16_1_TrainStableAttractorMoE}"
TRACKING_GROUP="${SWANLAB_TRACKING_GROUP:-v16-1-train-stable-attractor-moe}"
TRACKING_LOG_DIR="${SWANLAB_TRACKING_LOG_DIR:-${ROOT}/results/swanlog/${case_name}}"

"${PY}" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("CUDA is not available; refusing to run V16_1 training on CPU.", file=sys.stderr)
    sys.exit(3)
print("CUDA ready:", torch.cuda.get_device_name(0))
PY

mkdir -p "${out}" "${TRACKING_LOG_DIR}"
{
  echo "===== ${name} $(date -Is) ====="
  echo "case=${case_name}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
} | tee "${out}/run.log"

"${PY}" "${SCRIPT}" \
  --output-dir "${out}" \
  --experiment-name "${name}" \
  --experiment-tag "${case_name}" \
  --data-root /root/moe/ROM_PhysicsGeneralizable/data/Global_POD_AreaWeighted_L2 \
  --tensor-path /root/moe/ROM_PhysicsGeneralizable/data/semi_intrusive_galerkin_tensors_allRe100_areaWeightedL2_ru80_rp80_compact.npz \
  --pressure-surrogate-path /root/moe/ROM_PhysicsGeneralizable/data/pressure_poisson_surrogate_tensors_allRe100_areaWeightedL2_ru80_rp80.npz \
  --test-re-selection regime_default \
  --train-time-stride 1 \
  --train-re-stride 1 \
  --r-u "${ru}" \
  --r-p "${rp}" \
  --num-blocks 3 \
  --num-regime-groups 3 \
  --experts-per-group 6 \
  --num-experts 6 \
  --num-shared-experts 1 \
  --top-k 2 \
  --group-top-k 1 \
  --hidden-dim 224 \
  --expert-hidden 768 \
  --expert-blocks 3 \
  --quadratic-rank 4 \
  --quadratic-scale 0.05 \
  --batch-size 256 \
  --recon-dim 2048 \
  --rollout-steps 24 \
  --train-rollout-steps 16 \
  --rollout-batch 2 \
  --rollout-every-batches 1 \
  --curriculum-steps 4,8,12,16 \
  --dropout 0.04 \
  --temperature 0.95 \
  --gate-floor 0.0 \
  --group-temperature 0.90 \
  --group-gate-floor 0.0 \
  --shared-scale 1.0 \
  --routed-scale 0.85 \
  --rhs-target residual \
  --pressure-target closure \
  --pressure-input-mode pressure_only \
  --closure-mode adaptive_gate \
  --pressure-base-mode static \
  --attractor-balanced-sampling \
  --hopf-pair 0,1 \
  --hopf-re-min 47.0 \
  --hopf-re-max 59.5 \
  --hopf-focus-re 51.78645 \
  --hopf-focus-width 0.85 \
  --hopf-loss-warmup-epochs 30 \
  --hopf-overshoot-ratio 2.0 \
  --lambda-hopf-log-amp 0.0 \
  --lambda-hopf-energy 0.0 \
  --lambda-hopf-overshoot 0.0 \
  --hopf-sample-weight 1.0 \
  --hopf-focus-sample-weight 1.0 \
  --hopf-rollout-weight 1.0 \
  --hopf-diagnostic-horizons 8,16,24,32 \
  --lambda-steady-rhs 0.02 \
  --lambda-steady-state 0.01 \
  --lambda-attractor-hopf-radius "${lambda_hopf_radius}" \
  --lambda-attractor-hopf-overshoot "${lambda_hopf_overshoot}" \
  --lambda-attractor-hopf-onset "${lambda_hopf_onset}" \
  --v16-1-hopf-floor-quantile 10 \
  --v16-1-loss-warmup-epochs "${v16_1_warmup}" \
  --lambda-v16-1-hopf-growth "${lambda_v16_1_hopf_growth}" \
  --lambda-v16-1-hopf-false-growth "${lambda_v16_1_hopf_false_growth}" \
  --lambda-v16-1-hopf-floor-rel "${lambda_v16_1_hopf_floor_rel}" \
  --lambda-v16-1-steady-p-state "${lambda_v16_1_steady_p_state}" \
  --lambda-v16-1-steady-p-mean "${lambda_v16_1_steady_p_mean}" \
  --lambda-v16-1-steady-p-delta "${lambda_v16_1_steady_p_delta}" \
  --lambda-v16-1-steady-residual-damp "${lambda_v16_1_steady_residual_damp}" \
  --lambda-v16-1-steady-p-energy "${lambda_v16_1_steady_p_energy}" \
  --lambda-periodic-energy 0.02 \
  --lambda-periodic-radius 0.01 \
  --lambda-attractor-ce 0.0 \
  --lambda-proto-energy 0.0 \
  --lambda-proto-radius 0.0 \
  --attractor-r-floor 0.0 \
  --attractor-onset-threshold 5e-5 \
  "${extra_args[@]}" \
  --swanlab-mode "${TRACKING_MODE}" \
  --swanlab-project "${TRACKING_PROJECT}" \
  --swanlab-group "${TRACKING_GROUP}" \
  --swanlab-run-name "${case_name}" \
  --swanlab-log-dir "${TRACKING_LOG_DIR}" \
  --swanlab-required \
  --lambda-coeff 0.75 \
  --lambda-dyn 0.90 \
  --lambda-pressure 0.95 \
  --lambda-rollout 0.45 \
  --lambda-energy 0.05 \
  --lambda-trajectory-consistency 0.18 \
  --lambda-pressure-rollout 0.45 \
  --lambda-router-balance 0.06 \
  --lambda-router-entropy -0.002 \
  --lambda-group-balance 0.04 \
  --lambda-group-entropy 0.0 \
  --lambda-group-supervision 0.04 \
  --lambda-router-smooth 0.04 \
  --lambda-expert-diversity 0.006 \
  --lambda-regime-router 0.004 \
  --lambda-alpha-rel 0.04 \
  --lambda-rhs-rel 0.06 \
  --lambda-pressure-rel 0.70 \
  --scheduled-sampling-start 0.0 \
  --scheduled-sampling-end 0.85 \
  --scheduled-sampling-warmup-frac 0.70 \
  --rollout-relative-mix 0.35 \
  --relative-floor-frac 0.05 \
  --lr 5.5e-4 \
  --weight-decay 1.5e-4 \
  --seed "${V16_1_SEED:-1600}" \
  --epochs "${V16_1_EPOCHS:-240}" \
  --min-epochs "${V16_1_MIN_EPOCHS:-130}" \
  --patience 70 \
  --eval-every "${V16_1_EVAL_EVERY:-5}" \
  --eval-routing-every 20 \
  --early-stop-min-delta 8e-4 \
  --allow-tf32 \
  2>&1 | tee -a "${out}/run.log"
