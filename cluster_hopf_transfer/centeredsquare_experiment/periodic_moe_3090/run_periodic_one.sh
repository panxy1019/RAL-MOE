#!/usr/bin/env bash
set -euo pipefail

rank="${1:?usage: run_periodic_one.sh 32|80 OUTPUT_DIR [swanlab_mode]}"
output_dir="${2:?missing output directory}"
swanlab_mode="${3:-online}"
run_mode="${4:-formal}"

extra_args=()
if [ "$run_mode" = smoke ]; then
  extra_args=(--epochs 1 --min-epochs 1 --patience 1 --eval-every 1 --curriculum-steps 16 --rollout-batch 1)
elif [ "$run_mode" = smoke4 ]; then
  extra_args=(--epochs 1 --min-epochs 1 --patience 1 --eval-every 1 --curriculum-steps 4 --rollout-batch 1)
elif [ "$run_mode" = smoke1 ]; then
  extra_args=(--epochs 1 --min-epochs 1 --patience 1 --eval-every 1 --curriculum-steps 1 --rollout-batch 1)
elif [ "$run_mode" != formal ]; then
  echo "run mode must be formal, smoke, smoke4, or smoke1" >&2
  exit 2
fi

case "$rank" in
  80)
    asset_root=/cephfs/shared/V17_RegimeIndependentROM/artifacts/periodic
    run_name=V17-Periodic-V16Public-ru80-rp80
    curriculum=1,4,8,16
    ;;
  32)
    asset_root=/cephfs/shared/V17_IndependentMoEROM_v2/artifacts/periodic_r32_handoff/periodic
    run_name=V17-Periodic-V16Public-ru32-rp32
    curriculum=4,8,12,16
    ;;
  *)
    echo "rank must be 32 or 80" >&2
    exit 2
    ;;
esac

mkdir -p "$output_dir" "$output_dir/swanlog"
python /root/V17indepentMOEV2/periodic_v16_public/code/train_periodic_moe.py \
  --data-root "$asset_root/Global_POD_AreaWeighted_L2" \
  --tensor-path "$asset_root/velocity_rom_periodic.npz" \
  --pressure-surrogate-path "$asset_root/pressure_poisson_surrogate_periodic.npz" \
  --output-dir "$output_dir" \
  --experiment-name "periodic_v16_public_r${rank}" \
  --experiment-tag "Periodic-only V16_1 public-loss physical-zero-residual r${rank}" \
  --r-u "$rank" --r-p "$rank" \
  --test-re-selection values \
  --test-re-values 70.314635 100.352251 149.059229 189.862278 \
  --validation-re-values 66.970112 91.792204 121.050171 139.642302 169.244893 196.160723 \
  --phase-harmonics 4 \
  --hidden-dim 224 --num-blocks 3 \
  --num-regime-groups 3 --experts-per-group 6 \
  --group-top-k 1 --group-temperature 0.9 --group-gate-floor 0.0 \
  --top-k 2 --expert-hidden 768 --expert-blocks 3 \
  --quadratic-rank 4 --quadratic-scale 0.05 \
  --dropout 0.04 --temperature 0.8 --gate-floor 0.0 \
  --shared-scale 1.0 --routed-scale 0.75 \
  --epochs 240 --min-epochs 180 --patience 70 --eval-every 5 \
  --checkpoint-every-evals 1 \
  --batch-size 256 --lr 5.5e-4 --weight-decay 1.5e-4 \
  --integrator rk4 --rhs-target residual \
  --pressure-target closure --pressure-input-mode pressure_only \
  --pressure-base-mode static --closure-mode adaptive_gate \
  --adaptive-gate-initial-logit 6.0 \
  --history-len 3 --curriculum-steps "$curriculum" \
  --train-rollout-steps 16 --rollout-steps 24 \
  --rollout-batch 2 --rollout-every-batches 1 \
  --scheduled-sampling-start 1.0 --scheduled-sampling-end 1.0 \
  --scheduled-sampling-warmup-frac 0.0 \
  --recon-dim 2048 \
  --lambda-coeff 0.75 --lambda-dyn 0.90 --lambda-pressure 0.95 \
  --lambda-recon 0.08 --lambda-rollout 0.45 \
  --lambda-pressure-rollout 0.45 --lambda-consistency 0.15 \
  --lambda-router-balance 0.06 --lambda-router-entropy -0.002 \
  --lambda-group-balance 0.04 --lambda-group-entropy 0.0 \
  --lambda-group-supervision 0.0 --lambda-router-smooth 0.04 \
  --lambda-expert-diversity 0.006 --lambda-regime-router 0.0 \
  --lambda-energy 0.05 --lambda-trajectory-consistency 0.18 \
  --lambda-alpha-rel 0.04 --lambda-rhs-rel 0.06 \
  --lambda-pressure-rel 0.70 \
  --lambda-hopf-log-amp 0 --lambda-hopf-energy 0 --lambda-hopf-overshoot 0 \
  --lambda-attractor-ce 0 --lambda-steady-rhs 0 --lambda-steady-state 0 \
  --lambda-attractor-hopf-radius 0 --lambda-attractor-hopf-overshoot 0 \
  --lambda-attractor-hopf-onset 0 \
  --lambda-v16-1-hopf-growth 0 --lambda-v16-1-hopf-false-growth 0 \
  --lambda-v16-1-hopf-floor-rel 0 --lambda-v16-1-steady-p-state 0 \
  --lambda-v16-1-steady-p-mean 0 --lambda-v16-1-steady-p-delta 0 \
  --lambda-v16-1-steady-residual-damp 0 --lambda-v16-1-steady-p-energy 0 \
  --lambda-periodic-energy 0 --lambda-periodic-radius 0 \
  --lambda-proto-energy 0 --lambda-proto-radius 0 \
  --gpu-memory-fraction 0.20 --seed 1600 \
  --swanlab-mode "$swanlab_mode" \
  --swanlab-project V17indepentMOEV2 \
  --swanlab-group Periodic-V16_1-public-loss-from-scratch \
  --swanlab-run-name "$run_name" \
  --swanlab-log-dir "$output_dir/swanlog" \
  --swanlab-required --allow-tf32 \
  "${extra_args[@]}"
