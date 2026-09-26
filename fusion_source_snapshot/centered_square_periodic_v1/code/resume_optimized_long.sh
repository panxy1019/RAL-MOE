#!/usr/bin/env bash
set -euo pipefail

experiment_root=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/centered_square_periodic_v1
cd "$experiment_root"

task_swan_key=$(
  /root/miniconda3/envs/pt_env/bin/python -c \
    "import netrc; print(netrc.netrc('/root/.swanlab/.netrc').authenticators('https://api.swanlab.cn')[2])"
)

exec env \
  SWANLAB_API_KEY="<REDACTED>" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /root/miniconda3/envs/pt_env/bin/python \
  code/run_square_periodic_training_optimized.py \
  --trainer code/train_square_periodic_moe_optimized.py \
  --source-checkpoint /root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/checkpoint/FINAL_PERIODIC_SPECIALIST.pt \
  --data-root assets \
  --output-dir runs/optimized_long_seed1600 \
  --epochs 720 \
  --batch-size 1024 \
  --rollout-batch 56 \
  --rollout-batch-by-stage 56,56,40,28 \
  --rollout-updates-per-epoch 2 \
  --curriculum-steps 4,8,12,16 \
  --curriculum-stage-epochs 120,160,160,280 \
  --amp-mode bf16 \
  --batched-experts \
  --fixed-regime-group 1 \
  --no-dense-moe-training \
  --no-compile-model \
  --eval-every 10 \
  --resume-checkpoint runs/optimized_long_seed1600/latest.pt \
  --swanlab-mode online
