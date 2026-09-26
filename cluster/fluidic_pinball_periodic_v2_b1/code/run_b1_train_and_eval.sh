#!/usr/bin/env bash
set -euo pipefail

EXP=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/fluidic_pinball_periodic_v2_b1
PY=/root/miniconda3/envs/pt_env/bin/python
ASSETS="$EXP/assets"
RUNS="$EXP/runs"
NAME=FluidicPinballV2_B1_Deep_FNN_H3_seed1248

"$PY" -u "$EXP/code/train_b1_fluidic_pinball.py" \
  --variant b1 \
  --baseline-trainer "$EXP/code/b1_data_contract.py" \
  --coefficient-view "$ASSETS/fluidic_pinball_periodic_trainval_rank999.npz" \
  --galerkin-path "$ASSETS/fluidic_pinball_periodic_trainonly_galerkin_rank999.npz" \
  --pressure-path "$ASSETS/fluidic_pinball_periodic_trainonly_pressure_rank999.npz" \
  --asset-manifest "$ASSETS/TRAINING_ASSET_MANIFEST.json" \
  --fluctuation-contract "$ASSETS/trainonly_fluctuation_contract.npz" \
  --output-root "$RUNS" --experiment-name "$NAME" \
  --seed 1248 --max-steps 8000 --micro-batch 32 --grad-accum 1 \
  --gpu-memory-fraction 0.20 --amp --amp-dtype bfloat16 --allow-tf32 \
  --fused-adamw --eval-every 400 --long-eval-every 800 \
  --validation-windows-per-re 24 --swanlab-mode online \
  --swanlab-project FluidicPinballV2_B1_DeepFNN \
  --swanlab-group periodic-rank999-seed1248

"$PY" -u "$EXP/code/evaluate_b1_validation.py" \
  --trainer "$EXP/code/train_b1_fluidic_pinball.py" \
  --baseline-trainer "$EXP/code/b1_data_contract.py" \
  --coefficient-view "$ASSETS/fluidic_pinball_periodic_trainval_rank999.npz" \
  --galerkin-path "$ASSETS/fluidic_pinball_periodic_trainonly_galerkin_rank999.npz" \
  --pressure-path "$ASSETS/fluidic_pinball_periodic_trainonly_pressure_rank999.npz" \
  --asset-manifest "$ASSETS/TRAINING_ASSET_MANIFEST.json" \
  --checkpoint "$RUNS/$NAME/best_validation.pt" \
  --output-dir "$EXP/evaluation_validation" \
  --windows-per-re 64 --gpu-memory-fraction 0.20
