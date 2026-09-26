#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE
BASE="$ROOT/CenteredSquare_Hopf_H4_20260728"
EXP="$ROOT/KDA_PR_FNN_ROM_20260730"
PY=/root/miniconda3/envs/pt_env/bin/python
TRAIN="$EXP/code/train_kda_pr_fnn_rom.py"
COMMON=(
  --baseline-trainer "$BASE/code/train_centeredsquare_hopf_base.py"
  --coefficient-view "$BASE/assets_r11/centeredsquare_hopf_trainval_r11.npz"
  --galerkin-path "$BASE/assets_r11/centeredsquare_hopf_trainonly_galerkin_r11.npz"
  --pressure-path "$BASE/assets_r11/centeredsquare_hopf_trainonly_pressure_r11.npz"
  --asset-manifest "$BASE/assets_r11/TRAINING_ASSET_MANIFEST.json"
  --fluctuation-contract "$BASE/trainonly_contract_mb46/trainonly_fluctuation_contract.npz"
  --output-root "$EXP/runs_single_seed"
  --seed 1248 --max-steps 8000 --micro-batch 1536 --grad-accum 1
  --gpu-memory-fraction 0.94 --swanlab-mode online
)

"$PY" "$TRAIN" --variant b1 --experiment-name B1_Deep_FNN_H3_seed1248 "${COMMON[@]}"
"$PY" "$TRAIN" --variant b2 --experiment-name B2_KDA_PR_FNN_ROM_radial_off_seed1248 "${COMMON[@]}"
"$PY" "$TRAIN" --variant b3 --experiment-name B3_KDA_PR_FNN_ROM_radial_on_seed1248 "${COMMON[@]}"
