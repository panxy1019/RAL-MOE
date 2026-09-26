#!/usr/bin/env bash
set -uo pipefail
cd /root/panxy/particalMOE/Hopf/migrated_h4_expanded
source /root/miniconda3/bin/activate pt_env
python code/train_h4_expanded.py \
  --variant h4 \
  --baseline-trainer code/train_hopf_moe_expanded.py \
  --coefficient-view assets_r32/expanded_h4_trainval_r32.npz \
  --galerkin-path assets_r32/expanded_h4_trainonly_galerkin_r32.npz \
  --pressure-path assets_r32/expanded_h4_trainonly_pressure_r32.npz \
  --asset-manifest assets_r32/TRAINING_ASSET_MANIFEST.json \
  --contract trainonly_contract/trainonly_fluctuation_contract.npz \
  --output-root runs \
  --experiment-name HopfExpanded34_H4_NormalFormRadial_r32 \
  --micro-batch 232 \
  --swanlab-mode disabled
rc=$?
printf '%s\n' "$rc" > training.rc
exit "$rc"
