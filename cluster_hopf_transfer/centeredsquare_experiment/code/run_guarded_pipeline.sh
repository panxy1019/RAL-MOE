#!/usr/bin/env bash
set -uo pipefail

WORK=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/CenteredSquare_Hopf_H4_20260728
PYTHON=/root/miniconda3/envs/pt_env/bin/python
EXPERIMENT=CenteredSquareHopf34_H4_NormalFormRadial_r11
cd "$WORK"

stable_checks=0
while (( stable_checks < 3 )); do
  IFS=, read -r used util < <(
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
  )
  used=${used//[[:space:]]/}
  util=${util//[[:space:]]/}
  printf '%s gpu_wait memory_used_mib=%s utilization_pct=%s stable_checks=%s\n' \
    "$(date --iso-8601=seconds)" "$used" "$util" "$stable_checks"
  if (( used <= 2000 && util <= 5 )); then
    stable_checks=$((stable_checks + 1))
  else
    stable_checks=0
  fi
  if (( stable_checks < 3 )); then
    sleep 30
  fi
done

printf '%s preflight_start\n' "$(date --iso-8601=seconds)"
"$PYTHON" code/train_centeredsquare_h4.py \
  --variant h4 \
  --baseline-trainer code/train_centeredsquare_hopf_base.py \
  --coefficient-view assets_r11/centeredsquare_hopf_trainval_r11.npz \
  --galerkin-path assets_r11/centeredsquare_hopf_trainonly_galerkin_r11.npz \
  --pressure-path assets_r11/centeredsquare_hopf_trainonly_pressure_r11.npz \
  --asset-manifest assets_r11/TRAINING_ASSET_MANIFEST.json \
  --contract trainonly_contract/trainonly_fluctuation_contract.npz \
  --output-root preflight_runs \
  --experiment-name "$EXPERIMENT" \
  --micro-batch 92 \
  --gpu-memory-fraction 0.42 \
  --preflight-per-re \
  --preflight-windows 2 \
  --swanlab-mode disabled
preflight_rc=$?
printf '%s preflight_end rc=%s\n' "$(date --iso-8601=seconds)" "$preflight_rc"
if (( preflight_rc != 0 )); then
  exit "$preflight_rc"
fi

printf '%s training_start\n' "$(date --iso-8601=seconds)"
"$PYTHON" code/train_centeredsquare_h4.py \
  --variant h4 \
  --baseline-trainer code/train_centeredsquare_hopf_base.py \
  --coefficient-view assets_r11/centeredsquare_hopf_trainval_r11.npz \
  --galerkin-path assets_r11/centeredsquare_hopf_trainonly_galerkin_r11.npz \
  --pressure-path assets_r11/centeredsquare_hopf_trainonly_pressure_r11.npz \
  --asset-manifest assets_r11/TRAINING_ASSET_MANIFEST.json \
  --contract trainonly_contract/trainonly_fluctuation_contract.npz \
  --output-root runs \
  --experiment-name "$EXPERIMENT" \
  --micro-batch 92 \
  --gpu-memory-fraction 0.42 \
  --swanlab-mode online \
  --swanlab-project CenteredSquare_Hopf_H4 \
  --swanlab-group CenteredSquare_Hopf_r11
training_rc=$?
printf '%s training_end rc=%s\n' "$(date --iso-8601=seconds)" "$training_rc"
if (( training_rc != 0 )); then
  exit "$training_rc"
fi

run_dir="runs/$EXPERIMENT"
selected="$run_dir/best_validation.pt"
if [[ ! -f "$selected" ]]; then
  selected="$run_dir/latest.pt"
fi
mkdir -p final_evaluation
cp -p "$selected" final_evaluation/selected_validation.pt

printf '%s heldout_evaluation_start selected=%s\n' \
  "$(date --iso-8601=seconds)" "$selected"
"$PYTHON" code/evaluate_centeredsquare_h4.py \
  --baseline-trainer code/train_centeredsquare_hopf_base.py \
  --h4-trainer code/train_centeredsquare_h4.py \
  --coefficient-view assets_r11/centeredsquare_hopf_trainval_r11.npz \
  --heldout-view assets_r11/centeredsquare_hopf_heldout_r11.npz \
  --galerkin-path assets_r11/centeredsquare_hopf_trainonly_galerkin_r11.npz \
  --pressure-path assets_r11/centeredsquare_hopf_trainonly_pressure_r11.npz \
  --contract trainonly_contract/trainonly_fluctuation_contract.npz \
  --checkpoint final_evaluation/selected_validation.pt \
  --output-dir final_evaluation
evaluation_rc=$?
printf '%s heldout_evaluation_end rc=%s\n' "$(date --iso-8601=seconds)" "$evaluation_rc"
exit "$evaluation_rc"
