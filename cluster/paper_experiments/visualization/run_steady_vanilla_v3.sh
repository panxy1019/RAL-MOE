#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE
VIZ="$ROOT/paper_experiments/visualization"
"$ROOT/.runtime/pt_env/bin/python" "$VIZ/export_steady_rollouts.py" \
  --root "$ROOT" \
  --method vanilla \
  --checkpoint "$ROOT/paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/steady/vanilla-fnn/frozen_user_stop_step6200_20260724/steady_vanilla_final_ablation_step6200.pt" \
  --output "$VIZ/cache/steady_vanilla_v3/vanilla_rollouts.npz" \
  --window-stride 8
printf '0\n' > "$VIZ/cache/steady_vanilla_v3/RETURN_CODE"
