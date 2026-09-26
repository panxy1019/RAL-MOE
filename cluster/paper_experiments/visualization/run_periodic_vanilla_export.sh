#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE
VIZ="$ROOT/paper_experiments/visualization"
OUT="$VIZ/cache/periodic_vanilla"
mkdir -p "$OUT"
"$ROOT/.runtime/pt_env/bin/python" \
  "$ROOT/paper_experiments/code/run_vanilla_fnn_moe.py" \
  --ablation-mode vanilla-fnn \
  --regime periodic \
  --entrypoint "$VIZ/evaluate_periodic_export.py" \
  --contract-output "$OUT/allowlist.json" \
  --trainer "$ROOT/periodic_specialist_r32/code/train_periodic_moe.py" \
  --checkpoint "$ROOT/paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/periodic_v4/periodic/vanilla-fnn/training/best_validation.pt" \
  --existing-metrics "$ROOT/periodic_specialist_r32/evaluation/periodic_v16_public_r32_metrics.json" \
  --output-dir "$OUT" \
  --window-stride 8 \
  --data-root "$ROOT/periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2" \
  --tensor-path "$ROOT/periodic_specialist_r32/assets/velocity_rom_periodic.npz" \
  --pressure-surrogate-path "$ROOT/periodic_specialist_r32/assets/pressure_poisson_surrogate_periodic.npz" \
  --save-rollout-bundle "$OUT/vanilla_rollouts.npz"
printf '0\n' > "$OUT/RETURN_CODE"
