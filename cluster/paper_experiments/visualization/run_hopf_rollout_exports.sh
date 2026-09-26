#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE
VIZ="$ROOT/paper_experiments/visualization"
NAME=HopfExpanded34_H4_NormalFormRadial_r32
TRAINER="$ROOT/Hopf/migrated_h4_expanded/code/train_hopf_moe_expanded.py"
SOURCE="$ROOT/Hopf/artifacts/hopf"
CONTRACT="$ROOT/Hopf/migrated_h4_expanded/trainonly_contract/trainonly_fluctuation_contract.npz"

PROPOSED="$VIZ/cache/hopf_proposed_train_v3"
mkdir -p "$PROPOSED/$NAME"
ln -s \
  "$ROOT/Hopf/migrated_h4_expanded/final_evaluation/20260722_h4_expanded_final/$NAME/final.pt" \
  "$PROPOSED/$NAME/final.pt"
cd "$ROOT/Hopf/migrated_h4_expanded"
"$ROOT/.runtime/pt_env/bin/python" "$VIZ/evaluate_hopf_export.py" \
  --trainer "$TRAINER" \
  --source-root "$SOURCE" \
  --contract "$CONTRACT" \
  --finalization-dir "$PROPOSED" \
  --window-stride 8 \
  --experiments "$NAME" \
  --split train \
  --re-values 49.3,49.6,50.0 \
  --save-rollout-bundle "$PROPOSED/proposed_rollouts.npz"
printf '0\n' > "$PROPOSED/RETURN_CODE"

VANILLA="$VIZ/cache/hopf_vanilla_train_v3"
mkdir -p "$VANILLA/$NAME"
ln -s \
  "$ROOT/paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/hopf_v6/hopf/vanilla-fnn/training/$NAME/best_validation.pt" \
  "$VANILLA/$NAME/final.pt"
"$ROOT/.runtime/pt_env/bin/python" \
  "$ROOT/paper_experiments/code/run_vanilla_fnn_moe.py" \
  --ablation-mode vanilla-fnn \
  --regime hopf \
  --entrypoint "$VIZ/evaluate_hopf_export.py" \
  --contract-output "$VANILLA/allowlist.json" \
  --trainer "$TRAINER" \
  --source-root "$SOURCE" \
  --contract "$CONTRACT" \
  --finalization-dir "$VANILLA" \
  --window-stride 8 \
  --experiments "$NAME" \
  --split train \
  --re-values 49.3,49.6,50.0 \
  --save-rollout-bundle "$VANILLA/vanilla_rollouts.npz"
printf '0\n' > "$VANILLA/RETURN_CODE"
