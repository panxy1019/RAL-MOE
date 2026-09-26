#!/usr/bin/env bash
set -euo pipefail
variant=${1:?h3_or_h4}
mode=${2:-formal}
shift 2 || true
ROOT=/cephfs/shared/V17_HopfLocal32_H3H4_FluctuationNormalForm
BASE=/cephfs/shared/V17_HopfLocal32_V16CommonScaleAware
case "$variant" in
  h3) name=HopfLocal32_H3_FluctuationNormalized ;;
  h4) name=HopfLocal32_H4_NormalFormRadial ;;
  *) exit 2 ;;
esac
args=(--variant "$variant" --baseline-trainer "$BASE/code/hopf_moe_3090/train_hopf_moe.py"
 --coefficient-view "$BASE/assets/hopf_trainval_r32/hopf_local_r32_trainval_coefficients.npz"
 --galerkin-path "$BASE/assets/hopf_trainval_r32/velocity_rom_hopf_local_r32_trainnodes.npz"
 --pressure-path "$BASE/assets/hopf_trainval_r32/pressure_poisson_hopf_local_r32_trainnodes.npz"
 --asset-manifest "$BASE/assets/hopf_trainval_r32/TRAINING_ASSET_MANIFEST.json"
 --contract "$ROOT/assets/trainonly_contract_v5/trainonly_fluctuation_contract.npz"
 --output-root "$ROOT/runs" --experiment-name "$name" --seed 1248 --micro-batch 256 --grad-accum 1
 --max-steps 8000 --gpu-memory-fraction .30 --swanlab-project V17_HopfLocal32_H3H4 --swanlab-group H3H4_FluctuationNormalForm)
case "$mode" in
 smoke) args+=(--output-root "$ROOT/smoke" --smoke-only --swanlab-mode disabled) ;;
 bench) args+=(--output-root "$ROOT/benchmark" --benchmark-steps 100 --benchmark-horizon 8 --swanlab-mode disabled) ;;
 formal) : ;;
 *) exit 2 ;;
esac
exec python "$ROOT/code/hopf_h34/train_h3h4.py" "${args[@]}" "$@"
