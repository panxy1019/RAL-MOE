#!/usr/bin/env bash
set -euo pipefail

ROOT=/cephfs/shared/V17_HopfExpanded34_POD_20260721
LEGACY=/cephfs/shared/V17_RegimeIndependentROM
CODE="$ROOT/code"
SOURCE="$ROOT/source_database"
ARTIFACTS="$ROOT/artifacts"
LOGS="$ROOT/logs"
MESH="$LEGACY/source_mesh/run_Re_100_7107.vtk"

mkdir -p "$LOGS" "$ARTIFACTS"
rm -f "$LOGS/pipeline.exit" "$LOGS/PIPELINE_DONE"
trap 'rc=$?; printf "%s\n" "$rc" > "$LOGS/pipeline.exit"' EXIT

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pt_env
test "${CONDA_DEFAULT_ENV:-}" = pt_env

python "$CODE/prepare_hopf_expanded_source.py" \
  --legacy-source "$LEGACY/source_database" \
  --new-source "$ROOT/incoming_new" \
  --output-source "$SOURCE" \
  --split-config "$CODE/hopf_expansion_split_config.json"

python "$CODE/build_independent_regime_pod.py" \
  --data-dir "$SOURCE" \
  --global-pod-dir "$SOURCE/Global_POD_AreaWeighted_L2" \
  --split-config "$CODE/hopf_expansion_split_config.json" \
  --output-root "$ARTIFACTS" \
  --regimes hopf

python "$LEGACY/code/compute_area_weighted_l2_semi_intrusive_galerkin_tensors.py" \
  --data-dir "$ARTIFACTS/hopf" \
  --mesh-vtu "$MESH" \
  --all-re --ru 80 --rp 80 --nu 1e-3 --chunk-size 2048 --skip-raw \
  --output "$ARTIFACTS/hopf/velocity_rom_hopf.npz" \
  --report "$ARTIFACTS/hopf/velocity_rom_hopf.md" \
  >"$LOGS/velocity_rom_hopf.log" 2>&1 &
VELOCITY_PID=$!

python "$LEGACY/code/compute_area_weighted_pressure_poisson_surrogate_tensors.py" \
  --data-dir "$ARTIFACTS/hopf" \
  --mesh-vtu "$MESH" \
  --all-re --ru 80 --rp 80 --chunk-size 2048 --pinv-rcond 1e-10 \
  --output "$ARTIFACTS/hopf/pressure_poisson_surrogate_hopf.npz" \
  --report "$ARTIFACTS/hopf/pressure_poisson_surrogate_hopf.md" \
  >"$LOGS/pressure_poisson_surrogate_hopf.log" 2>&1 &
PRESSURE_PID=$!

status=0
wait "$VELOCITY_PID" || status=1
wait "$PRESSURE_PID" || status=1
test "$status" -eq 0

python "$CODE/verify_hopf_expanded_pod.py" \
  --artifact-root "$ARTIFACTS" \
  --split-config "$CODE/hopf_expansion_split_config.json"

touch "$LOGS/PIPELINE_DONE"
