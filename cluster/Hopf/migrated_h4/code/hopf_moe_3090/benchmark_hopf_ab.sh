#!/usr/bin/env bash
set -euo pipefail

# Isolated 100-optimizer-step benchmarks.  Use a separate output root; these
# checkpoints are performance artifacts and must never be promoted as science.
: "${HOPF_BENCH_OUTPUT_ROOT:?set benchmark-only output root}"
export HOPF_OUTPUT_ROOT="${HOPF_BENCH_OUTPUT_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
fi
conda activate pt_env
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${HOPF_BENCH_OUTPUT_ROOT}"
MICRO_BATCH="${MICRO_BATCH:-256}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
GPU_FRACTION="${GPU_FRACTION:-0.30}"

for variant in common scale_aware; do
  name="HopfLocal32_V16Common_K1248"
  [[ "${variant}" == scale_aware ]] && name="HopfLocal32_V16ScaleAware_K1248"
  python -u "${SCRIPT_DIR}/train_hopf_moe.py" \
    --variant "${variant}" \
    --coefficient-view "${HOPF_COEFFICIENT_VIEW:?}" \
    --galerkin-path "${HOPF_GALERKIN:?}" \
    --pressure-path "${HOPF_PRESSURE:?}" \
    --asset-manifest "${HOPF_MANIFEST:?}" \
    --output-root "${HOPF_BENCH_OUTPUT_ROOT}" \
    --benchmark-steps 100 --benchmark-horizon 8 \
    --micro-batch "${MICRO_BATCH}" --grad-accum "${GRAD_ACCUM}" \
    --gpu-memory-fraction "${GPU_FRACTION}" \
    --swanlab-mode disabled --device cuda \
    >"${HOPF_BENCH_OUTPUT_ROOT}/${name}_benchmark.log" 2>&1
done

echo "100-step benchmark throughput JSON files:"
find "${HOPF_BENCH_OUTPUT_ROOT}" -name throughput.json -print
