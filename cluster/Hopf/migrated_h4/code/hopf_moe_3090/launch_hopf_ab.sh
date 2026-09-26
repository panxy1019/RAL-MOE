#!/usr/bin/env bash
set -euo pipefail

# Launch the strict A/B pair on one already-selected RTX 3090 without touching
# any existing GPU process.  Required environment variables keep site-specific
# sanitized asset paths out of the scientific configuration.
: "${HOPF_COEFFICIENT_VIEW:?set sanitized train+validation coefficient NPZ}"
: "${HOPF_GALERKIN:?set sanitized 12-train-Re Galerkin NPZ}"
: "${HOPF_PRESSURE:?set sanitized 12-train-Re pressure NPZ}"
: "${HOPF_MANIFEST:?set strict asset manifest JSON}"
: "${HOPF_OUTPUT_ROOT:?set output root}"

BASE_OUTPUT_ROOT="${HOPF_OUTPUT_ROOT}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
HOPF_OUTPUT_ROOT="${BASE_OUTPUT_ROOT}/runs/${RUN_ID}"

GPU_ID="${GPU_ID:-0}"
STAGGER_SECONDS="${STAGGER_SECONDS:-45}"
MICRO_BATCH="${MICRO_BATCH:-256}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
GPU_FRACTION="${GPU_FRACTION:-0.30}"
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

mkdir -p "${HOPF_OUTPUT_ROOT}/launcher_logs"
ln -sfn "${HOPF_OUTPUT_ROOT}" "${BASE_OUTPUT_ROOT}/latest"
nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
  | tee "${HOPF_OUTPUT_ROOT}/launcher_logs/gpu_processes_before.txt"
nvidia-smi -i "${GPU_ID}" --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader | tee "${HOPF_OUTPUT_ROOT}/launcher_logs/gpu_state_before.txt"
free_mib="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if (( free_mib < 15000 )); then
  echo "Refusing A/B launch: only ${free_mib} MiB is free; existing GPU work is protected." >&2
  exit 3
fi

common=(
  python -u "${SCRIPT_DIR}/train_hopf_moe.py"
  --coefficient-view "${HOPF_COEFFICIENT_VIEW}"
  --galerkin-path "${HOPF_GALERKIN}"
  --pressure-path "${HOPF_PRESSURE}"
  --asset-manifest "${HOPF_MANIFEST}"
  --output-root "${HOPF_OUTPUT_ROOT}"
  --micro-batch "${MICRO_BATCH}"
  --grad-accum "${GRAD_ACCUM}"
  --gpu-memory-fraction "${GPU_FRACTION}"
  --device cuda
  --amp
  --allow-tf32
  --swanlab-mode online
)

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${common[@]}" --variant common \
  >"${HOPF_OUTPUT_ROOT}/launcher_logs/common.log" 2>&1 &
pid_a=$!
echo "common pid=${pid_a}"
sleep "${STAGGER_SECONDS}"

# No process is killed or paused here.  This second snapshot documents the A
# process and any unrelated workloads before B is admitted.
nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
  | tee "${HOPF_OUTPUT_ROOT}/launcher_logs/gpu_processes_before_B.txt"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${common[@]}" --variant scale_aware \
  >"${HOPF_OUTPUT_ROOT}/launcher_logs/scale_aware.log" 2>&1 &
pid_b=$!
echo "scale_aware pid=${pid_b}"

printf '%s\n' "${pid_a}" >"${HOPF_OUTPUT_ROOT}/launcher_logs/common.pid"
printf '%s\n' "${pid_b}" >"${HOPF_OUTPUT_ROOT}/launcher_logs/scale_aware.pid"
cat >"${HOPF_OUTPUT_ROOT}/launcher_logs/launch_manifest.json" <<EOF
{"run_id":"${RUN_ID}","micro_batch":${MICRO_BATCH},"gradient_accumulation":${GRAD_ACCUM},"effective_batch":$((MICRO_BATCH*GRAD_ACCUM)),"gpu_fraction_per_process":${GPU_FRACTION},"stagger_seconds":${STAGGER_SECONDS},"common_pid":${pid_a},"scale_aware_pid":${pid_b}}
EOF
echo "Both processes launched. Logs are under ${HOPF_OUTPUT_ROOT}/launcher_logs."
