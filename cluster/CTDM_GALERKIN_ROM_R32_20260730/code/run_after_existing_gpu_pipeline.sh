#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/CTDM_GALERKIN_ROM_R32_20260730
EXISTING_PARENT_PID=120276
STATUS="${EXPERIMENT_ROOT}/logs/post_existing_preflight_status.json"

mkdir -p "${EXPERIMENT_ROOT}/logs"

write_status() {
  local state=$1
  local detail=$2
  printf '{"state":"%s","detail":"%s","unix_time":%s,"pid":%s}\n' \
    "${state}" "${detail}" "$(date +%s)" "$$" > "${STATUS}.tmp"
  mv "${STATUS}.tmp" "${STATUS}"
}

fail_status() {
  local exit_code=$?
  write_status failed "line_${BASH_LINENO[0]}_exit_${exit_code}"
  exit "${exit_code}"
}
trap fail_status ERR

while existing_state="$(ps -o stat= -p "${EXISTING_PARENT_PID}" 2>/dev/null)"; \
  [[ -n "${existing_state}" && "${existing_state:0:1}" != "Z" ]]
do
  write_status waiting "existing_parent_${EXISTING_PARENT_PID}_running"
  sleep 30
done

idle_samples=0
while (( idle_samples < 3 )); do
  if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
    idle_samples=$((idle_samples + 1))
    write_status waiting "gpu_idle_sample_${idle_samples}_of_3"
  else
    idle_samples=0
    write_status waiting "other_gpu_compute_process_present"
  fi
  sleep 30
done

write_status running cuda_smoke
bash "${EXPERIMENT_ROOT}/code/run_smoke_all.sh"

write_status running b4_k56_batch_benchmark
bash "${EXPERIMENT_ROOT}/code/run_b4_batch_benchmark.sh"

write_status running single_seed_training_and_validation
bash "${EXPERIMENT_ROOT}/code/run_single_seed_pipeline.sh"

write_status complete single_seed_training_and_validation_complete
