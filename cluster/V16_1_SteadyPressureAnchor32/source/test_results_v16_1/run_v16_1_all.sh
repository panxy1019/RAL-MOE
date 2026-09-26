#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/moe/V16_1_TrainStableAttractorMoE/test_results_v16_1"
cases=(
  V16_1_HopfOnsetGrowthLoss32
  V16_1_SteadyPressureAnchor32
  V16_1_TrainStableCombined32
)

mapfile -t gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}')
if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
  echo "No visible GPU; refusing to launch V16_1." >&2
  exit 3
fi

mkdir -p "${ROOT}/results"
echo "Visible GPUs: ${gpu_ids[*]}"

for i in "${!cases[@]}"; do
  case_name="${cases[$i]}"
  gpu="${gpu_ids[$((i % ${#gpu_ids[@]}))]}"
  if pgrep -af "train_v16_1_train_stable_attractor_moe.py.*--experiment-tag ${case_name}" >/dev/null 2>&1; then
    echo "${case_name} already running; skip"
    continue
  fi
  log_dir="${ROOT}/results/${case_name}"
  mkdir -p "${log_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${ROOT}/run_v16_1_one.sh" "${case_name}" \
    >"${log_dir}/nohup.log" 2>&1 &
  echo $! >"${log_dir}/launcher.pid"
  echo "started ${case_name} on visible GPU ${gpu}; launcher pid $(cat "${log_dir}/launcher.pid")"
  sleep 8
done
