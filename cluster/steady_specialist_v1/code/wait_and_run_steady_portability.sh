#!/usr/bin/env bash
set -euo pipefail

root=/root/panxy/particalMOE/steady_specialist_v1
status="$root/portability_runs/WAITER_STATUS.json"
log="$root/portability_runs/steady_portability.log"
mkdir -p "$root/portability_runs"
idle_count=0

while true; do
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
  if [[ -z "$pids" ]]; then
    idle_count=$((idle_count + 1))
  else
    idle_count=0
  fi
  printf '{"status":"WAITING_FOR_EXCLUSIVE_GPU","busy_pids":"%s","idle_checks":%d,"time":%d}\n' \
    "$(echo "$pids" | paste -sd, -)" "$idle_count" "$(date +%s)" > "$status.tmp"
  mv "$status.tmp" "$status"
  if [[ "$idle_count" -ge 2 ]]; then
    break
  fi
  sleep 60
done

printf '{"status":"STARTING","time":%d}\n' "$(date +%s)" > "$status"
if "$root/code/run_steady_portability_when_gpu_idle.sh" > "$log" 2>&1; then
  printf '{"status":"PASS","time":%d}\n' "$(date +%s)" > "$status"
else
  code=$?
  printf '{"status":"FAILED","exit_code":%d,"time":%d}\n' "$code" "$(date +%s)" > "$status"
  exit "$code"
fi
