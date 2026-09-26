#!/usr/bin/env bash
set -euo pipefail

root=/root/V17indepentMOEV2/periodic_v16_public
stamp="$(date +%Y%m%d_%H%M%S)"
run_root="$root/runs/$stamp"
mkdir -p "$run_root"

gpu_csv="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
gpu_snapshot="$(nvidia-smi --query-gpu=timestamp,memory.total,memory.used,memory.free,utilization.gpu,power.draw --format=csv,noheader,nounits)"
cat >"$run_root/PREFLIGHT.txt" <<EOF
$gpu_snapshot
$gpu_csv
EOF

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [ "$free_mib" -lt 15000 ]; then
  echo "Refusing launch: only ${free_mib} MiB free; existing GPU task is protected." >&2
  exit 3
fi

for rank in 80 32; do
  out="$run_root/ru${rank}_rp${rank}"
  mkdir -p "$out"
  nohup bash "$root/code/run_periodic_one.sh" "$rank" "$out" online \
    >"$out/train.log" 2>&1 &
  pid=$!
  printf '{"status":"RUNNING","pid":%s,"rank":%s,"started_at":"%s","output_dir":"%s"}\n' \
    "$pid" "$rank" "$(date --iso-8601=seconds)" "$out" >"$out/RUNNING.json"
  echo "$pid" >"$out/train.pid"
  sleep 8
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "rank $rank exited during launch" >&2
    tail -80 "$out/train.log" >&2
    exit 4
  fi
done

ln -sfn "$run_root" "$root/runs/latest"
echo "$run_root"
