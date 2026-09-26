#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pt_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ROOT=/cephfs/shared/V17_HopfLocal32_H3H4_FluctuationNormalForm
GATE="$ROOT/assets/warmstart_exact_formal_path_gate_v5_final.json"
python - "$ROOT" "$GATE" <<'PY'
import json, pathlib, sys
r=pathlib.Path(sys.argv[1]);g=json.loads(pathlib.Path(sys.argv[2]).read_text())
assert g['passed'] and not g['unsafe_starts'] and g['parent_output_equivalence']['passed']
for v in ('HopfLocal32_H3_FluctuationNormalized','HopfLocal32_H4_NormalFormRadial'):
 s=json.loads((r/'preflight_exact'/v/'per_re_forward_backward_smoke.json').read_text())
 assert s['passed'] and len(s['rows'])==48
 assert (r/'benchmark_exact'/v/'throughput.json').is_file()
PY
if pgrep -af 'train_h3h4.py.*HopfLocal32_H[34]_' >/dev/null; then
 echo 'Refusing duplicate H3/H4 formal trainer' >&2;exit 3
fi
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
(( free_mib >= 16000 )) || { echo "Insufficient free GPU memory: ${free_mib} MiB" >&2;exit 4; }
stamp=${1:-$(date +%Y%m%d_%H%M%S)}
RUNROOT="$ROOT/formal/$stamp";mkdir -p "$RUNROOT/logs"
python - "$ROOT" "$RUNROOT" "$GATE" <<'PY'
import hashlib,json,pathlib,sys,time
r,run,g=map(pathlib.Path,sys.argv[1:]);trainer=r/'code/hopf_h34/train_h3h4.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(run/'launch_manifest.json').write_text(json.dumps({'created_unix':time.time(),'trainer':str(trainer),'trainer_sha256':sha(trainer),'exact_forward_gate':str(g),'exact_forward_gate_sha256':sha(g),'parent_sha256':'5a11867f6defdae4f5ad020dc2830defcad00728bad57acbad06fecfb7225c88','seed':1248,'micro_batch':256,'grad_accum':1,'gpu_memory_fraction_each':.30,'stagger_seconds':45,'heldout_used':False},indent=2))
PY
nohup bash "$ROOT/code/hopf_h34/run_h3h4.sh" h3 formal --output-root "$RUNROOT" --gpu-memory-fraction .30 >"$RUNROOT/logs/h3.log" 2>&1 &
h3=$!;echo "$h3" >"$RUNROOT/h3.pid"
sleep 45
nohup bash "$ROOT/code/hopf_h34/run_h3h4.sh" h4 formal --output-root "$RUNROOT" --gpu-memory-fraction .30 >"$RUNROOT/logs/h4.log" 2>&1 &
h4=$!;echo "$h4" >"$RUNROOT/h4.pid"
printf '%s\n' "$RUNROOT" >"$ROOT/formal/CURRENT_RUN"
echo "RUNROOT=$RUNROOT H3_PID=$h3 H4_PID=$h4"
