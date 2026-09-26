#!/usr/bin/env python3
"""Read-only remote progress snapshot for the KDA single-seed pipeline."""
from __future__ import annotations

import os
import sys

import paramiko

HOST = "10.10.164.243"
PORT = 20073
USER = "root"
ROOT = (
    "/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/"
    "panxy/particalMOE/KDA_PR_FNN_ROM_20260730"
)

COMMAND = f"""
set -eu
EXP={ROOT}
echo '--- PROCESSES ---'
ps -eo pid,ppid,etime,%cpu,%mem,args | grep -E \
  'run_single_seed_pipeline|train_kda_pr_fnn_rom.py' | grep -v grep || true
echo '--- GPU ---'
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
  --format=csv,noheader
echo '--- RUN DIRECTORIES ---'
find "$EXP/runs_single_seed" -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' | sort
echo '--- STATUS FILES ---'
for d in "$EXP"/runs_single_seed/*; do
  echo "=== $(basename "$d")"
  test -f "$d/runtime_status.json" && cat "$d/runtime_status.json" || true
  test -f "$d/THROUGHPUT.json" && cat "$d/THROUGHPUT.json" || true
done
echo '--- VALIDATION SUMMARY ---'
/root/miniconda3/envs/pt_env/bin/python - <<'PY'
import json
from pathlib import Path
root=Path("{ROOT}")/"runs_single_seed"
for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    p=d/"VALIDATION_HISTORY.json"
    print("===", d.name)
    if not p.is_file():
        print("validation: missing")
        continue
    history=json.loads(p.read_text())["history"]
    print("entries",len(history),"latest_step",history[-1]["step"])
    for row in history[-3:]:
        print({{
            "step":row["step"], "score":row["score"],
            "hard_gate":row["hard_gate"],
            "finite_fraction":row["finite_fraction"],
            "divergent_windows":row["divergent_windows"],
            "k56":row.get("k56"),
        }})
    k56=[row for row in history if "k56" in row]
    if k56:
        best=min(k56,key=lambda row:row["score"])
        print("best_k56",{{
            "step":best["step"], "score":best["score"],
            "hard_gate":best["hard_gate"],
            "finite_fraction":best["finite_fraction"],
            "divergent_windows":best["divergent_windows"],
            "k56":best["k56"],
        }})
PY
echo '--- PIPELINE LOG ---'
tail -100 "$EXP/logs/single_seed_pipeline.log"
"""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    password = os.environ["CLUSTER_SSH_PASSWORD"]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        port=PORT,
        username=USER,
        password=password,
        timeout=15,
        banner_timeout=15,
        auth_timeout=15,
    )
    try:
        _, stdout, stderr = client.exec_command(COMMAND, timeout=60)
        output = stdout.read().decode("utf-8", errors="replace")
        error = stderr.read().decode("utf-8", errors="replace")
        print(output, end="")
        if error:
            print(error, end="")
        status = stdout.channel.recv_exit_status()
        if status:
            raise SystemExit(status)
    finally:
        client.close()


if __name__ == "__main__":
    main()
