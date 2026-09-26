#!/usr/bin/env python3
"""Upload and run the validation-only rollout evaluator on the training cluster."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import paramiko

HOST, PORT, USER = "10.10.164.243", 20073, "root"
ROOT = (
    "/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/"
    "panxy/particalMOE"
)
EXP = f"{ROOT}/KDA_PR_FNN_ROM_20260730"
BASE = f"{ROOT}/CenteredSquare_Hopf_H4_20260728"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    local = Path(__file__).with_name("evaluate_validation_rollout.py")
    remote = f"{EXP}/code/evaluate_validation_rollout.py"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST, port=PORT, username=USER,
        password=os.environ["CLUSTER_SSH_PASSWORD"],
        timeout=15, banner_timeout=15, auth_timeout=15,
    )
    try:
        with client.open_sftp() as sftp:
            sftp.put(str(local), remote)
        command = (
            f"/root/miniconda3/envs/pt_env/bin/python {remote} "
            f"--trainer {EXP}/code/train_kda_pr_fnn_rom.py "
            f"--baseline-trainer {BASE}/code/train_centeredsquare_hopf_base.py "
            f"--coefficient-view {BASE}/assets_r11/centeredsquare_hopf_trainval_r11.npz "
            f"--galerkin-path {BASE}/assets_r11/centeredsquare_hopf_trainonly_galerkin_r11.npz "
            f"--pressure-path {BASE}/assets_r11/centeredsquare_hopf_trainonly_pressure_r11.npz "
            f"--asset-manifest {BASE}/assets_r11/TRAINING_ASSET_MANIFEST.json "
            f"--runs-root {EXP}/runs_single_seed "
            f"--output-dir {EXP}/rollout_evaluation_validation "
            f"--windows-per-re 64 --device cuda"
        )
        _, stdout, stderr = client.exec_command(command, timeout=None)
        channel = stdout.channel
        while not channel.exit_status_ready():
            while channel.recv_ready():
                sys.stdout.write(channel.recv(65536).decode("utf-8", errors="replace"))
                sys.stdout.flush()
            while channel.recv_stderr_ready():
                sys.stderr.write(channel.recv_stderr(65536).decode("utf-8", errors="replace"))
                sys.stderr.flush()
            time.sleep(0.2)
        while channel.recv_ready():
            sys.stdout.write(channel.recv(65536).decode("utf-8", errors="replace"))
        while channel.recv_stderr_ready():
            sys.stderr.write(channel.recv_stderr(65536).decode("utf-8", errors="replace"))
        status = channel.recv_exit_status()
        print(f"\n__EVAL_EXIT_STATUS__={status}")
        if status:
            raise SystemExit(status)
    finally:
        client.close()


if __name__ == "__main__":
    main()
