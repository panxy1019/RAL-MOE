#!/usr/bin/env python3
"""Fetch validation rollout artifacts from the training cluster."""
from __future__ import annotations

import os
from pathlib import Path

import paramiko

HOST, PORT, USER = "10.10.164.243", 20073, "root"
REMOTE = (
    "/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/"
    "KDA_PR_FNN_ROM_20260730/rollout_evaluation_validation"
)
FILES = (
    "ROLLOUT_RESULTS.csv",
    "ERROR_GROWTH_CURVES.csv",
    "MEMORY_DIAGNOSTICS.csv",
    "ROLLOUT_SUMMARY.json",
)


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "rollout_evaluation_validation"
    output.mkdir(parents=True, exist_ok=True)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST, port=PORT, username=USER,
        password=os.environ["CLUSTER_SSH_PASSWORD"],
        timeout=15, banner_timeout=15, auth_timeout=15,
    )
    try:
        with client.open_sftp() as sftp:
            for name in FILES:
                sftp.get(f"{REMOTE}/{name}", str(output / name))
                print(name, (output / name).stat().st_size)
    finally:
        client.close()


if __name__ == "__main__":
    main()
