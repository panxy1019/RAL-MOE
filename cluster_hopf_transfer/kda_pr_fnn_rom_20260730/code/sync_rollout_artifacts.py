#!/usr/bin/env python3
"""Upload finalized local rollout report artifacts to the experiment directory."""
from __future__ import annotations

import os
import hashlib
from pathlib import Path

import paramiko

HOST, PORT, USER = "10.10.164.243", 20073, "root"
REMOTE = (
    "/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/"
    "KDA_PR_FNN_ROM_20260730/rollout_evaluation_validation"
)


def main() -> None:
    local = Path(__file__).resolve().parents[1] / "rollout_evaluation_validation"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST, port=PORT, username=USER,
        password=os.environ["CLUSTER_SSH_PASSWORD"],
        timeout=15, banner_timeout=15, auth_timeout=15,
    )
    try:
        with client.open_sftp() as sftp:
            for path in sorted(local.iterdir()):
                if path.is_file():
                    remote = f"{REMOTE}/{path.name}"
                    sftp.put(str(path), remote)
                    local_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                    digest = hashlib.sha256()
                    with sftp.open(remote, "rb") as stream:
                        for block in iter(lambda: stream.read(1 << 20), b""):
                            digest.update(block)
                    if digest.hexdigest() != local_hash:
                        raise RuntimeError(f"SHA-256 mismatch after upload: {path.name}")
                    print(path.name, path.stat().st_size, local_hash)
    finally:
        client.close()


if __name__ == "__main__":
    main()
