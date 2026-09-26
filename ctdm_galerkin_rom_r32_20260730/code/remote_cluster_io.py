#!/usr/bin/env python3
"""Execute an encoded command or transfer files on the CTDM training cluster."""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path, PurePosixPath

import paramiko


HOST = "10.10.164.243"
PORT = 20073
USER = "root"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    execute = subparsers.add_parser("exec")
    execute.add_argument("command_base64")
    download = subparsers.add_parser("get")
    download.add_argument("remote")
    download.add_argument("local", type=Path)
    upload = subparsers.add_parser("put")
    upload.add_argument("local", type=Path)
    upload.add_argument("remote")
    return parser.parse_args()


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        port=PORT,
        username=USER,
        password=os.environ["CLUSTER_SSH_PASSWORD"],
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return client


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    client = connect()
    try:
        if args.operation == "exec":
            command = base64.b64decode(args.command_base64).decode("utf-8")
            _, stdout, stderr = client.exec_command(command, timeout=None)
            for block in iter(lambda: stdout.read(1 << 16), b""):
                print(block.decode("utf-8", errors="replace"), end="")
            for block in iter(lambda: stderr.read(1 << 16), b""):
                print(block.decode("utf-8", errors="replace"), end="")
            raise SystemExit(stdout.channel.recv_exit_status())
        with client.open_sftp() as sftp:
            if args.operation == "get":
                args.local.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(args.remote, str(args.local))
                print(f"downloaded {args.remote} -> {args.local}")
            else:
                if not args.local.is_file():
                    raise FileNotFoundError(args.local)
                parent = str(PurePosixPath(args.remote).parent)
                client.exec_command(f"mkdir -p -- {parent!r}")[1].channel.recv_exit_status()
                sftp.put(str(args.local), args.remote)
                print(f"uploaded {args.local} -> {args.remote}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
