#!/usr/bin/env python3
"""Write a compact reproducibility manifest for the active runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy
import torch


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "swanlab": importlib.metadata.version("swanlab"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "gpu_count": torch.cuda.device_count(),
        "nvidia_smi": command(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ),
        "git_commit": None,
        "working_directory": os.getcwd(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
