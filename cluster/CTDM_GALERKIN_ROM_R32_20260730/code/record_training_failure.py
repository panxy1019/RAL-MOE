#!/usr/bin/env python3
"""Write an immutable, machine-readable fail-closed training record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status_path = args.run_dir / "runtime_status.json"
    runtime = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.is_file()
        else {}
    )
    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    error_lines = [
        line.strip()
        for line in log_text.splitlines()
        if "Error" in line or "nonfinite" in line
    ]
    payload = {
        "schema_version": 1,
        "status": "training_failed",
        "variant": args.variant,
        "seed": args.seed,
        "exit_code": args.exit_code,
        "last_finite_step": runtime.get("optimizer_step"),
        "last_horizon": runtime.get("horizon"),
        "error": error_lines[-1] if error_lines else "nonzero training exit",
        "run_dir": str(args.run_dir),
        "log": str(args.log),
        "log_sha256": hashlib.sha256(args.log.read_bytes()).hexdigest(),
        "final_training_checkpoint_present": (
            args.run_dir / "final_training.pt"
        ).is_file(),
        "heldout_evaluation_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
