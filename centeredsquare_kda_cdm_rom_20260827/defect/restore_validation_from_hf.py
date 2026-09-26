#!/usr/bin/env python3
"""Restore exactly the six preregistered periodic validation trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "panxy1019/centeredSquare"
VALIDATION_RE = [99.0, 101.5, 110.344827586, 125.862068966, 141.379310345, 150.0]
PREFIX = "/home/ray/Desktop/centeredSquare/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.canonical_cases.read_text())
    selected = []
    for re_value in VALIDATION_RE:
        matches = [item for item in records if abs(float(item["Re"]) - re_value) < 5e-7]
        if len(matches) != 1:
            raise RuntimeError(f"canonical match failed for validation Re={re_value}")
        source = str(matches[0]["path"])
        if not source.startswith(PREFIX):
            raise RuntimeError(f"unexpected canonical path: {source}")
        repository_path = source.removeprefix(PREFIX)
        local = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename=repository_path, local_dir=args.output,
        )
        selected.append({"Re": re_value, "repository_path": repository_path, "local_path": local})
    payload = {
        "schema_version": 1, "role": "periodic_validation", "cases": selected,
        "validation_case_count": len(selected), "heldout_snapshots_downloaded": False,
    }
    (args.output / "VALIDATION_RESTORE.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
