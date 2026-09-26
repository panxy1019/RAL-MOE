#!/usr/bin/env python3
"""Read-only checkpoint/schema inventory for the visualization reruns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def compact(value):
    if isinstance(value, dict):
        return {"keys": sorted(value)[:40], "count": len(value)}
    if hasattr(value, "shape"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return str(value)[:300]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    selections = [
        "paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/steady/"
        "vanilla-fnn/FROZEN_SELECTION.json",
        "paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/"
        "hopf_v6/hopf/vanilla-fnn/FROZEN_SELECTION.json",
        "paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/"
        "periodic_v4/periodic/vanilla-fnn/FROZEN_SELECTION.json",
    ]
    paths: list[Path] = []
    for relative in selections:
        path = root / relative
        print("SELECTION", path, path.exists())
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        print(json.dumps(payload, indent=2, default=str))
        for key, value in payload.items():
            if "checkpoint" in key.lower() and isinstance(value, str):
                candidate = Path(value)
                if candidate.exists():
                    paths.append(candidate)
    proposed_patterns = [
        "steady_specialist_v1/**/*best*.pt",
        "Hopf/migrated_h4_expanded/**/*best*.pt",
        "periodic_specialist_r32/**/*best*.pt",
    ]
    for pattern in proposed_patterns:
        candidates = sorted(root.glob(pattern))
        print("PATTERN", pattern, len(candidates))
        for path in candidates[-12:]:
            print("  ", path)
    for path in dict.fromkeys(paths):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        print("CHECKPOINT", path)
        print({key: compact(value) for key, value in checkpoint.items()})


if __name__ == "__main__":
    main()
