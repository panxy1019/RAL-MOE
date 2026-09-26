#!/usr/bin/env python3
"""Read-only inventory for the three frozen Pinball B1 expert bundles."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy")
EXPERT_ROOTS = {
    "steady": ROOT / "Pinball/steady_b1_rank999_20260806",
    "hopf": ROOT / "Pinball/hopf_b1_rank999_20260806",
    "periodic": ROOT / "particalMOE/fluidic_pinball_periodic_v2_b1",
}


def describe_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: {"shape": list(source[key].shape), "dtype": str(source[key].dtype)} for key in source.files}


def describe_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    result = {"keys": sorted(checkpoint)}
    for key in ("optimizer_step", "step", "variant", "shape_contract", "median_native_dt", "asset_hashes"):
        if key in checkpoint:
            value = checkpoint[key]
            result[key] = value.tolist() if isinstance(value, np.ndarray) else value
    if "norm_stats" in checkpoint:
        result["norm_stats"] = {key: list(np.asarray(value).shape) for key, value in checkpoint["norm_stats"].items()}
    return result


def main() -> None:
    report = {}
    for name, root in EXPERT_ROOTS.items():
        row = {"root": str(root), "exists": root.exists()}
        if root.exists():
            row["top_level"] = sorted(path.name for path in root.iterdir())
            row["files"] = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())
            row["npz"] = {str(path.relative_to(root)): describe_npz(path) for path in root.rglob("*.npz")}
            row["checkpoints"] = {
                str(path.relative_to(root)): describe_checkpoint(path)
                for path in root.rglob("best_validation.pt")
            }
        report[name] = row
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
