"""Create the immutable preregistration for the explicit 2026-07-23 Re split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


METHODS = (
    "T2-C_LearnedConvexCorrection_FieldBlend",
    "RiskPredictionRouter",
    "LookAheadShortRolloutRouter",
)
SPLIT = {
    "train": [40.711525, 43.093929, 43.20, 43.40, 43.60, 43.797394, 44.478355],
    "validation": [41.576575, 43.30, 43.70],
    "heldout": [42.359071, 43.50, 43.90],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-prereg-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for method in METHODS:
        source = args.source_prereg_dir / f"{method}.json"
        config = json.loads(source.read_text(encoding="utf-8"))
        config.update(
            {
                "protocol_version": "ONE_SIDED_NATIVE_SUPPORTED_RESPLIT_20260723_V1",
                "complete_Re_split": SPLIT,
                "windows_per_Re": 5,
                "window_selection": "Five deterministic evenly spaced valid K56 starts per complete Re trajectory; no random window split.",
                "optimizer_steps": 8000,
                "evaluation_every_steps": 100,
                "seed": 42001,
                "same_seed_across_methods": True,
                "development_cache_counts": {"train": 35, "validation": 15},
                "final_test_cache_count": 15,
                "selection_rule": [
                    "finite_fraction must equal 1 and divergent_windows must equal 0",
                    "illegal S-P and Top-3 activations must equal 0",
                    "minimize validation worst-Re joint physical-field error",
                    "then minimize validation mean joint physical-field error",
                    "then maximize attractor preservation",
                    "then prefer lower inference cost; complexity is never a positive tie-break",
                ],
                "test_access": "Final-test cache construction and evaluation forbidden until all three validation-selected checkpoints, thresholds, and decision are frozen.",
                "known_prior_test_disclosure": "These Re and earlier test results were already visible in prior work; this rerun is not blind, but test values are excluded from optimization/checkpoint/threshold/promotion decisions.",
            }
        )
        destination = args.output_dir / f"{method}.json"
        atomic_json(destination, config)
        hashes[destination.name] = sha256(destination)
    manifest = {
        "status": "PREREGISTERED",
        "protocol_version": "ONE_SIDED_NATIVE_SUPPORTED_RESPLIT_20260723_V1",
        "complete_Re_split": SPLIT,
        "windows_per_Re": 5,
        "same_seed": 42001,
        "same_optimizer_steps": 8000,
        "same_native_cache_across_methods": True,
        "specialists_frozen": True,
        "fusion_feedback": False,
        "test_sealed_until_validation_freeze": True,
        "known_prior_test_disclosure": True,
        "config_sha256": hashes,
    }
    atomic_json(args.output_dir / "PREREGISTRATION.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
