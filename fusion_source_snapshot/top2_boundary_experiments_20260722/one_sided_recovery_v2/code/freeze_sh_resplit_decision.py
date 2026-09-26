"""Freeze the validation-only promotion decision for the Re-resplit routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


COMPLEXITY = {
    "T2-C_LearnedConvexCorrection_FieldBlend": 0,
    "RiskPredictionRouter": 1,
    "LookAheadShortRolloutRouter": 2,
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
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    marker = args.training_dir / "ALL_ROUTES_VALIDATION_FROZEN.json"
    if not marker.is_file():
        raise RuntimeError("all routes are not validation-frozen")
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    candidates = []
    for method, payload in comparison["routes"].items():
        metrics = payload["validation"]
        if metrics["finite_fraction"] != 1.0 or metrics["divergent_windows"] != 0:
            continue
        worst_re_mean = max(float(item["mean"]) for item in metrics["per_Re"].values())
        candidates.append(
            {
                "method": method,
                "validation_worst_Re_mean": worst_re_mean,
                "validation_joint_all_mean": float(metrics["joint_all_mean"]),
                "validation_joint_all_worst": float(metrics["joint_all_worst"]),
                "complexity_rank": COMPLEXITY[method],
            }
        )
    if len(candidates) != len(COMPLEXITY):
        raise RuntimeError("one or more routes failed the finite/divergence gate")
    promoted = min(
        candidates,
        key=lambda item: (
            item["validation_worst_Re_mean"],
            item["validation_joint_all_mean"],
            item["complexity_rank"],
        ),
    )
    decision = {
        "status": "VALIDATION_DECISION_FROZEN_BEFORE_FINAL_TEST_CACHE",
        "decision_basis": "validation only",
        "selection_order": [
            "finite_fraction == 1 and divergent_windows == 0",
            "minimum validation worst-Re mean joint physical-field error",
            "minimum validation overall mean joint physical-field error",
            "lower complexity only as a final tie-break",
        ],
        "candidates": candidates,
        "promoted_method": promoted["method"],
        "comparison_sha256": sha256(args.comparison),
        "all_routes_marker_sha256": sha256(marker),
        "test_used_for_selection": False,
        "known_prior_test_disclosure": True,
    }
    atomic_json(args.output, decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
