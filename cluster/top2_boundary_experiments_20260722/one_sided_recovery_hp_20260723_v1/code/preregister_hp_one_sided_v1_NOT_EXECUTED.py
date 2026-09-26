#!/usr/bin/env python3
"""Create the immutable one-sided P-native H-P preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


METHODS = (
    "T2-C_LearnedConvexCorrection_FieldBlend",
    "RiskPredictionRouter",
    "LookAheadShortRolloutRouter",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-contract", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if preflight.get("decision") != "PASS":
        raise RuntimeError("formal G_HP_native_FULL_K56 has not passed")

    with np.load(args.projection_contract, allow_pickle=False) as archive:
        labels = archive["Re_labels"].astype(str)
        re_values = archive["Re_values"].astype(float)
        split_by_re = archive["split_by_Re"].astype(str)
    split = {
        role: [
            {"Re_label": str(label), "Re": float(re_value)}
            for label, re_value in zip(labels[split_by_re == role], re_values[split_by_re == role])
        ]
        for role in ("train", "validation", "heldout")
    }
    if {role: len(rows) for role, rows in split.items()} != {
        "train": 53,
        "validation": 6,
        "heldout": 4,
    }:
        raise RuntimeError("Periodic split must be 53/6/4 complete Re trajectories")
    validation_re = np.sort(
        np.asarray([row["Re"] for row in split["validation"]], dtype=np.float64)
    )
    region_threshold_candidates = (
        (0.5 * (validation_re[:-1] + validation_re[1:])).tolist() + [float("inf")]
    )
    boundary_upper_re = float(region_threshold_candidates[0])

    args.output_dir.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for method in METHODS:
        config = {
            "schema": "one_sided_native_supported_hp_route/v1",
            "method": method,
            "active_boundary": "H-P only",
            "source_database": "Periodic native trajectories",
            "allowed_pair": ["Hopf", "Periodic"],
            "forbidden": ["Steady-Periodic", "Top-3", "continuous_RHS_fusion", "fusion_feedback"],
            "specialists_frozen": True,
            "native_contract": {
                "Periodic": "native indexed phase/time and native pressure closure",
                "Hopf": "phase-free; same three physical history states projected to Hopf chart",
                "fusion": "strictly aligned physical-time output aggregation only",
                "trajectory_weight": "constant over complete K56 window/trajectory query",
            },
            "complete_Re_split": split,
            "windows_per_Re": 5,
            "horizon": 56,
            "window_selection": "five deterministic evenly spaced valid K56 starts per complete Re trajectory",
            "gate_features": [
                "Re",
                "chart-independent initial physical descriptors from the same three states/timestamps",
            ],
            "convex_weight_order": ["a_H", "a_P"],
            "velocity_policy": "convex H/P physical-field blend",
            "pressure_primary_policy": "pressure from the trajectory-level dominant expert",
            "pressure_diagnostic": "also report full convex H/P pressure blend",
            "optimizer_steps": 8000,
            "evaluation_every_steps": 100,
            "seed": 52001,
            "same_seed_across_methods": True,
            "boundary_upper_Re": boundary_upper_re,
            "boundary_rule": "midpoint between the first two sorted Periodic validation Re nodes, fixed from split metadata before route training",
            "region_threshold_candidates_recorded_but_not_searched": region_threshold_candidates,
            "region_policy": "H-P route at or below boundary_upper_Re; P-only above it",
            "selection_rule": [
                "finite_fraction equals 1 and divergent_windows equals 0",
                "illegal S-P and Top-3 activations equal 0",
                "minimize validation worst-Re K56 joint primary physical-field error",
                "then minimize validation mean K56 joint primary physical-field error",
                "then maximize periodic amplitude/frequency/phase/orbit preservation",
                "then prefer lower inference cost; complexity is never a positive tie-break",
            ],
            "test_access": "forbidden until all three validation-selected checkpoints, route thresholds, region threshold, and promotion decision are frozen",
            "known_prior_test_disclosure": "heldout Re identities were known from specialist work; this recovery does not claim a historically blind test and does not use heldout fields or metrics for tuning",
        }
        if method == "RiskPredictionRouter":
            config["additional_contract"] = {
                "risk_labels": "train/validation future native rollout error and divergence only",
                "selection_score_distinct_from_fusion_weight": True,
            }
        elif method == "LookAheadShortRolloutRouter":
            config["additional_contract"] = {
                "lookahead_inputs": "prediction-only finite/drift/energy/growth/amplitude/frequency/phase/orbit diagnostics",
                "future_truth_in_routing": False,
            }
        else:
            config["additional_contract"] = {
                "learned_quantity": "pair-specific trajectory-level convex correction"
            }
        path = args.output_dir / f"{method}.json"
        atomic_json(path, config)
        hashes[path.name] = sha256(path)

    manifest = {
        "status": "PREREGISTERED",
        "protocol_version": "ONE_SIDED_P_NATIVE_HP_20260723_V1",
        "split_counts": {role: len(rows) for role, rows in split.items()},
        "same_seed": 52001,
        "same_optimizer_steps": 8000,
        "same_native_cache_across_methods": True,
        "specialists_frozen": True,
        "test_sealed_until_validation_freeze": True,
        "formal_preflight_sha256": sha256(args.preflight),
        "projection_contract_sha256": sha256(args.projection_contract),
        "config_sha256": hashes,
    }
    atomic_json(args.output_dir / "PREREGISTRATION.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
