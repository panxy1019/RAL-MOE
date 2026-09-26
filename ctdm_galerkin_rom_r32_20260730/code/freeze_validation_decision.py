#!/usr/bin/env python3
"""Freeze the pre-registered validation-only decision and gated test access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--failure",
        type=Path,
        action="append",
        default=[],
        help="Recorded fail-closed training attempt; never authorizes test access.",
    )
    parser.add_argument(
        "--candidate-variant",
        choices=("b1", "b2", "b3", "b4"),
        action="append",
        required=True,
        help=(
            "Variant that was finite with zero divergence in the frozen "
            "seed-1248 screening run. Repeat once per candidate."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def relative_degradation(candidate: float, reference: float) -> float:
    return (candidate - reference) / max(abs(reference), 1.0e-12)


def consistency_score(payload: dict[str, Any]) -> float:
    rows = [
        row
        for row in payload["timestep_consistency"]
        if row["coarse_substeps"] == 1 and row["fine_substeps"] == 2
    ]
    values = []
    for row in rows:
        components = [row["C_u"], row["C_p"]]
        if np.isfinite(row["C_S"]):
            components.append(row["C_S"])
        values.append(float(np.mean(components)))
    return float(np.mean(values))


def main() -> None:
    args = parse_args()
    analyses = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.analysis
    ]
    failures = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.failure
    ]
    for failure in failures:
        if failure.get("status") != "training_failed":
            raise RuntimeError(f"invalid failure record: {failure}")
        if failure.get("variant") not in {"b1", "b2", "b3", "b4"}:
            raise RuntimeError(f"invalid failure variant: {failure}")
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in analyses:
        if payload.get("split") != "validation":
            raise RuntimeError("freeze inputs must be validation analyses")
        by_variant[payload["variant"]].append(payload)
    if set(by_variant) != {"b1", "b2", "b3", "b4"}:
        raise RuntimeError(f"expected B1--B4, got {sorted(by_variant)}")
    screening_candidates = set(args.candidate_variant)
    screening_by_variant: dict[str, dict[str, Any]] = {}
    for variant, payloads in by_variant.items():
        screening = [
            payload for payload in payloads if int(payload.get("seed", -1)) == 1248
        ]
        if len(screening) != 1:
            raise RuntimeError(
                f"{variant} requires exactly one seed-1248 screening analysis"
            )
        screening_by_variant[variant] = screening[0]
    observed_screening_candidates = {
        variant
        for variant, payload in screening_by_variant.items()
        if payload["main"]["aggregate"]["finite_fraction_min"] == 1.0
        and payload["main"]["aggregate"]["divergent_windows"] == 0
    }
    if screening_candidates != observed_screening_candidates:
        raise RuntimeError(
            "declared screening candidates do not match frozen seed-1248 "
            f"validation: declared={sorted(screening_candidates)}, "
            f"observed={sorted(observed_screening_candidates)}"
        )
    checkpoints = {sha256(path): path for path in args.checkpoint}
    for payload in analyses:
        if payload["checkpoint_sha256"] not in checkpoints:
            raise RuntimeError(
                f"missing checkpoint for {payload['checkpoint_sha256']}"
            )
    summary: dict[str, Any] = {}
    all_finite_candidates: list[str] = []
    for variant, payloads in sorted(by_variant.items()):
        aggregates = [payload["main"]["aggregate"] for payload in payloads]
        finite = all(
            row["finite_fraction_min"] == 1.0
            and row["divergent_windows"] == 0
            for row in aggregates
        )
        if finite:
            all_finite_candidates.append(variant)
        summary[variant] = {
            "seeds": len(payloads),
            "all_finite_zero_divergence": finite,
            "Eu_mean": float(np.mean([row["Eu"] for row in aggregates])),
            "Ep_mean": float(np.mean([row["Ep"] for row in aggregates])),
            "joint_mean": float(
                np.mean([row["Eu"] + row["Ep"] for row in aggregates])
            ),
            "joint_std": float(
                np.std([row["Eu"] + row["Ep"] for row in aggregates])
            ),
            "worst_window_mean": float(
                np.mean([row["worst_window_joint"] for row in aggregates])
            ),
            "consistency_mean": float(
                np.mean([consistency_score(payload) for payload in payloads])
            ),
        }
    b3, b4 = summary["b3"], summary["b4"]
    mean_improved = b4["joint_mean"] < b3["joint_mean"]
    worst_improved = b4["worst_window_mean"] < b3["worst_window_mean"]
    mean_degradation = relative_degradation(
        b4["joint_mean"], b3["joint_mean"]
    )
    worst_degradation = relative_degradation(
        b4["worst_window_mean"], b3["worst_window_mean"]
    )
    paired_prediction_gate = (
        (mean_improved and worst_degradation <= 0.05)
        or (worst_improved and mean_degradation <= 0.05)
    )
    timestep_gate = b4["consistency_mean"] < b3["consistency_mean"]
    memory_rows = [
        row
        for payload in by_variant["b4"]
        for row in payload["memory_diagnostics"]
        if row["warmup_length"] == 3
    ]
    memory_nondegenerate = bool(memory_rows) and (
        min(row["eta_mean"] for row in memory_rows) > 1.0e-5
        and min(row["memory_frobenius_mean"] for row in memory_rows) > 1.0e-8
        and min(row["memory_effective_rank_mean"] for row in memory_rows) > 1.05
    )
    three_seed_complete = all(
        len(by_variant[key]) >= 3 for key in screening_candidates
    )
    attempted_seeds = {
        variant: sorted(
            {int(payload["seed"]) for payload in by_variant[variant]}
            | {
                int(failure["seed"])
                for failure in failures
                if failure["variant"] == variant
            }
        )
        for variant in sorted(screening_candidates)
    }
    confirmatory_attempts_complete = all(
        set(attempted_seeds[variant]) >= {1248, 2248, 3248}
        for variant in screening_candidates
    )
    test_access = (
        three_seed_complete
        and not failures
        and {"b3", "b4"}.issubset(screening_candidates)
        and b4["all_finite_zero_divergence"]
        and b3["all_finite_zero_divergence"]
        and paired_prediction_gate
        and timestep_gate
        and memory_nondegenerate
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen: list[dict[str, Any]] = []
    for digest, source in sorted(checkpoints.items()):
        target = args.output_dir / f"{source.parent.name}_{source.name}"
        if target.exists():
            if sha256(target) != digest:
                raise RuntimeError(f"existing frozen file hash mismatch: {target}")
        else:
            shutil.copy2(source, target)
        frozen.append(
            {
                "source": str(source),
                "frozen": str(target),
                "sha256": digest,
                "bytes": target.stat().st_size,
            }
        )
    manifest = {
        "schema_version": 1,
        "selection_split": "validation_only",
        "validation_frozen": True,
        "seed1248_screening_candidates": sorted(screening_candidates),
        "three_seed_complete": three_seed_complete,
        "confirmatory_attempts_complete": confirmatory_attempts_complete,
        "attempted_seeds": attempted_seeds,
        "training_failures": failures,
        "finite_zero_divergence_candidates": all_finite_candidates,
        "summary": summary,
        "preregistered_gates": {
            "B4_vs_B3_prediction_gate": paired_prediction_gate,
            "B4_timestep_consistency_better_than_B3": timestep_gate,
            "B4_memory_nondegenerate": memory_nondegenerate,
        },
        "test_access_authorized": test_access,
        "authorized_checkpoint_sha256s": (
            [row["sha256"] for row in frozen] if test_access else []
        ),
        "frozen_checkpoints": frozen,
    }
    atomic_json(manifest, args.output_dir / "VALIDATION_FREEZE_MANIFEST.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
