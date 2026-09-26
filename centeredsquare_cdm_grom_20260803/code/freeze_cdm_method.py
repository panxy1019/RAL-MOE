#!/usr/bin/env python3
"""Freeze the CDM-GROM implementation after validation and before held-out test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-metrics", type=Path, required=True)
    parser.add_argument("--operator-asset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--operator-builder", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    validation = json.loads(args.validation_metrics.read_text())
    if validation.get("role") != "validation":
        raise RuntimeError("input metrics are not a validation run")
    if validation.get("heldout_loaded") is not False:
        raise RuntimeError("validation run is contaminated by held-out fields")
    operator_hash = sha256(args.operator_asset)
    if validation.get("operator_sha256") != operator_hash:
        raise RuntimeError("validation/operator hash mismatch")
    payload = {
        "schema_version": 1,
        "status": "FROZEN",
        "method": "CenteredSquare Hopf operator-based CDM-GROM r11+u53",
        "freeze_basis": "all choices fixed after validation and before held-out field access",
        "validation_status": validation.get("status"),
        "validation_aggregate": validation.get("aggregate"),
        "operator_sha256": operator_hash,
        "validation_metrics_sha256": sha256(args.validation_metrics),
        "config_sha256": sha256(args.config),
        "operator_builder_sha256": sha256(args.operator_builder),
        "evaluator_sha256": sha256(args.evaluator),
        "heldout_loaded_at_freeze": False,
        "immutable_choices": {
            "resolved_rank": 11,
            "unresolved_rank": 53,
            "warmup_intervals": validation["warmup_intervals"],
            "max_internal_step": validation["max_internal_step"],
            "parameter_interpolation": "piecewise linear with two-node linear extrapolation",
            "pressure": "algebraic pressure-Poisson map from predicted velocity coordinates",
            "stability": "node and interpolated A_uu uniform spectral-abscissa shift to margin",
        },
    }
    atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
