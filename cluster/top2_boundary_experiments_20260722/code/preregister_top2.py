"""Write the immutable train/validation-only protocol for three Top-2 routes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time


ROOT = Path("/root/panxy/particalMOE")
E2 = ROOT / "trajectory_router_e1_e2_e3_20260722/frozen_E2_baseline_candidate/best.pt"
SPECIALISTS = {
    "Steady": ROOT / "steady_specialist_v1/checkpoint/frozen_s4_validation_step_1200.pt",
    "Hopf": ROOT / "Hopf/migrated_h4_expanded/runs/HopfExpanded34_H4_NormalFormRadial_r32/best_validation.pt",
    "Periodic": ROOT / "periodic_specialist_r32/checkpoint/FINAL_PERIODIC_SPECIALIST.pt",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def common() -> dict[str, object]:
    return {
        "seed": 42001,
        "optimizer_steps": 8000,
        "batch_size": 256,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "evaluation_every_steps": 100,
        "windows_per_Re": 32,
        "history_length": 3,
        "split": "existing complete-Re and complete-trajectory train/validation/test isolation",
        "development_data_access": ["train", "validation"],
        "test_access": "forbidden until all three validation-selected checkpoints and configs are frozen",
        "allowed_pairs": ["Steady-Hopf", "Hopf-Periodic"],
        "forbidden_routes": ["Steady-Periodic", "Top-3"],
        "specialists": "frozen; native POD/scaler/history/integrator/pressure closure unchanged",
        "fusion": "trajectory-constant convex output-level physical-field aggregation at exactly aligned query times",
        "feedback": "fused state is never fed back into a specialist",
        "H_P_pressure_primary": "pressure from frozen-E2 dominant anchor specialist",
        "H_P_pressure_diagnostic": "full convex pressure blend is diagnostic only",
        "loss_scaling": "each physical/attractor component standardized by train-only median and MAD, then equal weighted",
        "boundary_threshold_grid": {
            "top1_confidence_max": [0.55, 0.65, 0.75, 0.85, 0.95],
            "top1_top2_margin_max": [0.1, 0.25, 0.5, 0.75],
            "selection": "shared threshold selected once using train group-CV then validation; frozen for all methods"
        },
        "selection_rule": [
            "finite_fraction must equal 1 and divergent_windows must equal 0",
            "illegal S-P and Top-3 activations must equal 0",
            "minimize validation worst-Re joint physical-field error",
            "then minimize validation mean joint physical-field error",
            "then maximize attractor preservation",
            "then prefer lower inference cost; complexity is never a positive tie-break"
        ],
        "metrics": [
            "area-weighted velocity and pressure field error at K1/K4/K8/K16/K56",
            "per-Re mean and worst, finite fraction, divergent windows",
            "fixed-point, growth, amplitude, frequency, phase and orbit preservation as applicable",
            "calibration, Top-2 usage, inference time/memory, oracle gap",
            "full H-P pressure blend diagnostic"
        ],
    }


def experiment_configs() -> dict[str, dict[str, object]]:
    base = common()
    return {
        "T2-C_LearnedConvexCorrection_FieldBlend": {
            **base,
            "method": "E2 selects a legal adjacent pair; a pair-specific convex gate predicts two nonnegative weights summing to one",
            "gate_inputs": ["Re", "8 chart-independent initial/history physical descriptors"],
            "gate": "pair-specific MLP 9->64->64->1; sigmoid convex weight",
            "initialization": "zero final correction, exactly reproducing masked normalized E2 pair probability at step 0",
            "trainable_modules": ["S-H convex correction gate", "H-P convex correction gate"],
            "loss_targets": ["decoded velocity field", "decoded pressure policy output", "pair-appropriate attractor diagnostics"],
        },
        "RiskPredictionRouter": {
            **base,
            "method": "separate candidate risk prediction from physical blending weight",
            "risk_inputs": ["Re", "chart-independent descriptors", "local representation error", "available no-future physical diagnostics", "candidate id"],
            "risk_outputs": ["long-field risk", "attractor risk", "divergence probability"],
            "risk_model": "shared 64x64 MLP with candidate embedding and three calibrated heads",
            "decision": "risk scores select Top-1 or legal adjacent Top-2; a separate pair-specific convex gate produces blend weights",
            "trainable_modules": ["risk predictor", "independent convex blend gate"],
            "calibration": "validation temperature scaling; test forbidden",
        },
        "LookAheadShortRolloutRouter": {
            **base,
            "method": "E2 selects a legal pair; both frozen candidates run the same future-truth-free short query interval before scoring",
            "lookahead_horizon_grid": [4, 8],
            "lookahead_inputs": ["finite", "drift", "energy", "growth", "fixed-point", "amplitude", "frequency", "phase", "orbit diagnostics when applicable"],
            "lookahead_model": "pair-specific 64x64 capability scorer plus independent convex blend gate",
            "decision": "capability score selects Top-1 or Top-2; blend gate controls physical aggregation only",
            "trainable_modules": ["lookahead capability scorer", "independent convex blend gate"],
            "truth_contract": "lookahead features use predictions and query clock only; no future true state or label",
        },
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty preregistration directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        "E2": {"path": str(E2), "sha256": sha256(E2)},
        **{name: {"path": str(path), "sha256": sha256(path)} for name, path in SPECIALISTS.items()},
    }
    configs = experiment_configs()
    config_hashes = {}
    for name, config in configs.items():
        path = args.output_dir / f"{name}.json"
        atomic_json(path, config)
        config_hashes[name] = sha256(path)
    atomic_json(args.output_dir / "PREREGISTRATION.json", {
        "status": "PRE_REGISTERED_PENDING_FAIL_CLOSED_PREFLIGHT",
        "created_unix": time.time(),
        "assets": assets,
        "config_sha256": config_hashes,
        "test_seal": {
            "state": "SEALED",
            "rule": "training and validation executables must reject split=test; a separate final evaluator may unlock only after three freeze manifests exist"
        },
        "known_history_disclosure": "Earlier project work already reported legacy test metrics. This run is mechanically test-sealed but is not represented as a never-observed blind test.",
    })
    for path in args.output_dir.iterdir():
        path.chmod(0o444)
    print(json.dumps({"status": "PRE_REGISTERED", "output": str(args.output_dir), "configs": config_hashes}, indent=2))


if __name__ == "__main__":
    main()
