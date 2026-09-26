#!/usr/bin/env python3
"""Evaluation-only completion of frozen DataOnly-MoE metrics for revision 10.

This script never trains, saves, or selects a checkpoint.  It reuses the
original DataOnly rollout contract and adds only coefficient errors plus the
previously omitted same-Re attractor diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_dataonly_missing_contracts as base


EPS = 1.0e-12


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def coefficient_errors(pred_a, pred_b, true_a, true_b) -> dict[str, float]:
    return {
        "velocity_coefficient_relative_l2": float(
            np.linalg.norm(pred_a - true_a) / (np.linalg.norm(true_a) + EPS)
        ),
        "pressure_coefficient_relative_l2": float(
            np.linalg.norm(pred_b - true_b) / (np.linalg.norm(true_b) + EPS)
        ),
    }


def rollout_fields(training, evaluator, checkpoint, data, starts, horizon):
    pred_a, pred_b, true_a, true_b = evaluator.dataonly_predictions(
        training, checkpoint, data, starts, horizon
    )
    return (
        base.physical_metrics(evaluator, data, pred_a, pred_b, true_a, true_b)
        | coefficient_errors(pred_a, pred_b, true_a, true_b),
        pred_a,
        pred_b,
        true_a,
        true_b,
    )


def evaluate_steady(root, training, evaluator, checkpoint, data):
    requested = [24.630436, 32.740068, 39.685479, 45.142703]
    legal = evaluator.legal_starts(data, "heldout", 56)
    selected = base.nearest_rows(data, requested, legal)
    terminal = base.steady_terminal_diagnostic(
        training, evaluator, checkpoint, data, requested
    )
    rows = {}
    for re_value, starts in selected.items():
        # Match the frozen evaluator's stride while retaining at least one
        # legal K56 window for every complete-Re heldout trajectory.
        starts = starts[::8]
        if not len(starts):
            starts = selected[re_value][:1]
        fields, *_ = rollout_fields(training, evaluator, checkpoint, data, starts, 56)
        key = f"{re_value:.6f}"
        terminal_row = terminal[key]
        rows[key] = {
            "split": "heldout",
            "K56": fields,
            "terminal_diagnostic": terminal_row,
            "strict_pressure_pass": bool(
                fields["finite_fraction"] == 1.0
                and fields["divergent_windows"] == 0
                and terminal_row["strict_pressure_pass"]
            ),
            "window_count": int(len(starts)),
        }
    return rows


def evaluate_hopf(root, training, evaluator, checkpoint, data, values, split):
    legal = evaluator.legal_starts(data, split, 56)
    selected = base.nearest_rows(data, values, legal)
    hopf = base.load_module(
        "revision10_hopf_evaluator",
        root / "Hopf/migrated_h4_expanded/code/evaluate_h4_expanded.py",
    )
    contract = np.load(
        root / "Hopf/migrated_h4_expanded/trainonly_contract/trainonly_fluctuation_contract.npz",
        allow_pickle=False,
    )
    rows = {}
    for requested, starts in selected.items():
        starts = base.evenly(starts, 5)
        fields, pred_a, _pred_b, true_a, _true_b = rollout_fields(
            training, evaluator, checkpoint, data, starts, 56
        )
        center = hopf.interp_np(contract["nodes"], contract["mean_a"], requested)
        radial_scale = float(
            hopf.interp_np(contract["nodes"], contract["radial_scale"], requested)
        )
        attractor = hopf.phase_metrics(
            true_a,
            pred_a,
            base.times_for(data, starts, 56),
            center,
            contract["plane"] / max(radial_scale, EPS),
            np.ones(2, dtype=np.float64),
            float(contract["radial_floor"]),
            float(contract["growth_tolerance"]),
        )
        strict = (
            fields["finite_fraction"] == 1.0
            and fields["divergent_windows"] == 0
            and fields["velocity_area_weighted_physical_relative_l2"] <= 0.05
            and fields["pressure_area_weighted_physical_relative_l2"] <= 0.05
            and attractor["rms_amplitude_error"] <= 0.10
            and attractor["peak_to_peak_amplitude_error"] <= 0.10
            and attractor["frequency_relative_error"] <= 0.05
            and abs(attractor["terminal_phase_drift_cycles"]) <= 0.25
            and attractor["normalized_orbit_distance"] <= 0.10
            and fields["velocity_energy_drift"] <= 0.10
            and fields["pressure_energy_drift"] <= 0.10
            and not attractor["false_growth"]
        )
        rows[f"{requested:.6f}"] = {
            "split": split,
            "K56": fields,
            "attractor_K56": attractor,
            "strict_preserved": bool(strict),
            "window_count": int(len(starts)),
        }
    return rows


def evaluate_periodic(root, training, evaluator, checkpoint, data):
    requested = [70.314635, 100.352251, 149.059229, 189.862278]
    legal = evaluator.legal_starts(data, "heldout", 48)
    selected = base.nearest_rows(data, requested, legal)
    periodic = base.load_module(
        "revision10_periodic_evaluator",
        root / "periodic_specialist_r32/code/evaluate_periodic_r32_portable.py",
    )
    rows = {}
    for re_value, starts in selected.items():
        starts = starts[::8]
        if not len(starts):
            starts = selected[re_value][:1]
        fields, pred_a, _pred_b, true_a, _true_b = rollout_fields(
            training, evaluator, checkpoint, data, starts, 48
        )
        ids = np.flatnonzero(np.isclose(data["re"], re_value, atol=5e-4, rtol=0))
        full_true = np.asarray(data["a"][ids], dtype=np.float64)
        center = np.mean(full_true, axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(full_true - center, full_matrices=False)
        components = vt[:2]
        orbit_scale = np.maximum(
            np.std((full_true - center) @ components.T, axis=0, keepdims=True), EPS
        )
        times = base.times_for(data, starts, 48)
        windows = [
            periodic.trajectory_metrics(
                true_a[index], pred_a[index], times[index], center, components, orbit_scale
            )
            for index in range(len(starts))
        ]
        cycle = {key: float(np.mean([row[key] for row in windows])) for key in windows[0]}
        strict = (
            fields["finite_fraction"] == 1.0
            and fields["divergent_windows"] == 0
            and fields["velocity_area_weighted_physical_relative_l2"] <= 0.05
            and fields["pressure_area_weighted_physical_relative_l2"] <= 0.05
            and cycle["rms_amplitude_error"] <= 0.10
            and cycle["peak_to_peak_amplitude_error"] <= 0.10
            and cycle["strouhal_error"] <= 0.05
            and abs(cycle["terminal_cycle_drift"]) <= 0.25
            and cycle["normalized_orbit_distance"] <= 0.10
            and fields["velocity_energy_drift"] <= 0.10
            and fields["pressure_energy_drift"] <= 0.10
        )
        rows[f"{re_value:.6f}"] = {
            "split": "heldout",
            "K48": fields,
            "cycle_K48": cycle,
            "strict_preserved": bool(strict),
            "window_count": int(len(starts)),
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    code = args.root / "paper_experiments/code"
    evaluator = base.load_module("revision10_base_evaluator", code / "evaluate_specialist_ablation_final.py")
    training = base.load_module("revision10_dataonly_training", code / "train_dataonly_moe.py")
    checkpoints = {
        "steady": args.root / "paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/steady/data-only/training/best_validation.pt",
        "hopf": args.root / "paper_experiments/runs/specialist_ablations_20260723_v2/hopf/dataonly/seed_1248/best_validation.pt",
        "periodic": args.root / "paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/periodic/data-only/training/best_validation.pt",
    }
    payload = {
        "schema": "revision10_dataonly_completion/v1",
        "policy": {"training": False, "checkpoint_modified": False, "test_used_for_selection": False},
        "results": {},
    }
    for regime, checkpoint_path in checkpoints.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        data = evaluator.load_full_data(args.root, regime, training)
        if regime == "steady":
            rows = evaluate_steady(args.root, training, evaluator, checkpoint, data)
        elif regime == "hopf":
            rows = {
                "heldout": evaluate_hopf(args.root, training, evaluator, checkpoint, data, [47.081356, 49.022357, 51.786450], "heldout"),
                "specified_train": evaluate_hopf(args.root, training, evaluator, checkpoint, data, [49.3, 49.6, 50.0], "train"),
            }
        else:
            rows = evaluate_periodic(args.root, training, evaluator, checkpoint, data)
        payload["results"][regime] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": digest(checkpoint_path),
            "checkpoint_step": int(checkpoint["step"]),
            "rows": rows,
        }
        print(json.dumps({"event": "regime_complete", "regime": regime}), flush=True)
        del checkpoint
        torch.cuda.empty_cache()
    output = args.output_dir / "REVISION10_DATAONLY_COMPLETION.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
