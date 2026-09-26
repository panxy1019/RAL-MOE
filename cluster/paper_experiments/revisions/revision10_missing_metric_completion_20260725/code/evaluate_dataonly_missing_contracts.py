#!/usr/bin/env python3
"""Evaluate frozen DataOnly checkpoints on missing same-Re attractor contracts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


EPS = 1.0e-12


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def nearest_rows(data, values, starts):
    output = {}
    for requested in values:
        mask = np.isclose(data["re"][starts], requested, atol=5.0e-4, rtol=0)
        selected = starts[mask]
        if not len(selected):
            raise RuntimeError(f"No legal window at Re={requested}")
        actual = float(np.mean(data["re"][selected]))
        if abs(actual - requested) > 5.0e-4:
            raise RuntimeError(f"Re mismatch: requested={requested}, actual={actual}")
        output[requested] = selected
    return output


def evenly(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) <= count:
        return values
    return values[np.linspace(0, len(values) - 1, count, dtype=np.int64)]


def times_for(data, starts, horizon):
    current = starts.copy()
    times = []
    for _ in range(horizon):
        current = data["next"][current]
        times.append(data["time"][current])
    return np.stack(times, axis=1)


def physical_metrics(evaluator, data, pred_a, pred_b, true_a, true_b):
    velocity_geometry = evaluator.geometry(
        data["phi_u"], data["mean_u"], data["areas"], True
    )
    pressure_geometry = evaluator.geometry(
        data["phi_p"], data["mean_p"], data["areas"], False
    )
    flat_pa = pred_a.reshape(-1, 32)
    flat_pb = pred_b.reshape(-1, 32)
    flat_ta = true_a.reshape(-1, 32)
    flat_tb = true_b.reshape(-1, 32)
    velocity_true_energy = evaluator.field_energy(flat_ta, velocity_geometry)
    pressure_true_energy = evaluator.field_energy(flat_tb, pressure_geometry)
    velocity_pred_energy = evaluator.field_energy(flat_pa, velocity_geometry)
    pressure_pred_energy = evaluator.field_energy(flat_pb, pressure_geometry)
    finite = np.isfinite(pred_a).all(2) & np.isfinite(pred_b).all(2)
    true_u_max = np.maximum(
        np.max(np.linalg.norm(true_a, axis=2), axis=1, keepdims=True), EPS
    )
    true_p_max = np.maximum(
        np.max(np.linalg.norm(true_b, axis=2), axis=1, keepdims=True), EPS
    )
    bad = (
        (~finite)
        | (np.linalg.norm(pred_a, axis=2) > 10.0 * true_u_max)
        | (np.linalg.norm(pred_b, axis=2) > 10.0 * true_p_max)
    )
    return {
        "velocity_area_weighted_physical_relative_l2": math.sqrt(
            float(
                np.nansum(
                    evaluator.error_energy(flat_pa - flat_ta, velocity_geometry)
                )
            )
            / (float(np.sum(velocity_true_energy)) + EPS)
        ),
        "pressure_area_weighted_physical_relative_l2": math.sqrt(
            float(
                np.nansum(
                    evaluator.error_energy(flat_pb - flat_tb, pressure_geometry)
                )
            )
            / (float(np.sum(pressure_true_energy)) + EPS)
        ),
        "velocity_energy_drift": abs(
            float(np.nanmean(velocity_pred_energy))
            - float(np.mean(velocity_true_energy))
        )
        / (abs(float(np.mean(velocity_true_energy))) + EPS),
        "pressure_energy_drift": abs(
            float(np.nanmean(pressure_pred_energy))
            - float(np.mean(pressure_true_energy))
        )
        / (abs(float(np.mean(pressure_true_energy))) + EPS),
        "finite_fraction": float(np.mean(finite)),
        "divergent_windows": int(np.sum(np.any(bad, axis=1))),
    }


def load_dataonly_model(training, checkpoint, data):
    args = SimpleNamespace(**checkpoint["args"])
    groups = 1 if args.regime == "hopf" else 3
    hidden = 256 if args.regime == "hopf" else 224
    expert_hidden = 1024 if args.regime == "hopf" else 768
    blocks = 4 if args.regime == "hopf" else 3
    model = training.HierarchicalSparseMoE(
        3 * 64 + 1 + 2 * data["phase_harmonics"],
        64,
        hidden,
        expert_hidden,
        groups,
        6,
        blocks,
        0.04,
    ).cuda()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    normalization = checkpoint["normalization"]
    mean = torch.as_tensor(normalization["mean"], device="cuda")
    scale = torch.as_tensor(normalization["scale"], device="cuda")
    train = training.load_data(Path(args.root), args.regime)
    re_train = train["re"][train["split"] == "train"]
    re_center = float(np.mean(re_train))
    re_scale = float(max(np.std(re_train), 1.0))
    return model, mean, scale, re_center, re_scale


def steady_terminal_diagnostic(training, evaluator, checkpoint, data, re_values):
    model, mean, scale, re_center, re_scale = load_dataonly_model(
        training, checkpoint, data
    )
    states = np.concatenate([data["a"], data["b"]], axis=1).astype(np.float32)
    fixed_ids = []
    for re_value in re_values:
        ids = np.flatnonzero(
            np.isclose(data["re"], re_value, atol=5.0e-4, rtol=0)
        )
        fixed_ids.append(int(ids[-1]))
    initial = torch.as_tensor(states[fixed_ids], device="cuda")
    history = initial[:, None].repeat(1, 3, 1)
    re_tensor = torch.as_tensor(
        ((data["re"][fixed_ids] - re_center) / re_scale)[:, None],
        device="cuda",
    )
    with torch.inference_mode():
        for _ in range(56):
            normalized = (history - mean) / scale
            delta, _ = model(torch.cat([normalized.flatten(1), re_tensor], 1))
            next_state = history[:, 0] + delta.float() * scale
            history = torch.cat(
                [next_state[:, None], history[:, :-1]], dim=1
            )
    final = next_state.cpu().numpy()
    pressure_geometry = evaluator.geometry(
        data["phi_p"], data["mean_p"], data["areas"], False
    )
    rows = {}
    for column, re_value in enumerate(re_values):
        true_b = states[fixed_ids[column], 32:]
        pred_b = final[column, 32:]
        fixed_error = math.sqrt(
            float(evaluator.error_energy(pred_b - true_b, pressure_geometry))
            / (
                float(evaluator.field_energy(true_b, pressure_geometry))
                + EPS
            )
        )
        drift = float(
            np.linalg.norm(pred_b - true_b) / (np.linalg.norm(true_b) + EPS)
        )
        rows[f"{re_value:.6f}"] = {
            "pressure_fixed_point_physical_relative_l2": fixed_error,
            "pressure_drift_coefficient_relative_l2": drift,
            "finite": bool(np.isfinite(final[column]).all()),
            "strict_pressure_pass": bool(
                np.isfinite(final[column]).all()
                and fixed_error <= 0.05
                and drift <= 0.05
            ),
        }
    del model
    torch.cuda.empty_cache()
    return rows


def evaluate_steady(training, evaluator, checkpoint, data, re_values):
    legal = evaluator.legal_starts(data, "train", 56)
    legal = np.concatenate(
        [
            legal,
            evaluator.legal_starts(data, "validation", 56),
            evaluator.legal_starts(data, "heldout", 56),
        ]
    )
    selected = nearest_rows(data, re_values, legal)
    output = {}
    for requested, starts in selected.items():
        starts = starts[::8]
        predictions = evaluator.dataonly_predictions(
            training, checkpoint, data, starts, 56
        )
        aggregate = evaluator.aggregate(data, starts, *predictions)
        actual_key = min(
            aggregate["by_re"],
            key=lambda key: abs(float(key) - requested),
        )
        output[f"{requested:.6f}"] = {
            "split": str(data["split"][starts[0]]),
            "K56": aggregate["by_re"][actual_key]["k56"],
        }
    terminal = steady_terminal_diagnostic(
        training, evaluator, checkpoint, data, re_values
    )
    for key, value in output.items():
        value["terminal_diagnostic"] = terminal[key]
    return output


def evaluate_hopf(root, training, evaluator, checkpoint, data):
    re_values = [49.3, 49.6, 50.0]
    legal = evaluator.legal_starts(data, "train", 56)
    selected = nearest_rows(data, re_values, legal)
    hopf_evaluator = load_module(
        "revision5_hopf_evaluator",
        root / "Hopf/migrated_h4_expanded/code/evaluate_h4_expanded.py",
    )
    contract = np.load(
        root
        / "Hopf/migrated_h4_expanded/trainonly_contract/"
        "trainonly_fluctuation_contract.npz",
        allow_pickle=False,
    )
    rows = {}
    for requested, starts in selected.items():
        starts = evenly(starts, 5)
        pred_a, pred_b, true_a, true_b = evaluator.dataonly_predictions(
            training, checkpoint, data, starts, 56
        )
        fields = physical_metrics(
            evaluator, data, pred_a, pred_b, true_a, true_b
        )
        center = hopf_evaluator.interp_np(
            contract["nodes"], contract["mean_a"], requested
        )
        radial_scale = float(
            hopf_evaluator.interp_np(
                contract["nodes"], contract["radial_scale"], requested
            )
        )
        attractor = hopf_evaluator.phase_metrics(
            true_a,
            pred_a,
            times_for(data, starts, 56),
            center,
            contract["plane"] / max(radial_scale, EPS),
            np.ones(2, dtype=np.float64),
            float(contract["radial_floor"]),
            float(contract["growth_tolerance"]),
        )
        preserved = (
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
        rows[f"{requested:.1f}"] = {
            "split": "train",
            "K56": fields,
            "attractor_K56": attractor,
            "strict_preserved": bool(preserved),
        }
    return rows


def evaluate_periodic(training, evaluator, checkpoint, data):
    re_values = [100.352249, 149.059235, 189.862274]
    legal = evaluator.legal_starts(data, "heldout", 48)
    selected = nearest_rows(data, re_values, legal)
    periodic = load_module(
        "revision5_periodic_evaluator",
        Path(__file__).with_name("evaluate_periodic_r32_portable.py"),
    )
    velocity_geometry = evaluator.geometry(
        data["phi_u"], data["mean_u"], data["areas"], True
    )
    rows = {}
    for requested, starts in selected.items():
        starts = starts[::8]
        pred_a, pred_b, true_a, true_b = evaluator.dataonly_predictions(
            training, checkpoint, data, starts, 48
        )
        fields = physical_metrics(
            evaluator, data, pred_a, pred_b, true_a, true_b
        )
        ids = np.flatnonzero(
            np.isclose(data["re"], requested, atol=5.0e-4, rtol=0)
        )
        full_true = np.asarray(data["a"][ids], dtype=np.float64)
        center = np.mean(full_true, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(full_true - center, full_matrices=False)
        components = vt[:2]
        full_orbit = (full_true - center) @ components.T
        orbit_scale = np.maximum(
            np.std(full_orbit, axis=0, keepdims=True), EPS
        )
        times = times_for(data, starts, 48)
        trajectories = [
            periodic.trajectory_metrics(
                true_a[i],
                pred_a[i],
                times[i],
                center,
                components,
                orbit_scale,
            )
            for i in range(len(starts))
        ]
        attractor = {
            key: float(np.mean([row[key] for row in trajectories]))
            for key in trajectories[0]
        }
        preserved = (
            fields["finite_fraction"] == 1.0
            and fields["divergent_windows"] == 0
            and fields["velocity_area_weighted_physical_relative_l2"] <= 0.05
            and fields["pressure_area_weighted_physical_relative_l2"] <= 0.05
            and attractor["rms_amplitude_error"] <= 0.10
            and attractor["peak_to_peak_amplitude_error"] <= 0.10
            and attractor["strouhal_error"] <= 0.05
            and abs(attractor["terminal_cycle_drift"]) <= 0.25
            and attractor["normalized_orbit_distance"] <= 0.10
            and fields["velocity_energy_drift"] <= 0.10
            and fields["pressure_energy_drift"] <= 0.10
        )
        rows[f"{requested:.6f}"] = {
            "split": "heldout",
            "K48": fields,
            "cycle_K48": attractor,
            "strict_preserved": bool(preserved),
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
    evaluator = load_module(
        "revision5_dataonly_base_evaluator",
        code / "evaluate_specialist_ablation_final.py",
    )
    training = load_module(
        "revision5_dataonly_training", code / "train_dataonly_moe.py"
    )
    run = (
        args.root
        / "paper_experiments/runs/"
        "specialist_ablations_single_seed_v3_20260724"
    )
    checkpoints = {
        "steady": run / "steady/data-only/training/best_validation.pt",
        "hopf": (
            args.root
            / "paper_experiments/runs/specialist_ablations_20260723_v2/"
            "hopf/dataonly/seed_1248/best_validation.pt"
        ),
        "periodic": run / "periodic/data-only/training/best_validation.pt",
    }
    payload = {
        "schema": "dataonly_missing_contracts/v1",
        "steady_fixed_point_qualified_Re": [
            34.737568,
            36.657768,
            38.357250,
            39.685479,
            40.711525,
            41.576576,
        ],
        "results": {},
    }
    for regime in ("steady", "hopf", "periodic"):
        checkpoint = torch.load(
            checkpoints[regime], map_location="cpu", weights_only=False
        )
        data = evaluator.load_full_data(args.root, regime, training)
        if regime == "steady":
            result = evaluate_steady(
                training,
                evaluator,
                checkpoint,
                data,
                payload["steady_fixed_point_qualified_Re"],
            )
        elif regime == "hopf":
            result = evaluate_hopf(
                args.root, training, evaluator, checkpoint, data
            )
        else:
            result = evaluate_periodic(
                training, evaluator, checkpoint, data
            )
        payload["results"][regime] = {
            "checkpoint": str(checkpoints[regime]),
            "checkpoint_step": int(checkpoint["step"]),
            "by_re": result,
        }
        print(
            json.dumps({"event": "regime_complete", "regime": regime}),
            flush=True,
        )

    output = args.output_dir / "DATAONLY_MISSING_CONTRACTS.json"
    output.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    print(json.dumps({"event": "complete", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
