#!/usr/bin/env python3
"""Evaluate one frozen Global MoE checkpoint on the frozen H/P windows.

This script is intended to run on the original cluster where the checkpoint,
global ROM assets, and sealed H/P test arrays are available.  It never trains.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shlex
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def center_pressure(phi: np.ndarray, mean: np.ndarray, weights: np.ndarray):
    phi = np.asarray(phi, dtype=np.float64).copy()
    mean = np.asarray(mean, dtype=np.float64).copy()
    phi -= (phi @ weights / weights.sum())[:, None]
    mean -= float(mean @ weights / weights.sum())
    return phi, mean


def cross_chart_geometry(
    global_phi: np.ndarray,
    global_mean: np.ndarray,
    local_phi: np.ndarray,
    local_mean: np.ndarray,
    weights: np.ndarray,
):
    delta_mean = np.asarray(global_mean, dtype=np.float64) - np.asarray(local_mean, dtype=np.float64)
    matrix = np.concatenate(
        [np.asarray(global_phi, dtype=np.float64), -np.asarray(local_phi, dtype=np.float64), delta_mean[None]],
        axis=0,
    )
    numerator_gram = (matrix * weights[None]) @ matrix.T
    truth_matrix = np.concatenate([np.asarray(local_phi, dtype=np.float64), np.asarray(local_mean, dtype=np.float64)[None]], axis=0)
    truth_gram = (truth_matrix * weights[None]) @ truth_matrix.T
    return numerator_gram, truth_gram


def gram_error(
    pred: np.ndarray,
    truth: np.ndarray,
    numerator_gram: np.ndarray,
    truth_gram: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([pred, truth, np.ones((len(pred), 1))], axis=1)
    truth_augmented = np.concatenate([truth, np.ones((len(truth), 1))], axis=1)
    numerator = np.einsum("bi,ij,bj->b", combined, numerator_gram, combined)
    denominator = np.einsum("bi,ij,bj->b", truth_augmented, truth_gram, truth_augmented)
    if not np.all(np.isfinite(denominator)) or not np.all(denominator > 0):
        raise RuntimeError(f"non-positive/non-finite reference norm: min={denominator.min()}")
    return 100.0 * np.sqrt(np.maximum(numerator, 0.0) / denominator), denominator


def global_args(trainer, argv: list[str]):
    original = sys.argv
    sys.argv = ["global_eval", *argv]
    try:
        return trainer.parse_args()
    finally:
        sys.argv = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--trainer", type=Path)
    args_cli = parser.parse_args()
    root = args_cli.root
    output = args_cli.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    checkpoint_path = args_cli.checkpoint or root / (
        "paper_experiments/revisions/revision5_supplemental_evaluation_20260725/"
        "recovery/global_k56_valid_history_windows_v5/"
        "V16_1_SteadyPressureAnchor32_ru32_rp32_Re_24p630436_checkpoint.pt"
    )
    trainer_path = args_cli.trainer or root / (
        "paper_experiments/revisions/revision10_missing_metric_completion_20260725/"
        "code/global_eval_revision10.py"
    )
    helper_path = root / "periodic_specialist_r32/code/evaluate_periodic_r32_portable.py"
    trainer = load_module("global_eval_reeval", trainer_path)
    helper = load_module("global_eval_helper", helper_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    frozen_argv = shlex.split(
        "--r-u 32 --r-p 32 --num-blocks 3 --num-regime-groups 3 "
        "--experts-per-group 6 --num-experts 6 --num-shared-experts 1 "
        "--top-k 2 --group-top-k 1 --hidden-dim 224 --expert-hidden 768 "
        "--expert-blocks 3 --quadratic-rank 4 --quadratic-scale 0.05 "
        "--dropout 0.04 --temperature 0.95 --gate-floor 0 "
        "--group-temperature 0.9 --group-gate-floor 0 --shared-scale 1 "
        "--routed-scale 0.85 --rhs-target residual --pressure-target closure "
        "--pressure-input-mode pressure_only --closure-mode adaptive_gate "
        "--pressure-base-mode static --attractor-balanced-sampling "
        "--test-re-selection regime_default --swanlab-mode offline --allow-tf32"
    )
    frozen = global_args(trainer, frozen_argv)
    model_args = SimpleNamespace(**vars(frozen))
    global_data = root / "V16_1_SteadyPressureAnchor32/assets/common_global_data"
    model_args.data_root = global_data / "Global_POD_AreaWeighted_L2"
    model_args.tensor_path = global_data / "semi_intrusive_galerkin_tensors_allRe100_areaWeightedL2_ru80_rp80_compact.npz"
    model_args.pressure_surrogate_path = global_data / "pressure_poisson_surrogate_tensors_allRe100_areaWeightedL2_ru80_rp80.npz"
    arrays, meta = trainer.build_arrays(model_args)
    tensors = np.load(model_args.tensor_path)
    pressure_tensors = np.load(model_args.pressure_surrogate_path)
    scalers = {
        key: trainer.Standardizer(
            mean=np.asarray(value["mean"], dtype=np.float32),
            scale=np.asarray(value["scale"], dtype=np.float32),
        )
        for key, value in checkpoint["scalers"].items()
    }
    device = torch.device("cuda")
    model = helper.instantiate_model(trainer, model_args, arrays, checkpoint["model_state"], device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    global_u = np.load(model_args.data_root / "global_velocity_pod_area_weighted_l2.npz")
    global_p = np.load(model_args.data_root / "global_pressure_pod_area_weighted_l2.npz")
    valid_samples = set(int(value) for value in arrays["sample_ids"].tolist())
    holdout_re = [float(value) for value in checkpoint.get("holdout_Re", [])]
    rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    direct_checks: list[dict[str, object]] = []
    split_audit: list[dict[str, object]] = []

    for regime_short, regime in (("H", "Hopf"), ("P", "Periodic")):
        sealed = np.load(root / f"experiments/external_rnn_fno_20260923/{regime_short}_sealed_test.npz")
        geometry = np.load(root / f"experiments/external_rnn_fno_20260923/{regime_short}_geometry.npz")
        local_u_phi = geometry["phi_u"].astype(np.float64)
        local_p_phi = geometry["phi_p"].astype(np.float64)
        local_u_mean = geometry["mean_u"].astype(np.float64)
        local_p_mean = geometry["mean_p"].astype(np.float64)
        weights = geometry["areas"].astype(np.float64)
        velocity_weights = np.tile(weights, 2)
        local_p_phi, local_p_mean = center_pressure(local_p_phi, local_p_mean, weights)
        np.testing.assert_allclose(global_u["point_areas"], weights, atol=1e-12, rtol=1e-8)
        np.testing.assert_allclose(global_p["point_areas"], weights, atol=1e-12, rtol=1e-8)
        if "points" in global_u.files and "points" in geometry.files:
            np.testing.assert_allclose(global_u["points"], geometry["points"], atol=1e-10, rtol=1e-8)

        starts = [int(value) for value in sealed["test_starts"]]
        for window_number, local_start in enumerate(starts):
            re_value = float(sealed["Re"][local_start])
            start_time = float(sealed["time"][local_start])
            matches = np.flatnonzero(
                np.isclose(arrays["re"], re_value, atol=2e-5, rtol=0)
                & np.isclose(arrays["time"], start_time, atol=1e-3, rtol=0)
            )
            if len(matches) != 1:
                raise RuntimeError(f"{regime} Re={re_value} time={start_time}: global matches={matches}")
            global_start = int(matches[0])
            if global_start not in valid_samples:
                raise RuntimeError(f"{regime}: global start {global_start} not a valid sample")
            mapping_rows.append(
                {
                    "regime": regime,
                    "window_number": window_number,
                    "window_id": f"{regime_short}:{int(sealed['global_ids'][local_start])}",
                    "Re": re_value,
                    "start_time": start_time,
                    "local_start": local_start,
                    "local_global_id": int(sealed["global_ids"][local_start]),
                    "global_model_start": global_start,
                }
            )

            global_re_index = int(np.argmin(np.abs(global_u["Re_values"] - re_value)))
            if abs(float(global_u["Re_values"][global_re_index]) - re_value) > 2e-5:
                raise RuntimeError(f"Global POD lacks Re={re_value}")
            global_u_phi = global_u["phi_uv"][:32].astype(np.float64)
            global_p_phi = global_p["phi_p"][:32].astype(np.float64)
            global_u_mean = global_u["mean_uv_by_Re"][global_re_index].astype(np.float64)
            global_p_mean = global_p["mean_p_by_Re"][global_re_index].astype(np.float64)
            global_p_phi, global_p_mean = center_pressure(global_p_phi, global_p_mean, weights)
            u_num_gram, u_truth_gram = cross_chart_geometry(
                global_u_phi, global_u_mean, local_u_phi, local_u_mean, velocity_weights
            )
            p_num_gram, p_truth_gram = cross_chart_geometry(
                global_p_phi, global_p_mean, local_p_phi, local_p_mean, weights
            )

            a_current = arrays["a"][global_start].copy()
            b_current = arrays["b"][global_start].copy()
            a_history, b_history, rhs_history = trainer.init_history_states_np(global_start, arrays)
            current = global_start
            predicted_a: list[np.ndarray] = []
            predicted_b: list[np.ndarray] = []
            with torch.inference_mode():
                for step in range(1, 49):
                    nxt = int(arrays["next_idx"][current])
                    if nxt < 0:
                        raise RuntimeError(f"{regime} window {window_number}: incomplete global chain")
                    expected_time = float(sealed["time"][local_start + step])
                    if not math.isclose(float(arrays["time"][nxt]), expected_time, abs_tol=1e-3, rel_tol=0):
                        raise RuntimeError(
                            f"{regime} window {window_number} step {step}: "
                            f"time {arrays['time'][nxt]} != {expected_time}"
                        )
                    dt = float(arrays["time"][nxt] - arrays["time"][current])
                    a_next, b_next, rhs_g = trainer.integrate_autonomous_step_np(
                        model,
                        a_current,
                        b_current,
                        current,
                        dt,
                        a_history,
                        b_history,
                        rhs_history,
                        arrays,
                        scalers,
                        tensors,
                        pressure_tensors,
                        model_args,
                        device,
                    )
                    if not (np.isfinite(a_next).all() and np.isfinite(b_next).all()):
                        for failed_step in range(step, 49):
                            rows.append(
                                {
                                    "regime": regime,
                                    "seed": "single_checkpoint",
                                    "Re": re_value,
                                    "window_id": f"{regime_short}:{int(sealed['global_ids'][local_start])}",
                                    "start_time": start_time,
                                    "step": failed_step,
                                    "physical_time": float(sealed["time"][local_start + failed_step]),
                                    "Eu": float("nan"),
                                    "Ep": float("nan"),
                                    "finite": False,
                                }
                            )
                        break
                    predicted_a.append(a_next.astype(np.float64, copy=True))
                    predicted_b.append(b_next.astype(np.float64, copy=True))
                    a_history = np.concatenate([a_next[None, None], a_history[:, :-1]], axis=1)
                    b_history = np.concatenate([b_next[None, None], b_history[:, :-1]], axis=1)
                    rhs_history = np.concatenate([rhs_g[None, None], rhs_history[:, :-1]], axis=1)
                    a_current, b_current, current = a_next, b_next, nxt

            if len(predicted_a) == 48:
                pred_a = np.asarray(predicted_a)
                pred_b = np.asarray(predicted_b)
                truth = sealed["state"][local_start + np.arange(1, 49)].astype(np.float64)
                truth_a, truth_b = truth[:, :32], truth[:, 32:]
                eu, u_denominator = gram_error(pred_a, truth_a, u_num_gram, u_truth_gram)
                ep, p_denominator = gram_error(pred_b, truth_b, p_num_gram, p_truth_gram)
                for step in range(1, 49):
                    rows.append(
                        {
                            "regime": regime,
                            "seed": "single_checkpoint",
                            "Re": re_value,
                            "window_id": f"{regime_short}:{int(sealed['global_ids'][local_start])}",
                            "start_time": start_time,
                            "step": step,
                            "physical_time": float(sealed["time"][local_start + step]),
                            "Eu": float(eu[step - 1]),
                            "Ep": float(ep[step - 1]),
                            "finite": True,
                        }
                    )
                if window_number in {0, len(starts) - 1}:
                    for step in (1, 24, 48):
                        index = step - 1
                        predicted_u_field = pred_a[index] @ global_u_phi + global_u_mean
                        truth_u_field = truth_a[index] @ local_u_phi + local_u_mean
                        predicted_p_field = pred_b[index] @ global_p_phi + global_p_mean
                        truth_p_field = truth_b[index] @ local_p_phi + local_p_mean
                        direct_eu = 100.0 * math.sqrt(
                            float(((predicted_u_field - truth_u_field) ** 2) @ velocity_weights)
                            / float((truth_u_field**2) @ velocity_weights)
                        )
                        direct_ep = 100.0 * math.sqrt(
                            float(((predicted_p_field - truth_p_field) ** 2) @ weights)
                            / float((truth_p_field**2) @ weights)
                        )
                        direct_checks.append(
                            {
                                "regime": regime,
                                "window_number": window_number,
                                "step": step,
                                "Eu_gram": float(eu[index]),
                                "Eu_direct": direct_eu,
                                "Eu_abs_difference": abs(float(eu[index]) - direct_eu),
                                "Ep_gram": float(ep[index]),
                                "Ep_direct": direct_ep,
                                "Ep_abs_difference": abs(float(ep[index]) - direct_ep),
                                "u_reference_energy": float(u_denominator[index]),
                                "p_reference_energy": float(p_denominator[index]),
                            }
                        )

        for re_value in sorted(set(float(sealed["Re"][value]) for value in starts)):
            label_ids = np.unique(arrays["label_id"][np.isclose(arrays["re"], re_value, atol=2e-5, rtol=0)])
            split_audit.append(
                {
                    "regime": regime,
                    "Re": re_value,
                    "global_label_ids": [int(value) for value in label_ids],
                    "listed_checkpoint_holdout": any(abs(value - re_value) < 2e-5 for value in holdout_re),
                }
            )

    rows.sort(key=lambda row: (str(row["regime"]), float(row["Re"]), str(row["window_id"]), int(row["step"])))
    with (output / "per_step_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "window_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)
    write_json(output / "cross_basis_direct_decode_audit.json", direct_checks)
    write_json(
        output / "model_identity.json",
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "trainer": str(trainer_path),
            "trainer_sha256": sha256(trainer_path),
            "checkpoint_keys": list(checkpoint),
            "test_label_id": checkpoint.get("test_label_id"),
            "holdout_label_ids": checkpoint.get("holdout_label_ids"),
            "holdout_Re": checkpoint.get("holdout_Re"),
            "test_Re_label": checkpoint.get("test_Re_label"),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_val_score": checkpoint.get("best_val_score"),
            "source_default_seed": int(frozen.seed),
            "source_default_seed_is_not_a_verified_launch_record": True,
            "model_class": type(model).__name__,
            "r_u": model_args.r_u,
            "r_p": model_args.r_p,
            "history_len": model_args.history_len,
            "phase_harmonics": model_args.phase_harmonics,
            "integrator": model_args.integrator,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "active_parameters": None,
            "active_parameters_note": "not counted without a validated execution-path hook audit",
            "target_split_audit": split_audit,
            "global_data_meta": meta,
            "evaluation_reference": "regime-local POD reconstructed truth in common physical space",
            "pressure_gauge": "area-weighted mean removed independently from Global prediction and local reference",
        },
    )
    max_direct_difference = max(
        max(float(row["Eu_abs_difference"]), float(row["Ep_abs_difference"])) for row in direct_checks
    )
    write_json(
        output / "COMPLETED.json",
        {
            "completed": True,
            "per_step_rows": len(rows),
            "windows": len(mapping_rows),
            "finite_rows": sum(bool(row["finite"]) for row in rows),
            "max_cross_basis_vs_direct_abs_difference_percentage_points": max_direct_difference,
            "runtime_seconds": time.time(),
        },
    )


if __name__ == "__main__":
    main()
