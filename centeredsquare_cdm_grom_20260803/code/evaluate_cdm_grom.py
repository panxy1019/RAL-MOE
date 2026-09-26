#!/usr/bin/env python3
"""Project and evaluate frozen operator-based CDM-GROM rollouts.

Validation and held-out Reynolds numbers are mutually exclusive roles.  A
held-out run is rejected unless a frozen-method manifest exists and matches
the exact operator asset hash.  No field file outside the selected role is
opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


VALIDATION_RE = np.asarray([94.5, 95.25, 95.5, 97.5, 99.0, 101.5])
HELDOUT_RE = np.asarray([95.1, 95.3, 96.5, 100.5, 102.0])
ROLE_RE = {"validation": VALIDATION_RE, "heldout": HELDOUT_RE}
RE_TOL = 5.0e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=sorted(ROLE_RE), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--operator-asset", type=Path, required=True)
    parser.add_argument("--canonical-cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-method", type=Path)
    parser.add_argument("--warmup-intervals", type=int, default=2)
    parser.add_argument("--max-internal-step", type=float, default=0.1)
    parser.add_argument("--consistency-step", type=float, default=0.2)
    parser.add_argument("--divergence-norm", type=float, default=1.0e8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def interpolation_weights(nodes: np.ndarray, value: float) -> tuple[int, int, float]:
    if value <= nodes[0]:
        left, right = 0, 1
    elif value >= nodes[-1]:
        left, right = len(nodes) - 2, len(nodes) - 1
    else:
        right = int(np.searchsorted(nodes, value))
        left = right - 1
    fraction = float((value - nodes[left]) / (nodes[right] - nodes[left]))
    return left, right, fraction


def interpolate(array: np.ndarray, bracket: tuple[int, int, float]) -> np.ndarray:
    left, right, fraction = bracket
    return (1.0 - fraction) * array[left] + fraction * array[right]


def rk4_advance(state: np.ndarray, duration: float, max_step: float, rhs) -> np.ndarray:
    steps = max(1, int(math.ceil(duration / max_step)))
    step = duration / steps
    current = state.copy()
    for _ in range(steps):
        k1 = rhs(current)
        k2 = rhs(current + 0.5 * step * k1)
        k3 = rhs(current + 0.5 * step * k2)
        k4 = rhs(current + step * k3)
        current += (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return current


def project_case(
    path: Path,
    velocity_mean: np.ndarray,
    velocity_weights: np.ndarray,
    velocity_weighted_modes: np.ndarray,
    pressure_mean: np.ndarray,
    pressure_weights: np.ndarray,
    pressure_weighted_modes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    with np.load(path, allow_pickle=False) as source:
        times = np.asarray(source["times"], dtype=np.float64)
        velocity = np.asarray(source["U"], dtype=np.float64)
        pressure = np.asarray(source["p"], dtype=np.float64)
        reynolds = float(source["Re"])
    velocity_centered = (velocity - velocity_mean[None]).reshape(len(times), -1)
    velocity_weighted = velocity_centered * velocity_weights[None]
    velocity_coefficients = velocity_weighted @ velocity_weighted_modes.T
    pressure -= (
        pressure @ (pressure_weights * pressure_weights)
        / float(np.sum(pressure_weights * pressure_weights))
    )[:, None]
    pressure_weighted = (pressure - pressure_mean[None]) * pressure_weights[None]
    pressure_coefficients = pressure_weighted @ pressure_weighted_modes.T
    velocity_total = float(np.sum(velocity_weighted * velocity_weighted))
    pressure_total = float(np.sum(pressure_weighted * pressure_weighted))
    audit = {
        "Re": reynolds,
        "velocity_projection_relative_residual": float(
            math.sqrt(max(0.0, 1.0 - np.sum(velocity_coefficients**2) / velocity_total))
        ),
        "pressure_projection_relative_residual": float(
            math.sqrt(max(0.0, 1.0 - np.sum(pressure_coefficients**2) / pressure_total))
        ),
    }
    return times, velocity_coefficients, pressure_coefficients, audit


def polynomial(c: np.ndarray, a: np.ndarray, h: np.ndarray, state: np.ndarray) -> np.ndarray:
    return c + a @ state + np.einsum("ijk,j,k->i", h, state, state, optimize=True)


def rollout(
    times: np.ndarray,
    truth: np.ndarray,
    operator: dict[str, np.ndarray],
    max_step: float,
    warmup_intervals: int,
    divergence_norm: float,
    use_memory: bool,
) -> tuple[np.ndarray, int | None]:
    resolved_rank = int(operator["resolved_rank"])
    forecast_start = warmup_intervals
    if forecast_start >= len(times) - 1:
        raise ValueError("warm-up leaves no forecast interval")
    predicted = np.full((len(times), operator["full_rank"]), np.nan)
    if use_memory:
        reference = operator["reference"]
        unresolved = np.zeros(operator["full_rank"] - resolved_rank)
        for interval in range(forecast_start):
            duration = float(times[interval + 1] - times[interval])
            a0 = truth[interval, :resolved_rank]
            a1 = truth[interval + 1, :resolved_rank]

            def warm_rhs(y: np.ndarray) -> np.ndarray:
                # Midpoint forcing is deterministic and uses only warm-up observations.
                a_mid = 0.5 * (a0 + a1)
                return (
                    operator["f_reference"][resolved_rank:]
                    + operator["A_ur"] @ (a_mid - reference[:resolved_rank])
                    + operator["A_uu"] @ y
                )

            unresolved = rk4_advance(unresolved, duration, max_step, warm_rhs)
        joint = np.concatenate([truth[forecast_start, :resolved_rank], unresolved])
        predicted[forecast_start, :resolved_rank] = joint[:resolved_rank]
        predicted[forecast_start, resolved_rank:] = reference[resolved_rank:] + joint[resolved_rank:]

        def rhs(state: np.ndarray) -> np.ndarray:
            resolved = state[:resolved_rank]
            memory = state[resolved_rank:]
            resolved_rhs = polynomial(
                operator["base_c"], operator["base_A"], operator["base_H"], resolved
            ) + operator["A_ru"] @ memory
            unresolved_rhs = (
                operator["f_reference"][resolved_rank:]
                + operator["A_ur"] @ (resolved - reference[:resolved_rank])
                + operator["A_uu"] @ memory
            )
            return np.concatenate([resolved_rhs, unresolved_rhs])

        for index in range(forecast_start, len(times) - 1):
            joint = rk4_advance(
                joint, float(times[index + 1] - times[index]), max_step, rhs
            )
            if not np.all(np.isfinite(joint)) or np.linalg.norm(joint) > divergence_norm:
                return predicted, index + 1
            predicted[index + 1, :resolved_rank] = joint[:resolved_rank]
            predicted[index + 1, resolved_rank:] = reference[resolved_rank:] + joint[resolved_rank:]
    else:
        state = truth[forecast_start, :resolved_rank].copy()
        predicted[forecast_start, :resolved_rank] = state
        predicted[forecast_start, resolved_rank:] = 0.0

        def rhs(state_value: np.ndarray) -> np.ndarray:
            return polynomial(
                operator["trunc_c"], operator["trunc_A"], operator["trunc_H"], state_value
            )

        for index in range(forecast_start, len(times) - 1):
            state = rk4_advance(
                state, float(times[index + 1] - times[index]), max_step, rhs
            )
            if not np.all(np.isfinite(state)) or np.linalg.norm(state) > divergence_norm:
                return predicted, index + 1
            predicted[index + 1, :resolved_rank] = state
            predicted[index + 1, resolved_rank:] = 0.0
    return predicted, None


def relative_l2(predicted: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - truth) / max(np.linalg.norm(truth), 1.0e-14))


def optional_relative_l2(predicted: np.ndarray, truth: np.ndarray) -> float | None:
    if predicted.shape[0] == 0:
        return None
    return relative_l2(predicted, truth)


def pressure_from_velocity(operator: dict[str, np.ndarray], velocity: np.ndarray) -> np.ndarray:
    return (
        operator["pressure_c"][None]
        + velocity @ operator["pressure_A"].T
        + np.einsum("mjk,tj,tk->tm", operator["pressure_H"], velocity, velocity, optimize=True)
    )


def load_selected_cases(canonical_path: Path, requested: np.ndarray) -> list[dict[str, Any]]:
    entries = json.loads(canonical_path.read_text())
    selected = []
    for value in requested:
        matches = [entry for entry in entries if abs(float(entry["Re"]) - value) <= RE_TOL]
        if len(matches) != 1:
            raise RuntimeError(f"expected one canonical case for Re={value}, found {len(matches)}")
        selected.append(matches[0])
    return selected


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.max_internal_step <= 0 or args.consistency_step <= args.max_internal_step:
        raise ValueError("require 0 < max-internal-step < consistency-step")
    operator_hash = sha256(args.operator_asset)
    if args.role == "heldout":
        if args.frozen_method is None or not args.frozen_method.is_file():
            raise RuntimeError("held-out evaluation requires --frozen-method")
        frozen = json.loads(args.frozen_method.read_text())
        if frozen.get("status") != "FROZEN" or frozen.get("operator_sha256") != operator_hash:
            raise RuntimeError("frozen method does not match operator asset")

    with np.load(args.dataset_root / "pod/weighted_pod_velocity.npz", allow_pickle=False) as pod:
        velocity_mean = np.asarray(pod["mean"], dtype=np.float64)
        velocity_weights = np.asarray(pod["weights"], dtype=np.float64)
        velocity_modes = np.asarray(pod["weighted_modes"], dtype=np.float64)
    with np.load(args.dataset_root / "pod/weighted_pod_pressure.npz", allow_pickle=False) as pod:
        pressure_mean = np.asarray(pod["mean"], dtype=np.float64)
        pressure_weights = np.asarray(pod["weights"], dtype=np.float64)
        pressure_modes = np.asarray(pod["weighted_modes"], dtype=np.float64)
    with np.load(args.operator_asset, allow_pickle=False) as asset:
        raw = {key: np.asarray(asset[key]) for key in asset.files}
    nodes = raw["Re_nodes"].astype(np.float64)
    resolved_rank = int(raw["resolved_rank"])
    full_rank = int(raw["full_velocity_rank"])
    selected = load_selected_cases(args.canonical_cases, ROLE_RE[args.role])
    args.output_dir.mkdir(parents=True)

    cases = []
    consistency_values = []
    for entry in selected:
        reynolds = float(entry["Re"])
        case_path = Path(entry["path"])
        if sha256(case_path) != entry["sha256"]:
            raise RuntimeError(f"canonical SHA256 mismatch for {case_path}")
        times, truth_velocity, truth_pressure, projection_audit = project_case(
            case_path,
            velocity_mean,
            velocity_weights,
            velocity_modes,
            pressure_mean,
            pressure_weights,
            pressure_modes,
        )
        bracket = interpolation_weights(nodes, reynolds)
        interpolated_a_uu = interpolate(raw["A_uu_stable"], bracket)
        interpolated_abscissa = float(np.max(np.linalg.eigvals(interpolated_a_uu).real))
        stability_margin = float(raw["stability_margin"])
        interpolation_shift = max(0.0, interpolated_abscissa + stability_margin)
        interpolated_a_uu -= interpolation_shift * np.eye(interpolated_a_uu.shape[0])
        operator = {
            "resolved_rank": resolved_rank,
            "full_rank": full_rank,
            "reference": interpolate(raw["reference"], bracket),
            "f_reference": interpolate(raw["f_reference"], bracket),
            "base_c": interpolate(raw["base_c"], bracket),
            "base_A": interpolate(raw["base_A"], bracket),
            "base_H": interpolate(raw["base_H"], bracket),
            "trunc_c": interpolate(raw["trunc_c"], bracket),
            "trunc_A": interpolate(raw["trunc_A"], bracket),
            "trunc_H": interpolate(raw["trunc_H"], bracket),
            "A_ru": interpolate(raw["A_ru"], bracket),
            "A_ur": interpolate(raw["A_ur"], bracket),
            "A_uu": interpolated_a_uu,
            "pressure_c": interpolate(raw["pressure_c"], bracket),
            "pressure_A": interpolate(raw["pressure_A"], bracket),
            "pressure_H": raw["pressure_H"],
        }
        cdm, cdm_diverged = rollout(
            times, truth_velocity, operator, args.max_internal_step,
            args.warmup_intervals, args.divergence_norm, True,
        )
        baseline, baseline_diverged = rollout(
            times, truth_velocity, operator, args.max_internal_step,
            args.warmup_intervals, args.divergence_norm, False,
        )
        coarse, coarse_diverged = rollout(
            times, truth_velocity, operator, args.consistency_step,
            args.warmup_intervals, args.divergence_norm, True,
        )
        start = args.warmup_intervals + 1
        cdm_stop = cdm_diverged if cdm_diverged is not None else len(times)
        baseline_stop = baseline_diverged if baseline_diverged is not None else len(times)
        cdm_pressure = pressure_from_velocity(operator, cdm[start:cdm_stop])
        baseline_pressure = pressure_from_velocity(operator, baseline[start:baseline_stop])
        metrics = {
            "Re": reynolds,
            "forecast_initial_time": float(times[args.warmup_intervals]),
            "metric_initial_time": float(times[start]),
            "forecast_final_time_requested": float(times[-1]),
            "cdm_diverged_at_index": cdm_diverged,
            "baseline_diverged_at_index": baseline_diverged,
            "consistency_diverged_at_index": coarse_diverged,
            "cdm_finite_final_time": float(times[cdm_stop - 1]),
            "baseline_finite_final_time": float(times[baseline_stop - 1]),
            "interpolated_A_uu_raw_spectral_abscissa": interpolated_abscissa,
            "interpolated_A_uu_additional_shift": interpolation_shift,
            "cdm_velocity_coeff_relative_l2": optional_relative_l2(
                cdm[start:cdm_stop], truth_velocity[start:cdm_stop]
            ),
            "baseline_velocity_coeff_relative_l2": optional_relative_l2(
                baseline[start:baseline_stop], truth_velocity[start:baseline_stop]
            ),
            "cdm_resolved_relative_l2": optional_relative_l2(
                cdm[start:cdm_stop, :resolved_rank], truth_velocity[start:cdm_stop, :resolved_rank]
            ),
            "baseline_resolved_relative_l2": optional_relative_l2(
                baseline[start:baseline_stop, :resolved_rank], truth_velocity[start:baseline_stop, :resolved_rank]
            ),
            "cdm_pressure_coeff_relative_l2": optional_relative_l2(
                cdm_pressure, truth_pressure[start:cdm_stop]
            ),
            "baseline_pressure_coeff_relative_l2": optional_relative_l2(
                baseline_pressure, truth_pressure[start:baseline_stop]
            ),
            **projection_audit,
        }
        consistency_stop = min(
            value for value in [cdm_diverged, coarse_diverged, len(times)] if value is not None
        )
        consistency = optional_relative_l2(
            coarse[start:consistency_stop, :resolved_rank],
            cdm[start:consistency_stop, :resolved_rank],
        )
        metrics["rk4_step_consistency_relative_l2"] = consistency
        consistency_values.append(consistency)
        cases.append(metrics)
        np.savez_compressed(
            args.output_dir / f"trajectory_Re{reynolds:010.6f}.npz",
            Re=np.asarray(reynolds), times=times, truth_velocity=truth_velocity,
            truth_pressure=truth_pressure, cdm_velocity=cdm,
            baseline_velocity=baseline,
        )

    numeric_keys = [
        "cdm_velocity_coeff_relative_l2", "baseline_velocity_coeff_relative_l2",
        "cdm_resolved_relative_l2", "baseline_resolved_relative_l2",
        "cdm_pressure_coeff_relative_l2", "baseline_pressure_coeff_relative_l2",
    ]
    aggregate = {}
    for key in numeric_keys:
        finite_values = [case[key] for case in cases if case[key] is not None]
        aggregate[key] = float(np.mean(finite_values)) if finite_values else None
    aggregate["cdm_better_velocity_case_count"] = int(sum(
        case["cdm_diverged_at_index"] is None
        and case["baseline_diverged_at_index"] is None
        and case["cdm_velocity_coeff_relative_l2"] < case["baseline_velocity_coeff_relative_l2"]
        for case in cases
    ))
    finite_consistency = [value for value in consistency_values if value is not None]
    aggregate["maximum_rk4_step_consistency_relative_l2"] = (
        float(max(finite_consistency)) if finite_consistency else None
    )
    aggregate["cdm_full_horizon_case_count"] = int(sum(
        case["cdm_diverged_at_index"] is None for case in cases
    ))
    aggregate["baseline_full_horizon_case_count"] = int(sum(
        case["baseline_diverged_at_index"] is None for case in cases
    ))
    manifest = {
        "schema_version": 1,
        "status": "PASS" if all(case["cdm_diverged_at_index"] is None for case in cases) else "DIVERGED",
        "role": args.role,
        "requested_Re": ROLE_RE[args.role].tolist(),
        "field_files_opened": [entry["path"] for entry in selected],
        "validation_loaded": args.role == "validation",
        "heldout_loaded": args.role == "heldout",
        "warmup_intervals": args.warmup_intervals,
        "max_internal_step": args.max_internal_step,
        "consistency_step": args.consistency_step,
        "operator_sha256": operator_hash,
        "canonical_manifest_sha256": sha256(args.canonical_cases),
        "cases": cases,
        "aggregate": aggregate,
    }
    atomic_json(manifest, args.output_dir / f"{args.role.upper()}_METRICS.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
