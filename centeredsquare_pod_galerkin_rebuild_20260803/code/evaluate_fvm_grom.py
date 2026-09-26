#!/usr/bin/env python3
"""Evaluate the OpenFOAM-discrete r11 GROM on validation cases only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--velocity-pod", type=Path, required=True)
    parser.add_argument("--pressure-pod", type=Path, required=True)
    parser.add_argument("--fvm-tensors", type=Path, required=True)
    parser.add_argument("--pressure-closure", type=Path, required=True)
    parser.add_argument("--canonical-cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-step", type=float, default=0.05)
    parser.add_argument("--warmup-intervals", type=int, default=2)
    parser.add_argument("--divergence-norm", type=float, default=100.0)
    parser.add_argument(
        "--target-re",
        "--validation-re",
        dest="target_re",
        type=float,
        nargs="+",
        default=[94.5, 95.25, 95.5, 97.5, 99.0, 101.5],
        help="Reynolds numbers to evaluate; --validation-re is retained as a compatibility alias.",
    )
    parser.add_argument("--role", choices=["validation", "heldout"], default="validation")
    parser.add_argument("--frozen-method", type=Path)
    return parser.parse_args()


def quadratic_features(state):
    i, j = np.triu_indices(state.shape[-1])
    return state[..., i] * state[..., j]


def pressure_features(state, viscosity, kind):
    state_2d = np.atleast_2d(state)
    nu = np.full((len(state_2d), 1), viscosity)
    columns = [np.ones((len(state_2d), 1)), nu, state_2d, nu * state_2d]
    quadratic = quadratic_features(state_2d)
    if kind in {"quadratic", "quadratic_nu"}:
        columns.append(quadratic)
    if kind == "quadratic_nu":
        columns.append(nu * quadratic)
    result = np.concatenate(columns, axis=1)
    return result[0] if np.ndim(state) == 1 else result


def rk4(state, duration, max_step, rhs):
    count = max(1, int(math.ceil(duration / max_step)))
    step = duration / count
    value = state.copy()
    for _ in range(count):
        k1 = rhs(value)
        k2 = rhs(value + 0.5 * step * k1)
        k3 = rhs(value + 0.5 * step * k2)
        k4 = rhs(value + step * k3)
        value += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return value


def relative(prediction, truth):
    return float(np.linalg.norm(prediction - truth) / max(np.linalg.norm(truth), 1e-14))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def project_case(path, u_mean, u_weights, u_modes, p_mean, p_weights, p_modes):
    with np.load(path, allow_pickle=False) as source:
        times = np.asarray(source["times"], dtype=np.float64)
        u = np.asarray(source["U"], dtype=np.float64)
        p = np.asarray(source["p"], dtype=np.float64)
        re_value = float(source["Re"])
    a = ((u - u_mean[None]).reshape(len(u), -1) * u_weights[None]) @ u_modes.T
    b = ((p - p_mean[None]) * p_weights[None]) @ p_modes.T
    return re_value, times, a, b


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.role == "heldout":
        if args.frozen_method is None or not args.frozen_method.is_file():
            raise RuntimeError("heldout evaluation requires --frozen-method")
        frozen = json.loads(args.frozen_method.read_text())
        if frozen.get("status") != "FROZEN":
            raise RuntimeError("method is not frozen")
        if frozen["fvm_tensors_sha256"] != sha256(args.fvm_tensors):
            raise RuntimeError("frozen FVM tensor hash mismatch")
        if frozen["pressure_closure_sha256"] != sha256(args.pressure_closure):
            raise RuntimeError("frozen pressure closure hash mismatch")
    canonical = json.loads(args.canonical_cases.read_text())
    with np.load(args.fvm_tensors) as source:
        tensor = {key: np.asarray(source[key]) for key in source.files}
    velocity_rank = int(tensor["velocity_rank"])
    pressure_rank = int(tensor["pressure_rank"])
    with np.load(args.velocity_pod) as pod:
        u_mean = np.asarray(pod["mean"], dtype=np.float64)
        u_weights = np.asarray(pod["weights"], dtype=np.float64)
        u_modes = np.asarray(pod["weighted_modes"], dtype=np.float64)[:velocity_rank]
    with np.load(args.pressure_pod) as pod:
        p_mean = np.asarray(pod["mean"], dtype=np.float64)
        p_weights = np.asarray(pod["weights"], dtype=np.float64)
        p_modes = np.asarray(pod["weighted_modes"], dtype=np.float64)[:pressure_rank]
    with np.load(args.pressure_closure) as source:
        pressure_kind = str(source["kind"])
        pressure_scale = np.asarray(source["feature_scale"], dtype=np.float64)
        pressure_weights = np.asarray(source["weights"], dtype=np.float64)

    args.output_dir.mkdir(parents=True)
    reports = []
    target_values = np.asarray(args.target_re, dtype=np.float64)
    for target_re in target_values:
        matches = [entry for entry in canonical if abs(float(entry["Re"]) - target_re) < 5e-7]
        if len(matches) != 1:
            raise RuntimeError(f"canonical case mismatch for Re={target_re}")
        re_value, times, truth_a, truth_b = project_case(
            Path(matches[0]["path"]), u_mean, u_weights, u_modes,
            p_mean, p_weights, p_modes,
        )
        viscosity = 1.0 / re_value

        def pressure(state):
            x = pressure_features(state, viscosity, pressure_kind) / pressure_scale
            return x @ pressure_weights

        def rhs(state):
            return (
                tensor["c_conv"] + viscosity * tensor["c_diff"] + tensor["c_pressure"]
                + (tensor["A_conv"] + viscosity * tensor["A_diff"]) @ state
                + np.einsum("ijk,j,k->i", tensor["H_conv"], state, state, optimize=True)
                + tensor["P"] @ pressure(state)
            )

        start = args.warmup_intervals
        predicted = np.full_like(truth_a, np.nan)
        predicted[start] = truth_a[start]
        diverged = None
        for index in range(start, len(times) - 1):
            predicted[index + 1] = rk4(
                predicted[index], times[index + 1] - times[index], args.max_step, rhs
            )
            if not np.all(np.isfinite(predicted[index + 1])) or np.linalg.norm(predicted[index + 1]) > args.divergence_norm:
                diverged = index + 1
                break
        stop = diverged if diverged is not None else len(times)
        metric_start = start + 1
        predicted_b = np.asarray([pressure(state) for state in predicted[metric_start:stop]])

        one_step = []
        for index in range(start, len(times) - 1):
            one_step.append(rk4(
                truth_a[index], times[index + 1] - times[index], args.max_step, rhs
            ))
        one_step = np.asarray(one_step)
        central = (truth_a[2:] - truth_a[:-2]) / (times[2:, None] - times[:-2, None])
        rhs_truth = np.asarray([rhs(state) for state in truth_a[1:-1]])
        report = {
            "Re": re_value,
            "diverged_at_index": diverged,
            "finite_final_time": float(times[stop - 1]),
            "velocity_rollout_relative_l2": relative(predicted[metric_start:stop], truth_a[metric_start:stop]),
            "pressure_rollout_relative_l2": relative(predicted_b, truth_b[metric_start:stop]),
            "one_step_velocity_relative_l2": relative(one_step, truth_a[start + 1:]),
            "coarse_derivative_relative_l2": relative(rhs_truth, central),
            "truth_final_norm": float(np.linalg.norm(truth_a[-1])),
            "predicted_final_norm": float(np.linalg.norm(predicted[stop - 1])),
        }
        reports.append(report)
        np.savez_compressed(
            args.output_dir / f"trajectory_Re{re_value:010.6f}.npz",
            Re=np.asarray(re_value), times=times, truth_velocity=truth_a,
            truth_pressure=truth_b, predicted_velocity=predicted,
        )
    keys = ["velocity_rollout_relative_l2", "pressure_rollout_relative_l2", "one_step_velocity_relative_l2", "coarse_derivative_relative_l2"]
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(item["diverged_at_index"] is None for item in reports) else "DIVERGED",
        "method": f"OpenFOAM-discrete FVM-Galerkin r{velocity_rank} with calibrated {pressure_kind} pressure closure",
        "max_step": args.max_step,
        "warmup_intervals": args.warmup_intervals,
        "cases": reports,
        "aggregate": {key: float(np.mean([item[key] for item in reports])) for key in keys},
        "role": args.role,
        "validation_loaded": args.role == "validation",
        "heldout_loaded": args.role == "heldout",
    }
    (args.output_dir / f"{args.role.upper()}_METRICS.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
