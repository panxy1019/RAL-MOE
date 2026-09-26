#!/usr/bin/env python3
"""Fit and validate a fixed-address continuous Delta-memory FVM-GROM closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


RATE_SETS = [
    (),
    (0.02,),
    (0.01, 0.05),
    (0.005, 0.02, 0.08),
]
RIDGES = [0.1, 1.0, 10.0, 100.0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-pod", type=Path, required=True)
    parser.add_argument("--pressure-pod", type=Path, required=True)
    parser.add_argument("--fvm-tensors", type=Path, required=True)
    parser.add_argument("--pressure-closure", type=Path, required=True)
    parser.add_argument("--canonical-cases", type=Path, required=True)
    parser.add_argument("--baseline-frozen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-re", type=float, nargs="+", required=True)
    parser.add_argument("--warmup-intervals", type=int, default=2)
    parser.add_argument("--selection-max-step", type=float, default=0.2)
    parser.add_argument("--final-max-step", type=float, default=0.05)
    parser.add_argument("--train-pair-stride", type=int, default=4)
    parser.add_argument("--divergence-norm", type=float, default=100.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tag_to_re(tag: str) -> float:
    return float(tag[2:].replace("p", "."))


def pressure_features(state: np.ndarray, viscosity: np.ndarray) -> np.ndarray:
    state = np.atleast_2d(state)
    viscosity = np.broadcast_to(np.asarray(viscosity, dtype=np.float64), (len(state),))
    return np.concatenate([
        np.ones((len(state), 1)), viscosity[:, None], state,
        viscosity[:, None] * state,
    ], axis=1)


class FrozenBaseline:
    def __init__(self, tensor_path: Path, pressure_path: Path):
        with np.load(tensor_path, allow_pickle=False) as source:
            self.tensor = {key: np.asarray(source[key], dtype=np.float64) for key in source.files}
        with np.load(pressure_path, allow_pickle=False) as source:
            self.pressure_kind = str(source["kind"])
            self.pressure_scale = np.asarray(source["feature_scale"], dtype=np.float64)
            self.pressure_weights = np.asarray(source["weights"], dtype=np.float64)
        if self.pressure_kind != "linear":
            raise ValueError("this CDM implementation requires the frozen linear pressure closure")

    def pressure(self, state: np.ndarray, viscosity: np.ndarray) -> np.ndarray:
        single = np.ndim(state) == 1
        result = pressure_features(state, viscosity) / self.pressure_scale[None]
        result = result @ self.pressure_weights
        return result[0] if single else result

    def rhs(self, state: np.ndarray, viscosity: np.ndarray) -> np.ndarray:
        single = np.ndim(state) == 1
        state = np.atleast_2d(state)
        viscosity = np.broadcast_to(np.asarray(viscosity, dtype=np.float64), (len(state),))
        tensor = self.tensor
        linear = np.einsum("ij,nj->ni", tensor["A_conv"], state)
        linear += viscosity[:, None] * np.einsum("ij,nj->ni", tensor["A_diff"], state)
        quadratic = np.einsum("ijk,nj,nk->ni", tensor["H_conv"], state, state, optimize=True)
        pressure = self.pressure(state, viscosity)
        result = (
            tensor["c_conv"][None] + viscosity[:, None] * tensor["c_diff"][None]
            + tensor["c_pressure"][None] + linear + quadratic
            + np.einsum("ij,nj->ni", tensor["P"], pressure)
        )
        return result[0] if single else result


def rk4_batch(state: np.ndarray, duration: np.ndarray, max_step: float, rhs) -> np.ndarray:
    duration = np.asarray(duration, dtype=np.float64)
    counts = np.ceil(duration / max_step).astype(int)
    if np.any(counts != counts[0]):
        raise ValueError("batched RK4 requires equal snapshot spacing")
    step = duration / counts
    value = state.copy()
    for _ in range(int(counts[0])):
        h = step[:, None]
        k1 = rhs(value)
        k2 = rhs(value + 0.5 * h * k1)
        k3 = rhs(value + 0.5 * h * k2)
        k4 = rhs(value + h * k3)
        value += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return value


def rk4_single(state: np.ndarray, duration: float, max_step: float, rhs) -> np.ndarray:
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


def relative(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - truth) / max(np.linalg.norm(truth), 1e-14))


def algebraic_features(state: np.ndarray, viscosity: float, a_scale: np.ndarray, nu_mean: float, nu_scale: float) -> np.ndarray:
    a_hat = state / a_scale
    nu_hat = (viscosity - nu_mean) / nu_scale
    return np.concatenate(([1.0, nu_hat], a_hat, nu_hat * a_hat))


def design_features(state, memory, rates, viscosity, a_scale, nu_mean, nu_scale):
    base = algebraic_features(state, viscosity, a_scale, nu_mean, nu_scale)
    if not rates:
        return base
    a_hat = state / a_scale
    return np.concatenate([base, *(memory[index] - a_hat for index in range(len(rates)))])


def build_train_design(train_a, times, tags, rates, a_scale, nu_mean, nu_scale):
    rows = []
    indices = []
    start = 0
    while start < len(tags):
        stop = start + 1
        while stop < len(tags) and tags[stop] == tags[start]:
            stop += 1
        viscosity = 1.0 / tag_to_re(tags[start])
        memory = np.repeat((train_a[start] / a_scale)[None], len(rates), axis=0)
        for index in range(start, stop - 1):
            rows.append(design_features(
                train_a[index], memory, rates, viscosity, a_scale, nu_mean, nu_scale
            ))
            indices.append(index)
            dt = times[index + 1] - times[index]
            if rates:
                decay = np.exp(-np.asarray(rates) * dt)[:, None]
                memory = decay * memory + (1.0 - decay) * (train_a[index] / a_scale)[None]
        start = stop
    return np.asarray(rows), np.asarray(indices, dtype=int)


def fit_weights(raw_design: np.ndarray, target: np.ndarray, ridge: float):
    scale = np.std(raw_design, axis=0)
    scale[scale < 1e-10] = 1.0
    scale[0] = 1.0
    design = raw_design / scale[None]
    regularizer = np.eye(design.shape[1]) * ridge
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
    return scale, weights, relative(design @ weights, target)


def project_case(entry, u_mean, u_weights, u_modes, p_mean, p_weights, p_modes):
    with np.load(entry["path"], allow_pickle=False) as source:
        times = np.asarray(source["times"], dtype=np.float64)
        velocity = np.asarray(source["U"], dtype=np.float64)
        pressure = np.asarray(source["p"], dtype=np.float64)
        re_value = float(source["Re"])
    a = ((velocity - u_mean[None]).reshape(len(velocity), -1) * u_weights[None]) @ u_modes.T
    b = ((pressure - p_mean[None]) * p_weights[None]) @ p_modes.T
    return re_value, times, a, b


def truth_memory(states, times, rates, a_scale):
    result = np.empty((len(states), len(rates), states.shape[1]), dtype=np.float64)
    if not rates:
        return result
    result[0] = np.repeat((states[0] / a_scale)[None], len(rates), axis=0)
    rate_array = np.asarray(rates)[:, None]
    for index in range(len(states) - 1):
        decay = np.exp(-rate_array * (times[index + 1] - times[index]))
        result[index + 1] = decay * result[index] + (1.0 - decay) * (states[index] / a_scale)[None]
    return result


def periodic_quality(times, truth, prediction, tail_start=100.0):
    ids = np.flatnonzero(times >= tail_start)
    truth = truth[ids] - truth[ids].mean(axis=0, keepdims=True)
    prediction = prediction[ids] - prediction[ids].mean(axis=0, keepdims=True)
    truth_amp = float(np.sqrt(np.mean(np.sum(truth**2, axis=1))))
    pred_amp = float(np.sqrt(np.mean(np.sum(prediction**2, axis=1))))
    frequencies = np.fft.rfftfreq(len(ids), d=float(times[1] - times[0]))
    truth_power = np.sum(np.abs(np.fft.rfft(truth, axis=0)) ** 2, axis=1)
    pred_power = np.sum(np.abs(np.fft.rfft(prediction, axis=0)) ** 2, axis=1)
    truth_index = int(np.argmax(truth_power[1:]) + 1)
    pred_index = int(np.argmax(pred_power[1:]) + 1)
    return {
        "amplitude_relative_error": abs(pred_amp - truth_amp) / max(truth_amp, 1e-14),
        "dominant_frequency_relative_error": abs(frequencies[pred_index] - frequencies[truth_index]) / max(frequencies[truth_index], 1e-14),
    }


def evaluate_case(case, baseline, rates, scale, weights, a_scale, nu_mean, nu_scale, warmup, max_step, divergence_norm, compute_one_step):
    re_value, times, truth_a, truth_b = case
    viscosity = 1.0 / re_value
    memory_truth = truth_memory(truth_a, times, rates, a_scale)
    predicted = np.full_like(truth_a, np.nan)
    predicted[warmup] = truth_a[warmup]
    initial_memory = memory_truth[warmup].copy()
    joint = np.concatenate([predicted[warmup], initial_memory.reshape(-1)])

    def joint_rhs(value):
        state = value[: truth_a.shape[1]]
        memory = value[truth_a.shape[1]:].reshape(len(rates), truth_a.shape[1])
        raw = design_features(state, memory, rates, viscosity, a_scale, nu_mean, nu_scale)
        correction = (raw / scale) @ weights
        if rates:
            memory_rhs = np.asarray(rates)[:, None] * (state[None] / a_scale - memory)
            return np.concatenate([baseline.rhs(state, viscosity) + correction, memory_rhs.reshape(-1)])
        return baseline.rhs(state, viscosity) + correction

    diverged = None
    for index in range(warmup, len(times) - 1):
        joint = rk4_single(joint, float(times[index + 1] - times[index]), max_step, joint_rhs)
        predicted[index + 1] = joint[: truth_a.shape[1]]
        if not np.all(np.isfinite(joint)) or np.linalg.norm(predicted[index + 1]) > divergence_norm:
            diverged = index + 1
            break
    stop = diverged if diverged is not None else len(times)
    metric_start = warmup + 1

    one_step_error = float("nan")
    if compute_one_step:
        one_step = []
        for index in range(warmup, len(times) - 1):
            local = np.concatenate([truth_a[index], memory_truth[index].reshape(-1)])
            local = rk4_single(local, float(times[index + 1] - times[index]), max_step, joint_rhs)
            one_step.append(local[: truth_a.shape[1]])
        one_step_error = relative(np.asarray(one_step), truth_a[warmup + 1:])
    predicted_pressure = baseline.pressure(predicted[metric_start:stop], viscosity)
    quality = periodic_quality(times[metric_start:stop], truth_a[metric_start:stop], predicted[metric_start:stop]) if diverged is None else {
        "amplitude_relative_error": float("inf"), "dominant_frequency_relative_error": float("inf")
    }
    return {
        "Re": re_value,
        "diverged_at_index": diverged,
        "finite_final_time": float(times[stop - 1]),
        "velocity_rollout_relative_l2": relative(predicted[metric_start:stop], truth_a[metric_start:stop]),
        "pressure_rollout_relative_l2": relative(predicted_pressure, truth_b[metric_start:stop]),
        "one_step_velocity_relative_l2": one_step_error,
        **quality,
    }, predicted


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frozen = json.loads(args.baseline_frozen.read_text())
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("baseline is not frozen")
    if sha256(args.fvm_tensors) != frozen["fvm_tensors_sha256"]:
        raise RuntimeError("baseline tensor hash mismatch")
    if sha256(args.pressure_closure) != frozen["pressure_closure_sha256"]:
        raise RuntimeError("baseline pressure hash mismatch")
    baseline = FrozenBaseline(args.fvm_tensors, args.pressure_closure)

    with np.load(args.velocity_pod, allow_pickle=False) as pod:
        train_a = np.asarray(pod["coefficients"], dtype=np.float64)[:, :28]
        train_times = np.asarray(pod["snapshot_times"], dtype=np.float64)
        train_tags = np.asarray(pod["snapshot_case_tags"]).astype(str)
        u_mean = np.asarray(pod["mean"], dtype=np.float64)
        u_weights = np.asarray(pod["weights"], dtype=np.float64)
        u_modes = np.asarray(pod["weighted_modes"], dtype=np.float64)[:28]
    with np.load(args.pressure_pod, allow_pickle=False) as pod:
        p_mean = np.asarray(pod["mean"], dtype=np.float64)
        p_weights = np.asarray(pod["weights"], dtype=np.float64)
        p_modes = np.asarray(pod["weighted_modes"], dtype=np.float64)[:24]
    a_scale = np.std(train_a, axis=0)
    a_scale[a_scale < 1e-8] = 1.0
    train_nu = np.asarray([1.0 / tag_to_re(tag) for tag in train_tags])
    nu_mean = float(np.mean(train_nu))
    nu_scale = float(np.std(train_nu))

    pair_indices = []
    position_in_case = 0
    for index in range(len(train_a) - 1):
        if train_tags[index] == train_tags[index + 1]:
            if position_in_case % args.train_pair_stride == 0:
                pair_indices.append(index)
            position_in_case += 1
        else:
            position_in_case = 0
    pair_indices = np.asarray(pair_indices, dtype=int)
    pair_dt = train_times[pair_indices + 1] - train_times[pair_indices]
    pair_nu = train_nu[pair_indices]
    baseline_next = rk4_batch(
        train_a[pair_indices], pair_dt, args.selection_max_step,
        lambda value: baseline.rhs(value, pair_nu),
    )
    target = (train_a[pair_indices + 1] - baseline_next) / pair_dt[:, None]

    canonical = json.loads(args.canonical_cases.read_text())
    validation_cases = []
    for value in args.validation_re:
        matches = [entry for entry in canonical if abs(float(entry["Re"]) - value) < 5e-7]
        if len(matches) != 1:
            raise RuntimeError(f"canonical validation match failed for Re={value}")
        validation_cases.append(project_case(
            matches[0], u_mean, u_weights, u_modes, p_mean, p_weights, p_modes
        ))

    args.output_dir.mkdir(parents=True)
    candidate_reports = []
    candidate_assets = []
    for rates in RATE_SETS:
        raw_design, design_indices = build_train_design(
            train_a, train_times, train_tags, rates, a_scale, nu_mean, nu_scale
        )
        chosen = np.searchsorted(design_indices, pair_indices)
        if not np.array_equal(design_indices[chosen], pair_indices):
            raise RuntimeError("training pair selection mismatch")
        raw_design = raw_design[chosen]
        for ridge in RIDGES:
            scale, weights, train_error = fit_weights(raw_design, target, ridge)
            case_reports = []
            trajectories = []
            for case in validation_cases:
                report, trajectory = evaluate_case(
                    case, baseline, rates, scale, weights, a_scale, nu_mean, nu_scale,
                    args.warmup_intervals, args.selection_max_step, args.divergence_norm, False,
                )
                case_reports.append(report)
                trajectories.append(trajectory)
            stable = all(item["diverged_at_index"] is None for item in case_reports)
            aggregate_keys = [
                "velocity_rollout_relative_l2", "pressure_rollout_relative_l2",
                "amplitude_relative_error",
                "dominant_frequency_relative_error",
            ]
            aggregate = {
                key: float(np.mean([item[key] for item in case_reports]))
                for key in aggregate_keys
            }
            report = {
                "rates": list(rates), "ridge": ridge, "memory_state_dimension": 28 * len(rates),
                "train_residual_fit_relative_l2": train_error, "stable": stable,
                "aggregate": aggregate, "cases": case_reports,
            }
            candidate_reports.append(report)
            candidate_assets.append((report, scale, weights, trajectories))
            print(json.dumps({
                "rates": list(rates), "ridge": ridge, "stable": stable,
                "rollout": aggregate["velocity_rollout_relative_l2"],
            }))

    eligible = [item for item in candidate_assets if item[0]["stable"]]
    if not eligible:
        raise RuntimeError("all CDM candidates diverged")
    best_report, best_scale, best_weights, best_trajectories = min(
        eligible,
        key=lambda item: (
            item[0]["aggregate"]["velocity_rollout_relative_l2"],
        ),
    )
    final_case_reports = []
    best_trajectories = []
    for case in validation_cases:
        report, trajectory = evaluate_case(
            case, baseline, tuple(best_report["rates"]), best_scale, best_weights,
            a_scale, nu_mean, nu_scale, args.warmup_intervals, args.final_max_step,
            args.divergence_norm, True,
        )
        final_case_reports.append(report)
        best_trajectories.append(trajectory)
    final_keys = [
        "velocity_rollout_relative_l2", "pressure_rollout_relative_l2",
        "one_step_velocity_relative_l2", "amplitude_relative_error",
        "dominant_frequency_relative_error",
    ]
    selected_final = {
        **{key: value for key, value in best_report.items() if key not in {"aggregate", "cases"}},
        "aggregate": {
            key: float(np.mean([item[key] for item in final_case_reports])) for key in final_keys
        },
        "cases": final_case_reports,
        "final_max_step": args.final_max_step,
    }
    np.savez_compressed(
        args.output_dir / "cdm_validation_selected.npz",
        rates=np.asarray(best_report["rates"], dtype=np.float64), ridge=np.asarray(best_report["ridge"]),
        feature_scale=best_scale, weights=best_weights, a_scale=a_scale,
        nu_mean=np.asarray(nu_mean), nu_scale=np.asarray(nu_scale),
        velocity_rank=np.asarray(28), pressure_rank=np.asarray(24),
    )
    for case, trajectory in zip(validation_cases, best_trajectories):
        np.savez_compressed(
            args.output_dir / f"trajectory_Re{case[0]:010.6f}.npz",
            Re=np.asarray(case[0]), times=case[1], truth_velocity=case[2],
            truth_pressure=case[3], predicted_velocity=trajectory,
        )
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "fixed-address continuous Delta-memory residual on frozen periodic FVM-GROM",
        "selection_rule": "minimum stable validation velocity rollout relative L2",
        "selection_max_step": args.selection_max_step,
        "selected": selected_final,
        "candidates": candidate_reports,
        "baseline_hashes": {
            "fvm_tensors": sha256(args.fvm_tensors),
            "pressure_closure": sha256(args.pressure_closure),
            "frozen_manifest": sha256(args.baseline_frozen),
        },
        "train_pairs": int(len(pair_indices)),
        "target": "(truth next coefficient - frozen-baseline RK4 next coefficient) / snapshot interval",
        "validation_loaded": True,
        "heldout_loaded": False,
    }
    (args.output_dir / "VALIDATION_SELECTION.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"selected": selected_final, "heldout_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
