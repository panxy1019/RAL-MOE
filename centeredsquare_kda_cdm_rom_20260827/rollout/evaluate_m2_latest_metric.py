#!/usr/bin/env python3
"""Refit legacy fixed-pole M2 and evaluate it with the current J500 metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


RATES = np.asarray([0.01, 0.05], dtype=np.float64)
RIDGE = 10.0
PAIR_STRIDE = 4


def tag_to_re(tag: str) -> float:
    return float(tag[2:].replace("p", "."))


def pressure_features(state: np.ndarray, viscosity: np.ndarray) -> np.ndarray:
    state = np.atleast_2d(state)
    viscosity = np.broadcast_to(np.asarray(viscosity, dtype=np.float64), (len(state),))
    return np.concatenate([
        np.ones((len(state), 1)), viscosity[:, None], state,
        viscosity[:, None] * state,
    ], axis=1)


class NumpyBaseline:
    def __init__(self, tensor_path: Path, pressure_path: Path):
        with np.load(tensor_path, allow_pickle=False) as source:
            self.tensor = {key: np.asarray(source[key], dtype=np.float64) for key in source.files}
        with np.load(pressure_path, allow_pickle=False) as source:
            self.pressure_scale = np.asarray(source["feature_scale"], dtype=np.float64)
            self.pressure_weights = np.asarray(source["weights"], dtype=np.float64)

    def rhs(self, state: np.ndarray, viscosity: np.ndarray) -> np.ndarray:
        single = np.ndim(state) == 1
        state = np.atleast_2d(state)
        viscosity = np.broadcast_to(np.asarray(viscosity, dtype=np.float64), (len(state),))
        pressure = (pressure_features(state, viscosity) / self.pressure_scale[None]) @ self.pressure_weights
        result = (
            self.tensor["c_conv"][None] + viscosity[:, None] * self.tensor["c_diff"][None]
            + self.tensor["c_pressure"][None]
            + state @ self.tensor["A_conv"].T
            + viscosity[:, None] * (state @ self.tensor["A_diff"].T)
            + np.einsum("ijk,nj,nk->ni", self.tensor["H_conv"], state, state, optimize=True)
            + pressure @ self.tensor["P"].T
        )
        return result[0] if single else result


def rk4_batch(state: np.ndarray, duration: np.ndarray, max_step: float, rhs) -> np.ndarray:
    counts = np.ceil(duration / max_step).astype(int)
    if np.any(counts != counts[0]):
        raise ValueError("training snapshot spacings must match")
    step = duration / counts
    value = state.copy()
    for _ in range(int(counts[0])):
        h = step[:, None]
        k1 = rhs(value)
        k2 = rhs(value + 0.5 * h * k1)
        k3 = rhs(value + 0.5 * h * k2)
        k4 = rhs(value + h * k3)
        value += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return value


def algebraic_features(state, viscosity, a_scale, nu_mean, nu_scale):
    a_hat = state / a_scale
    nu_hat = (viscosity - nu_mean) / nu_scale
    return np.concatenate(([1.0, nu_hat], a_hat, nu_hat * a_hat))


def design_features(state, memory, viscosity, a_scale, nu_mean, nu_scale):
    return np.concatenate([
        algebraic_features(state, viscosity, a_scale, nu_mean, nu_scale),
        *(memory[index] - state / a_scale for index in range(len(RATES))),
    ])


def build_design(states, times, tags, a_scale, nu_mean, nu_scale):
    rows, indices = [], []
    start = 0
    while start < len(tags):
        stop = start + 1
        while stop < len(tags) and tags[stop] == tags[start]:
            stop += 1
        viscosity = 1.0 / tag_to_re(tags[start])
        memory = np.repeat((states[start] / a_scale)[None], len(RATES), axis=0)
        for index in range(start, stop - 1):
            rows.append(design_features(states[index], memory, viscosity, a_scale, nu_mean, nu_scale))
            indices.append(index)
            decay = np.exp(-RATES * (times[index + 1] - times[index]))[:, None]
            memory = decay * memory + (1.0 - decay) * (states[index] / a_scale)[None]
        start = stop
    return np.asarray(rows), np.asarray(indices, dtype=int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--velocity-pod", type=Path, required=True)
    parser.add_argument("--fvm-tensors", type=Path, required=True)
    parser.add_argument("--pressure-closure", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-max-step", type=float, default=0.2)
    parser.add_argument("--final-max-step", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-init", choices=["legacy", "zero"], default="legacy")
    parser.add_argument("--swanlab-project")
    parser.add_argument("--swanlab-mode", choices=["online", "cloud", "local", "offline"], default="local")
    args = parser.parse_args()

    swanlab = None
    if args.swanlab_project:
        import swanlab as swanlab_module
        swanlab = swanlab_module
        swanlab.init(
            project=args.swanlab_project, experiment_name="m2-current-j500-supplement",
            mode=args.swanlab_mode,
            config={"rates": RATES.tolist(), "ridge": RIDGE, "heldout_loaded": False},
        )

    baseline = NumpyBaseline(args.fvm_tensors, args.pressure_closure)
    with np.load(args.velocity_pod, allow_pickle=False) as source:
        train_a = np.asarray(source["coefficients"], dtype=np.float64)[:, :28]
        train_times = np.asarray(source["snapshot_times"], dtype=np.float64)
        train_tags = np.asarray(source["snapshot_case_tags"]).astype(str)
        mean = np.asarray(source["mean"], dtype=np.float64)
        cell_weights = np.asarray(source["weights"], dtype=np.float64)
        modes = np.asarray(source["weighted_modes"], dtype=np.float64)[:28]
    a_scale = np.std(train_a, axis=0)
    a_scale[a_scale < 1e-8] = 1.0
    train_nu = np.asarray([1.0 / tag_to_re(tag) for tag in train_tags])
    nu_mean, nu_scale = float(np.mean(train_nu)), float(np.std(train_nu))

    pair_indices, position = [], 0
    for index in range(len(train_a) - 1):
        if train_tags[index] == train_tags[index + 1]:
            if position % PAIR_STRIDE == 0:
                pair_indices.append(index)
            position += 1
        else:
            position = 0
    pair_indices = np.asarray(pair_indices, dtype=int)
    pair_dt = train_times[pair_indices + 1] - train_times[pair_indices]
    pair_nu = train_nu[pair_indices]
    baseline_next = rk4_batch(
        train_a[pair_indices], pair_dt, args.selection_max_step,
        lambda value: baseline.rhs(value, pair_nu),
    )
    target = (train_a[pair_indices + 1] - baseline_next) / pair_dt[:, None]
    raw_design, design_indices = build_design(train_a, train_times, train_tags, a_scale, nu_mean, nu_scale)
    chosen = np.searchsorted(design_indices, pair_indices)
    if not np.array_equal(design_indices[chosen], pair_indices):
        raise RuntimeError("training pair selection mismatch")
    raw_design = raw_design[chosen]
    feature_scale = np.std(raw_design, axis=0)
    feature_scale[feature_scale < 1e-10] = 1.0
    feature_scale[0] = 1.0
    design = raw_design / feature_scale[None]
    regularizer = np.eye(design.shape[1]) * RIDGE
    regularizer[0, 0] = 0.0
    readout = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
    train_error = float(np.linalg.norm(design @ readout - target) / max(np.linalg.norm(target), 1e-14))

    dtype, device = torch.float64, torch.device(args.device)
    with np.load(args.fvm_tensors, allow_pickle=False) as source:
        tensors = {key: torch.as_tensor(source[key], dtype=dtype, device=device) for key in source.files}
    with np.load(args.pressure_closure, allow_pickle=False) as source:
        pscale = torch.as_tensor(source["feature_scale"], dtype=dtype, device=device)
        pweights = torch.as_tensor(source["weights"], dtype=dtype, device=device)
    t_feature_scale = torch.as_tensor(feature_scale, dtype=dtype, device=device)
    t_readout = torch.as_tensor(readout, dtype=dtype, device=device)
    t_a_scale = torch.as_tensor(a_scale, dtype=dtype, device=device)
    t_rates = torch.as_tensor(RATES, dtype=dtype, device=device)

    validation = json.loads(args.validation_manifest.read_text())
    cases = []
    for entry in validation["cases"]:
        path = args.validation_root / Path(entry["local_path"]).name
        with np.load(path, allow_pickle=False) as source:
            times = np.asarray(source["times"], dtype=np.float64)
            velocity = np.asarray(source["U"], dtype=np.float64)
            re_value = float(source["Re"])
        truth = ((velocity - mean[None]).reshape(len(times), -1) * cell_weights[None]) @ modes.T
        fluctuation_energy = np.sum(((velocity - mean[None]).reshape(len(times), -1) * cell_weights[None]) ** 2, axis=1)
        unresolved = np.maximum(0.0, fluctuation_energy - np.sum(truth**2, axis=1))
        denominator = float(np.sum((velocity.reshape(len(times), -1) * cell_weights[None]) ** 2))
        cases.append((re_value, times, truth, unresolved, denominator))
    if any(not np.array_equal(case[1], cases[0][1]) for case in cases[1:]):
        raise RuntimeError("validation time grids must match for batched rollout")

    re_values = torch.as_tensor([case[0] for case in cases], dtype=dtype, device=device)
    viscosity = 1.0 / re_values
    state = torch.as_tensor(np.stack([case[2][0] for case in cases]), dtype=dtype, device=device)
    if args.memory_init == "legacy":
        memory = (state[:, None, :] / t_a_scale[None, None, :]).repeat(1, len(RATES), 1)
    else:
        memory = torch.zeros((len(cases), len(RATES), 28), dtype=dtype, device=device)
    trajectory = np.empty((len(cases), len(cases[0][1]), 28), dtype=np.float64)
    memory_norm = np.empty((len(cases), len(cases[0][1])), dtype=np.float64)

    def torch_baseline(value):
        pfeature = torch.cat([
            torch.ones((len(value), 1), dtype=dtype, device=device), viscosity[:, None],
            value, viscosity[:, None] * value,
        ], dim=1)
        pressure = (pfeature / pscale) @ pweights
        return (
            tensors["c_conv"] + viscosity[:, None] * tensors["c_diff"] + tensors["c_pressure"]
            + value @ tensors["A_conv"].T
            + viscosity[:, None] * (value @ tensors["A_diff"].T)
            + torch.einsum("ijk,bj,bk->bi", tensors["H_conv"], value, value)
            + pressure @ tensors["P"].T
        )

    def joint_rhs(value, mem):
        a_hat = value / t_a_scale
        nu_hat = (viscosity - nu_mean) / nu_scale
        raw = torch.cat([
            torch.ones((len(value), 1), dtype=dtype, device=device), nu_hat[:, None],
            a_hat, nu_hat[:, None] * a_hat,
            mem[:, 0] - a_hat, mem[:, 1] - a_hat,
        ], dim=1)
        correction = (raw / t_feature_scale) @ t_readout
        mem_rhs = t_rates[None, :, None] * (a_hat[:, None, :] - mem)
        return torch_baseline(value) + correction, mem_rhs

    def advance(value, mem, duration):
        count = max(1, int(np.ceil(duration / args.final_max_step)))
        step = duration / count
        for _ in range(count):
            k1a, k1m = joint_rhs(value, mem)
            k2a, k2m = joint_rhs(value + 0.5 * step * k1a, mem + 0.5 * step * k1m)
            k3a, k3m = joint_rhs(value + 0.5 * step * k2a, mem + 0.5 * step * k2m)
            k4a, k4m = joint_rhs(value + step * k3a, mem + step * k3m)
            value = value + step * (k1a + 2.0 * k2a + 2.0 * k3a + k4a) / 6.0
            mem = mem + step * (k1m + 2.0 * k2m + 2.0 * k3m + k4m) / 6.0
        return value, mem

    times = cases[0][1]
    for index in range(len(times)):
        trajectory[:, index] = state.detach().cpu().numpy()
        memory_norm[:, index] = torch.linalg.vector_norm(memory.reshape(len(cases), -1), dim=1).detach().cpu().numpy()
        if index + 1 < len(times):
            state, memory = advance(state, memory, float(times[index + 1] - times[index]))

    phase = json.loads(args.phase_summary.read_text())
    pair = (int(phase["phase_mode_p"]), int(phase["phase_mode_q"]))
    per_case = []
    for case_index, (re_value, case_times, truth, unresolved, denominator) in enumerate(cases):
        predicted = trajectory[case_index]
        truth_phase = np.unwrap(np.arctan2(truth[:, pair[1]], truth[:, pair[0]]))
        predicted_phase = np.unwrap(np.arctan2(predicted[:, pair[1]], predicted[:, pair[0]]))
        truth_frequency = float(np.polyfit(case_times, truth_phase, 1)[0] / (2.0 * np.pi))
        predicted_frequency = float(np.polyfit(case_times, predicted_phase, 1)[0] / (2.0 * np.pi))
        numerator = float(np.sum((predicted - truth) ** 2) + np.sum(unresolved))
        finite = bool(np.all(np.isfinite(predicted)) and np.all(np.isfinite(memory_norm[case_index])))
        item = {
            "Re": re_value,
            "J500": float(np.sqrt(numerator / denominator)),
            "finite": finite,
            "coefficient_norm_max": float(np.max(np.linalg.norm(predicted, axis=1))),
            "memory_norm_max": float(np.max(memory_norm[case_index])),
            "phase_error_final_abs": float(abs(predicted_phase[-1] - truth_phase[-1])),
            "frequency_relative_error": float(abs(predicted_frequency - truth_frequency) / max(abs(truth_frequency), 1e-14)),
            "amplitude_relative_error": float(abs(
                np.mean(np.hypot(predicted[:, pair[0]], predicted[:, pair[1]]))
                - np.mean(np.hypot(truth[:, pair[0]], truth[:, pair[1]]))
            ) / max(np.mean(np.hypot(truth[:, pair[0]], truth[:, pair[1]])), 1e-14)),
        }
        per_case.append(item)
        if swanlab is not None:
            swanlab.log({"validation/Re": re_value, "validation/M2_J500": item["J500"]})

    values = [item["J500"] for item in per_case]
    stable = all(item["finite"] and item["coefficient_norm_max"] < 1e6 and item["memory_norm_max"] < 1e6 for item in per_case)
    aggregate = {
        "name": "M2_rates_0.01_0.05_ridge_10", "kind": 2, "stable_6_of_6": stable,
        "J500_mean": float(np.mean(values)), "J500_median": float(np.median(values)),
        "J500_max": float(np.max(values)),
    }
    payload = {
        "schema_version": 1,
        "method": "legacy fixed-pole CDM refit unchanged; evaluated with current absolute full-field J500",
        "rates": RATES.tolist(), "ridge": RIDGE, "train_pair_stride": PAIR_STRIDE,
        "selection_max_step": args.selection_max_step, "final_max_step": args.final_max_step,
        "train_pairs": int(len(pair_indices)), "train_residual_fit_relative_l2": train_error,
        "initialization": (
            "truth_a_at_t0_and_memory_equal_a0_over_scale"
            if args.memory_init == "legacy" else "truth_a_at_t0_and_zero_memory"
        ),
        "phase_pair": list(pair), "per_case": per_case, "aggregate": aggregate,
        "heldout_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".npz"), rates=RATES, ridge=np.asarray(RIDGE),
        feature_scale=feature_scale, readout=readout, a_scale=a_scale,
        nu_mean=np.asarray(nu_mean), nu_scale=np.asarray(nu_scale), trajectory=trajectory,
    )
    if swanlab is not None:
        swanlab.log({"summary/M2_J500_mean": aggregate["J500_mean"], "summary/stable_6_of_6": int(stable)})
        swanlab.finish()
    print(json.dumps({"aggregate": aggregate, "train_fit": train_error, "heldout_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
