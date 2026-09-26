#!/usr/bin/env python3
"""Select one scalar eddy damping on Hopf validation without held-out access."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from evaluate_fvm_grom import pressure_features, project_case


VALIDATION_RE = np.asarray([94.5, 95.25, 95.5, 97.5, 99.0, 101.5])
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-pod", type=Path, required=True)
    parser.add_argument("--pressure-pod", type=Path, required=True)
    parser.add_argument("--fvm-tensors", type=Path, required=True)
    parser.add_argument("--pressure-closure", type=Path, required=True)
    parser.add_argument("--mean-calibration", type=Path, required=True)
    parser.add_argument("--canonical-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-step", type=float, default=0.1)
    parser.add_argument("--damping-kind", choices=["linear", "barrier"], default="linear")
    return parser.parse_args()


def interpolate(nodes, array, value):
    if value <= nodes[0]:
        left, right = 0, 1
    elif value >= nodes[-1]:
        left, right = len(nodes) - 2, len(nodes) - 1
    else:
        right = int(np.searchsorted(nodes, value)); left = right - 1
    fraction = (value - nodes[left]) / (nodes[right] - nodes[left])
    return (1 - fraction) * array[left] + fraction * array[right]


def main():
    args = parse_args()
    gammas = np.asarray(
        [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
        if args.damping_kind == "linear"
        else [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0]
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    canonical = json.loads(args.canonical_cases.read_text())
    with np.load(args.velocity_pod) as pod:
        u_mean = np.asarray(pod["mean"], dtype=np.float64)
        u_weights = np.asarray(pod["weights"], dtype=np.float64)
        u_modes = np.asarray(pod["weighted_modes"], dtype=np.float64)[:11]
    with np.load(args.pressure_pod) as pod:
        p_mean = np.asarray(pod["mean"], dtype=np.float64)
        p_weights = np.asarray(pod["weights"], dtype=np.float64)
        p_modes = np.asarray(pod["weighted_modes"], dtype=np.float64)[:11]
    with np.load(args.fvm_tensors) as source:
        tensor = {key: np.asarray(source[key]) for key in source.files}
    with np.load(args.pressure_closure) as source:
        kind = str(source["kind"]); scale = np.asarray(source["feature_scale"]); weights = np.asarray(source["weights"])
    with np.load(args.mean_calibration) as source:
        nodes = np.asarray(source["Re_nodes"]); centers = np.asarray(source["center"]); corrections = np.asarray(source["constant_correction"]); radii = np.asarray(source["radius"])

    total_squared_error = np.zeros(len(gammas))
    total_squared_truth = np.zeros(len(gammas))
    full_horizon = np.ones(len(gammas), dtype=bool)
    case_reports = []
    for target_re in VALIDATION_RE:
        entry = next(item for item in canonical if abs(float(item["Re"]) - target_re) < 5e-7)
        re_value, times, truth, _ = project_case(
            Path(entry["path"]), u_mean, u_weights, u_modes, p_mean, p_weights, p_modes
        )
        viscosity = 1.0 / re_value
        center = interpolate(nodes, centers, re_value)
        correction = interpolate(nodes, corrections, re_value)
        radius = float(interpolate(nodes, radii, re_value))
        states = np.repeat(truth[2][None], len(gammas), axis=0)
        predictions = np.full((len(gammas), len(times), 11), np.nan)
        predictions[:, 2] = states
        alive = np.ones(len(gammas), dtype=bool)

        def rhs(value):
            p_x = pressure_features(value, viscosity, kind) / scale[None]
            pressure = p_x @ weights
            physical = (
                tensor["c_conv"][None] + viscosity * tensor["c_diff"][None] + tensor["c_pressure"][None]
                + value @ (tensor["A_conv"] + viscosity * tensor["A_diff"]).T
                + np.einsum("ijk,nj,nk->ni", tensor["H_conv"], value, value, optimize=True)
                + pressure @ tensor["P"].T
            )
            delta = value - center[None]
            if args.damping_kind == "linear":
                damping = gammas[:, None] * delta
            else:
                excess = np.maximum(0.0, np.sum(delta * delta, axis=1) / (radius * radius) - 1.0)
                damping = gammas[:, None] * excess[:, None] * delta
            return physical + correction[None] - damping

        for index in range(2, len(times) - 1):
            substeps = max(1, int(math.ceil((times[index + 1] - times[index]) / args.max_step)))
            step = (times[index + 1] - times[index]) / substeps
            for _ in range(substeps):
                k1 = rhs(states); k2 = rhs(states + 0.5 * step * k1)
                k3 = rhs(states + 0.5 * step * k2); k4 = rhs(states + step * k3)
                states += step * (k1 + 2*k2 + 2*k3 + k4) / 6
            alive &= np.all(np.isfinite(states), axis=1) & (np.linalg.norm(states, axis=1) < 100)
            states[~alive] = 0.0
            predictions[alive, index + 1] = states[alive]
        errors = []
        for gamma_index in range(len(gammas)):
            if not alive[gamma_index]:
                errors.append(None); full_horizon[gamma_index] = False; continue
            error = predictions[gamma_index, 3:] - truth[3:]
            total_squared_error[gamma_index] += np.sum(error**2)
            total_squared_truth[gamma_index] += np.sum(truth[3:]**2)
            errors.append(float(np.linalg.norm(error) / np.linalg.norm(truth[3:])))
        case_reports.append({"Re": re_value, "relative_l2_by_gamma": errors})
    scores = np.sqrt(total_squared_error / np.maximum(total_squared_truth, 1e-30))
    scores[~full_horizon] = np.inf
    selected_index = int(np.argmin(scores))
    payload = {
        "schema_version": 1, "status": "PASS",
        "gammas": gammas.tolist(), "aggregate_relative_l2": [None if not np.isfinite(x) else float(x) for x in scores],
        "full_horizon": full_horizon.tolist(), "selected_gamma": float(gammas[selected_index]),
        "selected_relative_l2": float(scores[selected_index]), "cases": case_reports,
        "closure": (
            "linear eddy damping: -gamma(a-center)"
            if args.damping_kind == "linear"
            else "tail-attractor energy barrier: -gamma max(0, ||a-center||^2/radius^2-1)(a-center)"
        ),
        "validation_loaded": True, "heldout_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
