#!/usr/bin/env python3
"""Build train-only mean-drift calibration for the FVM-Galerkin ROM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_fvm_grom import pressure_features


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-pod", type=Path, required=True)
    parser.add_argument("--fvm-tensors", type=Path, required=True)
    parser.add_argument("--pressure-closure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-snapshots", type=int, default=32)
    return parser.parse_args()


def tag_to_re(tag):
    return float(tag[2:].replace("p", "."))


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.velocity_pod) as pod:
        coefficients = np.asarray(pod["coefficients"], dtype=np.float64)[:, :11]
        tags = np.asarray(pod["snapshot_case_tags"]).astype(str)
        times = np.asarray(pod["snapshot_times"], dtype=np.float64)
    with np.load(args.fvm_tensors) as source:
        tensor = {key: np.asarray(source[key]) for key in source.files}
    with np.load(args.pressure_closure) as source:
        kind = str(source["kind"])
        scale = np.asarray(source["feature_scale"], dtype=np.float64)
        weights = np.asarray(source["weights"], dtype=np.float64)

    reynolds = np.asarray(sorted({tag_to_re(tag) for tag in tags}))
    centers = []
    corrections = []
    raw_mean_rhs_norms = []
    radii = []
    for re_value in reynolds:
        tag = ("Re" + f"{re_value:010.6f}").replace(".", "p")
        indices = np.flatnonzero(tags == tag)
        state = coefficients[indices][-args.tail_snapshots :]
        case_times = times[indices][-args.tail_snapshots :]
        viscosity = 1.0 / re_value
        p_features = pressure_features(state, viscosity, kind) / scale[None]
        pressure = p_features @ weights
        rhs = (
            tensor["c_conv"][None] + viscosity * tensor["c_diff"][None] + tensor["c_pressure"][None]
            + state @ (tensor["A_conv"] + viscosity * tensor["A_diff"]).T
            + np.einsum("ijk,tj,tk->ti", tensor["H_conv"], state, state, optimize=True)
            + pressure @ tensor["P"].T
        )
        target_mean_derivative = (state[-1] - state[0]) / (case_times[-1] - case_times[0])
        mean_rhs = rhs.mean(axis=0)
        center = state.mean(axis=0)
        centers.append(center)
        radii.append(max(0.005, 1.2 * float(np.max(np.linalg.norm(state - center[None], axis=1)))))
        corrections.append(target_mean_derivative - mean_rhs)
        raw_mean_rhs_norms.append(float(np.linalg.norm(mean_rhs - target_mean_derivative)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, Re_nodes=reynolds, center=np.asarray(centers),
        constant_correction=np.asarray(corrections), radius=np.asarray(radii),
    )
    report = {
        "schema_version": 1, "status": "PASS", "train_nodes": len(reynolds),
        "maximum_raw_mean_drift_norm": max(raw_mean_rhs_norms),
        "minimum_raw_mean_drift_norm": min(raw_mean_rhs_norms),
        "mean_raw_mean_drift_norm": float(np.mean(raw_mean_rhs_norms)),
        "tail_snapshots": args.tail_snapshots,
        "radius_range": [float(min(radii)), float(max(radii))],
        "validation_loaded": False, "heldout_loaded": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
