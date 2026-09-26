#!/usr/bin/env python3
"""Audit Galerkin sign conventions and closure error on validation projections only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_cdm_grom import (
    VALIDATION_RE,
    interpolation_weights,
    interpolate,
    load_selected_cases,
    project_case,
    relative_l2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--canonical-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def one(directory: Path, pattern: str) -> Path:
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {pattern}, found {paths}")
    return paths[0]


def main() -> None:
    args = parse_args()
    with np.load(args.dataset_root / "pod/weighted_pod_velocity.npz") as pod:
        u_mean = np.asarray(pod["mean"], dtype=np.float64)
        u_weights = np.asarray(pod["weights"], dtype=np.float64)
        u_modes = np.asarray(pod["weighted_modes"], dtype=np.float64)
    with np.load(args.dataset_root / "pod/weighted_pod_pressure.npz") as pod:
        p_mean = np.asarray(pod["mean"], dtype=np.float64)
        p_weights = np.asarray(pod["weights"], dtype=np.float64)
        p_modes = np.asarray(pod["weighted_modes"], dtype=np.float64)
    with np.load(one(args.tensor_dir, "semi_intrusive_*compact.npz")) as velocity:
        nodes = np.asarray(velocity["Re_list"], dtype=np.float64)
        c_all = np.asarray(velocity["c_all"], dtype=np.float64)
        a_all = np.asarray(velocity["A_all"], dtype=np.float64)
        h = np.asarray(velocity["H"], dtype=np.float64)
        p_coupling = np.asarray(velocity["P"], dtype=np.float64)
    with np.load(one(args.tensor_dir, "pressure_poisson_*.npz")) as pressure:
        pc_all = np.asarray(pressure["c_tilde_all"], dtype=np.float64)
        pa_all = np.asarray(pressure["A_tilde_all"], dtype=np.float64)
        ph = np.asarray(pressure["H_tilde"], dtype=np.float64)

    reports = []
    for entry in load_selected_cases(args.canonical_cases, VALIDATION_RE):
        times, coeff, pressure_truth, _ = project_case(
            Path(entry["path"]), u_mean, u_weights, u_modes,
            p_mean, p_weights, p_modes,
        )
        bracket = interpolation_weights(nodes, float(entry["Re"]))
        c = interpolate(c_all, bracket)
        a = interpolate(a_all, bracket)
        pc = interpolate(pc_all, bracket)
        pa = interpolate(pa_all, bracket)
        pressure_pred = (
            pc[None] + coeff @ pa.T
            + np.einsum("mjk,tj,tk->tm", ph, coeff, coeff, optimize=True)
        )
        base = c[None] + coeff @ a.T + np.einsum(
            "ijk,tj,tk->ti", h, coeff, coeff, optimize=True
        )
        pressure_force = pressure_pred @ p_coupling.T
        derivative = (coeff[2:] - coeff[:-2]) / (times[2:, None] - times[:-2, None])
        options = {
            "plus_pressure": base[1:-1] + pressure_force[1:-1],
            "minus_pressure": base[1:-1] - pressure_force[1:-1],
            "no_pressure": base[1:-1],
        }
        reports.append({
            "Re": float(entry["Re"]),
            "pressure_algebraic_relative_l2": relative_l2(pressure_pred, pressure_truth),
            "projected_derivative_rms": float(np.sqrt(np.mean(derivative**2))),
            "rhs_plus_pressure_rms": float(np.sqrt(np.mean(options["plus_pressure"]**2))),
            "derivative_relative_residual": {
                key: relative_l2(value, derivative) for key, value in options.items()
            },
            "resolved_derivative_relative_residual": {
                key: relative_l2(value[:, :11], derivative[:, :11])
                for key, value in options.items()
            },
        })
    payload = {
        "role": "validation_dynamics_audit",
        "heldout_loaded": False,
        "cases": reports,
        "mean": {
            key: float(np.mean([r["derivative_relative_residual"][key] for r in reports]))
            for key in ["plus_pressure", "minus_pressure", "no_pressure"]
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
