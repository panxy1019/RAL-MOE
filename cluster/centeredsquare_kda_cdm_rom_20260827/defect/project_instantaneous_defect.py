#!/usr/bin/env python3
"""Project OpenFOAM snapshot RHS and compare it with the frozen tensor FVM-GROM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pressure_features(state, viscosity):
    return np.concatenate([np.ones((len(state), 1)), np.full((len(state), 1), viscosity), state, viscosity * state], axis=1)


def make_rhs(tensor, feature_scale, pressure_weights, viscosity):
    linear = tensor["A_conv"] + viscosity * tensor["A_diff"]
    constant = tensor["c_conv"] + viscosity * tensor["c_diff"] + tensor["c_pressure"]

    def rhs(state):
        state = np.asarray(state, dtype=np.float64)
        features = np.concatenate(([1.0, viscosity], state, viscosity * state))
        pressure = (features / feature_scale) @ pressure_weights
        return (
            constant + linear @ state
            + np.einsum("ijk,j,k->i", tensor["H_conv"], state, state, optimize=True)
            + tensor["P"] @ pressure
        )

    return rhs


def rk4_advance(state, duration, rhs, max_step):
    count = max(1, int(np.ceil(duration / max_step)))
    step = duration / count
    result = np.asarray(state, dtype=np.float64).copy()
    for _ in range(count):
        k1 = rhs(result)
        k2 = rhs(result + 0.5 * step * k1)
        k3 = rhs(result + 0.5 * step * k2)
        k4 = rhs(result + step * k3)
        result += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foam-case", type=Path, required=True)
    parser.add_argument("--velocity-pod", type=Path, required=True)
    parser.add_argument("--pressure-pod", type=Path)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--fvm-tensors", type=Path, required=True)
    parser.add_argument("--pressure-closure", type=Path, required=True)
    parser.add_argument("--dataset-tools-dir", type=Path, required=True)
    parser.add_argument("--max-step", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = json.loads((args.foam_case / "snapshot_manifest.json").read_text())
    import sys
    sys.path.insert(0, str(args.dataset_tools_dir))
    from dataset_tools import read_internal_field
    with np.load(manifest["snapshot_npz"], allow_pickle=False) as source:
        velocity = np.asarray(source["U"], dtype=np.float64)
        pressure = np.asarray(source["p"], dtype=np.float64)
        times = np.asarray(source["times"], dtype=np.float64)
    with np.load(args.mesh, allow_pickle=False) as source:
        volumes = np.asarray(source["cellVolumes"], dtype=np.float64)
    with np.load(args.velocity_pod, allow_pickle=False) as source:
        u_mean = np.asarray(source["mean"], dtype=np.float64)
        u_weights = np.asarray(source["weights"], dtype=np.float64)
        u_weighted_modes = np.asarray(source["weighted_modes"], dtype=np.float64)[:28]
        u_modes = np.asarray(source["modes"], dtype=np.float64)[:28].reshape(28, len(volumes), 2)
    with np.load(args.fvm_tensors, allow_pickle=False) as source:
        tensor = {key: np.asarray(source[key], dtype=np.float64) for key in source.files}
    with np.load(args.pressure_closure, allow_pickle=False) as source:
        feature_scale = np.asarray(source["feature_scale"], dtype=np.float64)
        pressure_weights = np.asarray(source["weights"], dtype=np.float64)

    a_cfd = ((velocity - u_mean[None]).reshape(len(times), -1) * u_weights[None]) @ u_weighted_modes.T
    b_cfd = None
    if args.pressure_pod is not None:
        with np.load(args.pressure_pod, allow_pickle=False) as source:
            p_mean = np.asarray(source["mean"], dtype=np.float64)
            p_weights = np.asarray(source["weights"], dtype=np.float64)
            p_modes = np.asarray(source["weighted_modes"], dtype=np.float64)[:24]
        b_cfd = ((pressure - p_mean[None]) * p_weights[None]) @ p_modes.T
    adot = []
    for entry in manifest["entries"]:
        field = read_internal_field(args.foam_case / str(entry["foam_time"]) / "romRhs", "vector", len(volumes))[:, :2]
        adot.append(np.einsum("n,inc,nc->i", volumes, u_modes, field, optimize=True))
    adot = np.asarray(adot)
    viscosity = float(manifest["nu"])
    predicted_b = (pressure_features(a_cfd, viscosity) / feature_scale[None]) @ pressure_weights
    fvm_rhs = (
        tensor["c_conv"][None] + viscosity * tensor["c_diff"][None] + tensor["c_pressure"][None]
        + np.einsum("ij,nj->ni", tensor["A_conv"] + viscosity * tensor["A_diff"], a_cfd)
        + np.einsum("ijk,nj,nk->ni", tensor["H_conv"], a_cfd, a_cfd, optimize=True)
        + np.einsum("ij,nj->ni", tensor["P"], predicted_b)
    )
    defect = adot - fvm_rhs
    rhs = make_rhs(tensor, feature_scale, pressure_weights, viscosity)
    flowmap_defect = np.asarray([
        (a_cfd[index + 1] - rk4_advance(a_cfd[index], times[index + 1] - times[index], rhs, args.max_step))
        / (times[index + 1] - times[index])
        for index in range(len(times) - 1)
    ])
    flat_instant = defect[:-1].ravel()
    flat_flowmap = flowmap_defect.ravel()
    correlation = float(np.corrcoef(flat_instant, flat_flowmap)[0, 1])
    modal_energy = np.mean(defect**2, axis=0)
    modal_energy_fraction = modal_energy / max(float(np.sum(modal_energy)), 1e-30)
    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "a_cfd.npy", a_cfd)
    np.save(args.output_dir / "adot_cfd_projected.npy", adot)
    np.save(args.output_dir / "fvm_rhs_tensor.npy", fvm_rhs)
    np.save(args.output_dir / "instantaneous_defect.npy", defect)
    np.save(args.output_dir / "flowmap_defect_old.npy", flowmap_defect)
    if b_cfd is not None:
        np.save(args.output_dir / "pressure_coefficients_cfd.npy", b_cfd)
    np.save(args.output_dir / "time.npy", times)
    payload = {
        "schema_version": 1, "Re": float(manifest["Re"]), "snapshot_count": len(times),
        "instantaneous_defect_relative_l2": float(np.linalg.norm(defect) / max(np.linalg.norm(adot), 1e-14)),
        "instantaneous_defect_rms": float(np.sqrt(np.mean(defect**2))),
        "flowmap_defect_rms": float(np.sqrt(np.mean(flowmap_defect**2))),
        "flowmap_instantaneous_correlation": correlation,
        "instantaneous_defect_modal_energy_fraction": modal_energy_fraction.tolist(),
        "finite": bool(np.all(np.isfinite(defect))), "validation_loaded": False, "heldout_loaded": False,
    }
    if b_cfd is not None:
        payload["pressure_closure_relative_l2"] = float(
            np.linalg.norm(predicted_b - b_cfd) / max(np.linalg.norm(b_cfd), 1e-14)
        )
    (args.output_dir / "Re.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
