#!/usr/bin/env python3
"""Assemble projected finite-volume Galerkin tensors from OpenFOAM probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/ray/Desktop/centeredSquare/scripts")
from dataset_tools import read_internal_field  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-case", type=Path, required=True)
    parser.add_argument("--velocity-pod", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.probe_case / "probe_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ru = int(manifest["velocity_rank"])
    rp = int(manifest["pressure_rank"])
    eps_u = float(manifest["velocity_epsilon"])
    eps_p = float(manifest["pressure_epsilon"])
    with np.load(args.mesh) as mesh:
        volumes = np.asarray(mesh["cellVolumes"], dtype=np.float64)
    with np.load(args.velocity_pod) as pod:
        modes = np.asarray(pod["modes"], dtype=np.float64)[:ru].reshape(ru, len(volumes), 2)

    state_values = {}
    for state in manifest["states"]:
        directory = args.probe_case / str(state["index"])
        projected = {}
        for field in ["romConvection", "romDiffusion", "romPressure"]:
            value = read_internal_field(directory / field, "vector", len(volumes))[:, :2]
            projected[field] = np.einsum("n,inc,nc->i", volumes, modes, value, optimize=True)
        state_values[state["label"]] = projected

    base = state_values["base"]
    c_conv = -base["romConvection"]
    c_diff = base["romDiffusion"]
    c_pressure = -base["romPressure"]
    a_conv = np.zeros((ru, ru))
    a_diff = np.zeros((ru, ru))
    h_conv = np.zeros((ru, ru, ru))
    pressure_coupling = np.zeros((ru, rp))
    for i in range(ru):
        minus = state_values[f"u_{i}_-1"]
        plus = state_values[f"u_{i}_+1"]
        a_conv[:, i] = -(plus["romConvection"] - minus["romConvection"]) / (2 * eps_u)
        a_diff[:, i] = (plus["romDiffusion"] - minus["romDiffusion"]) / (2 * eps_u)
        h_conv[:, i, i] = -(
            plus["romConvection"] + minus["romConvection"] - 2 * base["romConvection"]
        ) / (2 * eps_u**2)
    for i in range(ru):
        for j in range(i + 1, ru):
            mixed = np.zeros(ru)
            for sign_i in (-1, 1):
                for sign_j in (-1, 1):
                    coefficient = sign_i * sign_j
                    mixed += coefficient * state_values[
                        f"u_{i}_{sign_i:+d}_{j}_{sign_j:+d}"
                    ]["romConvection"]
            h_value = -mixed / (8 * eps_u**2)
            h_conv[:, i, j] = h_value
            h_conv[:, j, i] = h_value
    for i in range(rp):
        minus = state_values[f"p_{i}_-1"]["romPressure"]
        plus = state_values[f"p_{i}_+1"]["romPressure"]
        pressure_coupling[:, i] = -(plus - minus) / (2 * eps_p)

    diffusion_axis_nonlinearity = 0.0
    for i in range(ru):
        minus = state_values[f"u_{i}_-1"]["romDiffusion"]
        plus = state_values[f"u_{i}_+1"]["romDiffusion"]
        diffusion_axis_nonlinearity = max(
            diffusion_axis_nonlinearity,
            float(np.linalg.norm(plus + minus - 2 * base["romDiffusion"])),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        c_conv=c_conv, c_diff=c_diff, c_pressure=c_pressure,
        A_conv=a_conv, A_diff=a_diff, H_conv=h_conv,
        P=pressure_coupling, velocity_rank=np.asarray(ru), pressure_rank=np.asarray(rp),
        velocity_epsilon=np.asarray(eps_u), pressure_epsilon=np.asarray(eps_p),
    )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "velocity_rank": ru,
        "pressure_rank": rp,
        "probe_states": len(manifest["states"]),
        "mass_gram_max_abs": float(np.max(np.abs(np.einsum("n,inc,jnc->ij", volumes, modes, modes) - np.eye(ru)))),
        "diffusion_axis_nonlinearity_max_norm": diffusion_axis_nonlinearity,
        "tensor_norms": {
            "c_conv": float(np.linalg.norm(c_conv)), "c_diff": float(np.linalg.norm(c_diff)),
            "c_pressure": float(np.linalg.norm(c_pressure)), "A_conv": float(np.linalg.norm(a_conv)),
            "A_diff": float(np.linalg.norm(a_diff)), "H_conv": float(np.linalg.norm(h_conv)),
            "P": float(np.linalg.norm(pressure_coupling)),
        },
        "sha256": {"tensor": sha256(args.output), "probe_manifest": sha256(manifest_path)},
        "validation_loaded": False,
        "heldout_loaded": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
