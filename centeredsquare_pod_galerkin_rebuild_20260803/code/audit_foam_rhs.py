#!/usr/bin/env python3
"""Compare OpenFOAM-discrete spatial RHS, solver ddt, and POD derivatives."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/ray/Desktop/centeredSquare/scripts")
from dataset_tools import read_internal_field  # noqa: E402


CASE = Path(
    "/home/ray/Desktop/centeredSquare/runs/cn09_graded_20260708_000142/Re100"
)
TIME = CASE / "100"
NPZ = Path(
    "/home/ray/Desktop/centeredSquare/dataset_Re50_150_N100_npz/"
    "cases_npz/snapshots_Re100p000000.npz"
)
DATASET = Path("/home/ray/Desktop/centeredSquare/cdm_grom_hopf_r64_v1")


def relative(value: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> float:
    numerator = np.sum(weights[:, None] * (value - reference) ** 2)
    denominator = np.sum(weights[:, None] * reference**2)
    return float(np.sqrt(numerator / denominator))


def rms(value: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights[:, None] * value**2) / np.sum(weights)))


def main() -> None:
    with np.load(DATASET / "mesh/mesh_metadata.npz") as mesh:
        volumes = np.asarray(mesh["cellVolumes"], dtype=np.float64)
    fields = {
        name: read_internal_field(TIME / name, "vector", expected_count=len(volumes))[:, :2]
        for name in ["U", "romDdt", "romConvection", "romDiffusion", "romPressure", "romRhs"]
    }
    with np.load(NPZ, allow_pickle=False) as source:
        times = np.asarray(source["times"], dtype=np.float64)
        velocity = np.asarray(source["U"], dtype=np.float64)
    index = int(np.flatnonzero(np.isclose(times, 100.0))[0])
    coarse_ddt = (velocity[index + 1] - velocity[index - 1]) / (
        times[index + 1] - times[index - 1]
    )
    with np.load(DATASET / "pod/weighted_pod_velocity.npz") as pod:
        modes = np.asarray(pod["modes"], dtype=np.float64).reshape(-1, len(volumes), 2)

    def project(field: np.ndarray) -> np.ndarray:
        return np.einsum("n,inc,nc->i", volumes, modes, field, optimize=True)

    ddt_coeff = project(fields["romDdt"])
    rhs_coeff = project(fields["romRhs"])
    coarse_coeff = project(coarse_ddt)
    term_coefficients = {
        name: project(fields[name])
        for name in ["romConvection", "romDiffusion", "romPressure"]
    }
    payload = {
        "npz_openfoam_U_max_abs": float(np.max(np.abs(velocity[index] - fields["U"]))),
        "field_weighted_rms": {name: rms(value, volumes) for name, value in fields.items() if name != "U"},
        "field_relative_error": {
            "spatial_rhs_vs_solver_ddt": relative(fields["romRhs"], fields["romDdt"], volumes),
            "coarse_centered_ddt_vs_solver_ddt": relative(coarse_ddt, fields["romDdt"], volumes),
        },
        "pod64_relative_error": {
            "spatial_rhs_vs_solver_ddt": float(
                np.linalg.norm(rhs_coeff - ddt_coeff) / np.linalg.norm(ddt_coeff)
            ),
            "coarse_centered_ddt_vs_solver_ddt": float(
                np.linalg.norm(coarse_coeff - ddt_coeff) / np.linalg.norm(ddt_coeff)
            ),
        },
        "pod11_relative_error": {
            "spatial_rhs_vs_solver_ddt": float(
                np.linalg.norm(rhs_coeff[:11] - ddt_coeff[:11]) / np.linalg.norm(ddt_coeff[:11])
            ),
            "coarse_centered_ddt_vs_solver_ddt": float(
                np.linalg.norm(coarse_coeff[:11] - ddt_coeff[:11]) / np.linalg.norm(ddt_coeff[:11])
            ),
        },
        "pod64_norm": {
            "solver_ddt": float(np.linalg.norm(ddt_coeff)),
            "spatial_rhs": float(np.linalg.norm(rhs_coeff)),
            "coarse_ddt": float(np.linalg.norm(coarse_coeff)),
        },
        "projected_term_norm": {
            rank: {
                name: float(np.linalg.norm(value[:count]))
                for name, value in term_coefficients.items()
            }
            for rank, count in {"r11": 11, "r64": 64}.items()
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
