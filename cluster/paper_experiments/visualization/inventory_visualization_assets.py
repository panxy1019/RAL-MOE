#!/usr/bin/env python3
"""Read-only inventory helper for frozen visualization assets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    try:
        import matplotlib

        print(f"VERSION\tmatplotlib\t{matplotlib.__version__}")
    except Exception as error:
        print(f"IMPORT_ERROR\tmatplotlib\t{type(error).__name__}\t{error}")
    try:
        import vtk

        print(f"VERSION\tvtk\t{vtk.vtkVersion.GetVTKVersion()}")
    except Exception as error:
        print(f"IMPORT_ERROR\tvtk\t{type(error).__name__}\t{error}")
    try:
        import pyvista

        print(f"VERSION\tpyvista\t{pyvista.__version__}")
        template = (
            root
            / "steady_specialist_v1/expanded_validation_20260723/visualization/"
            "Re_43p500000_last_snapshot_uvp.vtk"
        )
        mesh = pyvista.read(template)
        print(
            "TEMPLATE\t"
            f"points={mesh.n_points}\tcells={mesh.n_cells}\t"
            f"bounds={tuple(float(value) for value in mesh.bounds)}\t"
            f"point_arrays={list(mesh.point_data.keys())}"
        )
    except Exception as error:
        print(f"IMPORT_ERROR\tpyvista\t{type(error).__name__}\t{error}")
    relative_paths = [
        "paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/steady/"
        "vanilla-fnn/frozen_user_stop_step6200_20260724/evaluation/test/heldout_bank.npz",
        "paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/periodic/"
        "data-only/training/VanillaFNN_periodic_seed1600_pressure_anchor_stats.npz",
        "paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/"
        "periodic_v4/periodic/vanilla-fnn/training/"
        "VanillaFNN_periodic_seed1600_pressure_anchor_stats.npz",
        "top2_boundary_experiments_20260722/one_sided_recovery_v2/"
        "resplit_20260723_v1/cache_final_test/G_SH_resplit_final_test_cache.npz",
    ]
    for relative in relative_paths:
        path = root / relative
        print(f"PATH\t{path}\texists={path.exists()}")
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=True) as archive:
                for key in archive.files:
                    value = archive[key]
                    print(f"  {key}\tshape={value.shape}\tdtype={value.dtype}")
        except Exception as error:  # inventory must continue across corrupt assets
            print(f"  ERROR\t{type(error).__name__}\t{error}")


if __name__ == "__main__":
    main()
