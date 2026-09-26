#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from common import ROOT, load_config, validate_npz


def parse_time_from_path(path: Path) -> float:
    nums = re.findall(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", path.stem)
    if not nums:
        raise ValueError(f"Cannot parse time from VTK filename: {path}")
    return float(nums[-1])


def find_vtk_files(vtk_dir: Path, retain_start: float | None = None) -> list[tuple[float, Path]]:
    files: list[tuple[float, Path]] = []
    candidates = sorted(vtk_dir.glob("*.vtk"))
    if not candidates:
        candidates = [
            path
            for path in sorted(vtk_dir.rglob("*.vtk"))
            if path.parent == vtk_dir or path.parent.name.lower() in {"internal", "volume"}
        ]
    for path in candidates:
        try:
            t = parse_time_from_path(path)
        except ValueError:
            continue
        if retain_start is not None and t + 1e-10 < retain_start:
            continue
        files.append((t, path))
    files.sort(key=lambda item: item[0])
    return files


def _load_with_pyvista(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("pyvista not available") from exc

    mesh = pv.read(path)
    if "U" in mesh.point_data and "p" in mesh.point_data:
        coords = np.asarray(mesh.points)
        u = np.asarray(mesh.point_data["U"])
        p = np.asarray(mesh.point_data["p"])
        source = "point_data"
    elif "U" in mesh.cell_data and "p" in mesh.cell_data:
        centers = mesh.cell_centers()
        coords = np.asarray(centers.points)
        u = np.asarray(mesh.cell_data["U"])
        p = np.asarray(mesh.cell_data["p"])
        source = "cell_data"
    else:
        raise RuntimeError(
            f"{path} does not contain U and p in point_data or cell_data. "
            f"point_data={list(mesh.point_data.keys())}, cell_data={list(mesh.cell_data.keys())}"
        )
    return coords, u, p, source


def _load_with_meshio(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    try:
        import meshio
    except ImportError as exc:
        raise RuntimeError("meshio not available") from exc

    mesh = meshio.read(path)
    if "U" in mesh.point_data and "p" in mesh.point_data:
        return (
            np.asarray(mesh.points),
            np.asarray(mesh.point_data["U"]),
            np.asarray(mesh.point_data["p"]),
            "point_data",
        )
    if "U" in mesh.cell_data_dict and "p" in mesh.cell_data_dict:
        raise RuntimeError("meshio found cell data, but cell-center extraction is not implemented")
    raise RuntimeError(f"{path} does not contain U and p")


def load_vtk(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    errors = []
    for loader in (_load_with_pyvista, _load_with_meshio):
        try:
            return loader(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{loader.__name__}: {exc}")
    raise RuntimeError(
        "Could not read VTK. Install at least one reader with:\n"
        "  python3 -m pip install pyvista meshio\n"
        + "\n".join(errors)
    )


def collapse_to_xy(
    coords3: np.ndarray, u_raw: np.ndarray, p_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords3 = np.asarray(coords3, dtype=float)
    xy = coords3[:, :2]
    u = np.asarray(u_raw, dtype=float)
    p = np.asarray(p_raw, dtype=float).reshape(-1)
    if u.ndim == 1:
        raise ValueError("U array is scalar; expected vector data")
    u2 = u[:, :2]

    rounded = np.round(xy, decimals=12)
    unique_xy, inverse = np.unique(rounded, axis=0, return_inverse=True)
    if unique_xy.shape[0] == xy.shape[0]:
        return xy, u2, p

    counts = np.bincount(inverse).astype(float)
    ux = np.bincount(inverse, weights=u2[:, 0]) / counts
    uy = np.bincount(inverse, weights=u2[:, 1]) / counts
    pp = np.bincount(inverse, weights=p) / counts
    return unique_xy, np.column_stack([ux, uy]), pp


def metadata_from_args(args: argparse.Namespace, cfg: dict[str, Any], source: str) -> dict[str, Any]:
    mode_cfg = cfg.get("modes", {}).get(args.mode, {})
    return {
        "case_name": args.case_dir.name,
        "Re": args.re,
        "nu": args.nu,
        "nProcs": args.nprocs,
        "OpenFOAM solver sequence": [
            "blockMesh",
            "checkMesh",
            "decomposePar",
            "simpleFoam -parallel",
            "pimpleFoam spin-up -parallel",
            "pimpleFoam retained -parallel",
            "reconstructPar",
            "foamToVTK",
        ],
        "mesh parameters": cfg.get("mesh", {}),
        "geometry": cfg.get("geometry", {}),
        "time settings": mode_cfg,
        "retained time range": [args.retain_start, args.retain_end],
        "vtk_source": source,
    }


def convert(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config)
    vtk_files = find_vtk_files(args.vtk_dir, args.retain_start)
    if not vtk_files:
        raise RuntimeError(f"No retained VTK files found under {args.vtk_dir}")

    coords_ref: np.ndarray | None = None
    u_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []
    times: list[float] = []
    source = ""

    for time_value, path in vtk_files:
        coords3, u_raw, p_raw, source = load_vtk(path)
        coords, u, p = collapse_to_xy(coords3, u_raw, p_raw)
        if coords_ref is None:
            coords_ref = coords
        elif coords.shape != coords_ref.shape or not np.allclose(coords, coords_ref, atol=1e-10):
            raise RuntimeError(f"Coordinate mismatch at {path}")
        if not np.isfinite(u).all() or not np.isfinite(p).all():
            raise RuntimeError(f"NaN/Inf in {path}")
        times.append(time_value)
        u_list.append(u.astype(np.float32))
        p_list.append(p.astype(np.float32))

    assert coords_ref is not None
    order = np.argsort(np.asarray(times))
    times_arr = np.asarray(times, dtype=np.float64)[order]
    u_arr = np.stack(u_list, axis=0)[order]
    p_arr = np.stack(p_list, axis=0)[order]
    metadata = metadata_from_args(args, cfg, source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        coords=coords_ref.astype(np.float64),
        times=times_arr,
        U=u_arr,
        p=p_arr,
        Re=np.asarray(args.re, dtype=np.float64),
        nu=np.asarray(args.nu, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, indent=2)),
    )

    ok, reason = validate_npz(args.output)
    if not ok:
        raise RuntimeError(f"Written npz failed validation: {reason}")

    if not args.keep_vtk:
        shutil.rmtree(args.vtk_dir, ignore_errors=True)
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OpenFOAM VTK output to one compressed npz file.")
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--vtk-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--re", type=float, required=True)
    parser.add_argument("--nu", type=float, required=True)
    parser.add_argument("--nprocs", type=int, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--retain-start", type=float, required=True)
    parser.add_argument("--retain-end", type=float, required=True)
    parser.add_argument("--config", default=str(ROOT / "config" / "sweep.yaml"))
    parser.add_argument("--keep-vtk", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.re) or args.re <= 0:
        raise SystemExit("--re must be positive")
    output = convert(args)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
