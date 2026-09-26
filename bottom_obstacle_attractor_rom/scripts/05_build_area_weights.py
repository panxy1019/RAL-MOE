#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from common import ROOT, load_config


def coord_key(x: float, y: float) -> tuple[float, float]:
    # foamToVTK writes legacy VTK coordinates with limited decimal precision.
    # Four decimals is still well below the minimum mesh spacing here, but
    # tolerant of legacy VTK values such as 10.022222... being written as
    # 10.0222.
    return (round(float(x), 4), round(float(y), 4))


def first_npz(npz_dir: Path) -> Path:
    files = sorted(npz_dir.glob("Re_*.npz"))
    if not files:
        raise RuntimeError(f"No Re_*.npz files found in {npz_dir}")
    return files[0]


def add_vertex_lumped_block(
    weights: dict[tuple[float, float], float],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    for i in range(len(x_edges) - 1):
        dx = float(x_edges[i + 1] - x_edges[i])
        for j in range(len(y_edges) - 1):
            dy = float(y_edges[j + 1] - y_edges[j])
            area = dx * dy
            corners = [
                coord_key(x_edges[i], y_edges[j]),
                coord_key(x_edges[i + 1], y_edges[j]),
                coord_key(x_edges[i + 1], y_edges[j + 1]),
                coord_key(x_edges[i], y_edges[j + 1]),
            ]
            for key in corners:
                weights[key] = weights.get(key, 0.0) + area / 4.0


def add_cell_center_block(
    weights: dict[tuple[float, float], float],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    for i in range(len(x_edges) - 1):
        dx = float(x_edges[i + 1] - x_edges[i])
        xc = 0.5 * float(x_edges[i + 1] + x_edges[i])
        for j in range(len(y_edges) - 1):
            dy = float(y_edges[j + 1] - y_edges[j])
            yc = 0.5 * float(y_edges[j + 1] + y_edges[j])
            weights[coord_key(xc, yc)] = dx * dy


def build_weights_from_config(cfg: dict, *, centers: bool = False) -> dict[tuple[float, float], float]:
    g = cfg["geometry"]
    m = cfg["mesh"]
    blocks = [
        (g["x_min"], g["x_obstacle_front"], g["y_min"], g["y_obstacle_top"], m["nx_upstream"], m["ny_lower"]),
        (g["x_min"], g["x_obstacle_front"], g["y_obstacle_top"], g["y_max"], m["nx_upstream"], m["ny_upper"]),
        (g["x_obstacle_front"], g["x_obstacle_back"], g["y_obstacle_top"], g["y_max"], m["nx_obstacle"], m["ny_upper"]),
        (g["x_obstacle_back"], g["x_max"], g["y_min"], g["y_obstacle_top"], m["nx_downstream"], m["ny_lower"]),
        (g["x_obstacle_back"], g["x_max"], g["y_obstacle_top"], g["y_max"], m["nx_downstream"], m["ny_upper"]),
    ]
    weights: dict[tuple[float, float], float] = {}
    for x0, x1, y0, y1, nx, ny in blocks:
        x_edges = np.linspace(float(x0), float(x1), int(nx) + 1)
        y_edges = np.linspace(float(y0), float(y1), int(ny) + 1)
        if centers:
            add_cell_center_block(weights, x_edges, y_edges)
        else:
            add_vertex_lumped_block(weights, x_edges, y_edges)
    return weights


def align_weights(coords: np.ndarray, weights: dict[tuple[float, float], float]) -> np.ndarray:
    area = np.zeros(coords.shape[0], dtype=np.float64)
    xs = np.asarray(sorted({key[0] for key in weights}), dtype=np.float64)
    ys = np.asarray(sorted({key[1] for key in weights}), dtype=np.float64)
    dx_min = np.min(np.diff(xs)[np.diff(xs) > 0])
    dy_min = np.min(np.diff(ys)[np.diff(ys) > 0])
    tol = 0.25 * min(float(dx_min), float(dy_min))

    def nearest(values: np.ndarray, value: float) -> tuple[float, float]:
        idx = int(np.searchsorted(values, value))
        candidates = []
        if idx < values.size:
            candidates.append(values[idx])
        if idx > 0:
            candidates.append(values[idx - 1])
        best = min(candidates, key=lambda item: abs(float(item) - float(value)))
        return float(best), abs(float(best) - float(value))

    missing = []
    for i, (x, y) in enumerate(coords):
        x_key, x_err = nearest(xs, float(x))
        y_key, y_err = nearest(ys, float(y))
        key = (x_key, y_key)
        value = weights.get(key)
        if value is None or x_err > tol or y_err > tol:
            missing.append(coord_key(float(x), float(y)))
        else:
            area[i] = value
    if missing:
        sample = ", ".join(map(str, missing[:5]))
        raise RuntimeError(f"{len(missing)} coordinates were not found in generated mesh weights. Sample: {sample}")
    return area


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lumped nodal area weights for area-weighted POD.")
    parser.add_argument("--config", default=str(ROOT / "config" / "sweep.yaml"))
    parser.add_argument("--coords-npz", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "npz" / "point_area.npz")
    args = parser.parse_args()

    cfg = load_config(args.config)
    coords_file = args.coords_npz or first_npz(ROOT / "data" / "npz")
    with np.load(coords_file, allow_pickle=True) as data:
        coords = np.asarray(data["coords"], dtype=np.float64)

    try:
        weight_map = build_weights_from_config(cfg, centers=False)
        area = align_weights(coords, weight_map)
        kind = "vertex_lumped"
    except RuntimeError as vertex_error:
        weight_map = build_weights_from_config(cfg, centers=True)
        try:
            area = align_weights(coords, weight_map)
            kind = "cell_center"
            print(f"WARNING: using cell-center areas after vertex mapping failed: {vertex_error}")
        except RuntimeError:
            raise vertex_error

    if not np.all(area > 0):
        raise RuntimeError("Area weights must be strictly positive")
    fluid_area = (
        (cfg["geometry"]["x_max"] - cfg["geometry"]["x_min"])
        * (cfg["geometry"]["y_max"] - cfg["geometry"]["y_min"])
        - (cfg["geometry"]["x_obstacle_back"] - cfg["geometry"]["x_obstacle_front"])
        * (cfg["geometry"]["y_obstacle_top"] - cfg["geometry"]["y_min"])
    )
    total = float(np.sum(area))
    rel_err = abs(total - fluid_area) / fluid_area
    print(f"area_kind={kind}")
    print(f"sum(area)={total:.12g}")
    print(f"expected fluid area={fluid_area:.12g}")
    print(f"relative error={rel_err:.3e}")
    if rel_err > 1e-8:
        raise RuntimeError("Area sum does not match the expected fluid-domain area")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, coords=coords, area=area, area_kind=np.asarray(kind))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
