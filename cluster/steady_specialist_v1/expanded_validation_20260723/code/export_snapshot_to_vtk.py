#!/usr/bin/env python3
"""Export one u/v/p snapshot from the raw NPZ archive to a legacy VTK file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--topology-vtk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=int, default=-1)
    parser.add_argument("--re", type=float, required=True)
    return parser.parse_args()


def add_point_array(grid: vtk.vtkDataSet, values: np.ndarray, name: str) -> None:
    array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
    array.SetName(name)
    grid.GetPointData().AddArray(array)


def add_field_scalar(grid: vtk.vtkDataSet, value: float, name: str) -> None:
    array = numpy_to_vtk(np.asarray([value], dtype=np.float64), deep=True)
    array.SetName(name)
    grid.GetFieldData().AddArray(array)


def main() -> None:
    args = parse_args()
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(str(args.topology_vtk))
    reader.Update()
    source = reader.GetOutput()
    if source is None or source.GetNumberOfPoints() == 0:
        raise RuntimeError(f"failed to read topology: {args.topology_vtk}")

    grid = vtk.vtkUnstructuredGrid()
    grid.DeepCopy(source)
    grid.GetPointData().Initialize()

    with np.load(args.raw, allow_pickle=False) as archive:
        points = np.asarray(archive["points"], dtype=np.float32)
        times = np.asarray(archive["times"], dtype=np.float64)
        index = args.snapshot if args.snapshot >= 0 else times.size + args.snapshot
        if not 0 <= index < times.size:
            raise IndexError(f"snapshot {args.snapshot} resolves to {index}, count={times.size}")
        u = np.asarray(archive["u"][index], dtype=np.float32)
        v = np.asarray(archive["v"][index], dtype=np.float32)
        p = np.asarray(archive["p"][index], dtype=np.float32)

    topology_points = vtk_to_numpy(grid.GetPoints().GetData())
    if topology_points.shape != points.shape:
        raise ValueError(f"point-count mismatch: {topology_points.shape} vs {points.shape}")
    max_point_delta = float(np.max(np.abs(topology_points - points)))
    if max_point_delta > 5e-6:
        raise ValueError(f"mesh point ordering mismatch, max delta={max_point_delta}")

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))
    grid.SetPoints(vtk_points)
    velocity = np.column_stack((u, v, np.zeros_like(u)))
    add_point_array(grid, velocity, "U")
    add_point_array(grid, u, "u")
    add_point_array(grid, v, "v")
    add_point_array(grid, p, "p")
    add_field_scalar(grid, float(args.re), "Re")
    add_field_scalar(grid, float(times[index]), "time")
    add_field_scalar(grid, float(index), "snapshot_index")
    grid.GetPointData().SetActiveVectors("U")
    grid.GetPointData().SetActiveScalars("p")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(str(args.output))
    writer.SetInputData(grid)
    writer.SetFileTypeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"failed to write {args.output}")
    print(
        f"WROTE {args.output} points={grid.GetNumberOfPoints()} "
        f"cells={grid.GetNumberOfCells()} time={times[index]:.12g} Re={args.re}"
    )


if __name__ == "__main__":
    main()
