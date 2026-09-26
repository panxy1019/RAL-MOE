#!/usr/bin/env python3
"""Minimal PyVista example for the retained Steady flow-field VTK."""

from pathlib import Path

import pyvista as pv


vtk_path = Path(__file__).resolve().parents[1] / "visualization" / "Re_43p500000_last_snapshot_uvp.vtk"
mesh = pv.read(vtk_path)
mesh.plot(scalars="p", cmap="coolwarm", show_edges=False)

