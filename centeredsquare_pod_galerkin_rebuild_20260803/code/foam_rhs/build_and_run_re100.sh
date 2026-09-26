#!/usr/bin/env bash
set -eo pipefail
source /opt/openfoam13/etc/bashrc >/dev/null 2>&1 || true
cd /home/ray/Desktop/centeredSquare/pod_galerkin_rebuild_code/foam_rhs
wmake
cd /home/ray/Desktop/centeredSquare/runs/cn09_graded_20260708_000142/Re100
romSpatialRhs -nu 0.01 -time '100'
