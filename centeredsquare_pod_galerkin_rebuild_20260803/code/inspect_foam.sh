#!/usr/bin/env bash
set -o pipefail
source /opt/openfoam13/etc/bashrc >/dev/null 2>&1 || true
case_root=/home/ray/Desktop/centeredSquare/Re100
find "$case_root/100" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
cd "$case_root"
postProcess -func 'grad(U)' -time 100
postProcess -func 'div(phi,U)' -time 100
postProcess -func 'laplacian(U)' -time 100
postProcess -func 'grad(p)' -time 100
find "$case_root/100" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
