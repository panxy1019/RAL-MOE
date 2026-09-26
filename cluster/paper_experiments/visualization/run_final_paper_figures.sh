#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE
VIZ="$ROOT/paper_experiments/visualization"
PY="$ROOT/.runtime/pt_env/bin/python"
export PYTHONPATH="$VIZ/vendor"
export MPLBACKEND=Agg
TEMPLATE="$ROOT/steady_specialist_v1/expanded_validation_20260723/visualization/Re_43p500000_last_snapshot_uvp.vtk"

"$PY" "$VIZ/plot_field_figures.py" \
  --mode sh_fusion --config "$VIZ/config_sh_fusion.json" \
  --template-vtk "$TEMPLATE" --candidate-scan --split heldout \
  --output-dir "$VIZ/results/final_v2/sh_fusion" --save-vtk \
  --prefer-last-frame --allow-best-frame-search --dpi 400

"$PY" "$VIZ/plot_field_figures.py" \
  --mode specialist --config "$VIZ/config_steady_specialist.json" \
  --template-vtk "$TEMPLATE" --candidate-scan --split heldout \
  --output-dir "$VIZ/results/final_v2/steady" --save-vtk \
  --prefer-last-frame --allow-best-frame-search --dpi 400

"$PY" "$VIZ/plot_field_figures.py" \
  --mode specialist --config "$VIZ/config_hopf_specialist.json" \
  --template-vtk "$TEMPLATE" --candidate-scan --split train \
  --output-dir "$VIZ/results/final_v2/hopf" --save-vtk \
  --prefer-last-frame --allow-best-frame-search --dpi 400

"$PY" "$VIZ/plot_field_figures.py" \
  --mode specialist --config "$VIZ/config_periodic_specialist.json" \
  --template-vtk "$TEMPLATE" --candidate-scan --split heldout \
  --output-dir "$VIZ/results/final_v2/periodic" --save-vtk \
  --prefer-last-frame --allow-best-frame-search --dpi 400
