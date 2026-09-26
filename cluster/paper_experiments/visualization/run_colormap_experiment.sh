#!/bin/bash
# run_colormap_experiment.sh
# Runs all 8 figures with 3 different error colormaps, each in its own directory.
# Usage: bash run_colormap_experiment.sh [--output-base /path/to/output]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE="${1:-$SCRIPT_DIR/results/cmap_experiment}"

CMAPS=("haline_r" "thermal_r" "Reds")
LABELS=("batlowW_style_haline" "lajolla_style_thermal" "classic_Reds")

echo "============================================================"
echo "Colormap experiment: 3 cmaps x 4 cases = 12 runs"
echo "Output base: $OUTPUT_BASE"
echo "============================================================"

for i in "${!CMAPS[@]}"; do
    cmap="${CMAPS[$i]}"
    label="${LABELS[$i]}"
    out_dir="$OUTPUT_BASE/$label"
    echo ""
    echo ">>> CMAP $((i+1))/3: $cmap -> $out_dir"
    python3 "$SCRIPT_DIR/run_all_figures.py" \
        --output-dir "$out_dir" \
        --error-cmap "$cmap"
    echo "<<< $label DONE"
done

echo ""
echo "============================================================"
echo "All done. Results:"
find "$OUTPUT_BASE" -name "*.png" | sort
echo "============================================================"
