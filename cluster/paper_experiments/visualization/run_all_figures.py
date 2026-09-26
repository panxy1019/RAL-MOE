#!/usr/bin/env python3
"""Single-command reproduction of the paper field figures.

Usage:
    python3 run_all_figures.py --output-dir ./figures [--error-cmap magma]

Generates in <output-dir>/:
    Steady_specialist_best_case_physical_fields.{png,pdf}
    Steady_specialist_best_case_pointwise_errors.{png,pdf}
    Hopf_specialist_best_case_physical_fields.{png,pdf}
    Hopf_specialist_best_case_pointwise_errors.{png,pdf}
    Periodic_specialist_best_case_physical_fields.{png,pdf}
    Periodic_specialist_best_case_pointwise_errors.{png,pdf}
    SH_boundary_best_case_physical_fields.{png,pdf}
    SH_boundary_best_case_pointwise_errors.{png,pdf}

Plus manifests and a candidate scan per case.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

CONFIGS = {
    "steady": THIS_DIR / "config_steady_specialist.json",
    "hopf": THIS_DIR / "config_hopf_specialist.json",
    "periodic": THIS_DIR / "config_periodic_specialist.json",
    "sh_fusion": THIS_DIR / "config_sh_fusion.json",
}

VTK_TEMPLATE = (
    "/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/"
    "steady_specialist_v1/expanded_validation_20260723/visualization/"
    "Re_43p500000_last_snapshot_uvp.vtk"
)


def run_case(
    output_dir: Path,
    case_name: str,
    config_path: Path,
    mode: str,
    error_cmap: str,
    save_vtk: bool = False,
    png_only: bool = False,
) -> int:
    sub_dir = output_dir / case_name
    sub_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(THIS_DIR / "plot_field_figures.py"),
        "--mode", mode,
        "--config", str(config_path),
        "--template-vtk", VTK_TEMPLATE,
        "--output-dir", str(sub_dir),
        "--error-cmap", error_cmap,
        "--dpi", "400",
    ]
    if save_vtk:
        cmd.append("--save-vtk")
    if png_only:
        cmd.append("--png-only")
    print(f"[run] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FAIL] {mode}: {result.stderr.strip()}")
    else:
        try:
            info = json.loads(result.stdout.strip())
            print(f"[OK] {mode}: {info['selected']['Re']:.6f} step {info['selected']['time_index']}")
        except Exception:
            print(f"[OK] {mode}")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate all paper field figures")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for output figures")
    parser.add_argument("--error-cmap", type=str, default="magma",
                        choices=("magma", "Reds", "thermal_r", "haline_r"),
                        help="Colormap for error panels")
    parser.add_argument("--save-vtk", action="store_true",
                        help="Also save VTK files for selected frames")
    parser.add_argument("--png-only", action="store_true",
                        help="Write PNG figures only; omit PDF outputs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ret = 0
    for name, config_path in CONFIGS.items():
        mode = "sh_fusion" if name == "sh_fusion" else "specialist"
        rc = run_case(
            args.output_dir,
            name,
            config_path,
            mode,
            args.error_cmap,
            args.save_vtk,
            args.png_only,
        )
        if rc != 0:
            ret = 1

    # Summary
    print(f"\n{'='*60}")
    if ret == 0:
        print(f"All 8 figures written to {args.output_dir.resolve()}")
    else:
        print(f"Some cases failed (see above). Output in {args.output_dir.resolve()}")
    print(f"{'='*60}")
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
