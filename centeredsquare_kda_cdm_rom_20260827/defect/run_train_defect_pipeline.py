#!/usr/bin/env python3
"""Build instantaneous OpenFOAM defects for train cases encoded in a frozen POD."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-pod", type=Path, required=True)
    parser.add_argument("--pressure-pod", type=Path)
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--source-case", type=Path, required=True)
    parser.add_argument("--template-time", default="100")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--fvm-tensors", type=Path, required=True)
    parser.add_argument("--pressure-closure", type=Path, required=True)
    parser.add_argument("--dataset-tools-dir", type=Path, required=True)
    parser.add_argument("--rom-spatial-rhs", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--expected-train-cases", type=int, default=27)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with np.load(args.velocity_pod, allow_pickle=False) as pod:
        tags = sorted(set(np.asarray(pod["snapshot_case_tags"]).astype(str).tolist()))
    if len(tags) != args.expected_train_cases:
        raise RuntimeError(f"expected {args.expected_train_cases} train cases, found {len(tags)}")
    case_paths = [args.cases_dir / f"snapshots_{tag}.npz" for tag in tags]
    missing = [str(path) for path in case_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing train snapshots:\n" + "\n".join(missing))
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.work_root.mkdir(parents=True, exist_ok=True)

    contract = {
        "schema_version": 1,
        "train_tags": tags,
        "train_case_count": len(tags),
        "velocity_pod_sha256": sha256(args.velocity_pod),
        "pressure_pod_sha256": sha256(args.pressure_pod) if args.pressure_pod else None,
        "fvm_tensors_sha256": sha256(args.fvm_tensors),
        "pressure_closure_sha256": sha256(args.pressure_closure),
        "validation_loaded": False,
        "heldout_loaded": False,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(contract, indent=2))
        return

    args.output_root.mkdir(parents=True)
    (args.output_root / "TRAIN_ONLY_CONTRACT.json").write_text(json.dumps(contract, indent=2) + "\n")
    script_root = Path(__file__).resolve().parent
    reports = []
    for tag, snapshot in zip(tags, case_paths):
        output = args.output_root / tag
        with tempfile.TemporaryDirectory(prefix=f"cdm_{tag}_", dir=args.work_root) as temporary:
            foam_case = Path(temporary) / "case"
            subprocess.run([
                sys.executable, str(script_root / "build_snapshot_case.py"),
                "--source-case", str(args.source_case), "--snapshot-npz", str(snapshot),
                "--template-time", args.template_time, "--output-case", str(foam_case),
            ], check=True)
            with np.load(snapshot, allow_pickle=False) as source:
                count = len(source["times"])
                viscosity = 1.0 / float(source["Re"])
            subprocess.run([
                args.rom_spatial_rhs, "-case", str(foam_case), "-nu", f"{viscosity:.17g}",
                "-time", f"1:{count}", "-noDdt",
            ], check=True)
            project_command = [
                sys.executable, str(script_root / "project_instantaneous_defect.py"),
                "--foam-case", str(foam_case), "--velocity-pod", str(args.velocity_pod),
                "--mesh", str(args.mesh),
                "--fvm-tensors", str(args.fvm_tensors), "--pressure-closure", str(args.pressure_closure),
                "--dataset-tools-dir", str(args.dataset_tools_dir), "--output-dir", str(output),
            ]
            if args.pressure_pod:
                project_command.extend(["--pressure-pod", str(args.pressure_pod)])
            subprocess.run(project_command, check=True)
        reports.append(json.loads((output / "Re.json").read_text()))

    summary = {
        **contract,
        "dry_run": False,
        "all_finite": bool(all(report["finite"] for report in reports)),
        "mean_instantaneous_defect_relative_l2": float(np.mean([
            report["instantaneous_defect_relative_l2"] for report in reports
        ])),
        "mean_flowmap_instantaneous_correlation": float(np.mean([
            report["flowmap_instantaneous_correlation"] for report in reports
        ])),
        "case_reports": reports,
    }
    (args.output_root / "TRAIN_DEFECT_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
