#!/usr/bin/env python3
"""Assemble the explicit 34-Re Hopf source view without copying legacy fields."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-source", type=Path, required=True)
    parser.add_argument("--new-source", type=Path, required=True)
    parser.add_argument("--output-source", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    return parser.parse_args()


def replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Refusing to replace existing path: {link}")
    link.symlink_to(target.resolve())


def main() -> None:
    args = parse_args()
    config = json.loads(args.split_config.read_text(encoding="utf-8"))
    labels = set(config["include_labels"]["hopf"])
    if len(labels) != 34:
        raise ValueError(f"Expected 34 unique Hopf labels, received {len(labels)}")

    legacy_index = args.legacy_source / "Global_POD_AreaWeighted_L2" / "pod_snapshot_index.csv"
    with legacy_index.open(newline="", encoding="utf-8-sig") as handle:
        legacy_rows = list(csv.DictReader(handle))
    legacy_labels = {row["Re_label"] for row in legacy_rows}
    new_labels = labels - legacy_labels
    old_labels = labels & legacy_labels
    if len(old_labels) != 17 or len(new_labels) != 17:
        raise ValueError(f"Expected 17 legacy + 17 new labels, found {len(old_labels)} + {len(new_labels)}")

    args.output_source.mkdir(parents=True, exist_ok=True)
    global_dir = args.output_source / "Global_POD_AreaWeighted_L2"
    global_dir.mkdir(exist_ok=True)
    legacy_global = args.legacy_source / "Global_POD_AreaWeighted_L2"
    for name in (
        "global_velocity_pod_area_weighted_l2.npz",
        "global_pressure_pod_area_weighted_l2.npz",
        "mesh_l2_point_area_weights.npz",
    ):
        replace_symlink(global_dir / name, legacy_global / name)

    with np.load(legacy_global / "mesh_l2_point_area_weights.npz", allow_pickle=False) as archive:
        reference_points = np.asarray(archive["points"], dtype=np.float64)

    rows: list[dict[str, object]] = [row for row in legacy_rows if row["Re_label"] in old_labels]
    for label in sorted(new_labels):
        npz_path = args.new_source / f"{label}_uvp_pointData.npz"
        metadata_path = args.new_source / f"{label}_uvp_metadata.json"
        if not npz_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing new simulation outputs for {label}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as archive:
            times = np.asarray(archive["times"], dtype=np.float64)
            points = np.asarray(archive["points"], dtype=np.float64)
            shapes = {name: archive[name].shape for name in ("u", "v", "p")}
            finite = all(np.isfinite(archive[name]).all() for name in ("times", "u", "v", "p"))
        if points.shape != reference_points.shape or not np.allclose(points, reference_points, rtol=0.0, atol=1e-10):
            raise ValueError(f"{label}: mesh points differ from the frozen database")
        if not finite or any(shape != (times.size, points.shape[0]) for shape in shapes.values()):
            raise ValueError(f"{label}: invalid or non-finite field arrays: {shapes}")
        re_value = float(metadata["Re"])
        regime = str(metadata["regime"])
        if label != f"Re_{re_value:.6f}".replace(".", "p"):
            raise ValueError(f"{label}: metadata Reynolds number does not reproduce label")
        if re_value < 47.0 and times.size not in (63, 64):
            raise ValueError(f"{label}: pre-onset case has {times.size} snapshots, expected 63/64")
        if re_value >= 47.0 and times.size < 161:
            raise ValueError(f"{label}: periodic case has only {times.size} snapshots")
        estimated_period = metadata.get("diagnostics", {}).get("estimated_period", "")
        for local_index, time_value in enumerate(times):
            rows.append(
                {
                    "Re": f"{re_value:.12g}",
                    "Re_label": label,
                    "regime": regime,
                    "time": f"{float(time_value):.12g}",
                    "estimated_period": "" if estimated_period in (None, "") else f"{float(estimated_period):.12g}",
                    "local_snapshot_index": local_index,
                }
            )
        replace_symlink(args.output_source / npz_path.name, npz_path)
        replace_symlink(args.output_source / metadata_path.name, metadata_path)

    for label in old_labels:
        for suffix in ("_uvp_pointData.npz", "_uvp_metadata.json"):
            replace_symlink(args.output_source / f"{label}{suffix}", args.legacy_source / f"{label}{suffix}")

    rows.sort(key=lambda row: (float(row["Re"]), int(row["local_snapshot_index"])))
    fieldnames = ["snapshot_id", "Re", "Re_label", "regime", "time", "estimated_period", "local_snapshot_index"]
    with (global_dir / "pod_snapshot_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for snapshot_id, row in enumerate(rows):
            writer.writerow({"snapshot_id": snapshot_id, **row})

    counts = {split: 0 for split in ("train", "validation", "heldout")}
    validation = set(config["validation_labels"]["hopf"])
    heldout = set(config["heldout_labels"]["hopf"])
    for label in labels:
        counts["heldout" if label in heldout else "validation" if label in validation else "train"] += 1
    manifest = {
        "status": "PASS",
        "legacy_cases": len(old_labels),
        "new_cases": len(new_labels),
        "total_cases": len(labels),
        "total_snapshots": len(rows),
        "split_case_counts": counts,
        "new_labels": sorted(new_labels),
        "legacy_labels": sorted(old_labels),
    }
    (args.output_source / "SOURCE_ASSEMBLY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
