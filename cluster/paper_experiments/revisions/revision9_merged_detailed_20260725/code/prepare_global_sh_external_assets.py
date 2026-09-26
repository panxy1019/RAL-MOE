#!/usr/bin/env python3
"""Create read-only-derived Global MoE evaluation assets for exact S-H Re nodes.

The frozen Global POD basis/checkpoint are not modified. The two external
S-native trajectories are projected into the frozen Global coordinates, and
the nearest already-frozen Galerkin/pressure operators are copied into a new
evaluation-only asset bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


EXTERNAL = (
    (43.5, "Re_43p500000", "Re_43p797402"),
    (43.9, "Re_43p900000", "Re_43p797402"),
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-data-root", type=Path, required=True)
    parser.add_argument("--galerkin", type=Path, required=True)
    parser.add_argument("--pressure", type=Path, required=True)
    parser.add_argument("--external-raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    velocity_path = args.global_data_root / "global_velocity_pod_area_weighted_l2.npz"
    pressure_path = args.global_data_root / "global_pressure_pod_area_weighted_l2.npz"
    index_path = args.global_data_root / "pod_snapshot_index.csv"
    with np.load(velocity_path, allow_pickle=False) as archive:
        velocity = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(pressure_path, allow_pickle=False) as archive:
        pressure = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.galerkin, allow_pickle=False) as archive:
        galerkin = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.pressure, allow_pickle=False) as archive:
        pressure_ops = {key: np.asarray(archive[key]) for key in archive.files}

    points = velocity["points"].astype(np.float32)
    weights = velocity["point_areas"].astype(np.float64)
    sqrt_w = np.sqrt(weights)
    if not np.array_equal(points, pressure["points"]):
        raise RuntimeError("frozen Global velocity/pressure mesh mismatch")
    if not np.allclose(sqrt_w, velocity["sqrt_point_areas"], rtol=2e-6):
        raise RuntimeError("frozen Global point weights are inconsistent")

    uv_coeff, p_coeff, uv_means, p_means = [], [], [], []
    labels, re_values, regimes = [], [], []
    index_rows = list(csv.DictReader(index_path.open(newline="", encoding="utf-8")))
    next_snapshot = max(int(row["snapshot_id"]) for row in index_rows) + 1
    diagnostics = []
    uv_modes_w = velocity["phi_uv_weighted"].astype(np.float64)
    p_modes_w = pressure["phi_p_weighted"].astype(np.float64)
    velocity_sqrt = np.concatenate((sqrt_w, sqrt_w))

    for re_value, label, operator_label in EXTERNAL:
        raw_path = args.external_raw_root / f"{label}_uvp_pointData.npz"
        with np.load(raw_path, allow_pickle=False) as archive:
            raw_points = np.asarray(archive["points"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float64)
            u = np.asarray(archive["u"], dtype=np.float64)
            v = np.asarray(archive["v"], dtype=np.float64)
            p = np.asarray(archive["p"], dtype=np.float64)
        if not np.allclose(raw_points, points, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"{label}: mesh mismatch")
        if len(times) != 64 or not np.all(np.diff(times) > 0):
            raise RuntimeError(f"{label}: invalid time trajectory")
        if not all(np.isfinite(x).all() for x in (times, u, v, p)):
            raise RuntimeError(f"{label}: non-finite raw fields")

        uv = np.concatenate((u, v), axis=1)
        p_gauged = p - (p @ weights / weights.sum())[:, None]
        uv_mean = uv.mean(axis=0)
        p_mean = p_gauged.mean(axis=0)
        uv_centered_w = (uv - uv_mean) * velocity_sqrt
        p_centered_w = (p_gauged - p_mean) * sqrt_w
        a = uv_centered_w @ uv_modes_w.T
        b = p_centered_w @ p_modes_w.T
        uv_reconstructed_w = a[:, :32] @ uv_modes_w[:32]
        p_reconstructed_w = b[:, :32] @ p_modes_w[:32]
        uv_floor = np.linalg.norm(uv_centered_w - uv_reconstructed_w) / max(
            np.linalg.norm(uv * velocity_sqrt), 1e-30
        )
        p_floor = np.linalg.norm(p_centered_w - p_reconstructed_w) / max(
            np.linalg.norm(p_gauged * sqrt_w), 1e-30
        )

        uv_coeff.append(a.astype(np.float32))
        p_coeff.append(b.astype(np.float32))
        uv_means.append(uv_mean.astype(np.float32))
        p_means.append(p_mean.astype(np.float32))
        labels.append(label)
        re_values.append(re_value)
        regimes.append("pre_hopf_steady")
        for local, time_value in enumerate(times.tolist()):
            index_rows.append(
                {
                    "snapshot_id": str(next_snapshot),
                    "Re": f"{re_value:.10f}",
                    "Re_label": label,
                    "regime": "pre_hopf_steady",
                    "time": f"{time_value:.12g}",
                    "estimated_period": "",
                    "local_snapshot_index": str(local),
                }
            )
            next_snapshot += 1
        diagnostics.append(
            {
                "Re": re_value,
                "label": label,
                "raw_sha256": digest(raw_path),
                "snapshots": len(times),
                "time_start": float(times[0]),
                "time_end": float(times[-1]),
                "pressure_gauge": "subtract_area_mean_per_snapshot",
                "global_r32_velocity_projection_floor": float(uv_floor),
                "global_r32_pressure_projection_floor": float(p_floor),
                "operator_source_label": operator_label,
            }
        )

        gal_labels = galerkin["Re_labels_computed"].astype(str).tolist()
        source_row = gal_labels.index(operator_label)
        galerkin["c_all"] = np.concatenate(
            (galerkin["c_all"], galerkin["c_all"][source_row : source_row + 1]), axis=0
        )
        galerkin["A_all"] = np.concatenate(
            (galerkin["A_all"], galerkin["A_all"][source_row : source_row + 1]), axis=0
        )
        galerkin["Re_labels_computed"] = np.concatenate(
            (galerkin["Re_labels_computed"].astype(str), np.asarray([label]))
        )
        if "Re_values_computed" in galerkin:
            galerkin["Re_values_computed"] = np.concatenate(
                (galerkin["Re_values_computed"], np.asarray([re_value], dtype=np.float64))
            )
        pressure_ops[f"{label}_c_tilde"] = pressure_ops[f"{operator_label}_c_tilde"].copy()
        pressure_ops[f"{label}_A_tilde"] = pressure_ops[f"{operator_label}_A_tilde"].copy()

    velocity["coeff_uv"] = np.concatenate((velocity["coeff_uv"], *uv_coeff), axis=0)
    velocity["mean_uv_by_Re"] = np.concatenate(
        (velocity["mean_uv_by_Re"], np.stack(uv_means)), axis=0
    )
    velocity["Re_values"] = np.concatenate(
        (velocity["Re_values"], np.asarray(re_values, dtype=np.float64))
    )
    velocity["Re_labels"] = np.concatenate(
        (velocity["Re_labels"].astype(str), np.asarray(labels))
    )
    velocity["regimes"] = np.concatenate(
        (velocity["regimes"].astype(str), np.asarray(regimes))
    )
    pressure["coeff_p"] = np.concatenate((pressure["coeff_p"], *p_coeff), axis=0)
    pressure["mean_p_by_Re"] = np.concatenate(
        (pressure["mean_p_by_Re"], np.stack(p_means)), axis=0
    )
    pressure["Re_values"] = velocity["Re_values"].copy()
    pressure["Re_labels"] = velocity["Re_labels"].copy()
    pressure["regimes"] = velocity["regimes"].copy()

    save_npz(args.output_dir / velocity_path.name, velocity)
    save_npz(args.output_dir / pressure_path.name, pressure)
    save_npz(args.output_dir / args.galerkin.name, galerkin)
    save_npz(args.output_dir / args.pressure.name, pressure_ops)
    fieldnames = [
        "snapshot_id", "Re", "Re_label", "regime", "time",
        "estimated_period", "local_snapshot_index",
    ]
    with (args.output_dir / "pod_snapshot_index.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    manifest = {
        "schema": "global_external_sh_eval_assets/v1",
        "status": "PASS",
        "checkpoint_modified": False,
        "global_pod_basis_modified": False,
        "global_scaler_modified": False,
        "representation": "transductive per-Re mean, matching frozen Global baseline",
        "external_nodes": diagnostics,
        "sources": {
            "velocity_pod": {"path": str(velocity_path), "sha256": digest(velocity_path)},
            "pressure_pod": {"path": str(pressure_path), "sha256": digest(pressure_path)},
            "snapshot_index": {"path": str(index_path), "sha256": digest(index_path)},
            "galerkin": {"path": str(args.galerkin), "sha256": digest(args.galerkin)},
            "pressure": {"path": str(args.pressure), "sha256": digest(args.pressure)},
        },
    }
    for name in (
        velocity_path.name,
        pressure_path.name,
        args.galerkin.name,
        args.pressure.name,
        "pod_snapshot_index.csv",
    ):
        path = args.output_dir / name
        manifest.setdefault("derived", {})[name] = {
            "path": str(path),
            "sha256": digest(path),
        }
    (args.output_dir / "PREPARE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
