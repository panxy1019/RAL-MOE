#!/usr/bin/env python3
"""Build a train/validation-only Periodic modal bundle from per-Re raw files.

This utility deliberately never reads ``coeff_uv`` or ``coeff_p`` from the
legacy mixed POD archives and never opens a heldout raw trajectory.  It reuses
the frozen checkpoint POD basis, per-Re means, point ordering, area weights,
pressure gauge, and indexed-phase contract without fitting any asset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np


VALIDATION_LABELS = {
    "Re_66p970112",
    "Re_91p792204",
    "Re_121p050171",
    "Re_139p642302",
    "Re_169p244893",
    "Re_196p160723",
}
HELDOUT_LABELS = {
    "Re_70p314635",
    "Re_100p352251",
    "Re_149p059229",
    "Re_189p862278",
}


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_member(path: Path, key: str) -> np.ndarray:
    """Read exactly one .npy member from an NPZ, never adjacent coefficient data."""
    member = f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if member not in names:
            raise KeyError(f"{member} absent from {path}")
        with archive.open(member) as handle:
            return np.load(io.BytesIO(handle.read()), allow_pickle=False)


def load_index(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["Re_label"], []).append(row)
    for label in grouped:
        grouped[label].sort(key=lambda row: int(float(row["local_snapshot_index"])))
    return grouped


def relative_weighted_error(
    truth: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    numerator = np.sum((truth - prediction) ** 2 * weights[None, :], axis=1)
    denominator = np.sum(truth**2 * weights[None, :], axis=1)
    return np.sqrt(numerator / np.maximum(denominator, 1.0e-30))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--pod-root", type=Path)
    parser.add_argument("--projection-contract", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    if (args.pod_root is None) == (args.projection_contract is None):
        raise ValueError("provide exactly one of --pod-root or --projection-contract")
    pod_root = args.pod_root.resolve() if args.pod_root is not None else None
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    if pod_root is None:
        contract_path = args.projection_contract.resolve()
        with np.load(contract_path, allow_pickle=False) as contract:
            contract_arrays = {key: contract[key] for key in contract.files}
        velocity_path = pressure_path = contract_path
        index_path = raw_root / "Regime_ROM_Library/periodic/Global_POD_AreaWeighted_L2/pod_snapshot_index.csv"
        member = lambda _path, key: np.asarray(contract_arrays[key])
    else:
        contract_path = None
        velocity_path = pod_root / "global_velocity_pod_area_weighted_l2.npz"
        pressure_path = pod_root / "global_pressure_pod_area_weighted_l2.npz"
        index_path = pod_root / "pod_snapshot_index.csv"
        member = load_member
    index = load_index(index_path)

    labels = member(velocity_path, "Re_labels").astype(str)
    re_values = member(velocity_path, "Re_values").astype(np.float64)
    if contract_path is not None:
        fit_split = str(np.asarray(contract_arrays["fit_split"]).item())
        centering = str(np.asarray(contract_arrays["centering"]).item())
        pressure_gauge = str(np.asarray(contract_arrays["pressure_gauge"]).item())
        split_by_re = np.asarray(contract_arrays["split_by_Re"]).astype(str)
        if fit_split != "train":
            raise ValueError(f"unexpected POD fit split: {fit_split}")
        if centering != "single_train_only_regime_mean":
            raise ValueError(f"unexpected POD centering: {centering}")
        if pressure_gauge != "subtract_area_mean_per_snapshot":
            raise ValueError(f"unexpected pressure gauge: {pressure_gauge}")
        declared_validation = set(labels[split_by_re == "validation"].tolist())
        declared_heldout = set(labels[split_by_re == "heldout"].tolist())
        if declared_validation != VALIDATION_LABELS or declared_heldout != HELDOUT_LABELS:
            raise ValueError("projection contract split does not match preregistered split")
    if labels.shape != re_values.shape or set(labels) != set(index):
        raise ValueError("POD label/index contract mismatch")
    label_to_row = {label: i for i, label in enumerate(labels.tolist())}

    phi_uv = member(velocity_path, "phi_uv")[: args.rank].astype(np.float32)
    phi_p = member(pressure_path, "phi_p")[: args.rank].astype(np.float32)
    mean_uv_regime = member(velocity_path, "mean_uv_regime").astype(np.float32)
    mean_p_regime = member(pressure_path, "mean_p_regime").astype(np.float32)
    points = member(velocity_path, "points").astype(np.float32)
    weights = member(velocity_path, "point_areas").astype(np.float32)
    sqrt_weights = np.sqrt(weights)
    n_points = points.shape[0]
    if phi_uv.shape != (args.rank, 2 * n_points):
        raise ValueError(f"unexpected velocity basis shape {phi_uv.shape}")
    if phi_p.shape != (args.rank, n_points):
        raise ValueError(f"unexpected pressure basis shape {phi_p.shape}")

    # Frozen raw modes are area-orthonormal.  Projection is <phi, x-mean>_M.
    velocity_gram = (
        phi_uv[:, :n_points] @ (phi_uv[:, :n_points] * weights).T
        + phi_uv[:, n_points:] @ (phi_uv[:, n_points:] * weights).T
    )
    pressure_gram = phi_p @ (phi_p * weights).T
    orth_uv = float(np.max(np.abs(velocity_gram - np.eye(args.rank))))
    orth_p = float(np.max(np.abs(pressure_gram - np.eye(args.rank))))
    if orth_uv > 2.0e-5 or orth_p > 2.0e-5:
        raise ValueError(f"POD orthogonality failed: velocity={orth_uv}, pressure={orth_p}")

    split_payload: dict[str, dict[str, list[np.ndarray]]] = {
        "train": {},
        "validation": {},
    }
    manifest_rows: list[dict[str, object]] = []
    projection_rows: list[dict[str, object]] = []
    opened_labels: list[str] = []

    for label in labels.tolist():
        if label in HELDOUT_LABELS:
            continue
        split = "validation" if label in VALIDATION_LABELS else "train"
        source = raw_root / f"{label}_uvp_pointData.npz"
        if not source.exists():
            raise FileNotFoundError(source)
        opened_labels.append(label)
        row_id = label_to_row[label]
        rows = index[label]
        with np.load(source, allow_pickle=False) as archive:
            times = np.asarray(archive["times"], dtype=np.float64)
            source_points = np.asarray(archive["points"], dtype=np.float64)
            u = np.asarray(archive["u"], dtype=np.float32)
            v = np.asarray(archive["v"], dtype=np.float32)
            pressure = np.asarray(archive["p"], dtype=np.float32)
        if source_points.shape != points.shape or not np.allclose(
            source_points, points, rtol=0.0, atol=1.0e-7
        ):
            raise ValueError(f"{label}: point ordering mismatch")
        indexed_times = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
        if times.shape != indexed_times.shape or not np.allclose(
            times, indexed_times, rtol=0.0, atol=2.0e-4
        ):
            raise ValueError(f"{label}: raw/index time mismatch")

        centered_u = u - mean_uv_regime[:n_points]
        centered_v = v - mean_uv_regime[n_points:]
        pressure_area_mean = pressure @ weights / weights.sum()
        pressure_gauged = pressure - pressure_area_mean[:, None]
        centered_p = pressure_gauged - mean_p_regime
        a = (
            (centered_u * weights) @ phi_uv[:, :n_points].T
            + (centered_v * weights) @ phi_uv[:, n_points:].T
        )
        b = (centered_p * weights) @ phi_p.T

        reconstructed_u = mean_uv_regime[:n_points] + a @ phi_uv[:, :n_points]
        reconstructed_v = mean_uv_regime[n_points:] + a @ phi_uv[:, n_points:]
        reconstructed_p = mean_p_regime + b @ phi_p
        uv_truth = np.concatenate((u, v), axis=1)
        uv_pred = np.concatenate((reconstructed_u, reconstructed_v), axis=1)
        uv_weights = np.concatenate((weights, weights))
        uv_error = relative_weighted_error(uv_truth, uv_pred, uv_weights)
        p_error = relative_weighted_error(pressure_gauged, reconstructed_p, weights)

        period = np.asarray(
            [float(row.get("estimated_period", "0") or 0.0) for row in rows],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(period)) or np.any(period <= 0.0):
            raise ValueError(f"{label}: missing native estimated_period")
        phase = np.mod((times - times[0]) / period, 1.0)
        local_index = np.asarray(
            [int(float(row["local_snapshot_index"])) for row in rows], dtype=np.int64
        )
        re_column = np.full(times.shape, re_values[row_id], dtype=np.float64)
        label_column = np.full(times.shape, label)

        values = {
            "a": a.astype(np.float32),
            "b": b.astype(np.float32),
            "Re": re_column,
            "time": times,
            "phase": phase.astype(np.float32),
            "period": period.astype(np.float32),
            "local_snapshot_index": local_index,
            "Re_label": label_column,
        }
        target = split_payload[split]
        for key, value in values.items():
            target.setdefault(key, []).append(value)

        manifest_rows.append(
            {
                "split": split,
                "Re_label": label,
                "Re": float(re_values[row_id]),
                "snapshots": int(times.size),
                "source": str(source),
                "source_sha256": sha256_file(source),
            }
        )
        projection_rows.append(
            {
                "split": split,
                "Re_label": label,
                "velocity_relative_l2_mean": float(np.mean(uv_error)),
                "velocity_relative_l2_max": float(np.max(uv_error)),
                "pressure_relative_l2_mean": float(np.mean(p_error)),
                "pressure_relative_l2_max": float(np.max(p_error)),
                "pressure_area_mean_abs_max": float(np.max(np.abs(pressure_area_mean))),
            }
        )

    if set(opened_labels) & HELDOUT_LABELS:
        raise AssertionError("heldout raw trajectory was opened")
    expected_development = set(labels.tolist()) - HELDOUT_LABELS
    if set(opened_labels) != expected_development:
        raise ValueError("development label coverage mismatch")
    if set(labels.tolist()) & VALIDATION_LABELS != VALIDATION_LABELS:
        raise ValueError("validation labels absent from frozen POD")

    output_files: dict[str, dict[str, object]] = {}
    for split, parts in split_payload.items():
        packed = {key: np.concatenate(chunks, axis=0) for key, chunks in parts.items()}
        path = output_dir / f"periodic_{split}_modal_r{args.rank}.npz"
        np.savez_compressed(path, **packed)
        output_files[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "snapshots": int(packed["a"].shape[0]),
            "Re_labels": sorted(set(packed["Re_label"].tolist())),
        }

    manifest = {
        "schema": "periodic_development_modal_bundle/v1",
        "decision": "PASS",
        "contract": {
            "checkpoint_pod_reused_without_refit": True,
            "mixed_coeff_members_read": [],
            "heldout_raw_files_opened": [],
            "pressure_gauge": "preserve_raw_database_and_frozen_per_Re_mean_contract",
            "phase": "indexed time and estimated_period; phase=(time-time0)/period mod 1",
            "rank": args.rank,
        },
        "frozen_assets": {
            "velocity_pod": str(velocity_path),
            "velocity_pod_sha256": sha256_file(velocity_path),
            "pressure_pod": str(pressure_path),
            "pressure_pod_sha256": sha256_file(pressure_path),
            "projection_contract": str(contract_path) if contract_path else None,
            "snapshot_index": str(index_path),
            "snapshot_index_sha256": sha256_file(index_path),
        },
        "orthogonality_max_abs": {"velocity": orth_uv, "pressure": orth_p},
        "outputs": output_files,
        "sources_opened": manifest_rows,
        "projection_diagnostics": projection_rows,
        "heldout_labels_sealed": sorted(HELDOUT_LABELS),
    }
    manifest_path = output_dir / "PERIODIC_DEVELOPMENT_MODAL_BUNDLE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "MANIFEST.sha256").write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(manifest["outputs"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
