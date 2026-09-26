#!/usr/bin/env python3
"""Project additional CFD snapshots onto the frozen Steady POD without refitting it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def projection(centered_weighted: np.ndarray, basis_weighted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gram_inv = np.linalg.pinv(basis_weighted @ basis_weighted.T, rcond=1e-12)
    inner = centered_weighted @ basis_weighted.T
    coefficients = inner @ gram_inv
    residual_sq = np.einsum("ij,ij->i", centered_weighted, centered_weighted)
    residual_sq -= np.einsum("ij,ij->i", coefficients, inner)
    return coefficients, np.sqrt(np.maximum(residual_sq, 0.0))


def history_links(start: int, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = np.full((count, 3), -1, dtype=np.int64)
    previous = np.full(count, -1, dtype=np.int64)
    following = np.full(count, -1, dtype=np.int64)
    for local in range(count):
        global_index = start + local
        history[local, 2] = global_index
        if local >= 1:
            history[local, 1] = global_index - 1
            previous[local] = global_index - 1
        if local >= 2:
            history[local, 0] = global_index - 2
        if local + 1 < count:
            following[local] = global_index + 1
    return history, previous, following


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "worst": float(np.max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--steady-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = contract["cases"]
    labels = [case["label"] for case in cases]
    if len(labels) != len(set(labels)):
        raise ValueError("split manifest contains duplicate labels")
    expected_splits = {"train": 3, "validation": 2, "heldout": 2}
    actual_splits = {
        split: sum(case["split"] == split for case in cases) for split in expected_splits
    }
    if actual_splits != expected_splits:
        raise ValueError(f"split count mismatch: {actual_splits}")

    artifact_root = args.steady_root / "source_artifacts" / "steady"
    velocity_path = artifact_root / "velocity_pod_steady.npz"
    pressure_path = artifact_root / "pressure_pod_steady.npz"
    normalization_path = artifact_root / "normalization_steady.npz"
    frozen_hashes = {
        path.name: sha256(path)
        for path in (velocity_path, pressure_path, normalization_path)
    }

    with np.load(velocity_path, allow_pickle=False) as pod:
        points = np.asarray(pod["points"], dtype=np.float32)
        weights = np.asarray(pod["point_areas"], dtype=np.float64)
        sqrt_weights = np.asarray(pod["sqrt_point_areas"], dtype=np.float64)
        velocity_mean = np.asarray(pod["mean_uv_regime"], dtype=np.float64)
        velocity_basis = np.asarray(pod["phi_uv_weighted"][: args.rank], dtype=np.float64)
    with np.load(pressure_path, allow_pickle=False) as pod:
        pressure_mean = np.asarray(pod["mean_p_regime"], dtype=np.float64)
        pressure_basis = np.asarray(pod["phi_p_weighted"][: args.rank], dtype=np.float64)
        pressure_gauge = str(pod["pressure_gauge"].item())
    with np.load(normalization_path, allow_pickle=False) as normalization:
        a_mean = np.asarray(normalization["velocity_coeff_mean"][: args.rank], dtype=np.float64)
        a_std = np.asarray(normalization["velocity_coeff_std"][: args.rank], dtype=np.float64)
        b_mean = np.asarray(normalization["pressure_coeff_mean"][: args.rank], dtype=np.float64)
        b_std = np.asarray(normalization["pressure_coeff_std"][: args.rank], dtype=np.float64)

    if pressure_gauge != "subtract_area_mean_per_snapshot":
        raise ValueError(f"unexpected pressure gauge: {pressure_gauge}")
    if np.any(weights <= 0) or not np.allclose(np.sqrt(weights), sqrt_weights, rtol=2e-6):
        raise ValueError("invalid frozen POD quadrature weights")
    if min(np.min(a_std), np.min(b_std)) <= 0:
        raise ValueError("frozen normalization contains nonpositive scales")

    records: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "a_raw", "b_raw", "a_normalized", "b_normalized", "Re", "Re_label",
            "split", "requested_split", "time", "local_snapshot_index",
            "history_index", "prev_index", "next_index", "dt_next",
        )
    }
    diagnostics: list[dict[str, object]] = []
    raw_hashes: dict[str, str] = {}
    offset = 0
    n_points = points.shape[0]
    velocity_sqrt = np.concatenate((sqrt_weights, sqrt_weights))
    velocity_mean_weighted = velocity_mean * velocity_sqrt
    pressure_mean_gauged = pressure_mean - float(pressure_mean @ weights / weights.sum())
    pressure_mean_weighted = pressure_mean_gauged * sqrt_weights

    for case in cases:
        path = args.raw_dir / f"{case['label']}_uvp_pointData.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        raw_hashes[path.name] = sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            raw_points = np.asarray(archive["points"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float64)
            u = np.asarray(archive["u"], dtype=np.float64)
            v = np.asarray(archive["v"], dtype=np.float64)
            p = np.asarray(archive["p"], dtype=np.float64)
        count = times.size
        if raw_points.shape != points.shape or not np.allclose(raw_points, points, rtol=0, atol=1e-6):
            raise ValueError(f"{case['label']}: mesh differs from frozen Steady POD")
        if u.shape != (count, n_points) or v.shape != u.shape or p.shape != u.shape:
            raise ValueError(f"{case['label']}: malformed field arrays")
        if count < 56 or not np.all(np.diff(times) > 0):
            raise ValueError(f"{case['label']}: insufficient or nonmonotonic snapshots")
        if not all(np.all(np.isfinite(array)) for array in (times, u, v, p)):
            raise ValueError(f"{case['label']}: nonfinite simulation output")

        velocity_full_weighted = np.concatenate((u * sqrt_weights, v * sqrt_weights), axis=1)
        velocity_centered = velocity_full_weighted - velocity_mean_weighted
        p_gauged = p - (p @ weights / weights.sum())[:, None]
        pressure_full_weighted = p_gauged * sqrt_weights
        pressure_centered = pressure_full_weighted - pressure_mean_weighted
        a_raw, velocity_residual = projection(velocity_centered, velocity_basis)
        b_raw, pressure_residual = projection(pressure_centered, pressure_basis)
        a_normalized = (a_raw - a_mean) / a_std
        b_normalized = (b_raw - b_mean) / b_std

        history, previous, following = history_links(offset, count)
        dt_next = np.full(count, np.nan, dtype=np.float64)
        dt_next[:-1] = np.diff(times)
        records["a_raw"].append(a_raw.astype(np.float32))
        records["b_raw"].append(b_raw.astype(np.float32))
        records["a_normalized"].append(a_normalized.astype(np.float32))
        records["b_normalized"].append(b_normalized.astype(np.float32))
        records["Re"].append(np.full(count, case["Re"], dtype=np.float64))
        records["Re_label"].append(np.full(count, case["label"]))
        records["split"].append(np.full(count, case["split"]))
        records["requested_split"].append(np.full(count, case["requested_split"]))
        records["time"].append(times)
        records["local_snapshot_index"].append(np.arange(count, dtype=np.int64))
        records["history_index"].append(history)
        records["prev_index"].append(previous)
        records["next_index"].append(following)
        records["dt_next"].append(dt_next)

        velocity_denominator = np.linalg.norm(velocity_full_weighted, axis=1)
        pressure_denominator = np.linalg.norm(pressure_full_weighted, axis=1)
        diagnostics.append(
            {
                **case,
                "snapshots": int(count),
                "time_start": float(times[0]),
                "time_end": float(times[-1]),
                "velocity_r32_full_relative_l2": summary(
                    velocity_residual / np.maximum(velocity_denominator, 1e-30)
                ),
                "pressure_r32_full_relative_l2": summary(
                    pressure_residual / np.maximum(pressure_denominator, 1e-30)
                ),
            }
        )
        offset += count

    merged = {key: np.concatenate(value, axis=0) for key, value in records.items()}
    source_snapshot_id = np.arange(offset, dtype=np.int64)
    split_indices = {
        split: np.flatnonzero(merged["split"] == split).astype(np.int64)
        for split in expected_splits
    }
    if any(set(split_indices[a]) & set(split_indices[b]) for a, b in (
        ("train", "validation"), ("train", "heldout"), ("validation", "heldout")
    )):
        raise AssertionError("snapshot splits overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    modal_path = args.output_dir / "steady_expanded_validation_modal_r32.npz"
    np.savez_compressed(
        modal_path,
        schema_version=np.asarray(1, dtype=np.int64),
        rank=np.asarray(args.rank, dtype=np.int64),
        source_snapshot_id=source_snapshot_id,
        **merged,
        train_index=split_indices["train"],
        validation_index=split_indices["validation"],
        heldout_index=split_indices["heldout"],
        final_test_index=split_indices["heldout"],
        pressure_gauge=np.asarray(pressure_gauge),
        centering=np.asarray("single_train_only_regime_mean"),
        pod_refit=np.asarray(False),
    )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "pod_refit": False,
        "rank": args.rank,
        "cases": diagnostics,
        "case_split_counts": actual_splits,
        "snapshot_split_counts": {
            key: int(value.size) for key, value in split_indices.items()
        },
        "frozen_artifact_sha256": frozen_hashes,
        "raw_sha256": raw_hashes,
        "modal_file": modal_path.name,
        "modal_sha256": sha256(modal_path),
    }
    (args.output_dir / "projection_validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
