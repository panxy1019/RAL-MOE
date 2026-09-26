#!/usr/bin/env python3
"""Build an independent train-only extended POD for CenteredSquare Hopf.

The source subset must contain exactly the 23 frozen training Reynolds-number
cases. Validation and held-out files are rejected if present. The randomized
SVD implementation and seed match the original three-regime construction,
but this script stores the first 64 modes instead of truncating at r999=11.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


TRAIN_RE = np.asarray(
    [
        94.0, 95.0, 95.05, 95.15, 95.2, 95.35, 95.4, 95.45,
        95.55, 95.6, 95.75, 96.0, 96.25, 96.75, 97.0, 97.25,
        97.75, 98.0, 98.5, 99.5, 100.0, 101.0, 101.724137931034,
    ],
    dtype=np.float64,
)
TOL = 5.0e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-subset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--store-rank", type=int, default=64)
    parser.add_argument("--analysis-rank", type=int, default=256)
    parser.add_argument("--oversample", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    os.link(source, target)


def re_tag(value: float) -> str:
    return ("Re" + f"{value:010.6f}").replace(".", "p")


def load_case(path: Path, volumes: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        reynolds = float(source["Re"])
        times = np.asarray(source["times"], dtype=np.float64)
        velocity = np.asarray(source["U"], dtype=np.float32)
        pressure = np.asarray(source["p"], dtype=np.float32)
    if velocity.shape != (len(times), len(volumes), 2):
        raise RuntimeError(f"velocity shape mismatch in {path}: {velocity.shape}")
    if pressure.shape != (len(times), len(volumes)):
        raise RuntimeError(f"pressure shape mismatch in {path}: {pressure.shape}")
    if not np.all(np.diff(times) > 0):
        raise RuntimeError(f"non-monotone times in {path}")
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(pressure)):
        raise RuntimeError(f"non-finite fields in {path}")
    pressure = pressure - (
        pressure.astype(np.float64) @ volumes / float(volumes.sum())
    ).astype(np.float32)[:, None]
    return reynolds, times, velocity, pressure


def randomized_svd(
    matrix: np.memmap,
    total_energy: float,
    max_modes: int,
    oversample: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, features = matrix.shape
    sample = min(max_modes + oversample, rows, features)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((features, sample), dtype=np.float32)
    projected = np.zeros((rows, sample), dtype=np.float64)
    for start in range(0, rows, 256):
        stop = min(start + 256, rows)
        projected[start:stop] = matrix[start:stop].astype(np.float64) @ omega
    q_factor, _ = np.linalg.qr(projected, mode="reduced")
    compressed = np.zeros((sample, features), dtype=np.float64)
    for start in range(0, rows, 256):
        stop = min(start + 256, rows)
        compressed += q_factor[start:stop].T @ matrix[start:stop].astype(np.float64)
    left, singular, modes = np.linalg.svd(compressed, full_matrices=False)
    keep = min(max_modes, singular.size)
    singular = singular[:keep]
    modes = modes[:keep]
    coefficients = (q_factor @ left[:, :keep]) * singular[None, :]
    cumulative = np.cumsum(singular * singular) / total_energy
    return singular, modes, coefficients, cumulative


def rank_for(cumulative: np.ndarray, threshold: float) -> int:
    hits = np.flatnonzero(cumulative >= threshold)
    if not len(hits):
        raise RuntimeError(
            f"analysis spectrum did not reach {threshold}; captured={cumulative[-1]}"
        )
    return int(hits[0] + 1)


def build_field(
    output_root: Path,
    cases: list[Path],
    volumes: np.ndarray,
    mean: np.ndarray,
    field: str,
    store_rank: int,
    analysis_rank: int,
    oversample: int,
    seed: int,
) -> dict[str, Any]:
    counts: list[int] = []
    for path in cases:
        with np.load(path, allow_pickle=False) as source:
            counts.append(int(np.asarray(source["times"]).size))
    snapshots = int(sum(counts))
    features = len(volumes) * (2 if field == "velocity" else 1)
    if not 0 < store_rank <= analysis_rank <= min(snapshots, features):
        raise ValueError(
            f"invalid ranks store={store_rank}, analysis={analysis_rank}, "
            f"snapshots={snapshots}, features={features}"
        )
    pod_dir = output_root / "pod"
    matrix_path = pod_dir / f"_tmp_{field}_weighted.dat"
    matrix = np.memmap(
        matrix_path, dtype="float32", mode="w+", shape=(snapshots, features)
    )
    sqrt_volumes = np.sqrt(volumes)
    weights = np.repeat(sqrt_volumes, 2) if field == "velocity" else sqrt_volumes
    row = 0
    total_energy = 0.0
    times_all: list[float] = []
    tags_all: list[str] = []
    offsets: list[list[Any]] = []
    for path, count in zip(cases, counts):
        reynolds, times, velocity, pressure = load_case(path, volumes)
        if field == "velocity":
            centered = (velocity - mean[None]).reshape(count, -1)
        else:
            centered = pressure - mean[None]
        weighted = centered.astype(np.float64) * weights[None]
        matrix[row : row + count] = weighted.astype(np.float32)
        total_energy += float(np.sum(weighted * weighted))
        tag = re_tag(reynolds)
        offsets.append([tag, row, row + count])
        times_all.extend(times.tolist())
        tags_all.extend([tag] * count)
        row += count
    matrix.flush()
    singular, weighted_modes, coefficients, cumulative = randomized_svd(
        matrix, total_energy, analysis_rank, oversample, seed
    )
    del matrix
    matrix_path.unlink(missing_ok=True)

    rank99 = rank_for(cumulative, 0.99)
    rank999 = rank_for(cumulative, 0.999)
    physical_modes = weighted_modes[:store_rank] / weights[None]
    gram = weighted_modes[:store_rank] @ weighted_modes[:store_rank].T
    gram_error = float(np.max(np.abs(gram - np.eye(store_rank))))
    if gram_error > 5.0e-6:
        raise RuntimeError(f"{field} weighted Gram error too large: {gram_error}")

    output = pod_dir / f"weighted_pod_{field}.npz"
    np.savez_compressed(
        output,
        singular_values=singular[:store_rank].astype(np.float64),
        cumulative_energy=cumulative[:store_rank].astype(np.float64),
        modes=physical_modes.astype(np.float32),
        weighted_modes=weighted_modes[:store_rank].astype(np.float32),
        coefficients=coefficients[:, :store_rank].astype(np.float32),
        mean=mean.astype(np.float32),
        weights=weights.astype(np.float32),
        total_energy=np.asarray(total_energy, dtype=np.float64),
        centered=np.asarray(True),
        pressure_gauge=np.asarray(
            "subtract_volume_mean_per_snapshot"
            if field == "pressure"
            else "not_applicable"
        ),
        snapshot_times=np.asarray(times_all, dtype=np.float64),
        snapshot_case_tags=np.asarray(tags_all),
        case_offsets=np.asarray(json.dumps(offsets)),
        analysis_rank=np.asarray(analysis_rank, dtype=np.int64),
        stored_modes=np.asarray(store_rank, dtype=np.int64),
        rank_99=np.asarray(rank99, dtype=np.int64),
        rank_999=np.asarray(rank999, dtype=np.int64),
        gram_max_abs=np.asarray(gram_error, dtype=np.float64),
    )
    return {
        "field": field,
        "path": str(output),
        "sha256": sha256(output),
        "rank_99": rank99,
        "rank_999": rank999,
        "analysis_rank": analysis_rank,
        "stored_modes": store_rank,
        "captured_energy": float(cumulative[store_rank - 1]),
        "gram_max_abs": gram_error,
    }


def subspace_audit(new_path: Path, old_path: Path, rank: int = 11) -> dict[str, float]:
    with np.load(new_path, allow_pickle=False) as new, np.load(
        old_path, allow_pickle=False
    ) as old:
        new_modes = np.asarray(new["weighted_modes"], dtype=np.float64)[:rank]
        old_modes = np.asarray(old["weighted_modes"], dtype=np.float64)[:rank]
    singular = np.linalg.svd(old_modes @ new_modes.T, compute_uv=False)
    return {
        "minimum_principal_cosine": float(singular.min()),
        "maximum_principal_angle_deg": float(
            np.degrees(np.arccos(np.clip(singular.min(), -1.0, 1.0)))
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite independent output: {args.output_root}"
        )
    source_cases = sorted((args.source_subset / "cases_npz").glob("snapshots_*.npz"))
    observed: list[float] = []
    for path in source_cases:
        with np.load(path, allow_pickle=False) as source:
            observed.append(float(source["Re"]))
    if len(observed) != len(TRAIN_RE) or not np.allclose(
        np.sort(observed), np.sort(TRAIN_RE), atol=TOL, rtol=0
    ):
        raise RuntimeError(
            "source subset is not the exact frozen 23-case train split: "
            f"{sorted(observed)}"
        )

    args.output_root.mkdir(parents=True)
    (args.output_root / "pod").mkdir()
    (args.output_root / "cases_npz").mkdir()
    (args.output_root / "mesh").mkdir()
    (args.output_root / "reference_vtk").mkdir()
    (args.output_root / "manifest").mkdir()
    for source in source_cases:
        hardlink(source, args.output_root / "cases_npz" / source.name)
    mesh_source = args.source_subset / "mesh" / "mesh_metadata.npz"
    hardlink(mesh_source, args.output_root / "mesh" / mesh_source.name)
    vtk_sources = sorted((args.source_subset / "reference_vtk").glob("*.vtk"))
    if len(vtk_sources) != 1 or not vtk_sources[0].is_file():
        raise RuntimeError(f"expected one reference VTK, found {vtk_sources}")
    hardlink(vtk_sources[0], args.output_root / "reference_vtk" / vtk_sources[0].name)

    rows = []
    by_re = sorted(zip(observed, source_cases), key=lambda item: item[0])
    for index, (reynolds, path) in enumerate(by_re, start=1):
        rows.append(
            {
                "index": index,
                "case_tag": path.stem.removeprefix("snapshots_"),
                "Re": f"{reynolds:.12f}",
                "nu": f"{1.0 / reynolds:.16g}",
                "segment": "target_hopf_train_cdm_extended",
                "source_regime": "operator_train_only",
                "source_dataset": args.source_subset.name,
            }
        )
    manifest_csv = args.output_root / "manifest" / "re_points_100.csv"
    with manifest_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with np.load(mesh_source, allow_pickle=False) as mesh:
        volumes = np.asarray(mesh["cellVolumes"], dtype=np.float64)
    if volumes.shape != (9400,) or np.any(volumes <= 0) or not np.all(np.isfinite(volumes)):
        raise RuntimeError(f"invalid volume array: {volumes.shape}")

    velocity_sum = np.zeros((9400, 2), dtype=np.float64)
    pressure_sum = np.zeros(9400, dtype=np.float64)
    snapshots = 0
    for _, path in by_re:
        _, _, velocity, pressure = load_case(path, volumes)
        velocity_sum += velocity.astype(np.float64).sum(axis=0)
        pressure_sum += pressure.astype(np.float64).sum(axis=0)
        snapshots += len(velocity)
    velocity_mean = velocity_sum / snapshots
    pressure_mean = pressure_sum / snapshots
    pressure_mean_gauge = float(volumes @ pressure_mean / volumes.sum())
    if abs(pressure_mean_gauge) > 2.0e-7:
        raise RuntimeError(f"pressure train mean violates gauge: {pressure_mean_gauge}")

    velocity_report = build_field(
        args.output_root,
        [item[1] for item in by_re],
        volumes,
        velocity_mean,
        "velocity",
        args.store_rank,
        args.analysis_rank,
        args.oversample,
        args.seed,
    )
    pressure_report = build_field(
        args.output_root,
        [item[1] for item in by_re],
        volumes,
        pressure_mean,
        "pressure",
        args.store_rank,
        args.analysis_rank,
        args.oversample,
        args.seed,
    )
    velocity_subspace = subspace_audit(
        Path(velocity_report["path"]),
        args.source_subset / "pod" / "weighted_pod_velocity.npz",
    )
    pressure_subspace = subspace_audit(
        Path(pressure_report["path"]),
        args.source_subset / "pod" / "weighted_pod_pressure.npz",
    )
    if velocity_subspace["minimum_principal_cosine"] < 0.999:
        raise RuntimeError(f"velocity r11 subspace mismatch: {velocity_subspace}")
    if pressure_subspace["minimum_principal_cosine"] < 0.999:
        raise RuntimeError(f"pressure r11 subspace mismatch: {pressure_subspace}")

    payload = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "CenteredSquare Hopf train-only extended POD",
        "source_subset": str(args.source_subset),
        "source_split_count": len(source_cases),
        "train_re": TRAIN_RE.tolist(),
        "validation_loaded": False,
        "heldout_loaded": False,
        "snapshots": snapshots,
        "cells": len(volumes),
        "pressure_mean_volume_gauge": pressure_mean_gauge,
        "velocity": velocity_report,
        "pressure": pressure_report,
        "r11_subspace_reproduction": {
            "velocity": velocity_subspace,
            "pressure": pressure_subspace,
        },
        "source_sha256": {
            "mesh": sha256(mesh_source),
            "reference_vtk": sha256(vtk_sources[0]),
            "split_manifest": sha256(args.source_subset / "manifest" / "re_points_100.csv"),
        },
    }
    atomic_json(payload, args.output_root / "POD_BUILD_MANIFEST.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
