#!/usr/bin/env python3
"""Build train-only pressure POD in the OpenFOAM outlet-fixed gauge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store-rank", type=int, default=32)
    parser.add_argument("--analysis-rank", type=int, default=128)
    parser.add_argument("--oversample", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--expected-cases", type=int, default=23)
    return parser.parse_args()


def randomized_svd(matrix, total_energy, rank, oversample, seed):
    rows, features = matrix.shape
    sample = min(rank + oversample, rows, features)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((features, sample), dtype=np.float32)
    projected = np.empty((rows, sample), dtype=np.float64)
    for start in range(0, rows, 256):
        stop = min(start + 256, rows)
        projected[start:stop] = matrix[start:stop].astype(np.float64) @ omega
    q_factor, _ = np.linalg.qr(projected, mode="reduced")
    compressed = np.zeros((sample, features), dtype=np.float64)
    for start in range(0, rows, 256):
        stop = min(start + 256, rows)
        compressed += q_factor[start:stop].T @ matrix[start:stop].astype(np.float64)
    left, singular, modes = np.linalg.svd(compressed, full_matrices=False)
    singular = singular[:rank]
    modes = modes[:rank]
    coefficients = (q_factor @ left[:, :rank]) * singular[None]
    cumulative = np.cumsum(singular**2) / total_energy
    return singular, modes, coefficients, cumulative


def threshold_rank(cumulative, threshold):
    indices = np.flatnonzero(cumulative >= threshold)
    return int(indices[0] + 1) if len(indices) else None


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cases = sorted((args.dataset_root / "cases_npz").glob("*.npz"))
    if len(cases) != args.expected_cases:
        raise RuntimeError(f"expected {args.expected_cases} train cases, found {len(cases)}")
    with np.load(args.dataset_root / "mesh/mesh_metadata.npz") as mesh:
        volumes = np.asarray(mesh["cellVolumes"], dtype=np.float64)
    counts = []
    pressure_sum = np.zeros(len(volumes), dtype=np.float64)
    tags = []
    times = []
    for path in cases:
        with np.load(path, allow_pickle=False) as source:
            pressure = np.asarray(source["p"], dtype=np.float64)
            case_times = np.asarray(source["times"], dtype=np.float64)
            re_value = float(source["Re"])
        counts.append(len(case_times))
        pressure_sum += pressure.sum(axis=0)
        tag = ("Re" + f"{re_value:010.6f}").replace(".", "p")
        tags.extend([tag] * len(case_times))
        times.extend(case_times.tolist())
    snapshots = sum(counts)
    mean = pressure_sum / snapshots
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".weighted.dat")
    matrix = np.memmap(temporary, mode="w+", dtype="float32", shape=(snapshots, len(volumes)))
    weights = np.sqrt(volumes)
    total_energy = 0.0
    row = 0
    for path, count in zip(cases, counts):
        with np.load(path, allow_pickle=False) as source:
            centered = np.asarray(source["p"], dtype=np.float64) - mean[None]
        weighted = centered * weights[None]
        matrix[row : row + count] = weighted.astype(np.float32)
        total_energy += float(np.sum(weighted**2))
        row += count
    matrix.flush()
    singular, weighted_modes, coefficients, cumulative = randomized_svd(
        matrix, total_energy, args.analysis_rank, args.oversample, args.seed
    )
    del matrix
    temporary.unlink()
    keep = args.store_rank
    modes = weighted_modes[:keep] / weights[None]
    gram_error = float(np.max(np.abs(weighted_modes[:keep] @ weighted_modes[:keep].T - np.eye(keep))))
    np.savez_compressed(
        args.output,
        singular_values=singular[:keep], cumulative_energy=cumulative[:keep],
        modes=modes.astype(np.float32), weighted_modes=weighted_modes[:keep].astype(np.float32),
        coefficients=coefficients[:, :keep].astype(np.float32), mean=mean.astype(np.float32),
        weights=weights.astype(np.float32), snapshot_times=np.asarray(times),
        snapshot_case_tags=np.asarray(tags), stored_modes=np.asarray(keep),
        analysis_rank=np.asarray(args.analysis_rank), rank_99=np.asarray(threshold_rank(cumulative, 0.99)),
        rank_999=np.asarray(threshold_rank(cumulative, 0.999)), gram_max_abs=np.asarray(gram_error),
        gauge=np.asarray("OpenFOAM fixed outlet p=0; no per-snapshot volume-mean subtraction"),
    )
    report = {
        "status": "PASS", "train_cases": len(cases), "train_snapshots": snapshots,
        "stored_modes": keep, "analysis_rank": args.analysis_rank,
        "rank_99": threshold_rank(cumulative, 0.99), "rank_999": threshold_rank(cumulative, 0.999),
        "captured_energy": float(cumulative[keep - 1]), "gram_max_abs": gram_error,
        "heldout_loaded": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
