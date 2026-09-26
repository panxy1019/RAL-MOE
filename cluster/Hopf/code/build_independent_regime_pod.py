#!/usr/bin/env python3
"""Build leakage-free, regime-independent velocity/pressure POD artifacts.

The fit contract is deliberately strict:

* Reynolds-number splits are disjoint.
* Only training Re cases contribute to the regime mean, POD basis, singular
  spectrum, rank recommendation, or normalization.
* Pressure uses one area-mean gauge for every snapshot before centering.
* A single train-only regime mean is used, so the steady base-flow branch is
  represented instead of being erased by per-Re temporal centering.
* Validation and held-out cases are projected only after the fit is frozen.

The output keeps the legacy V15/V16 array keys needed by the semi-intrusive
Galerkin and pressure-Poisson builders while adding explicit split/provenance
fields for the three later specialist MoE tasks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
import scipy.linalg


Array = np.ndarray

REGIME_GROUPS: dict[str, tuple[str, ...]] = {
    "steady": ("steady_wake", "pre_hopf_steady"),
    "hopf": ("hopf_transition",),
    "periodic": (
        "developing_periodic_shedding",
        "mature_periodic_shedding",
        "high_re_2d_periodic_near_modeA",
    ),
}


@dataclass(frozen=True)
class Case:
    label: str
    re_value: float
    source_regime: str
    target_regime: str
    split: str
    path: Path
    times: Array

    @property
    def n_snapshots(self) -> int:
        return int(self.times.size)


class Timer:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    def stamp(self) -> str:
        return f"{time.perf_counter() - self.started:9.1f}s"

    def elapsed(self) -> float:
        return time.perf_counter() - self.started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--global-pod-dir", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--canonical-contract", type=Path)
    parser.add_argument("--regimes", nargs="+", choices=tuple(REGIME_GROUPS), default=list(REGIME_GROUPS))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--control-rank", type=int)
    parser.add_argument("--oversampling", type=int)
    parser.add_argument("--power-iterations", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def area_gauge(block: Array, weights: Array) -> Array:
    means = np.asarray(block, dtype=np.float64) @ weights / float(weights.sum())
    return np.asarray(block, dtype=np.float64) - means[:, None]


def read_index(index_path: Path) -> list[dict[str, str]]:
    with index_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def group_index_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        label = row["Re_label"]
        item = grouped.setdefault(
            label,
            {
                "re_value": float(row["Re"]),
                "source_regime": row["regime"],
                "rows": [],
            },
        )
        if item["source_regime"] != row["regime"]:
            raise ValueError(f"{label} has inconsistent source regime labels")
        cast_rows = item["rows"]
        assert isinstance(cast_rows, list)
        cast_rows.append(row)
    return grouped


def resolve_cases(
    data_dir: Path,
    grouped: dict[str, dict[str, object]],
    split_config: dict[str, object],
    regime: str,
) -> list[Case]:
    validation = set(split_config["validation_labels"][regime])
    heldout = set(split_config["heldout_labels"][regime])
    if validation & heldout:
        raise ValueError(f"{regime}: validation/heldout overlap: {sorted(validation & heldout)}")

    configured_sources = split_config.get("source_regimes", {})
    allowed = set(configured_sources.get(regime, REGIME_GROUPS[regime]))
    configured_includes = split_config.get("include_labels", {})
    include_labels = set(configured_includes.get(regime, ()))
    selected: list[Case] = []
    for label, item in grouped.items():
        source_regime = str(item["source_regime"])
        if source_regime not in allowed or (include_labels and label not in include_labels):
            continue
        split = "heldout" if label in heldout else "validation" if label in validation else "train"
        path = data_dir / f"{label}_uvp_pointData.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            times = np.asarray(archive["times"], dtype=np.float64)
            shapes = {name: archive[name].shape for name in ("u", "v", "p")}
        if any(shape[0] != times.size for shape in shapes.values()):
            raise ValueError(f"{label}: inconsistent snapshot counts {shapes}, times={times.shape}")
        source_rows = item["rows"]
        assert isinstance(source_rows, list)
        if len(source_rows) != times.size:
            raise ValueError(f"{label}: index/file snapshot mismatch {len(source_rows)} != {times.size}")
        selected.append(
            Case(
                label=label,
                re_value=float(item["re_value"]),
                source_regime=source_regime,
                target_regime=regime,
                split=split,
                path=path,
                times=times,
            )
        )
    selected.sort(key=lambda case: case.re_value)

    found_validation = {case.label for case in selected if case.split == "validation"}
    found_heldout = {case.label for case in selected if case.split == "heldout"}
    found_labels = {case.label for case in selected}
    if include_labels and found_labels != include_labels:
        raise ValueError(f"{regime}: missing included labels {sorted(include_labels - found_labels)}")
    if found_validation != validation:
        raise ValueError(f"{regime}: missing validation labels {sorted(validation - found_validation)}")
    if found_heldout != heldout:
        raise ValueError(f"{regime}: missing heldout labels {sorted(heldout - found_heldout)}")
    if not any(case.split == "train" for case in selected):
        raise ValueError(f"{regime}: empty training split")
    return selected


def validate_canonical_heldout(
    split_config: dict[str, object],
    canonical_contract_path: Path | None,
    cases_by_regime: dict[str, list[Case]],
) -> dict[str, object]:
    configured = {
        case.label: case.re_value
        for cases in cases_by_regime.values()
        for case in cases
        if case.split == "heldout"
    }
    result: dict[str, object] = {
        "configured_labels": sorted(configured),
        "configured_values": sorted(configured.values()),
        "canonical_contract_checked": canonical_contract_path is not None,
    }
    if canonical_contract_path is None:
        return result
    contract = json.loads(canonical_contract_path.read_text(encoding="utf-8"))
    expected = np.sort(np.asarray(contract["training_holdout_re"], dtype=np.float64))
    actual = np.sort(np.asarray(list(configured.values()), dtype=np.float64))
    if expected.shape != actual.shape or not np.allclose(expected, actual, rtol=0.0, atol=2e-5):
        raise ValueError(f"heldout contract mismatch: canonical={expected.tolist()} configured={actual.tolist()}")
    result["canonical_path"] = str(canonical_contract_path)
    result["canonical_sha256"] = sha256_file(canonical_contract_path)
    result["status"] = "PASS"
    return result


def load_weights(global_pod_dir: Path) -> tuple[Array, Array, Array]:
    path = global_pod_dir / "mesh_l2_point_area_weights.npz"
    with np.load(path, allow_pickle=False) as archive:
        points = np.asarray(archive["points"], dtype=np.float64)
        weights = np.asarray(archive["point_areas"], dtype=np.float64)
        sqrt_weights = np.asarray(archive["sqrt_point_areas"], dtype=np.float64)
    if points.shape != (weights.size, 3) or sqrt_weights.shape != weights.shape:
        raise ValueError("invalid point-area artifact shapes")
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("point areas must be finite and positive")
    if not np.allclose(sqrt_weights * sqrt_weights, weights, rtol=2e-6, atol=1e-12):
        raise ValueError("sqrt_point_areas does not match point_areas")
    return points, weights, sqrt_weights


def verify_case_points(cases: list[Case], points: Array) -> None:
    for case in cases:
        with np.load(case.path, allow_pickle=False) as archive:
            candidate = np.asarray(archive["points"], dtype=np.float64)
        if candidate.shape != points.shape or not np.allclose(candidate, points, rtol=0.0, atol=1e-7):
            raise ValueError(f"{case.label}: point ordering differs from mass-weight artifact")


def fit_regime_mean(train_cases: list[Case], weights: Array, timer: Timer) -> tuple[Array, Array]:
    n_points = weights.size
    sum_uv = np.zeros(2 * n_points, dtype=np.float64)
    sum_p = np.zeros(n_points, dtype=np.float64)
    count = 0
    for idx, case in enumerate(train_cases, start=1):
        with np.load(case.path, allow_pickle=False) as archive:
            u = np.asarray(archive["u"], dtype=np.float64)
            v = np.asarray(archive["v"], dtype=np.float64)
            p = area_gauge(np.asarray(archive["p"], dtype=np.float64), weights)
        sum_uv[:n_points] += u.sum(axis=0)
        sum_uv[n_points:] += v.sum(axis=0)
        sum_p += p.sum(axis=0)
        count += case.n_snapshots
        print(f"[{timer.stamp()}] mean {case.target_regime} {idx}/{len(train_cases)} {case.label}", flush=True)
    mean_uv = sum_uv / count
    mean_p = sum_p / count
    mean_p -= float(mean_p @ weights / weights.sum())
    return mean_uv, mean_p


def offsets(cases: list[Case]) -> Array:
    out = np.zeros(len(cases) + 1, dtype=np.int64)
    for index, case in enumerate(cases):
        out[index + 1] = out[index] + case.n_snapshots
    return out


def make_velocity_loader(mean_uv: Array, sqrt_weights: Array) -> Callable[[Case], Array]:
    n_points = sqrt_weights.size

    def load(case: Case) -> Array:
        with np.load(case.path, allow_pickle=False) as archive:
            u = np.asarray(archive["u"], dtype=np.float64) - mean_uv[:n_points]
            v = np.asarray(archive["v"], dtype=np.float64) - mean_uv[n_points:]
        out = np.empty((case.n_snapshots, 2 * n_points), dtype=np.float64)
        out[:, :n_points] = u * sqrt_weights
        out[:, n_points:] = v * sqrt_weights
        return out

    return load


def make_pressure_loader(mean_p: Array, weights: Array, sqrt_weights: Array) -> Callable[[Case], Array]:
    def load(case: Case) -> Array:
        with np.load(case.path, allow_pickle=False) as archive:
            p = area_gauge(np.asarray(archive["p"], dtype=np.float64), weights)
        return (p - mean_p) * sqrt_weights

    return load


def randomized_weighted_pod(
    cases: list[Case],
    load_weighted_centered: Callable[[Case], Array],
    n_features: int,
    feature_sqrt_weights: Array,
    rank: int,
    oversampling: int,
    power_iterations: int,
    seed: int,
    timer: Timer,
    name: str,
) -> dict[str, Array | float]:
    case_offsets = offsets(cases)
    n_snapshots = int(case_offsets[-1])
    work_rank = min(n_snapshots, rank + oversampling)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n_snapshots, work_rank), dtype=np.float64)

    def x_times(matrix: Array, collect_energy: bool = False) -> tuple[Array, float]:
        result = np.zeros((n_features, matrix.shape[1]), dtype=np.float64)
        energy = 0.0
        for index, case in enumerate(cases):
            start, stop = int(case_offsets[index]), int(case_offsets[index + 1])
            block = load_weighted_centered(case)
            result += block.T @ matrix[start:stop]
            if collect_energy:
                energy += float(np.einsum("ij,ij->", block, block, optimize=True))
        return result, energy

    def xt_times(q_matrix: Array) -> Array:
        result = np.empty((n_snapshots, q_matrix.shape[1]), dtype=np.float64)
        for index, case in enumerate(cases):
            start, stop = int(case_offsets[index]), int(case_offsets[index + 1])
            result[start:stop] = load_weighted_centered(case) @ q_matrix
        return result

    print(
        f"[{timer.stamp()}] {name} randomized range n={n_snapshots}, features={n_features}, work_rank={work_rank}",
        flush=True,
    )
    y_matrix, total_energy = x_times(omega, collect_energy=True)
    q_matrix, _ = np.linalg.qr(y_matrix, mode="reduced")
    for iteration in range(power_iterations):
        print(f"[{timer.stamp()}] {name} power {iteration + 1}/{power_iterations}", flush=True)
        y_matrix, _ = x_times(xt_times(q_matrix), collect_energy=False)
        q_matrix, _ = np.linalg.qr(y_matrix, mode="reduced")

    small = xt_times(q_matrix).T
    u_hat, singular_values_all, _ = scipy.linalg.svd(small, full_matrices=False, lapack_driver="gesdd")
    retained = min(rank, singular_values_all.size)
    weighted_modes = (q_matrix @ u_hat[:, :retained]).T
    raw_modes = weighted_modes / feature_sqrt_weights[None, :]
    cumulative_all = np.cumsum(singular_values_all**2) / max(total_energy, np.finfo(np.float64).tiny)
    return {
        "raw_modes": raw_modes.astype(np.float32),
        "weighted_modes": weighted_modes.astype(np.float32),
        "singular_values": singular_values_all.astype(np.float64),
        "cumulative_energy": cumulative_all.astype(np.float64),
        "total_energy": float(total_energy),
    }


def load_full_weighted_velocity(case: Case, sqrt_weights: Array) -> Array:
    n_points = sqrt_weights.size
    with np.load(case.path, allow_pickle=False) as archive:
        u = np.asarray(archive["u"], dtype=np.float64)
        v = np.asarray(archive["v"], dtype=np.float64)
    out = np.empty((case.n_snapshots, 2 * n_points), dtype=np.float64)
    out[:, :n_points] = u * sqrt_weights
    out[:, n_points:] = v * sqrt_weights
    return out


def load_full_weighted_pressure(case: Case, weights: Array, sqrt_weights: Array) -> Array:
    with np.load(case.path, allow_pickle=False) as archive:
        return area_gauge(np.asarray(archive["p"], dtype=np.float64), weights) * sqrt_weights


def projection_from_weighted_basis(centered: Array, basis_weighted: Array, rank: int) -> tuple[Array, Array]:
    basis = np.asarray(basis_weighted[:rank], dtype=np.float64)
    gram = basis @ basis.T
    gram_inv = np.linalg.pinv(gram, rcond=1e-12)
    projected_inner = centered @ basis.T
    coeff = projected_inner @ gram_inv
    centered_norm_sq = np.einsum("ij,ij->i", centered, centered, optimize=True)
    residual_sq = centered_norm_sq - np.einsum("ij,ij->i", coeff, projected_inner, optimize=True)
    residual_sq = np.maximum(residual_sq, 0.0)
    return coeff, residual_sq


def summarize(values: Array) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "worst": float(np.max(values)),
    }


def gauge_basis_and_mean(phi_p: Array, mean_p: Array, weights: Array, sqrt_weights: Array) -> tuple[Array, Array]:
    basis = np.asarray(phi_p, dtype=np.float64)
    basis_means = basis @ weights / weights.sum()
    basis_gauged = basis - basis_means[:, None]
    mean_gauged = np.asarray(mean_p, dtype=np.float64) - float(mean_p @ weights / weights.sum())
    return basis_gauged * sqrt_weights[None, :], mean_gauged


def load_global_control(global_pod_dir: Path, weights: Array, sqrt_weights: Array) -> dict[str, object]:
    with np.load(global_pod_dir / "global_velocity_pod_area_weighted_l2.npz", allow_pickle=False) as velocity:
        data: dict[str, object] = {
            "re_labels": np.asarray(velocity["Re_labels"]).astype(str),
            "velocity_mean": np.asarray(velocity["mean_uv_by_Re"], dtype=np.float64),
            "velocity_basis_weighted": np.asarray(velocity["phi_uv_weighted"], dtype=np.float64),
        }
    with np.load(global_pod_dir / "global_pressure_pod_area_weighted_l2.npz", allow_pickle=False) as pressure:
        pressure_labels = np.asarray(pressure["Re_labels"]).astype(str)
        if not np.array_equal(data["re_labels"], pressure_labels):
            raise ValueError("frozen global velocity/pressure Re labels differ")
        phi_p = np.asarray(pressure["phi_p"], dtype=np.float64)
        mean_p = np.asarray(pressure["mean_p_by_Re"], dtype=np.float64)
    pressure_basis_weighted, _ = gauge_basis_and_mean(phi_p, mean_p[0], weights, sqrt_weights)
    pressure_mean_gauged = np.empty_like(mean_p)
    for index, row in enumerate(mean_p):
        pressure_mean_gauged[index] = row - float(row @ weights / weights.sum())
    data["pressure_basis_weighted"] = pressure_basis_weighted
    data["pressure_mean_gauged"] = pressure_mean_gauged
    data["label_to_index"] = {label: index for index, label in enumerate(data["re_labels"])}
    return data


def project_all_cases(
    cases: list[Case],
    variable: str,
    mean: Array,
    local_basis_weighted: Array,
    control_rank: int,
    retained_rank: int,
    weights: Array,
    sqrt_weights: Array,
    global_control: dict[str, object],
    timer: Timer,
) -> dict[str, object]:
    coefficients: list[Array] = []
    snapshot_records: list[dict[str, object]] = []
    split_errors: dict[str, dict[str, list[float]]] = {
        split: {"local_r32": [], "local_r80": [], "global_r32": []}
        for split in ("train", "validation", "heldout")
    }
    train_coefficients: list[Array] = []
    label_to_global = global_control["label_to_index"]
    assert isinstance(label_to_global, dict)

    for case_index, case in enumerate(cases, start=1):
        global_index = label_to_global.get(case.label)
        if variable == "velocity":
            full = load_full_weighted_velocity(case, sqrt_weights)
            n_points = sqrt_weights.size
            local_centered = full.copy()
            local_centered[:, :n_points] -= mean[:n_points] * sqrt_weights
            local_centered[:, n_points:] -= mean[n_points:] * sqrt_weights
            if global_index is not None:
                global_mean = np.asarray(global_control["velocity_mean"])[int(global_index)]
                global_centered = full.copy()
                global_centered[:, :n_points] -= global_mean[:n_points] * sqrt_weights
                global_centered[:, n_points:] -= global_mean[n_points:] * sqrt_weights
            global_basis = np.asarray(global_control["velocity_basis_weighted"])
        else:
            full = load_full_weighted_pressure(case, weights, sqrt_weights)
            local_centered = full - mean * sqrt_weights
            if global_index is not None:
                global_mean = np.asarray(global_control["pressure_mean_gauged"])[int(global_index)]
                global_centered = full - global_mean * sqrt_weights
            global_basis = np.asarray(global_control["pressure_basis_weighted"])

        coeff_r80, residual_r80 = projection_from_weighted_basis(local_centered, local_basis_weighted, retained_rank)
        coeff_r32, residual_r32 = projection_from_weighted_basis(local_centered, local_basis_weighted, control_rank)
        if global_index is None:
            residual_global = np.full(full.shape[0], np.nan, dtype=np.float64)
        else:
            _, residual_global = projection_from_weighted_basis(global_centered, global_basis, control_rank)
        full_norm_sq = np.maximum(np.einsum("ij,ij->i", full, full, optimize=True), np.finfo(np.float64).tiny)
        centered_norm_sq = np.maximum(
            np.einsum("ij,ij->i", local_centered, local_centered, optimize=True),
            np.finfo(np.float64).tiny,
        )
        errors = {
            "local_r32": np.sqrt(residual_r32 / full_norm_sq),
            "local_r80": np.sqrt(residual_r80 / full_norm_sq),
            "global_r32": np.sqrt(residual_global / full_norm_sq),
            "local_r32_centered": np.sqrt(residual_r32 / centered_norm_sq),
            "local_r80_centered": np.sqrt(residual_r80 / centered_norm_sq),
        }
        coefficients.append(coeff_r80.astype(np.float32))
        if case.split == "train":
            train_coefficients.append(coeff_r80)
        for metric in ("local_r32", "local_r80", "global_r32"):
            finite = errors[metric][np.isfinite(errors[metric])]
            split_errors[case.split][metric].extend(finite.tolist())
        for local_index, time_value in enumerate(case.times):
            snapshot_records.append(
                {
                    "Re_label": case.label,
                    "Re": case.re_value,
                    "source_regime": case.source_regime,
                    "target_regime": case.target_regime,
                    "split": case.split,
                    "local_snapshot_index": local_index,
                    "time": float(time_value),
                    **{key: float(values[local_index]) for key, values in errors.items()},
                }
            )
        print(
            f"[{timer.stamp()}] project {case.target_regime}/{variable} {case_index}/{len(cases)} {case.label} {case.split}",
            flush=True,
        )

    coeff_all = np.concatenate(coefficients, axis=0)
    coeff_train = np.concatenate(train_coefficients, axis=0)
    scale_mean = coeff_train.mean(axis=0)
    scale_std = coeff_train.std(axis=0)
    scale_std = np.where(scale_std > 1e-12, scale_std, 1.0)
    summary = {
        split: {metric: summarize(np.asarray(values)) for metric, values in metrics.items()}
        for split, metrics in split_errors.items()
    }
    return {
        "coefficients": coeff_all,
        "normalization_mean": scale_mean.astype(np.float64),
        "normalization_std": scale_std.astype(np.float64),
        "snapshot_records": snapshot_records,
        "summary": summary,
    }


def energy_rank(cumulative: Array, threshold: float, max_rank: int) -> int | None:
    candidates = np.flatnonzero(np.asarray(cumulative[:max_rank]) >= threshold)
    return int(candidates[0] + 1) if candidates.size else None


def write_snapshot_index(path: Path, records: list[dict[str, object]], variable: str) -> None:
    fieldnames = [
        "snapshot_id",
        "Re",
        "Re_label",
        "source_regime",
        "target_regime",
        "split",
        "time",
        "local_snapshot_index",
        f"{variable}_local_r32_full_relative_l2",
        f"{variable}_local_r80_full_relative_l2",
        f"{variable}_global_r32_full_relative_l2",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for snapshot_id, row in enumerate(records):
            writer.writerow(
                {
                    "snapshot_id": snapshot_id,
                    "Re": row["Re"],
                    "Re_label": row["Re_label"],
                    "source_regime": row["source_regime"],
                    "target_regime": row["target_regime"],
                    "split": row["split"],
                    "time": row["time"],
                    "local_snapshot_index": row["local_snapshot_index"],
                    f"{variable}_local_r32_full_relative_l2": row["local_r32"],
                    f"{variable}_local_r80_full_relative_l2": row["local_r80"],
                    f"{variable}_global_r32_full_relative_l2": row["global_r32"],
                }
            )


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def save_regime_artifacts(
    output_root: Path,
    regime: str,
    cases: list[Case],
    points: Array,
    weights: Array,
    sqrt_weights: Array,
    mean_uv: Array,
    mean_p: Array,
    velocity_pod: dict[str, Array | float],
    pressure_pod: dict[str, Array | float],
    velocity_projection: dict[str, object],
    pressure_projection: dict[str, object],
    config: dict[str, object],
    input_manifest: dict[str, object],
    timer: Timer,
) -> None:
    regime_dir = output_root / regime
    regime_dir.mkdir(parents=True, exist_ok=True)
    re_values = np.asarray([case.re_value for case in cases], dtype=np.float64)
    re_labels = np.asarray([case.label for case in cases])
    source_regimes = np.asarray([case.source_regime for case in cases])
    splits_by_re = np.asarray([case.split for case in cases])
    snapshot_splits = np.concatenate([np.repeat(case.split, case.n_snapshots) for case in cases])
    snapshot_re_labels = np.concatenate([np.repeat(case.label, case.n_snapshots) for case in cases])
    mean_uv_by_re = np.repeat(mean_uv[None, :], len(cases), axis=0).astype(np.float32)
    mean_p_by_re = np.repeat(mean_p[None, :], len(cases), axis=0).astype(np.float32)

    velocity_path = regime_dir / f"velocity_pod_{regime}.npz"
    pressure_path = regime_dir / f"pressure_pod_{regime}.npz"
    np.savez_compressed(
        velocity_path,
        phi_uv=velocity_pod["raw_modes"],
        phi_uv_weighted=velocity_pod["weighted_modes"],
        coeff_uv=velocity_projection["coefficients"],
        mean_uv_regime=mean_uv.astype(np.float32),
        mean_uv_by_Re=mean_uv_by_re,
        Re_values=re_values,
        Re_labels=re_labels,
        regimes=source_regimes,
        split_by_Re=splits_by_re,
        snapshot_splits=snapshot_splits,
        snapshot_Re_labels=snapshot_re_labels,
        points=points.astype(np.float32),
        point_areas=weights.astype(np.float32),
        sqrt_point_areas=sqrt_weights.astype(np.float32),
        singular_values_uv=velocity_pod["singular_values"],
        cumulative_energy_uv=velocity_pod["cumulative_energy"],
        total_weighted_energy_uv=np.asarray(velocity_pod["total_energy"], dtype=np.float64),
        coeff_train_mean=velocity_projection["normalization_mean"],
        coeff_train_std=velocity_projection["normalization_std"],
        fit_split=np.asarray("train"),
        centering=np.asarray("single_train_only_regime_mean"),
    )
    np.savez_compressed(
        pressure_path,
        phi_p=pressure_pod["raw_modes"],
        phi_p_weighted=pressure_pod["weighted_modes"],
        coeff_p=pressure_projection["coefficients"],
        mean_p_regime=mean_p.astype(np.float32),
        mean_p_by_Re=mean_p_by_re,
        Re_values=re_values,
        Re_labels=re_labels,
        regimes=source_regimes,
        split_by_Re=splits_by_re,
        snapshot_splits=snapshot_splits,
        snapshot_Re_labels=snapshot_re_labels,
        points=points.astype(np.float32),
        point_areas=weights.astype(np.float32),
        sqrt_point_areas=sqrt_weights.astype(np.float32),
        singular_values_p=pressure_pod["singular_values"],
        cumulative_energy_p=pressure_pod["cumulative_energy"],
        total_weighted_energy_p=np.asarray(pressure_pod["total_energy"], dtype=np.float64),
        coeff_train_mean=pressure_projection["normalization_mean"],
        coeff_train_std=pressure_projection["normalization_std"],
        fit_split=np.asarray("train"),
        centering=np.asarray("single_train_only_regime_mean"),
        pressure_gauge=np.asarray("subtract_area_mean_per_snapshot"),
    )
    np.savez_compressed(
        regime_dir / f"normalization_{regime}.npz",
        velocity_coeff_mean=velocity_projection["normalization_mean"],
        velocity_coeff_std=velocity_projection["normalization_std"],
        pressure_coeff_mean=pressure_projection["normalization_mean"],
        pressure_coeff_std=pressure_projection["normalization_std"],
        fit_split=np.asarray("train"),
        train_Re_labels=np.asarray([case.label for case in cases if case.split == "train"]),
    )

    compatibility_dir = regime_dir / "Global_POD_AreaWeighted_L2"
    compatibility_dir.mkdir(parents=True, exist_ok=True)
    hardlink_or_copy(velocity_path, compatibility_dir / "global_velocity_pod_area_weighted_l2.npz")
    hardlink_or_copy(pressure_path, compatibility_dir / "global_pressure_pod_area_weighted_l2.npz")
    np.savez_compressed(
        compatibility_dir / "mesh_l2_point_area_weights.npz",
        points=points.astype(np.float32),
        point_areas=weights.astype(np.float32),
        sqrt_point_areas=sqrt_weights.astype(np.float32),
    )

    write_snapshot_index(
        regime_dir / f"projection_snapshots_velocity_{regime}.csv",
        velocity_projection["snapshot_records"],
        "velocity",
    )
    write_snapshot_index(
        regime_dir / f"projection_snapshots_pressure_{regime}.csv",
        pressure_projection["snapshot_records"],
        "pressure",
    )
    # Compatibility index contains split metadata and matches coeff row order.
    with (compatibility_dir / "pod_snapshot_index.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["snapshot_id", "Re", "Re_label", "regime", "target_regime", "split", "time", "local_snapshot_index"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for snapshot_id, row in enumerate(velocity_projection["snapshot_records"]):
            writer.writerow(
                {
                    "snapshot_id": snapshot_id,
                    "Re": row["Re"],
                    "Re_label": row["Re_label"],
                    "regime": row["source_regime"],
                    "target_regime": regime,
                    "split": row["split"],
                    "time": row["time"],
                    "local_snapshot_index": row["local_snapshot_index"],
                }
            )

    retained_rank = int(config["rank"])
    diagnostics = {
        "schema_version": 1,
        "regime": regime,
        "fit_contract": {
            "fit_split": "train",
            "centering": config["centering"],
            "pressure_gauge": config["pressure_gauge"],
            "validation_or_heldout_used_for_fit": False,
            "train_Re_labels": [case.label for case in cases if case.split == "train"],
            "validation_Re_labels": [case.label for case in cases if case.split == "validation"],
            "heldout_Re_labels": [case.label for case in cases if case.split == "heldout"],
        },
        "counts": {
            split: {
                "Re": sum(case.split == split for case in cases),
                "snapshots": sum(case.n_snapshots for case in cases if case.split == split),
            }
            for split in ("train", "validation", "heldout")
        },
        "velocity_projection_full_field_relative_l2": velocity_projection["summary"],
        "pressure_projection_full_field_relative_l2_gauge_fixed": pressure_projection["summary"],
        "training_only_spectrum": {
            "velocity": {
                "rank_99pct": energy_rank(velocity_pod["cumulative_energy"], 0.99, retained_rank),
                "rank_99p9pct": energy_rank(velocity_pod["cumulative_energy"], 0.999, retained_rank),
                "rank_99p99pct": energy_rank(velocity_pod["cumulative_energy"], 0.9999, retained_rank),
                "captured_at_rank32": float(velocity_pod["cumulative_energy"][int(config["control_rank"]) - 1]),
                "captured_at_retained_rank": float(velocity_pod["cumulative_energy"][retained_rank - 1]),
            },
            "pressure": {
                "rank_99pct": energy_rank(pressure_pod["cumulative_energy"], 0.99, retained_rank),
                "rank_99p9pct": energy_rank(pressure_pod["cumulative_energy"], 0.999, retained_rank),
                "rank_99p99pct": energy_rank(pressure_pod["cumulative_energy"], 0.9999, retained_rank),
                "captured_at_rank32": float(pressure_pod["cumulative_energy"][int(config["control_rank"]) - 1]),
                "captured_at_retained_rank": float(pressure_pod["cumulative_energy"][retained_rank - 1]),
            },
            "rank_selection_used_validation_or_heldout": False,
            "note": "Hopf rank must additionally use training-only oscillatory-mode diagnostics; energy alone is not authoritative.",
        },
        "global_control_disclosure": config["global_control_policy"],
    }
    diagnostics_path = regime_dir / f"pod_projection_diagnostics_{regime}.json"
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "schema_version": 1,
        "method": "train-only regime-specific randomized block POD with lumped point-area L2",
        "regime": regime,
        "rank": config["rank"],
        "control_rank": config["control_rank"],
        "oversampling": config["oversampling"],
        "power_iterations": config["power_iterations"],
        "seed": config["seed"],
        "centering": config["centering"],
        "pressure_gauge": config["pressure_gauge"],
        "elapsed_seconds": timer.elapsed(),
    }
    (compatibility_dir / "pod_area_weighted_l2_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    artifact_paths = [
        velocity_path,
        pressure_path,
        regime_dir / f"normalization_{regime}.npz",
        diagnostics_path,
        compatibility_dir / "pod_snapshot_index.csv",
        compatibility_dir / "pod_area_weighted_l2_metadata.json",
    ]
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "regime": regime,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "split_config_sha256": sha256_json(config),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "inputs": input_manifest,
        "artifacts": {
            path.name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifact_paths
        },
        "leakage_audit": {
            "fit_rows": "train only",
            "basis_uses_validation": False,
            "basis_uses_heldout": False,
            "mean_uses_validation": False,
            "mean_uses_heldout": False,
            "normalization_uses_validation": False,
            "normalization_uses_heldout": False,
        },
    }
    (regime_dir / f"pod_artifact_manifest_{regime}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    velocity_heldout = diagnostics["velocity_projection_full_field_relative_l2"]["heldout"]
    pressure_heldout = diagnostics["pressure_projection_full_field_relative_l2_gauge_fixed"]["heldout"]
    report = f"""# Independent POD build: {regime}

Status: **PASS**

- Centering: `{config['centering']}`
- Pressure gauge: `{config['pressure_gauge']}`
- Fit population: {diagnostics['counts']['train']['Re']} training Re / {diagnostics['counts']['train']['snapshots']} snapshots
- Reporting-only population: {diagnostics['counts']['validation']['Re']} validation Re, {diagnostics['counts']['heldout']['Re']} held-out Re
- Retained rank: {config['rank']}; fixed fair-control rank: {config['control_rank']}
- No validation or held-out field contributed to the mean, basis, spectrum, rank selection, or normalization.

## Held-out full-field relative L2

| Variable | local r32 mean | frozen global r32 mean | local r80 mean |
|---|---:|---:|---:|
| velocity | {velocity_heldout['local_r32']['mean']:.8g} | {velocity_heldout['global_r32']['mean']:.8g} | {velocity_heldout['local_r80']['mean']:.8g} |
| pressure (gauge-fixed) | {pressure_heldout['local_r32']['mean']:.8g} | {pressure_heldout['global_r32']['mean']:.8g} | {pressure_heldout['local_r80']['mean']:.8g} |

The frozen global control is the existing transductive asset: it used all Re-specific means and all 100 Re cases. Its comparison is retained for continuity and is not a leakage-free control.
"""
    (regime_dir / f"POD_BUILD_REPORT_{regime}.md").write_text(report, encoding="utf-8")


def build_regime(
    data_dir: Path,
    global_pod_dir: Path,
    output_root: Path,
    cases: list[Case],
    config: dict[str, object],
    points: Array,
    weights: Array,
    sqrt_weights: Array,
    global_control: dict[str, object],
) -> None:
    regime = cases[0].target_regime
    timer = Timer()
    train_cases = [case for case in cases if case.split == "train"]
    verify_case_points(cases, points)
    mean_uv, mean_p = fit_regime_mean(train_cases, weights, timer)
    n_points = weights.size
    rank = int(config["rank"])
    oversampling = int(config["oversampling"])
    power_iterations = int(config["power_iterations"])
    base_seed = int(config["seed"])
    velocity_loader = make_velocity_loader(mean_uv, sqrt_weights)
    pressure_loader = make_pressure_loader(mean_p, weights, sqrt_weights)
    velocity_pod = randomized_weighted_pod(
        train_cases,
        velocity_loader,
        2 * n_points,
        np.concatenate([sqrt_weights, sqrt_weights]),
        rank,
        oversampling,
        power_iterations,
        base_seed + {"steady": 101, "hopf": 201, "periodic": 301}[regime],
        timer,
        f"{regime}/velocity",
    )
    pressure_pod = randomized_weighted_pod(
        train_cases,
        pressure_loader,
        n_points,
        sqrt_weights,
        rank,
        oversampling,
        power_iterations,
        base_seed + {"steady": 111, "hopf": 211, "periodic": 311}[regime],
        timer,
        f"{regime}/pressure",
    )
    control_rank = int(config["control_rank"])
    velocity_projection = project_all_cases(
        cases,
        "velocity",
        mean_uv,
        np.asarray(velocity_pod["weighted_modes"]),
        control_rank,
        rank,
        weights,
        sqrt_weights,
        global_control,
        timer,
    )
    pressure_projection = project_all_cases(
        cases,
        "pressure",
        mean_p,
        np.asarray(pressure_pod["weighted_modes"]),
        control_rank,
        rank,
        weights,
        sqrt_weights,
        global_control,
        timer,
    )
    input_manifest = {
        "data_dir": str(data_dir),
        "global_pod_dir": str(global_pod_dir),
        "global_velocity_sha256": sha256_file(global_pod_dir / "global_velocity_pod_area_weighted_l2.npz"),
        "global_pressure_sha256": sha256_file(global_pod_dir / "global_pressure_pod_area_weighted_l2.npz"),
        "weights_sha256": sha256_file(global_pod_dir / "mesh_l2_point_area_weights.npz"),
        "raw_case_count": len(cases),
        "raw_case_labels_sha256": sha256_json([case.label for case in cases]),
        "train_case_labels_sha256": sha256_json([case.label for case in train_cases]),
    }
    save_regime_artifacts(
        output_root,
        regime,
        cases,
        points,
        weights,
        sqrt_weights,
        mean_uv,
        mean_p,
        velocity_pod,
        pressure_pod,
        velocity_projection,
        pressure_projection,
        config,
        input_manifest,
        timer,
    )
    print(f"[{timer.stamp()}] completed {regime}: {output_root / regime}", flush=True)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    global_pod_dir = args.global_pod_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    split_config_path = args.split_config.expanduser().resolve()
    config = json.loads(split_config_path.read_text(encoding="utf-8"))
    for key, override in (
        ("rank", args.rank),
        ("control_rank", args.control_rank),
        ("oversampling", args.oversampling),
        ("power_iterations", args.power_iterations),
        ("seed", args.seed),
    ):
        if override is not None:
            config[key] = int(override)
    if int(config["control_rank"]) > int(config["rank"]):
        raise ValueError("control_rank cannot exceed retained rank")
    output_root.mkdir(parents=True, exist_ok=True)

    rows = read_index(global_pod_dir / "pod_snapshot_index.csv")
    grouped = group_index_rows(rows)
    cases_by_regime = {
        regime: resolve_cases(data_dir, grouped, config, regime)
        for regime in args.regimes
    }
    contract_audit = validate_canonical_heldout(config, args.canonical_contract, cases_by_regime)
    (output_root / "split_contract_resolved.json").write_text(
        json.dumps(
            {
                "config": config,
                "config_path": str(split_config_path),
                "config_sha256": sha256_file(split_config_path),
                "canonical_heldout_audit": contract_audit,
                "resolved": {
                    regime: [
                        {
                            "Re_label": case.label,
                            "Re": case.re_value,
                            "source_regime": case.source_regime,
                            "split": case.split,
                            "snapshots": case.n_snapshots,
                        }
                        for case in cases
                    ]
                    for regime, cases in cases_by_regime.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    points, weights, sqrt_weights = load_weights(global_pod_dir)
    global_control = load_global_control(global_pod_dir, weights, sqrt_weights)
    for regime in args.regimes:
        build_regime(
            data_dir,
            global_pod_dir,
            output_root,
            cases_by_regime[regime],
            config,
            points,
            weights,
            sqrt_weights,
            global_control,
        )


if __name__ == "__main__":
    main()
