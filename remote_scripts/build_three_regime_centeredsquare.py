#!/usr/bin/env python3
"""Create overlapping, Re-disjoint three-regime CenteredSquare POD datasets."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


SOURCE_ROOT = Path("/home/ray/Desktop/centeredSquare")
FORMAL = SOURCE_ROOT / "dataset_Re50_150_N100_npz"
REFINED = SOURCE_ROOT / "dataset_Re95_102_refined_N31_npz"
OUTPUT = SOURCE_ROOT / "three_regime_overlap_v1"

VALIDATION = {
    "steady": {55.0, 75.0, 90.0, 94.5, 95.25},
    "hopf": {94.5, 95.25, 95.5, 97.5, 99.0, 101.5},
    "periodic": {99.0, 101.5, 110.344827586, 125.862068966, 141.379310345, 150.0},
}
HELDOUT = {
    "steady": {60.0, 85.0, 95.1, 95.3},
    "hopf": {95.1, 95.3, 96.5, 100.5, 102.0},
    "periodic": {100.5, 102.0, 120.689655172, 144.827586207},
}


def re_tag(value: float) -> str:
    return ("Re" + f"{value:010.6f}").replace(".", "p")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close_to(value: float, values: set[float]) -> bool:
    return any(abs(value - item) < 5e-7 for item in values)


def source_regime(value: float) -> str:
    if value <= 95.30 + 1e-9:
        return "steady"
    if value <= 102.0 + 1e-9:
        return "hopf"
    return "periodic"


def belongs(value: float, regime: str) -> bool:
    if regime == "steady":
        return value <= 95.40 + 1e-9
    if regime == "hopf":
        return 94.0 - 1e-9 <= value <= 102.0 + 1e-9
    if regime == "periodic":
        return value >= 98.5 - 1e-9
    raise ValueError(regime)


def scan_cases(dataset: Path, source_name: str) -> dict[float, dict]:
    records = {}
    for path in sorted((dataset / "cases_npz").glob("snapshots_*.npz")):
        with np.load(path, allow_pickle=False) as z:
            value = float(z["Re"])
            times = np.asarray(z["times"])
            u_shape = list(np.asarray(z["U"]).shape)
            p_shape = list(np.asarray(z["p"]).shape)
        records[round(value, 12)] = {
            "Re": value,
            "tag": path.stem.removeprefix("snapshots_"),
            "path": path,
            "source_dataset": dataset.name,
            "source_name": source_name,
            "shape": {"times": list(times.shape), "U": u_shape, "p": p_shape},
            "time_min": float(times[0]),
            "time_max": float(times[-1]),
            "sha256": sha256(path),
        }
    return records


def canonical_records() -> tuple[list[dict], list[dict]]:
    formal = scan_cases(FORMAL, "formal")
    refined = scan_cases(REFINED, "refined")
    selected = dict(formal)
    duplicate_audit = []
    for key, record in refined.items():
        if key in formal:
            old = formal[key]
            if old["shape"] != record["shape"] or old["time_min"] != record["time_min"] or old["time_max"] != record["time_max"]:
                raise RuntimeError(f"incompatible duplicate Re={record['Re']}")
            duplicate_audit.append(
                {
                    "Re": record["Re"],
                    "formal_sha256": old["sha256"],
                    "refined_sha256": record["sha256"],
                    "selected_source": "refined",
                }
            )
        selected[key] = record
    records = [selected[key] for key in sorted(selected)]
    if len(records) != 120:
        raise RuntimeError(f"expected 120 unique Re cases, found {len(records)}")
    return records, duplicate_audit


def ensure_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target):
            return
        raise RuntimeError(f"refusing to overwrite existing different file: {target}")
    os.link(source, target)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def randomized_svd(x: np.memmap, total_energy: float, max_modes: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, features = x.shape
    sample = min(max_modes + 24, rows, features)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((features, sample), dtype=np.float32)
    y = np.zeros((rows, sample), dtype=np.float64)
    for start in range(0, rows, 256):
        y[start : start + 256] = x[start : start + 256].astype(np.float64) @ omega
    q, _ = np.linalg.qr(y, mode="reduced")
    b = np.zeros((sample, features), dtype=np.float64)
    for start in range(0, rows, 256):
        b += q[start : start + 256].T @ x[start : start + 256].astype(np.float64)
    u, singular, vt = np.linalg.svd(b, full_matrices=False)
    keep = min(max_modes, singular.size)
    singular = singular[:keep]
    modes = vt[:keep]
    coeff = (q @ u[:, :keep]) * singular[None, :]
    cumulative = np.cumsum(singular * singular) / total_energy
    return singular, modes, coeff, cumulative


def rank_for(cumulative: np.ndarray, threshold: float) -> int:
    hits = np.flatnonzero(cumulative >= threshold)
    if not len(hits):
        raise RuntimeError(f"retained spectrum did not reach {threshold:.4f}; increase analysis rank")
    return int(hits[0] + 1)


def load_case(record: dict, volumes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(record["path"], allow_pickle=False) as z:
        times = np.asarray(z["times"], dtype=np.float64)
        u = np.asarray(z["U"], dtype=np.float32)
        p = np.asarray(z["p"], dtype=np.float32)
    if u.ndim != 3 or u.shape[1:] != (len(volumes), 2) or p.shape != u.shape[:2]:
        raise RuntimeError(f"invalid field shapes for Re={record['Re']}")
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(p)) and np.all(np.diff(times) > 0)):
        raise RuntimeError(f"non-finite or non-monotone data for Re={record['Re']}")
    # Pressure is gauge-fixed independently for every snapshot before train-only centering.
    p = p - (p.astype(np.float64) @ volumes / volumes.sum()).astype(np.float32)[:, None]
    return times, u, p


def build_field_pod(regime_dir: Path, records: list[dict], volumes: np.ndarray, mean: np.ndarray, field: str, rank_cap: int) -> dict:
    counts = []
    total = 0
    for record in records:
        with np.load(record["path"], allow_pickle=False) as z:
            n = int(np.asarray(z["times"]).size)
        counts.append(n)
        total += n
    root = regime_dir / "pod"
    root.mkdir(parents=True, exist_ok=True)
    features = len(volumes) * (2 if field == "velocity" else 1)
    cap = min(rank_cap, total, features)
    matrix_path = root / f"_tmp_{field}_weighted.dat"
    x = np.memmap(matrix_path, dtype="float32", mode="w+", shape=(total, features))
    sqrt_v = np.sqrt(volumes)
    weights = np.repeat(sqrt_v, 2) if field == "velocity" else sqrt_v
    times_all, tags_all, offsets = [], [], []
    row = 0
    energy = 0.0
    for record, count in zip(records, counts):
        times, u, p = load_case(record, volumes)
        data = (u - mean[None, :, :]).reshape(count, -1) if field == "velocity" else p - mean[None, :]
        weighted = data.astype(np.float64) * weights[None, :]
        x[row : row + count] = weighted.astype(np.float32)
        energy += float(np.sum(weighted * weighted))
        offsets.append([record["tag"], row, row + count])
        times_all.extend(times.tolist())
        tags_all.extend([record["tag"]] * count)
        row += count
    x.flush()
    singular, weighted_modes, coefficients, cumulative = randomized_svd(x, energy, cap, seed=20260724)
    del x
    matrix_path.unlink(missing_ok=True)
    r99, r999 = rank_for(cumulative, 0.99), rank_for(cumulative, 0.999)
    keep = max(r99, r999)
    modes = weighted_modes[:keep] / weights[None, :]
    out = root / ("weighted_pod_velocity.npz" if field == "velocity" else "weighted_pod_pressure.npz")
    np.savez_compressed(
        out,
        singular_values=singular[:keep].astype(np.float64),
        cumulative_energy=cumulative[:keep].astype(np.float64),
        modes=modes.astype(np.float32),
        weighted_modes=weighted_modes[:keep].astype(np.float32),
        coefficients=coefficients[:, :keep].astype(np.float32),
        mean=mean.astype(np.float32),
        weights=weights.astype(np.float32),
        total_energy=np.asarray(energy, dtype=np.float64),
        centered=np.asarray(True),
        pressure_gauge=np.asarray("subtract_volume_mean_per_snapshot" if field == "pressure" else "not_applicable"),
        snapshot_times=np.asarray(times_all, dtype=np.float64),
        snapshot_case_tags=np.asarray(tags_all),
        case_offsets=np.asarray(json.dumps(offsets)),
        analysis_rank=np.asarray(cap, dtype=np.int64),
        rank_99=np.asarray(r99, dtype=np.int64),
        rank_999=np.asarray(r999, dtype=np.int64),
    )
    return {"field": field, "path": str(out), "rank_99": r99, "rank_999": r999, "stored_modes": keep, "analysis_rank": cap, "captured": float(cumulative[keep - 1])}


def validation_projection(records: list[dict], volumes: np.ndarray, upod: Path, ppod: Path) -> dict:
    uz, pz = np.load(upod, allow_pickle=False), np.load(ppod, allow_pickle=False)
    um, pm = uz["mean"], pz["mean"]
    phi_u, phi_p = uz["modes"], pz["modes"]
    w_u = np.repeat(volumes, 2)
    report = []
    for record in records:
        _, u, p = load_case(record, volumes)
        du = (u - um[None, :, :]).reshape(u.shape[0], -1)
        dp = p - pm[None, :]
        cu = (du * w_u[None, :]) @ phi_u.T
        cp = (dp * volumes[None, :]) @ phi_p.T
        ru = cu @ phi_u
        rp = cp @ phi_p
        eu = np.sqrt(np.sum((du - ru) ** 2 * w_u[None, :]) / np.sum(du**2 * w_u[None, :]))
        ep = np.sqrt(np.sum((dp - rp) ** 2 * volumes[None, :]) / np.sum(dp**2 * volumes[None, :]))
        report.append({"Re": record["Re"], "tag": record["tag"], "velocity_weighted_rel_l2": float(eu), "pressure_weighted_rel_l2": float(ep), "finite": bool(np.isfinite(cu).all() and np.isfinite(cp).all())})
    return {"cases": report}


def write_regime(regime: str, all_records: list[dict], mesh: Path, vtk_dir: Path) -> dict:
    regime_dir = OUTPUT / "subsets" / regime
    records = [item for item in all_records if belongs(item["Re"], regime)]
    train, validation, heldout = [], [], []
    for item in records:
        value = item["Re"]
        if close_to(value, VALIDATION[regime]):
            validation.append(item)
        elif close_to(value, HELDOUT[regime]):
            heldout.append(item)
        else:
            train.append(item)
    if not train or not validation or not heldout:
        raise RuntimeError(f"empty split in {regime}")
    for item in train:
        ensure_link(item["path"], regime_dir / "cases_npz" / item["path"].name)
    ensure_link(mesh, regime_dir / "mesh" / "mesh_metadata.npz")
    ref_target = regime_dir / "reference_vtk"
    ref_target.mkdir(parents=True, exist_ok=True)
    for source in vtk_dir.iterdir():
        if source.is_file():
            ensure_link(source, ref_target / source.name)
    rows = []
    for index, item in enumerate(train, start=1):
        rows.append({"index": index, "case_tag": item["tag"], "Re": f"{item['Re']:.12f}", "nu": f"{1.0 / item['Re']:.16g}", "segment": f"target_{regime}_train", "source_regime": source_regime(item["Re"]), "source_dataset": item["source_dataset"]})
    manifest = regime_dir / "manifest" / "re_points_100.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meshz = np.load(mesh, allow_pickle=False)
    volumes = np.asarray(meshz["cellVolumes"], dtype=np.float64)
    sum_u = np.zeros((len(volumes), 2), dtype=np.float64)
    sum_p = np.zeros(len(volumes), dtype=np.float64)
    snapshots = 0
    for item in train:
        _, u, p = load_case(item, volumes)
        sum_u += u.astype(np.float64).sum(axis=0)
        sum_p += p.astype(np.float64).sum(axis=0)
        snapshots += u.shape[0]
    upod = build_field_pod(regime_dir, train, volumes, sum_u / snapshots, "velocity", rank_cap=256)
    ppod = build_field_pod(regime_dir, train, volumes, sum_p / snapshots, "pressure", rank_cap=256)
    projection = validation_projection(validation, volumes, Path(upod["path"]), Path(ppod["path"]))
    write_json(regime_dir / "pod" / "validation_projection.json", projection)
    with (regime_dir / "pod" / "pod_energy_report.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["field", "rank_99", "rank_999", "stored_modes", "analysis_rank", "captured"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows([upod, ppod])
    summary = {
        "regime": regime,
        "domain_count": len(records),
        "train_count": len(train),
        "validation_count": len(validation),
        "heldout_count": len(heldout),
        "train_snapshots": snapshots,
        "validation_Re": [x["Re"] for x in validation],
        "heldout_Re": [x["Re"] for x in heldout],
        "pod": [upod, ppod],
    }
    write_json(regime_dir / "RUN_SUMMARY.json", summary)
    (regime_dir / "RUN_SUMMARY.md").write_text("# CenteredSquare regime subset\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n")
    return summary


def main() -> None:
    records, duplicates = canonical_records()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = {
        "contract_name": "centeredSquare_three_regime_train_only_pod_v1",
        "source_regime_boundary": {"steady_max": 95.30, "hopf_max": 102.0, "hopf_onset_estimate": 95.312},
        "specialist_domains": {"steady": [50.0, 95.4], "hopf": [94.0, 102.0], "periodic": [98.5, 150.0]},
        "validation": {key: sorted(value) for key, value in VALIDATION.items()},
        "heldout": {key: sorted(value) for key, value in HELDOUT.items()},
        "pressure_gauge": "subtract_volume_mean_per_snapshot",
        "centering": "single_train_only_regime_mean",
        "pod_rank_policy": "retain_only_rank_99_and_rank_999",
    }
    write_json(OUTPUT / "config" / "regime_split_config.json", config)
    write_json(OUTPUT / "config" / "canonical_cases.json", records)
    write_json(OUTPUT / "config" / "duplicate_audit.json", duplicates)
    mesh = FORMAL / "mesh" / "mesh_metadata.npz"
    vtk_dir = FORMAL / "reference_vtk"
    summaries = [write_regime(name, records, mesh, vtk_dir) for name in ("steady", "hopf", "periodic")]
    write_json(OUTPUT / "config" / "split_contract_resolved.json", {"config": config, "summaries": summaries})
    print(json.dumps({"output": str(OUTPUT), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
