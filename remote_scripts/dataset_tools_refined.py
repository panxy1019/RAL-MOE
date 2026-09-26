#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np


def re_tag(re_value: float) -> str:
    return ("Re" + f"{re_value:010.6f}").replace(".", "p")


def generate_re_points():
    left = np.linspace(50.0, 80.0, 30, endpoint=False)
    mid = np.linspace(80.0, 100.0, 40, endpoint=False)
    right = np.linspace(100.0, 150.0, 30, endpoint=True)
    values = np.concatenate([left, mid, right]).astype(np.float64)
    if len(values) != 100:
        raise RuntimeError(f"expected 100 Re points, got {len(values)}")
    if not np.all(np.diff(values) > 0):
        raise RuntimeError("Re points are not strictly increasing")
    rounded = [format(x, ".12f") for x in values]
    if len(set(rounded)) != len(rounded):
        raise RuntimeError("Re points are not unique after 12 decimal serialization")
    return values


def ensure_dirs(dataset: Path):
    for rel in [
        "mesh",
        "cases_npz",
        "probes",
        "logs",
        "manifest",
        "pod",
        "work_cases",
        "logs/case_configs",
    ]:
        (dataset / rel).mkdir(parents=True, exist_ok=True)


def write_manifest(dataset: Path):
    ensure_dirs(dataset)
    out = dataset / "manifest" / "re_points_100.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "case_tag", "Re", "nu", "segment"])
        writer.writeheader()
        for i, re_value in enumerate(generate_re_points(), start=1):
            if re_value < 80:
                segment = "left_50_80_endpoint_false"
            elif re_value < 100:
                segment = "hopf_dense_80_100_endpoint_false"
            else:
                segment = "right_100_150_endpoint_true"
            writer.writerow(
                {
                    "index": i,
                    "case_tag": re_tag(float(re_value)),
                    "Re": f"{float(re_value):.12f}",
                    "nu": f"{1.0 / float(re_value):.16g}",
                    "segment": segment,
                }
            )
    return out


def read_manifest(dataset: Path):
    path = dataset / "manifest" / "re_points_refined_31.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("manifest contains no Re rows")
    re_values = np.array([float(r["Re"]) for r in rows], dtype=np.float64)
    if not np.all(np.diff(re_values) > 0):
        raise RuntimeError("manifest Re values are not strictly increasing")
    if len({r["case_tag"] for r in rows}) != len(rows):
        raise RuntimeError("manifest case tags are not unique")
    return rows


def numeric_time_dirs(case_dir: Path):
    dirs = []
    for p in case_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            t = float(p.name)
        except ValueError:
            continue
        if (p / "U").exists() and (p / "p").exists():
            dirs.append((t, p))
    dirs.sort(key=lambda x: x[0])
    return dirs


def _parse_uniform_value(value: str, kind: str, count: int):
    value = value.strip()
    if kind == "vector":
        nums = np.fromstring(value.strip("()"), sep=" ", dtype=np.float64)
        if nums.size != 3:
            raise RuntimeError(f"cannot parse uniform vector: {value!r}")
        return np.tile(nums, (count, 1))
    return np.full((count,), float(value), dtype=np.float64)


def read_internal_field(path: Path, kind: str, expected_count: int | None = None):
    text = path.read_text(errors="replace")
    m = re.search(r"internalField\s+uniform\s+([^;]+);", text, flags=re.S)
    if m:
        if expected_count is None:
            raise RuntimeError(f"{path}: expected_count required for uniform field")
        return _parse_uniform_value(m.group(1), kind, expected_count)
    m = re.search(
        r"internalField\s+nonuniform\s+List<[^>]+>\s+(\d+)\s*\(\s*(.*?)\s*\)\s*;",
        text,
        flags=re.S,
    )
    if not m:
        raise RuntimeError(f"{path}: cannot find internalField")
    count = int(m.group(1))
    body = m.group(2)
    if expected_count is not None and count != expected_count:
        raise RuntimeError(f"{path}: count {count} != expected {expected_count}")
    if kind == "vector":
        entries = re.findall(r"\(([^()]+)\)", body)
        if len(entries) != count:
            raise RuntimeError(f"{path}: vector entries {len(entries)} != {count}")
        arr = np.empty((count, 3), dtype=np.float64)
        for i, entry in enumerate(entries):
            vals = np.fromstring(entry, sep=" ", dtype=np.float64)
            if vals.size != 3:
                raise RuntimeError(f"{path}: bad vector entry {entry!r}")
            arr[i] = vals
        return arr
    arr = np.fromstring(body, sep=" ", dtype=np.float64)
    if arr.size != count:
        raise RuntimeError(f"{path}: scalar entries {arr.size} != {count}")
    return arr


def read_probe_u(path: Path):
    times = []
    values = []
    with path.open(errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split("(")
            try:
                t = float(parts[0].strip())
            except ValueError:
                continue
            vecs = []
            for part in parts[1:]:
                vals = np.fromstring(part.split(")", 1)[0].strip(), sep=" ", dtype=np.float64)
                if vals.size >= 3:
                    vecs.append(vals[:3])
            if vecs:
                times.append(t)
                values.append(vecs)
    if not values:
        raise RuntimeError(f"no probe data in {path}")
    nprobe = len(values[0])
    if any(len(v) != nprobe for v in values):
        raise RuntimeError(f"inconsistent probe count in {path}")
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float32)


def atomic_npz(path: Path, **arrays):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def find_probe_file(case_dir: Path):
    candidates = sorted(case_dir.glob("postProcessing/wakeProbes/*/U"))
    if not candidates:
        candidates = sorted(case_dir.glob("processor0/postProcessing/wakeProbes/*/U"))
    if not candidates:
        raise FileNotFoundError(f"no wakeProbes U file under {case_dir}")
    return candidates[0]


def mesh_metadata(dataset: Path, template_case: Path):
    ensure_dirs(dataset)
    c_path = template_case / "0" / "C"
    v_path = template_case / "0" / "Vc"
    if not c_path.exists() or not v_path.exists():
        raise FileNotFoundError("expected postProcess outputs 0/C and 0/Vc in template case")
    centers = read_internal_field(c_path, "vector").astype(np.float64)
    volumes = read_internal_field(v_path, "scalar", expected_count=centers.shape[0]).astype(np.float64)
    if centers.shape[0] <= 0 or volumes.shape[0] != centers.shape[0]:
        raise RuntimeError("invalid mesh metadata shapes")
    if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
        raise RuntimeError("invalid centers/volumes")
    out = dataset / "mesh" / "mesh_metadata.npz"
    atomic_npz(
        out,
        cellCenters=centers,
        cellVolumes=volumes,
        Nc=np.asarray(centers.shape[0], dtype=np.int64),
        volume_total=np.asarray(float(volumes.sum()), dtype=np.float64),
        source_template=np.asarray(str(template_case)),
    )
    return out


def load_mesh(dataset: Path):
    path = dataset / "mesh" / "mesh_metadata.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path)
    centers = z["cellCenters"]
    volumes = z["cellVolumes"]
    nc = int(z["Nc"])
    if centers.shape != (nc, 3) or volumes.shape != (nc,):
        raise RuntimeError("mesh metadata shape mismatch")
    return centers, volumes, nc


def scan_bad_log(path: Path):
    if not path.exists():
        return {"fatal_count": 0, "nan_count": 0, "floating_point_count": 0}
    text = path.read_text(errors="replace")
    lower = text.lower()
    return {
        "fatal_count": text.count("FOAM FATAL"),
        "nan_count": len(re.findall(r"(?<![A-Za-z])nan(?![A-Za-z])", lower)),
        "floating_point_count": lower.count("floating point exception"),
    }


def count_vtk_paths(root: Path):
    count = 0
    if not root.exists():
        return 0
    for p in root.rglob("*"):
        name = p.name.lower()
        if p.is_dir() and name == "vtk":
            count += 1
        elif p.is_file() and (name.endswith(".vtk") or name.endswith(".vtu") or name.endswith(".vtp")):
            count += 1
    return count


def convert_case(dataset: Path, case_dir: Path, tag: str, re_value: float, nu: float):
    ensure_dirs(dataset)
    _, _, nc = load_mesh(dataset)
    time_dirs = numeric_time_dirs(case_dir)
    if not time_dirs:
        raise RuntimeError(f"{case_dir}: no reconstructed numeric time dirs with U and p")
    times = np.asarray([t for t, _ in time_dirs], dtype=np.float64)
    U = np.empty((len(time_dirs), nc, 2), dtype=np.float32)
    p_arr = np.empty((len(time_dirs), nc), dtype=np.float32)
    for i, (_, tdir) in enumerate(time_dirs):
        u_full = read_internal_field(tdir / "U", "vector", expected_count=nc)
        p_full = read_internal_field(tdir / "p", "scalar", expected_count=nc)
        U[i, :, :] = u_full[:, :2].astype(np.float32)
        p_arr[i, :] = p_full.astype(np.float32)
    probe_file = find_probe_file(case_dir)
    probe_time, probe_U = read_probe_u(probe_file)
    metadata = {
        "case_tag": tag,
        "Re": float(re_value),
        "nu": float(nu),
        "source_case": str(case_dir),
        "n_snapshots": int(len(times)),
        "n_cells": int(nc),
        "n_probe_samples": int(len(probe_time)),
        "probe_file": str(probe_file),
        "field_dtype": "float32",
        "created_by": "dataset_tools.py convert_case",
    }
    snap_path = dataset / "cases_npz" / f"snapshots_{tag}.npz"
    probe_path = dataset / "probes" / f"probe_{tag}.npz"
    atomic_npz(
        snap_path,
        Re=np.asarray(float(re_value), dtype=np.float64),
        nu=np.asarray(float(nu), dtype=np.float64),
        times=times,
        U=U,
        p=p_arr,
        regime_placeholder=np.asarray("UNLABELED"),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    atomic_npz(
        probe_path,
        Re=np.asarray(float(re_value), dtype=np.float64),
        nu=np.asarray(float(nu), dtype=np.float64),
        probe_time=probe_time,
        probe_U=probe_U,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    metrics = validate_case(dataset, tag, re_value, write_metrics=False)
    metrics.update(scan_bad_log(case_dir / "run.log"))
    metrics["vtk_count_before_cleanup"] = count_vtk_paths(case_dir)
    metrics["source_case"] = str(case_dir)
    (dataset / "logs" / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def validate_case(dataset: Path, tag: str, re_value: float | None = None, write_metrics: bool = True):
    _, _, nc = load_mesh(dataset)
    snap_path = dataset / "cases_npz" / f"snapshots_{tag}.npz"
    probe_path = dataset / "probes" / f"probe_{tag}.npz"
    if not snap_path.exists() or not probe_path.exists():
        raise FileNotFoundError(f"missing npz/probe for {tag}")
    s = np.load(snap_path)
    q = np.load(probe_path)
    times = s["times"]
    U = s["U"]
    p = s["p"]
    probe_time = q["probe_time"]
    probe_U = q["probe_U"]
    if U.ndim != 3 or U.shape[1:] != (nc, 2):
        raise RuntimeError(f"{tag}: U shape {U.shape} incompatible with Nc={nc}")
    if p.shape != (U.shape[0], nc):
        raise RuntimeError(f"{tag}: p shape {p.shape} incompatible with U")
    if times.shape != (U.shape[0],):
        raise RuntimeError(f"{tag}: times shape mismatch")
    if times.size < 2 or float(times[0]) > 1e-12 or abs(float(times[-1]) - 500.0) > 1e-8:
        raise RuntimeError(f"{tag}: time range invalid")
    if not np.all(np.diff(times) > 0):
        raise RuntimeError(f"{tag}: snapshot times not strictly increasing")
    if probe_time.size < 2 or abs(float(probe_time[0])) > 1e-12 or abs(float(probe_time[-1]) - 500.0) > 1e-8:
        raise RuntimeError(f"{tag}: probe time range invalid")
    if probe_U.ndim != 3 or probe_U.shape[0] != probe_time.shape[0] or probe_U.shape[2] != 3:
        raise RuntimeError(f"{tag}: probe_U shape invalid {probe_U.shape}")
    for name, arr in [("U", U), ("p", p), ("probe_U", probe_U)]:
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{tag}: non-finite values in {name}")
    re_npz = float(s["Re"])
    if re_value is not None and abs(re_npz - float(re_value)) > 5e-10:
        raise RuntimeError(f"{tag}: Re mismatch {re_npz} vs {re_value}")
    metrics = {
        "case_tag": tag,
        "Re": re_npz,
        "nu": float(s["nu"]),
        "status": "ok",
        "n_snapshots": int(U.shape[0]),
        "n_cells": int(nc),
        "time_min": float(times[0]),
        "time_max": float(times[-1]),
        "n_probe_samples": int(probe_time.shape[0]),
        "probe_time_min": float(probe_time[0]),
        "probe_time_max": float(probe_time[-1]),
        "snapshot_npz_bytes": int(snap_path.stat().st_size),
        "probe_npz_bytes": int(probe_path.stat().st_size),
    }
    if write_metrics:
        (dataset / "logs" / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def load_all_metrics(dataset: Path):
    metrics = []
    for row in read_manifest(dataset):
        p = dataset / "logs" / f"{row['case_tag']}_metrics.json"
        if p.exists():
            try:
                metrics.append(json.loads(p.read_text()))
            except Exception as exc:
                metrics.append({"case_tag": row["case_tag"], "Re": float(row["Re"]), "status": "bad_metrics", "error": str(exc)})
        else:
            metrics.append({"case_tag": row["case_tag"], "Re": float(row["Re"]), "status": "missing"})
    return metrics


def iter_case_npz(dataset: Path):
    for row in read_manifest(dataset):
        tag = row["case_tag"]
        path = dataset / "cases_npz" / f"snapshots_{tag}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        yield row, path


def compute_means(dataset: Path):
    _, _, nc = load_mesh(dataset)
    sum_u = np.zeros((nc, 2), dtype=np.float64)
    sum_p = np.zeros((nc,), dtype=np.float64)
    total = 0
    case_offsets = []
    all_times = []
    tags = []
    for row, path in iter_case_npz(dataset):
        z = np.load(path)
        U = z["U"]
        p = z["p"]
        n = U.shape[0]
        sum_u += U.astype(np.float64).sum(axis=0)
        sum_p += p.astype(np.float64).sum(axis=0)
        case_offsets.append((row["case_tag"], total, total + n))
        total += n
        all_times.extend([float(x) for x in z["times"]])
        tags.extend([row["case_tag"]] * n)
    if total <= 0:
        raise RuntimeError("no snapshots for POD")
    return sum_u / total, sum_p / total, total, case_offsets, np.asarray(all_times, dtype=np.float64), np.asarray(tags)


def fill_weighted_matrix(dataset: Path, field: str, mean, weights, total_snapshots: int, feature_count: int, tmp_path: Path):
    X = np.memmap(tmp_path, dtype="float32", mode="w+", shape=(total_snapshots, feature_count))
    row0 = 0
    total_energy = 0.0
    for _, path in iter_case_npz(dataset):
        z = np.load(path)
        if field == "velocity":
            data = z["U"].astype(np.float64) - mean[None, :, :]
            flat = data.reshape(data.shape[0], -1)
        else:
            flat = z["p"].astype(np.float64) - mean[None, :]
        flat *= weights[None, :]
        n = flat.shape[0]
        X[row0 : row0 + n, :] = flat.astype(np.float32)
        total_energy += float(np.sum(flat * flat))
        row0 += n
    X.flush()
    return X, total_energy


def _x_dot_omega(X, omega, chunk_rows=256):
    m = X.shape[0]
    y = np.zeros((m, omega.shape[1]), dtype=np.float64)
    om = omega.astype(np.float64, copy=False)
    for i in range(0, m, chunk_rows):
        y[i : i + chunk_rows] = X[i : i + chunk_rows].astype(np.float64) @ om
    return y


def _xt_dot_q(X, q, chunk_rows=256):
    out = np.zeros((X.shape[1], q.shape[1]), dtype=np.float64)
    for i in range(0, X.shape[0], chunk_rows):
        out += X[i : i + chunk_rows].astype(np.float64).T @ q[i : i + chunk_rows]
    return out


def _qt_dot_x(q, X, chunk_rows=256):
    out = np.zeros((q.shape[1], X.shape[1]), dtype=np.float64)
    for i in range(0, X.shape[0], chunk_rows):
        out += q[i : i + chunk_rows].T @ X[i : i + chunk_rows].astype(np.float64)
    return out


def randomized_svd_memmap(X, total_energy: float, max_modes: int, oversample: int = 24, n_iter: int = 1, seed: int = 20260708):
    m, f = X.shape
    l = min(max_modes + oversample, m, f)
    k = min(max_modes, l)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((f, l)).astype(np.float32)
    y = _x_dot_omega(X, omega)
    for _ in range(n_iter):
        q, _ = np.linalg.qr(y, mode="reduced")
        z = _xt_dot_q(X, q)
        y = _x_dot_omega(X, z)
    q, _ = np.linalg.qr(y, mode="reduced")
    b = _qt_dot_x(q, X)
    uhat, s, vt = np.linalg.svd(b, full_matrices=False)
    s = s[:k]
    vt = vt[:k]
    coeff = (q @ uhat[:, :k]) * s[None, :]
    cumulative = np.cumsum(s * s) / total_energy if total_energy > 0 else np.zeros_like(s)
    return s.astype(np.float64), vt.astype(np.float32), coeff.astype(np.float32), cumulative.astype(np.float64)


def rank_for(cumulative, threshold):
    idx = np.where(cumulative >= threshold)[0]
    if idx.size:
        return str(int(idx[0] + 1))
    return f">{len(cumulative)}"


def build_one_pod(dataset: Path, field: str, mean, weights, total_snapshots: int, snapshot_times, snapshot_tags, case_offsets, max_modes: int):
    pod_dir = dataset / "pod"
    pod_dir.mkdir(parents=True, exist_ok=True)
    feature_count = int(weights.shape[0])
    tmp_path = pod_dir / f"_tmp_{field}_weighted_matrix.dat"
    X, total_energy = fill_weighted_matrix(dataset, field, mean, weights, total_snapshots, feature_count, tmp_path)
    s, weighted_modes, coeff, cumulative = randomized_svd_memmap(X, total_energy, max_modes=max_modes)
    del X
    try:
        tmp_path.unlink()
    except FileNotFoundError:
        pass
    unweighted_modes = weighted_modes / weights[None, :].astype(np.float32)
    out = pod_dir / f"weighted_pod_{field}.npz"
    atomic_npz(
        out,
        singular_values=s,
        cumulative_energy=cumulative,
        modes=unweighted_modes.astype(np.float32),
        weighted_modes=weighted_modes.astype(np.float32),
        coefficients=coeff,
        mean=mean.astype(np.float32),
        weights=weights.astype(np.float32),
        total_energy=np.asarray(total_energy, dtype=np.float64),
        centered=np.asarray(True),
        randomized=np.asarray(True),
        snapshot_times=snapshot_times,
        snapshot_case_tags=snapshot_tags,
        case_offsets=np.asarray(json.dumps(case_offsets)),
        max_modes=np.asarray(max_modes, dtype=np.int64),
    )
    return {
        "field": field,
        "total_snapshots": int(total_snapshots),
        "features": feature_count,
        "total_energy": float(total_energy),
        "stored_modes": int(len(s)),
        "captured_at_stored_modes": float(cumulative[-1]) if len(cumulative) else 0.0,
        "rank_90": rank_for(cumulative, 0.90),
        "rank_95": rank_for(cumulative, 0.95),
        "rank_99": rank_for(cumulative, 0.99),
        "rank_999": rank_for(cumulative, 0.999),
        "output": str(out),
    }


def build_pod(dataset: Path, max_modes: int = 512):
    _, volumes, _ = load_mesh(dataset)
    mean_u, mean_p, total, case_offsets, snapshot_times, snapshot_tags = compute_means(dataset)
    sqrt_v = np.sqrt(volumes.astype(np.float64))
    reports = [
        build_one_pod(dataset, "velocity", mean_u, np.repeat(sqrt_v, 2).astype(np.float32), total, snapshot_times, snapshot_tags, case_offsets, max_modes),
        build_one_pod(dataset, "pressure", mean_p, sqrt_v.astype(np.float32), total, snapshot_times, snapshot_tags, case_offsets, max_modes),
    ]
    out = dataset / "pod" / "pod_energy_report.csv"
    with out.open("w", newline="") as f:
        fieldnames = [
            "field",
            "total_snapshots",
            "features",
            "total_energy",
            "stored_modes",
            "captured_at_stored_modes",
            "rank_90",
            "rank_95",
            "rank_99",
            "rank_999",
            "output",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in reports:
            writer.writerow(row)
    return out


def bytes_human(n):
    n = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def write_summary(dataset: Path):
    rows = read_manifest(dataset)
    metrics = {m.get("case_tag"): m for m in load_all_metrics(dataset)}
    vtk_residuals = []
    for p in dataset.rglob("*"):
        name = p.name.lower()
        if (p.is_dir() and name == "vtk") or (p.is_file() and (name.endswith(".vtk") or name.endswith(".vtu") or name.endswith(".vtp"))):
            vtk_residuals.append(str(p))
    failed = []
    fatal_total = 0
    nan_total = 0
    lines = [
        "# Refined Re95-102 N31 NPZ Dataset Summary",
        "",
        f"- Dataset root: `{dataset}`",
        "- Re formula: `concat(np.arange(95.00,95.60+eps,0.05), np.arange(95.75,98.00+eps,0.25), np.arange(98.50,102.00+eps,0.50))`",
        "- Solver: OpenFOAM 13 `icoFoam -parallel`, `3` MPI ranks per case, max `5` concurrent cases",
        "- Numerics: `CrankNicolson 0.9` and previously verified compatible graded mesh",
        "- VTK policy: `foamToVTK` not executed; no VTK directory is expected inside the dataset",
        "",
        "## Re Points",
        "",
        ", ".join([f"{float(r['Re']):.12g}" for r in rows]),
        "",
        "## Case Status",
        "",
        "| # | tag | Re | status | probe samples | snapshots | npz size | fatal | nan |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        tag = r["case_tag"]
        m = metrics.get(tag, {"status": "missing"})
        status = m.get("status", "missing")
        if status != "ok":
            failed.append(tag)
        fatal = int(m.get("fatal_count", 0) or 0)
        nan = int(m.get("nan_count", 0) or 0)
        fatal_total += fatal
        nan_total += nan
        snap_bytes = int(m.get("snapshot_npz_bytes", 0) or 0)
        probe_bytes = int(m.get("probe_npz_bytes", 0) or 0)
        lines.append(
            f"| {r['index']} | `{tag}` | {float(r['Re']):.12g} | {status} | "
            f"{int(m.get('n_probe_samples', 0) or 0)} | {int(m.get('n_snapshots', 0) or 0)} | "
            f"{bytes_human(snap_bytes + probe_bytes)} | {fatal} | {nan} |"
        )
    lines += [
        "",
        "## Failure And Log Scan",
        "",
        f"- Failed cases: `{', '.join(failed) if failed else 'none'}`",
        f"- FOAM FATAL count across case logs: `{fatal_total}`",
        f"- nan token count across case logs: `{nan_total}`",
        f"- VTK residual files/directories inside dataset: `{len(vtk_residuals)}`",
    ]
    if vtk_residuals:
        for p in vtk_residuals[:20]:
            lines.append(f"  - `{p}`")
    lines += ["", "## Weighted POD Energy", ""]
    pod_report = dataset / "pod" / "pod_energy_report.csv"
    if pod_report.exists():
        with pod_report.open(newline="") as f:
            pod_rows = list(csv.DictReader(f))
        lines.append("| field | snapshots | features | stored modes | captured | rank 90 | rank 95 | rank 99 | rank 99.9 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for pr in pod_rows:
            lines.append(
                f"| {pr['field']} | {pr['total_snapshots']} | {pr['features']} | {pr['stored_modes']} | "
                f"{float(pr['captured_at_stored_modes']):.6f} | {pr['rank_90']} | {pr['rank_95']} | {pr['rank_99']} | {pr['rank_999']} |"
            )
    else:
        lines.append("POD report not generated.")
    out = dataset / "RUN_SUMMARY.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--dataset", type=Path, required=True)
    p = sub.add_parser("mesh-metadata")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--template-case", type=Path, required=True)
    p = sub.add_parser("convert")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--case-dir", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--re", type=float, required=True)
    p.add_argument("--nu", type=float, required=True)
    p = sub.add_parser("validate-case")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--re", type=float, default=None)
    p = sub.add_parser("build-pod")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--max-modes", type=int, default=512)
    p = sub.add_parser("summary")
    p.add_argument("--dataset", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "manifest":
        print(write_manifest(args.dataset))
    elif args.cmd == "mesh-metadata":
        print(mesh_metadata(args.dataset, args.template_case))
    elif args.cmd == "convert":
        print(json.dumps(convert_case(args.dataset, args.case_dir, args.tag, args.re, args.nu), indent=2, sort_keys=True))
    elif args.cmd == "validate-case":
        print(json.dumps(validate_case(args.dataset, args.tag, args.re), indent=2, sort_keys=True))
    elif args.cmd == "build-pod":
        print(build_pod(args.dataset, args.max_modes))
    elif args.cmd == "summary":
        print(write_summary(args.dataset))


if __name__ == "__main__":
    main()
