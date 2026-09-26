#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    ROOT,
    case_dir,
    case_name,
    ensure_project_dirs,
    load_config,
    log_path,
    parse_numeric_table,
    re_tag,
)


PARAVIEW_ROOT = ROOT / "data" / "paraview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run/reuse scan_long_viz cases and prepare ParaView classification outputs."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "sweep.yaml")
    parser.add_argument("--nprocs", type=int, default=4)
    parser.add_argument("--max-pimple-end-time", type=float, default=15.0)
    parser.add_argument("--case-timeout-seconds", type=int, default=3600)
    parser.add_argument("--overwrite", action="store_true", help="Force rerun selected scan_long_viz cases.")
    parser.add_argument(
        "--selected-re",
        nargs="*",
        type=float,
        help="Optional explicit Re list. Defaults to Re20, Re150, and Re80/Re100 when available.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_refined_pilot_lookup() -> dict[int, dict[str, str]]:
    path = ROOT / "data" / "manifest" / "pilot_fast_refined_diagnostics.csv"
    lookup: dict[int, dict[str, str]] = {}
    for row in read_csv_rows(path):
        if row.get("Re"):
            lookup[int(round(as_float(row["Re"])))] = row
    return lookup


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_ratio(value: float, baseline: float) -> float:
    if not math.isfinite(value) or not math.isfinite(baseline) or abs(baseline) < 1e-15:
        return math.nan
    return value / baseline


def available_re_values(cfg: dict[str, Any]) -> set[int]:
    values: set[int] = set()
    for key in ["pilot_fast", "scan_long_viz", "pilot_viz", "probe_long"]:
        section = cfg.get(key, {})
        for item in section.get("re_list", section.get("reynolds", [])):
            values.add(int(round(float(item))))
    for row in read_csv_rows(ROOT / "data" / "manifest" / "pilot_fast_refined_diagnostics.csv"):
        if row.get("Re"):
            values.add(int(round(as_float(row["Re"]))))
    for row in read_csv_rows(ROOT / "data" / "manifest" / "pilot_fast_diagnostics.csv"):
        if row.get("Re"):
            values.add(int(round(as_float(row["Re"]))))
    return values


def select_re_values(cfg: dict[str, Any], explicit: list[float] | None) -> list[float]:
    if explicit:
        return sorted({float(x) for x in explicit})

    available = available_re_values(cfg)
    selected: list[int] = []
    if 20 in available:
        selected.append(20)
    else:
        selected.append(20)

    for preferred in [80, 100, 90, 70, 60, 50]:
        if preferred in available and preferred not in selected:
            selected.append(preferred)
        if len([x for x in selected if 50 <= x <= 100]) >= 2:
            break

    if 150 in available and 150 not in selected:
        selected.append(150)
    elif 150 not in selected:
        selected.append(150)

    ordered = [20, 80, 100, 150]
    return [float(x) for x in ordered if x in selected] + [
        float(x) for x in selected if x not in ordered
    ]


def write_runtime_config(cfg: dict[str, Any], selected: list[float], max_pimple_end: float) -> tuple[Path, float, float]:
    import yaml

    runtime = deepcopy(cfg)
    mode_cfg = runtime.setdefault("scan_long_viz", {})
    configured = float(mode_cfg.get("pimpleFoam_endTime", 30.0))
    actual = min(configured, float(max_pimple_end)) if max_pimple_end > 0 else configured
    mode_cfg["re_list"] = selected
    mode_cfg["pimpleFoam_endTime"] = actual
    mode_cfg["keep_vtk"] = True
    mode_cfg["copy_vtk_preview"] = True
    mode_cfg["make_npz"] = False
    mode_cfg["make_pod"] = False
    mode_cfg["make_vtk"] = True

    PARAVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    path = PARAVIEW_ROOT / "scan_long_viz_runtime.yaml"
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(runtime, fh, sort_keys=False)
    return path, configured, actual


def vtk_files_in(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(path.glob("*.vtk"))


def organized_case_root(reynolds: float) -> Path:
    return PARAVIEW_ROOT / f"Re_{re_tag(reynolds)}"


def find_vtk_source(reynolds: float) -> Path | None:
    organized = organized_case_root(reynolds) / "vtk"
    if vtk_files_in(organized):
        return organized
    case_vtk = case_dir(reynolds) / "VTK"
    if vtk_files_in(case_vtk):
        return case_vtk
    preview = ROOT / "data" / "vtk_preview" / f"Re_{re_tag(reynolds)}"
    if vtk_files_in(preview):
        return preview
    return None


def run_scan_long_viz(reynolds: float, runtime_config: Path, nprocs: int, timeout_seconds: int) -> bool:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "01_run_single_re.py"),
        "--re",
        str(reynolds),
        "--mode",
        "scan_long_viz",
        "--config",
        str(runtime_config),
        "--nprocs",
        str(nprocs),
        "--overwrite",
        "--keep-vtk",
        "--copy-vtk-preview",
    ]
    driver_log = log_path(reynolds, "scan_long_viz_driver")
    driver_log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[scan_long_viz] Re={reynolds:g}: running OpenFOAM; log={driver_log}")
    with driver_log.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"# command: {cmd}\n")
        log.write(f"# started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=True,
        )
        try:
            output, _ = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            log.write(f"\n# timeout after {timeout_seconds} seconds; terminating process group\n")
            terminate_process_group(proc)
            return False
        log.write(output or "")
        code = proc.returncode
        log.write(f"\n# exit_code: {code}\n")
    return code == 0


def terminate_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def ensure_foam_entry(reynolds: float, case_root: Path) -> tuple[Path, Path]:
    case_path = case_dir(reynolds)
    actual_foam = case_path / f"{case_name(reynolds)}.foam"
    actual_foam.parent.mkdir(parents=True, exist_ok=True)
    actual_foam.touch()

    openfoam_dir = case_root / "openfoam_case"
    openfoam_dir.mkdir(parents=True, exist_ok=True)
    link_foam = openfoam_dir / actual_foam.name
    if link_foam.exists() or link_foam.is_symlink():
        link_foam.unlink()
    try:
        link_foam.symlink_to(actual_foam)
    except OSError:
        link_foam.touch()
    return actual_foam, link_foam


def copy_vtk_to_paraview(reynolds: float, case_root: Path) -> Path | None:
    source = find_vtk_source(reynolds)
    if source is None:
        return None
    target = case_root / "vtk"
    if source.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    return target


def vtk_time_from_name(path: Path, fallback: float) -> float:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", path.stem)
    if not matches:
        return fallback
    return float(matches[-1])


def convert_vtk_to_vtu(case_root: Path, vtk_dir: Path) -> tuple[str, str, str]:
    try:
        import pyvista as pv
    except Exception as exc:  # noqa: BLE001
        return "", "", f"pyvista unavailable: {exc}"

    vtu_dir = case_root / "vtu"
    if vtu_dir.exists():
        shutil.rmtree(vtu_dir)
    vtu_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[tuple[float, Path]] = []
    for idx, vtk_path in enumerate(vtk_files_in(vtk_dir)):
        try:
            mesh = pv.read(str(vtk_path))
            vtu_path = vtu_dir / f"{vtk_path.stem}.vtu"
            mesh.save(str(vtu_path))
            datasets.append((vtk_time_from_name(vtk_path, float(idx)), vtu_path))
        except Exception as exc:  # noqa: BLE001
            return str(vtu_dir), "", f"VTK->VTU failed for {vtk_path.name}: {exc}"

    if not datasets:
        return str(vtu_dir), "", "no VTK files to convert"

    pvd_path = case_root / f"{case_root.name}.pvd"
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for timestep, vtu_path in sorted(datasets, key=lambda item: item[0]):
        rel = vtu_path.relative_to(case_root).as_posix()
        lines.append(f'    <DataSet timestep="{timestep:.12g}" group="" part="0" file="{rel}"/>')
    lines += ["  </Collection>", "</VTKFile>"]
    pvd_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(vtu_dir), str(pvd_path), "ok"


def mesh_points_and_values(mesh: Any, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    if name in mesh.point_data:
        return np.asarray(mesh.points), np.asarray(mesh.point_data[name])
    if name in mesh.cell_data:
        centers = mesh.cell_centers()
        return np.asarray(centers.points), np.asarray(mesh.cell_data[name])
    return None


def save_scatter_plot(path: Path, points: np.ndarray, values: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    sc = ax.scatter(points[:, 0], points[:, 1], c=values, s=1.2, cmap="viridis", linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_quicklooks(case_root: Path, vtk_dir: Path) -> tuple[str, str]:
    try:
        import pyvista as pv
    except Exception as exc:  # noqa: BLE001
        return "", f"quicklook skipped; pyvista unavailable: {exc}"

    files = vtk_files_in(vtk_dir)
    if not files:
        return "", "quicklook skipped; no VTK files"
    latest = max(files, key=vtk_time_from_name_for_sort)
    quicklook_dir = case_root / "quicklook"
    try:
        mesh = pv.read(str(latest))
        u_data = mesh_points_and_values(mesh, "U")
        if u_data is not None:
            points, u = u_data
            u_mag = np.linalg.norm(u, axis=1) if u.ndim == 2 else np.asarray(u, dtype=float)
            save_scatter_plot(quicklook_dir / "U_magnitude_latest.png", points, u_mag, "|U| latest")
        p_data = mesh_points_and_values(mesh, "p")
        if p_data is not None:
            points, p = p_data
            save_scatter_plot(quicklook_dir / "p_latest.png", points, np.asarray(p).reshape(-1), "p latest")
    except Exception as exc:  # noqa: BLE001
        return str(quicklook_dir), f"quicklook failed: {exc}"
    return str(quicklook_dir), "ok"


def vtk_time_from_name_for_sort(path: Path) -> float:
    return vtk_time_from_name(path, 0.0)


def find_force_files(case_path: Path) -> list[Path]:
    patterns = [
        "postProcessing/forceCoeffs*/**/forceCoeffs.dat",
        "postProcessing/forceCoeffs*/**/coefficient.dat",
        "processor0/postProcessing/forceCoeffs*/**/forceCoeffs.dat",
        "processor0/postProcessing/forceCoeffs*/**/coefficient.dat",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(case_path.glob(pattern))
    return sorted(set(files))


def find_probe_files(case_path: Path) -> list[Path]:
    files = list(case_path.glob("postProcessing/probes*/**/U"))
    files.extend(case_path.glob("processor0/postProcessing/probes*/**/U"))
    return sorted(set(files))


def read_force_series(case_path: Path) -> tuple[list[str], np.ndarray]:
    columns: list[str] = []
    rows: list[list[float]] = []
    for path in find_force_files(case_path):
        cols, values = parse_numeric_table(path)
        if cols:
            columns = cols
        rows.extend(values)
    if not rows:
        return columns, np.empty((0, 0))
    arr = np.asarray(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    _, idx = np.unique(arr[:, 0], return_index=True)
    idx.sort()
    return columns, arr[idx]


def read_probe_v_series(case_path: Path) -> tuple[np.ndarray, np.ndarray]:
    vector_re = re.compile(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)")
    rows: list[tuple[float, list[float]]] = []
    for path in find_probe_files(case_path):
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                t = float(line.split(maxsplit=1)[0])
            except (IndexError, ValueError):
                continue
            vecs = vector_re.findall(line)
            if vecs:
                rows.append((t, [float(v[1]) for v in vecs]))
    if not rows:
        return np.asarray([]), np.empty((0, 0))
    rows.sort(key=lambda item: item[0])
    times = np.asarray([item[0] for item in rows], dtype=float)
    values = np.asarray([item[1] for item in rows], dtype=float)
    _, idx = np.unique(times, return_index=True)
    idx.sort()
    return times[idx], values[idx]


def tail_mask(times: np.ndarray, fraction: float) -> np.ndarray:
    if times.size == 0:
        return np.asarray([], dtype=bool)
    start = float(np.min(times) + (1.0 - fraction) * (np.max(times) - np.min(times)))
    return times >= start


def peak_details(times: np.ndarray, signal: np.ndarray) -> tuple[int, float, float, float, float]:
    if times.size < 5 or signal.size < 5:
        return 0, math.nan, math.nan, math.nan, math.nan
    y = signal - float(np.mean(signal))
    std = float(np.std(y))
    if std <= 0:
        return 0, math.nan, math.nan, math.nan, math.nan
    peak_idx: list[int] = []
    threshold = 0.25 * std
    for i in range(1, y.size - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] > threshold:
            peak_idx.append(i)
    if len(peak_idx) < 3:
        return len(peak_idx), math.nan, math.nan, math.nan, math.nan
    peak_idx = peak_idx[-6:]
    peak_times = times[peak_idx]
    periods = np.diff(peak_times)
    amplitudes = np.abs(y[peak_idx])
    period = float(np.median(periods))
    period_cv = float(np.std(periods) / period) if period > 0 else math.nan
    amp_mean = float(np.mean(amplitudes))
    amp_cv = float(np.std(amplitudes) / amp_mean) if amp_mean > 0 else math.nan
    freq = float(1.0 / period) if period > 0 else math.nan
    return len(peak_idx), period, period_cv, amp_cv, freq


def compute_scan_diagnostics(
    reynolds: float,
    *,
    simple_end: float,
    tail_fraction: float,
    status: str,
    vtk_path: str,
    vtu_path: str,
    pvd_path: str,
    foam_path: str,
) -> dict[str, Any]:
    case_path = case_dir(reynolds)
    columns, force = read_force_series(case_path)
    probe_t, probe_v = read_probe_v_series(case_path)

    row: dict[str, Any] = {
        "Re": float(reynolds),
        "status": status,
        "classification": "no-diagnostics",
        "classification_reason": "No scan diagnostics were readable.",
        "Cd_tail_mean": math.nan,
        "Cd_tail_std": math.nan,
        "Cd_tail_relative_std": math.nan,
        "Cl_tail_mean": math.nan,
        "Cl_tail_std": math.nan,
        "Cl_tail_relative_std": math.nan,
        "probe_v_tail_std_max": math.nan,
        "Cd_std_ratio_to_Re20": math.nan,
        "Cl_std_ratio_to_Re20": math.nan,
        "probe_v_std_ratio_to_Re20": math.nan,
        "num_detected_peaks": 0,
        "peak_period_mean": math.nan,
        "peak_period_cv": math.nan,
        "peak_amplitude_cv": math.nan,
        "estimated_frequency": math.nan,
        "Strouhal": math.nan,
        "pilot_refined_status": "",
        "pilot_Cl_std_ratio_to_Re20": math.nan,
        "pilot_probe_v_std_ratio_to_Re20": math.nan,
        "VTK_path": vtk_path,
        "VTU_path_if_available": vtu_path,
        "PVD_path_if_available": pvd_path,
        "OpenFOAM_foam_path": foam_path,
        "ParaView_open_hint": "Open PVD if available; otherwise open VTK files or the .foam case marker.",
    }

    time_sources: list[np.ndarray] = []
    tail_force = np.empty((0, 0))
    cl_tail = np.asarray([])
    cl_time = np.asarray([])

    if force.size:
        force = force[force[:, 0] >= simple_end - 1e-10]
        if force.size:
            time_sources.append(force[:, 0])
            mask = tail_mask(force[:, 0], tail_fraction)
            tail_force = force[mask]
            cd_idx = columns.index("Cd") if "Cd" in columns else (2 if force.shape[1] > 2 else None)
            cl_idx = columns.index("Cl") if "Cl" in columns else (3 if force.shape[1] > 3 else None)
            if cd_idx is not None and tail_force.size:
                cd_tail = tail_force[:, cd_idx]
                row["Cd_tail_mean"] = float(np.mean(cd_tail))
                row["Cd_tail_std"] = float(np.std(cd_tail))
                if abs(row["Cd_tail_mean"]) > 1e-15:
                    row["Cd_tail_relative_std"] = float(row["Cd_tail_std"] / abs(row["Cd_tail_mean"]))
            if cl_idx is not None and tail_force.size:
                cl_tail = tail_force[:, cl_idx]
                cl_time = tail_force[:, 0]
                row["Cl_tail_mean"] = float(np.mean(cl_tail))
                row["Cl_tail_std"] = float(np.std(cl_tail))
                if abs(row["Cl_tail_mean"]) > 1e-15:
                    row["Cl_tail_relative_std"] = float(row["Cl_tail_std"] / abs(row["Cl_tail_mean"]))

    if probe_t.size and probe_v.size:
        mask = probe_t >= simple_end - 1e-10
        probe_t = probe_t[mask]
        probe_v = probe_v[mask]
        if probe_t.size:
            time_sources.append(probe_t)
            tmask = tail_mask(probe_t, tail_fraction)
            tail_probe = probe_v[tmask]
            if tail_probe.size:
                row["probe_v_tail_std_max"] = float(np.max(np.std(tail_probe, axis=0)))

    if cl_tail.size:
        peak_count, period, period_cv, amp_cv, freq = peak_details(cl_time, cl_tail)
        row["num_detected_peaks"] = peak_count
        row["peak_period_mean"] = period
        row["peak_period_cv"] = period_cv
        row["peak_amplitude_cv"] = amp_cv
        row["estimated_frequency"] = freq
        row["Strouhal"] = freq

    if time_sources:
        row["classification_reason"] = "Diagnostics parsed; classification ratios filled after Re20 baseline."
    return row


def classify_rows(
    rows: list[dict[str, Any]],
    baseline_re: float,
    pilot_lookup: dict[int, dict[str, str]],
) -> None:
    baseline_candidates = [r for r in rows if abs(float(r["Re"]) - baseline_re) < 1e-12]
    if not baseline_candidates:
        return
    baseline = baseline_candidates[0]
    b_cd = as_float(baseline["Cd_tail_std"])
    b_cl = as_float(baseline["Cl_tail_std"])
    b_probe = as_float(baseline["probe_v_tail_std_max"])

    for row in rows:
        pilot = pilot_lookup.get(int(round(float(row["Re"]))), {})
        row["pilot_refined_status"] = pilot.get("refined_status", "")
        row["pilot_Cl_std_ratio_to_Re20"] = as_float(pilot.get("Cl_std_ratio_to_Re20"))
        row["pilot_probe_v_std_ratio_to_Re20"] = as_float(pilot.get("probe_v_std_ratio_to_Re20"))

        cd_std = as_float(row["Cd_tail_std"])
        cl_std = as_float(row["Cl_tail_std"])
        probe_std = as_float(row["probe_v_tail_std_max"])
        row["Cd_std_ratio_to_Re20"] = finite_ratio(cd_std, b_cd)
        row["Cl_std_ratio_to_Re20"] = finite_ratio(cl_std, b_cl)
        row["probe_v_std_ratio_to_Re20"] = finite_ratio(probe_std, b_probe)

        cl_ratio = as_float(row["Cl_std_ratio_to_Re20"])
        probe_ratio = as_float(row["probe_v_std_ratio_to_Re20"])
        peak_count = int(as_float(row["num_detected_peaks"], 0))
        period_cv = as_float(row["peak_period_cv"])
        amp_cv = as_float(row["peak_amplitude_cv"])
        has_stable_period = peak_count >= 3 and math.isfinite(period_cv) and period_cv <= 0.18
        has_stable_amp = math.isfinite(amp_cv) and amp_cv <= 0.35
        pilot_unsteady = pilot.get("refined_status") in {"unsteady-candidate", "periodic-candidate"}
        near_baseline = (
            (not math.isfinite(cl_ratio) or cl_ratio <= 2.5)
            and (not math.isfinite(probe_ratio) or probe_ratio <= 2.5)
        )

        if row["status"] not in {"ok", "reused"}:
            row["classification"] = "no-diagnostics"
            row["classification_reason"] = f"Case status is {row['status']}."
        elif near_baseline and not has_stable_period:
            if pilot_unsteady:
                row["classification"] = "startup-adjustment / relaxation"
                row["classification_reason"] = (
                    "Refined pilot was elevated, but the scan_long_viz tail relaxed near the Re20 baseline "
                    "and did not resolve stable peaks."
                )
            else:
                row["classification"] = "steady-like"
                row["classification_reason"] = "Tail Cl/probe fluctuations stay near the Re20 baseline."
        elif peak_count >= 5 and period_cv <= 0.12 and has_stable_amp and (
            cl_ratio > 10 or probe_ratio > 10
        ):
            row["classification"] = "strong periodic / mature shedding"
            row["classification_reason"] = "Multiple stable peaks plus large tail fluctuation ratios."
        elif has_stable_period and has_stable_amp:
            row["classification"] = "periodic-candidate"
            row["classification_reason"] = "Repeated peaks have reasonably stable periods and amplitudes."
        else:
            row["classification"] = "near-onset / weak-unsteady candidate"
            row["classification_reason"] = (
                "Tail fluctuations exceed baseline, but this scan did not resolve stable repeated peaks."
            )


def write_diagnostics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Re",
        "status",
        "classification",
        "classification_reason",
        "Cd_tail_mean",
        "Cd_tail_std",
        "Cd_tail_relative_std",
        "Cl_tail_mean",
        "Cl_tail_std",
        "Cl_tail_relative_std",
        "probe_v_tail_std_max",
        "Cd_std_ratio_to_Re20",
        "Cl_std_ratio_to_Re20",
        "probe_v_std_ratio_to_Re20",
        "num_detected_peaks",
        "peak_period_mean",
        "peak_period_cv",
        "peak_amplitude_cv",
        "estimated_frequency",
        "Strouhal",
        "pilot_refined_status",
        "pilot_Cl_std_ratio_to_Re20",
        "pilot_probe_v_std_ratio_to_Re20",
        "VTK_path",
        "VTU_path_if_available",
        "PVD_path_if_available",
        "OpenFOAM_foam_path",
        "ParaView_open_hint",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: float(item["Re"])):
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_open_readme(path: Path, rows: list[dict[str, Any]], configured_end: float, actual_end: float) -> None:
    lines = [
        "# Open In ParaView",
        "",
        "These files are for visual regime inspection only. They are not formal ROM training labels.",
        "",
        f"Configured scan_long_viz pimpleFoam_endTime: `{configured_end:g}`.",
        f"Actual automated run cap used here: `{actual_end:g}`.",
        "",
        "## Opening Methods",
        "",
        "1. OpenFOAM native: open the `.foam` file listed in `scan_long_viz_diagnostics.csv`.",
        "2. Legacy VTK: open one or more `.vtk` files under each `Re_*/vtk/` directory.",
        "3. VTU/PVD time series: when conversion succeeded, open `Re_*/Re_*.pvd`.",
        "",
        "## What To Inspect",
        "",
        "- `mag(U)` for wake speed and recirculation shape.",
        "- `p` for pressure recovery and wake asymmetry.",
        "- Vorticity/curl of `U` for alternating shedding signatures.",
        "- Streamlines or glyphs seeded behind the obstacle.",
        "",
        "Expected qualitative reading: Re=20 should show stable recirculation; Re=80/100 may show wake wiggle or near-onset behavior; Re=150 is the strongest shedding candidate from the fast probe scan.",
        "",
        "## Generated Entries",
        "",
        "| Re | preferred file | VTK directory | .foam path |",
        "|---:|---|---|---|",
    ]
    for row in sorted(rows, key=lambda item: float(item["Re"])):
        preferred = row.get("PVD_path_if_available") or row.get("VTK_path") or row.get("OpenFOAM_foam_path")
        lines.append(
            f"| {float(row['Re']):g} | `{preferred}` | `{row.get('VTK_path', '')}` | `{row.get('OpenFOAM_foam_path', '')}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_classification_summary(path: Path, rows: list[dict[str, Any]], notes: list[str]) -> None:
    lines = [
        "# ParaView Scan Classification Summary",
        "",
        "Classification is metadata for choosing the next experiment. It must not be treated as a training label.",
        "",
        "## Result Table",
        "",
        "| Re | classification | scan Cl ratio | scan probe-v ratio | pilot status | pilot probe-v ratio | peaks | period CV | files |",
        "|---:|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda item: float(item["Re"])):
        files = []
        if row.get("VTK_path"):
            files.append("vtk")
        if row.get("VTU_path_if_available"):
            files.append("vtu")
        if row.get("PVD_path_if_available"):
            files.append("pvd")
        if row.get("OpenFOAM_foam_path"):
            files.append("foam")
        lines.append(
            "| {Re:g} | {cls} | {cl:.3g} | {probe:.3g} | {pilot} | {pilot_probe:.3g} | {peaks} | {cv:.3g} | {files} |".format(
                Re=float(row["Re"]),
                cls=row["classification"],
                cl=as_float(row["Cl_std_ratio_to_Re20"]),
                probe=as_float(row["probe_v_std_ratio_to_Re20"]),
                pilot=row.get("pilot_refined_status", ""),
                pilot_probe=as_float(row.get("pilot_probe_v_std_ratio_to_Re20")),
                peaks=int(as_float(row["num_detected_peaks"], 0)),
                cv=as_float(row["peak_period_cv"]),
                files=", ".join(files) if files else "none",
            )
        )
    lines += [
        "",
        "## Criteria",
        "",
        "- steady-like: tail Cl/probe fluctuations stay near the Re20 baseline and no stable repeated peaks are detected.",
        "- startup-adjustment / relaxation: reserved for cases where early transients decay toward baseline within the scan window.",
        "- near-onset / weak-unsteady candidate: tail fluctuations exceed baseline but stable repeated peaks are not resolved.",
        "- periodic-candidate: at least three peaks with reasonably stable period and amplitude.",
        "- strong periodic / mature shedding: at least five stable peaks with large fluctuation ratios.",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_case(
    reynolds: float,
    *,
    runtime_config: Path,
    cfg: dict[str, Any],
    nprocs: int,
    timeout_seconds: int,
    overwrite: bool,
) -> tuple[dict[str, Any], list[str]]:
    case_root = organized_case_root(reynolds)
    case_root.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    existing = find_vtk_source(reynolds)
    status = "reused" if existing and not overwrite else "ok"
    if overwrite or existing is None:
        ok = run_scan_long_viz(reynolds, runtime_config, nprocs, timeout_seconds)
        if not ok:
            status = "run_failed"
            notes.append(f"Re={reynolds:g}: scan_long_viz failed or timed out; outputs may be incomplete.")
        else:
            status = "ok"

    actual_foam, _ = ensure_foam_entry(reynolds, case_root)
    vtk_dir = copy_vtk_to_paraview(reynolds, case_root)
    if vtk_dir is None or not vtk_files_in(vtk_dir):
        notes.append(f"Re={reynolds:g}: no VTK files found after scan.")
        vtk_path = ""
    else:
        vtk_path = str(vtk_dir)

    vtu_path = ""
    pvd_path = ""
    if vtk_path and bool(cfg.get("scan_long_viz", {}).get("convert_vtk_to_vtu", True)):
        vtu_path, pvd_path, convert_note = convert_vtk_to_vtu(case_root, Path(vtk_path))
        if convert_note != "ok":
            notes.append(f"Re={reynolds:g}: {convert_note}")

        quicklook_dir, quicklook_note = generate_quicklooks(case_root, Path(vtk_path))
        if quicklook_note != "ok":
            notes.append(f"Re={reynolds:g}: {quicklook_note}")
        elif quicklook_dir:
            notes.append(f"Re={reynolds:g}: quicklook images written to {quicklook_dir}.")

    mode_cfg = cfg.get("scan_long_viz", {})
    row = compute_scan_diagnostics(
        reynolds,
        simple_end=float(mode_cfg.get("simpleFoam_endTime", 80)),
        tail_fraction=float(mode_cfg.get("tail_fraction", 0.5)),
        status=status if vtk_path else "no-vtk",
        vtk_path=vtk_path,
        vtu_path=vtu_path,
        pvd_path=pvd_path,
        foam_path=str(actual_foam),
    )
    return row, notes


def main() -> None:
    args = parse_args()
    ensure_project_dirs(ROOT)
    PARAVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    selected = select_re_values(cfg, args.selected_re)
    runtime_config, configured_end, actual_end = write_runtime_config(
        cfg,
        selected,
        args.max_pimple_end_time,
    )
    runtime_cfg = load_config(runtime_config)
    notes = [
        f"Auto-selected Re values: {', '.join(f'{x:g}' for x in selected)}.",
        f"Runtime config: {runtime_config}.",
    ]
    if actual_end < configured_end:
        notes.append(
            f"pimpleFoam_endTime was reduced from {configured_end:g} to {actual_end:g} to avoid long CPU burn."
        )

    rows: list[dict[str, Any]] = []
    for reynolds in selected:
        row, case_notes = process_case(
            reynolds,
            runtime_config=runtime_config,
            cfg=runtime_cfg,
            nprocs=args.nprocs,
            timeout_seconds=args.case_timeout_seconds,
            overwrite=args.overwrite,
        )
        rows.append(row)
        notes.extend(case_notes)

    baseline_re = float(runtime_cfg.get("scan_long_viz", {}).get("baseline_Re", 20))
    classify_rows(rows, baseline_re, load_refined_pilot_lookup())

    csv_path = PARAVIEW_ROOT / "scan_long_viz_diagnostics.csv"
    report_path = PARAVIEW_ROOT / "classification_summary.md"
    readme_path = PARAVIEW_ROOT / "README_OPEN_IN_PARAVIEW.md"
    write_diagnostics_csv(csv_path, rows)
    write_classification_summary(report_path, rows, notes)
    write_open_readme(readme_path, rows, configured_end, actual_end)

    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {readme_path}")
    for row in sorted(rows, key=lambda item: float(item["Re"])):
        print(f"Re={float(row['Re']):g}: {row['classification']} ({row['status']})")


if __name__ == "__main__":
    main()
