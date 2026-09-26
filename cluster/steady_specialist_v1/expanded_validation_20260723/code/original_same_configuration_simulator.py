#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import vtk
from scipy.sparse.linalg import eigsh
from vtk.util.numpy_support import vtk_to_numpy


HOME = Path.home()
BRANCH = Path(__file__).resolve().parent
BASE_CASE = BRANCH / "base_case_physics_generalizable"
DEFAULT_ROOT = HOME / "Desktop" / "Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re"
MESH_TMP_CASE = BRANCH / "tmp_area_weight_mesh_case"

NU = 1.0e-3
D = 1.0
RHO = 1.0
DOMAIN_THICKNESS_FALLBACK = 0.1

RE_MIN = 20.0
RE_MAX = 200.0
N_RE = 100
HOPF_RE = 47.0

PERIODIC_TRANSIENT_CYCLES = 5.0
PERIODIC_RETAINED_CYCLES = 20.0
FRAMES_PER_CYCLE = 8
STEADY_SAVE_SNAPSHOTS = 64
STEADY_SAVE_CONVECTIVE_TIMES = 12.0

MAX_CO = 0.8
MAX_DELTA_T = 0.2
MAX_MODES = 80
NPROCS_DEFAULT = min(8, os.cpu_count() or 1)

MIN_FREE_START_GB = 10.0
MIN_FREE_RUN_GB = 2.0
MIN_FREE_POD_GB = 4.0


@dataclass
class CasePlan:
    Re: float
    label: str
    regime: str
    U: float
    estimated_St: float | None
    estimated_period: float | None
    nominal_transient_time: float
    nominal_retained_time: float
    nominal_write_interval: float


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1024**3


def require_free(path: Path, required_gb: float, stage: str) -> None:
    available = free_gb(path)
    if available < required_gb:
        raise RuntimeError(
            f"Not enough free disk for {stage}: {available:.2f} GiB available, "
            f"{required_gb:.2f} GiB required."
        )


def re_label(re_value: float) -> str:
    return f"Re_{re_value:.6f}".replace(".", "p")


def smooth_step(x: np.ndarray, center: float, width: float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh((x - center) / width))


def smooth_box(x: np.ndarray, left: float, right: float, width: float) -> np.ndarray:
    return smooth_step(x, left, width) - smooth_step(x, right, width)


def sampling_density(x: np.ndarray) -> np.ndarray:
    """Continuous density focused on flow-regime changes instead of uniform Re spacing."""
    return (
        0.35
        + 1.20 * smooth_box(x, 20.0, 40.0, 2.2)
        + 3.10 * smooth_box(x, 40.0, 60.0, 2.0)
        + 1.55 * smooth_box(x, 60.0, 100.0, 4.0)
        + 1.00 * smooth_box(x, 100.0, 150.0, 5.0)
        + 0.72 * smooth_box(x, 150.0, 200.0, 6.0)
        + 1.45 * np.exp(-0.5 * ((x - HOPF_RE) / 4.5) ** 2)
    )


def make_re_values(n: int = N_RE) -> np.ndarray:
    grid = np.linspace(RE_MIN, RE_MAX, 200_001)
    density = sampling_density(grid)
    increments = 0.5 * (density[1:] + density[:-1]) * np.diff(grid)
    cdf = np.r_[0.0, np.cumsum(increments)]
    cdf /= cdf[-1]
    targets = np.linspace(0.0, 1.0, n)
    values = np.interp(targets, cdf, grid)
    values[0] = RE_MIN
    values[-1] = RE_MAX
    return values.astype(np.float64)


RE_VALUES = make_re_values()


def classify_regime(re_value: float) -> str:
    if re_value < 40.0:
        return "steady_wake"
    if re_value < HOPF_RE:
        return "pre_hopf_steady"
    if re_value < 60.0:
        return "hopf_transition"
    if re_value < 100.0:
        return "developing_periodic_shedding"
    if re_value < 150.0:
        return "mature_periodic_shedding"
    return "high_re_2d_periodic_near_modeA"


def shedding_st(re_value: float) -> float:
    # Low-Re cylinder estimate used only for scheduling; detected lift periods supersede it.
    return max(0.08, 0.198 * (1.0 - 19.7 / re_value))


def estimated_period(re_value: float) -> float:
    velocity = re_value * NU / D
    return D / (shedding_st(re_value) * velocity)


def case_plan(re_value: float) -> CasePlan:
    u = re_value * NU / D
    regime = classify_regime(re_value)
    if re_value >= HOPF_RE:
        period = estimated_period(re_value)
        write_interval = period / FRAMES_PER_CYCLE
        return CasePlan(
            Re=float(re_value),
            label=re_label(float(re_value)),
            regime=regime,
            U=float(u),
            estimated_St=float(shedding_st(re_value)),
            estimated_period=float(period),
            nominal_transient_time=float(PERIODIC_TRANSIENT_CYCLES * period),
            nominal_retained_time=float(PERIODIC_RETAINED_CYCLES * period),
            nominal_write_interval=float(write_interval),
        )
    convective = D / u
    save_window = STEADY_SAVE_CONVECTIVE_TIMES * convective
    return CasePlan(
        Re=float(re_value),
        label=re_label(float(re_value)),
        regime=regime,
        U=float(u),
        estimated_St=None,
        estimated_period=None,
        nominal_transient_time=float(40.0 * convective),
        nominal_retained_time=float(save_window),
        nominal_write_interval=float(save_window / max(1, STEADY_SAVE_SNAPSHOTS - 1)),
    )


def run(cmd: str, cwd: Path, log_path: Path | None = None) -> None:
    env_cmd = f"source /opt/openfoam13/etc/bashrc && {cmd}"
    if log_path is None:
        subprocess.run(["bash", "-lc", env_cmd], cwd=cwd, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write(f"$ {cmd}\n\n")
        log.flush()
        subprocess.run(
            ["bash", "-lc", env_cmd],
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def replace_text(path: Path, pattern: str, replacement: str, flags: int = 0) -> None:
    text = path.read_text()
    if not re.search(pattern, text, flags=flags):
        raise RuntimeError(f"Pattern not found in {path}: {pattern}")
    new = re.sub(pattern, replacement, text, flags=flags)
    path.write_text(new)


def set_dict_entry(path: Path, key: str, value: str) -> None:
    text = path.read_text()
    pattern = rf"^({re.escape(key)}\s+)([^;]+);"
    if re.search(pattern, text, flags=re.MULTILINE):
        text = re.sub(pattern, rf"\g<1>{value};", text, flags=re.MULTILINE)
    else:
        text = text.replace("functions\n{", f"{key}          {value};\n\nfunctions\n{{")
    path.write_text(text)


def foam_functions(velocity: float, write_interval_steps: int = 10) -> str:
    area_ref = D * DOMAIN_THICKNESS_FALLBACK
    return f"""functions
{{
    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        patches         (cylinder);
        rho             rhoInf;
        rhoInf          {RHO:g};
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        pitchAxis       (0 0 1);
        CofR            (0 0 0);
        magUInf         {velocity:.12g};
        lRef            {D:g};
        Aref            {area_ref:.12g};
        writeControl    timeStep;
        writeInterval   {write_interval_steps};
        log             true;
    }}

    fieldStats
    {{
        type            volFieldValue;
        libs            ("libfieldFunctionObjects.so");
        log             true;
        writeControl    timeStep;
        writeInterval   {max(5 * write_interval_steps, 50)};
        writeFields     false;
        cellZone        all;
        operation       volAverage;
        fields          (p U);
    }}
}}
"""


def patch_control_dict(
    case_dir: Path,
    velocity: float,
    end_time: float,
    write_interval: float,
    max_delta_t: float,
    purge_write: int,
    start_from: str = "latestTime",
) -> None:
    control = case_dir / "system" / "controlDict"
    set_dict_entry(control, "application", "pimpleFoam")
    set_dict_entry(control, "startFrom", start_from)
    set_dict_entry(control, "stopAt", "endTime")
    set_dict_entry(control, "endTime", f"{end_time:.12g}")
    set_dict_entry(control, "adjustTimeStep", "yes")
    set_dict_entry(control, "maxCo", f"{MAX_CO:.12g}")
    set_dict_entry(control, "maxDeltaT", f"{max_delta_t:.12g}")
    set_dict_entry(control, "writeControl", "adjustableRunTime")
    set_dict_entry(control, "writeInterval", f"{write_interval:.12g}")
    set_dict_entry(control, "purgeWrite", str(purge_write))
    set_dict_entry(control, "writeFormat", "binary")
    set_dict_entry(control, "writeCompression", "on")
    replace_text(
        control,
        r"functions\s*\{[\s\S]*\}\s*$",
        foam_functions(velocity),
        flags=re.MULTILINE,
    )


def patch_steady_control_dict(
    case_dir: Path,
    velocity: float,
    end_time: float,
    write_interval: float,
    purge_write: int,
    start_from: str = "startTime",
) -> None:
    control = case_dir / "system" / "controlDict"
    set_dict_entry(control, "application", "simpleFoam")
    set_dict_entry(control, "solver", "incompressibleFluid")
    set_dict_entry(control, "startFrom", start_from)
    set_dict_entry(control, "stopAt", "endTime")
    set_dict_entry(control, "endTime", f"{end_time:.12g}")
    set_dict_entry(control, "deltaT", "1")
    set_dict_entry(control, "adjustTimeStep", "no")
    set_dict_entry(control, "writeControl", "timeStep")
    set_dict_entry(control, "writeInterval", f"{write_interval:.12g}")
    set_dict_entry(control, "purgeWrite", str(purge_write))
    set_dict_entry(control, "writeFormat", "binary")
    set_dict_entry(control, "writeCompression", "on")
    replace_text(
        control,
        r"functions\s*\{[\s\S]*\}\s*$",
        foam_functions(velocity, write_interval_steps=20),
        flags=re.MULTILINE,
    )


def patch_initial_velocity(path: Path, velocity: float, periodic: bool) -> None:
    perturb = 0.002 * velocity if periodic else 0.0
    text = path.read_text()
    text = text.replace(
        "internalField   uniform (__INLET_VELOCITY__ 0 0);",
        f"internalField   uniform ({velocity:.12g} {perturb:.12g} 0);",
    )
    text = text.replace("__INLET_VELOCITY__", f"{velocity:.12g}")
    path.write_text(text)


def set_decompose(case_dir: Path, nprocs: int) -> None:
    replace_text(
        case_dir / "system" / "decomposeParDict",
        r"numberOfSubdomains\s+[^;]+;",
        f"numberOfSubdomains {nprocs};",
    )


def solver_command(nprocs: int) -> str:
    if nprocs <= 1:
        return "pimpleFoam"
    return f"mpirun -np {nprocs} pimpleFoam -parallel"


def steady_solver_command(nprocs: int) -> str:
    if nprocs <= 1:
        return "simpleFoam"
    return f"mpirun -np {nprocs} simpleFoam -parallel"


def latest_time(case_dir: Path) -> float:
    times: list[float] = [0.0]
    bases = [case_dir] + sorted(p for p in case_dir.glob("processor*") if p.is_dir())
    for base in bases:
        for p in base.iterdir():
            if not p.is_dir():
                continue
            try:
                times.append(float(p.name))
            except ValueError:
                pass
    return max(times)


def parse_numeric_table(path: Path) -> tuple[list[str], np.ndarray]:
    header: list[str] = []
    rows: list[list[float]] = []
    for line in path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            tokens = stripped.lstrip("#").split()
            if "Time" in tokens:
                header = tokens
            continue
        parts = stripped.replace("(", " ").replace(")", " ").split()
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    if not rows:
        return header, np.empty((0, 0), dtype=np.float64)
    return header, np.asarray(rows, dtype=np.float64)


def load_force_coefficients(case_dir: Path) -> dict[str, np.ndarray]:
    files = sorted(
        set(case_dir.glob("postProcessing/**/forceCoeffs.dat"))
        | set(case_dir.glob("postProcessing/**/coefficient.dat"))
    )
    all_rows: list[np.ndarray] = []
    header: list[str] = []
    for file in files:
        h, arr = parse_numeric_table(file)
        if arr.size:
            all_rows.append(arr)
            if h:
                header = h
    if not all_rows:
        return {"time": np.array([]), "Cd": np.array([]), "Cl": np.array([])}
    data = np.vstack(all_rows)
    data = data[np.argsort(data[:, 0])]
    _, unique_idx = np.unique(data[:, 0], return_index=True)
    data = data[np.sort(unique_idx)]

    def column(name: str, fallback: int) -> np.ndarray:
        if name in header:
            idx = header.index(name)
            if idx < data.shape[1]:
                return data[:, idx]
        if fallback < data.shape[1]:
            return data[:, fallback]
        return np.full(data.shape[0], np.nan)

    return {
        "time": data[:, 0],
        "Cd": column("Cd", 2),
        "Cl": column("Cl", 3),
    }


def load_field_stats(case_dir: Path) -> np.ndarray:
    files = sorted(case_dir.glob("postProcessing/**/volFieldValue.dat"))
    rows: list[np.ndarray] = []
    for file in files:
        _, arr = parse_numeric_table(file)
        if arr.size and arr.shape[1] > 1:
            rows.append(arr)
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    data = np.vstack(rows)
    data = data[np.argsort(data[:, 0])]
    _, unique_idx = np.unique(data[:, 0], return_index=True)
    return data[np.sort(unique_idx)]


def recent_residual_ok(log_files: list[Path], threshold: float = 5.0e-2) -> bool:
    values: list[float] = []
    pattern = re.compile(r"Final residual = ([0-9.eE+-]+)")
    for log_file in log_files[-4:]:
        for line in log_file.read_text(errors="ignore").splitlines():
            match = pattern.search(line)
            if match:
                values.append(float(match.group(1)))
    if len(values) < 20:
        return False
    recent = np.asarray(values[-240:], dtype=np.float64)
    return float(np.nanpercentile(recent, 95.0)) < threshold


def steady_converged(case_dir: Path, log_files: list[Path]) -> tuple[bool, dict]:
    coeffs = load_force_coefficients(case_dir)
    t = coeffs["time"]
    cl = coeffs["Cl"]
    cd = coeffs["Cd"]
    diagnostics: dict[str, float | bool | str] = {"samples": int(t.size)}
    if t.size < 80:
        diagnostics["reason"] = "not enough force samples"
        return False, diagnostics

    window = slice(max(0, t.size - 120), t.size)
    cl_w = cl[window]
    cd_w = cd[window]
    t_w = t[window]
    cd_mean = float(np.nanmean(cd_w))
    cl_std = float(np.nanstd(cl_w))
    cd_rel_std = float(np.nanstd(cd_w) / max(abs(cd_mean), 1.0e-12))
    cd_slope = float(abs(np.polyfit(t_w - t_w[0], cd_w, 1)[0]) / max(abs(cd_mean), 1.0e-12))
    residual_ok = recent_residual_ok(log_files)

    field = load_field_stats(case_dir)
    field_rel_std = math.inf
    field_abs_std = math.inf
    if field.size and field.shape[0] >= 20:
        fw = field[max(0, field.shape[0] - 40) :, 1:]
        means = np.abs(np.nanmean(fw, axis=0))
        stds = np.nanstd(fw, axis=0)
        nonzero = means > 1.0e-8
        field_abs_std = float(np.nanmax(stds))
        if np.any(nonzero):
            field_rel_std = float(np.nanmax(stds[nonzero] / means[nonzero]))

    diagnostics.update(
        {
            "Cl_std": cl_std,
            "Cd_rel_std": cd_rel_std,
            "Cd_rel_slope": cd_slope,
            "field_rel_std": field_rel_std,
            "field_abs_std": field_abs_std,
            "residual_ok": residual_ok,
        }
    )
    field_ok = (
        not field.size
        or field.shape[0] < 20
        or field_abs_std < 1.0e-5
        or field_rel_std < 1.0e-3
        or not np.isfinite(field_rel_std)
    )
    ok = (
        cl_std < 2.5e-4
        and cd_rel_std < 7.5e-4
        and cd_slope < 2.5e-5
        and field_ok
        and residual_ok
    )
    diagnostics["reason"] = "steady convergence criteria satisfied" if ok else "criteria not yet satisfied"
    return ok, diagnostics


def initialise_steady_wake(case_dir: Path, plan: CasePlan, nprocs: int) -> tuple[float, dict]:
    convective = D / plan.U
    max_iterations = min(max(1200.0, 35.0 * convective), 4500.0)
    patch_steady_control_dict(
        case_dir,
        plan.U,
        end_time=max_iterations,
        write_interval=max(100.0, max_iterations / 6.0),
        purge_write=2,
        start_from="startTime",
    )
    log = case_dir / "log.simpleFoam"
    run(steady_solver_command(nprocs), case_dir, log)
    current = latest_time(case_dir)
    ok, diag = steady_converged(case_dir, [log])
    diag.update(
        {
            "mode": "steady_simpleFoam_initialisation",
            "status": "steady_detected" if ok else "steady_solver_completed_fallback",
            "simpleFoam_end_time": current,
            "simpleFoam_max_iterations": max_iterations,
        }
    )
    return current, diag


def local_maxima(y: np.ndarray) -> np.ndarray:
    if y.size < 3:
        return np.array([], dtype=np.int64)
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1


def limit_cycle_detected(case_dir: Path, estimated: float, min_time: float) -> tuple[bool, dict]:
    coeffs = load_force_coefficients(case_dir)
    t = coeffs["time"]
    cl = coeffs["Cl"]
    diagnostics: dict[str, float | int | str] = {"samples": int(t.size)}
    if t.size < 100 or t[-1] < min_time:
        diagnostics["reason"] = "not enough lift samples or minimum transient time not reached"
        return False, diagnostics
    peaks = local_maxima(cl - np.nanmean(cl))
    peaks = peaks[t[peaks] > max(0.0, t[-1] - 8.0 * estimated)]
    if peaks.size < 6:
        diagnostics["reason"] = "not enough recent lift peaks"
        return False, diagnostics
    peak_times = t[peaks]
    peak_values = np.abs(cl[peaks] - np.nanmean(cl))
    periods = np.diff(peak_times)
    recent_amp = peak_values[-5:]
    recent_periods = periods[-4:]
    amp_cv = float(np.nanstd(recent_amp) / max(np.nanmean(recent_amp), 1.0e-12))
    period_cv = float(np.nanstd(recent_periods) / max(np.nanmean(recent_periods), 1.0e-12))
    detected_period = float(np.nanmean(recent_periods))
    diagnostics.update(
        {
            "recent_peak_count": int(peaks.size),
            "lift_peak_amplitude_cv": amp_cv,
            "period_cv": period_cv,
            "detected_period": detected_period,
        }
    )
    ok = amp_cv < 0.05 and period_cv < 0.04 and detected_period > 0.35 * estimated
    diagnostics["reason"] = "stable limit cycle detected" if ok else "limit cycle criteria not yet satisfied"
    return ok, diagnostics


def detect_start_for_steady(case_dir: Path, plan: CasePlan, nprocs: int) -> tuple[float, dict]:
    convective = D / plan.U
    chunk = min(max(8.0 * convective, 120.0), 650.0)
    max_time = 90.0 * convective
    current = latest_time(case_dir)
    logs: list[Path] = []
    diagnostics: dict = {"mode": "steady", "chunks": []}
    chunk_i = 0
    while current < max_time:
        chunk_i += 1
        end_time = current + chunk
        patch_control_dict(
            case_dir,
            plan.U,
            end_time=end_time,
            write_interval=chunk,
            max_delta_t=min(MAX_DELTA_T, 0.2 * convective),
            purge_write=2,
        )
        log = case_dir / f"log.detect_steady_{chunk_i:03d}.pimpleFoam"
        run(solver_command(nprocs), case_dir, log)
        logs.append(log)
        current = latest_time(case_dir)
        ok, diag = steady_converged(case_dir, logs)
        diagnostics["chunks"].append({"end_time": current, **diag})
        if ok:
            diagnostics["status"] = "steady_detected"
            return current, diagnostics
    diagnostics["status"] = "steady_fallback_max_time"
    return current, diagnostics


def detect_start_for_periodic(case_dir: Path, plan: CasePlan, nprocs: int) -> tuple[float, float, dict]:
    assert plan.estimated_period is not None
    period = plan.estimated_period
    min_transient = PERIODIC_TRANSIENT_CYCLES * period
    chunk = max(0.75 * period, 40.0)
    max_detect_time = max(12.0 * period, min_transient + 4.0 * period)
    current = latest_time(case_dir)
    diagnostics: dict = {"mode": "periodic", "estimated_period": period, "chunks": []}
    chunk_i = 0
    detected_period = period
    while current < max_detect_time:
        chunk_i += 1
        end_time = current + chunk
        patch_control_dict(
            case_dir,
            plan.U,
            end_time=end_time,
            write_interval=chunk,
            max_delta_t=min(MAX_DELTA_T, period / 80.0),
            purge_write=2,
        )
        log = case_dir / f"log.detect_periodic_{chunk_i:03d}.pimpleFoam"
        run(solver_command(nprocs), case_dir, log)
        current = latest_time(case_dir)
        ok, diag = limit_cycle_detected(case_dir, period, min_transient)
        diagnostics["chunks"].append({"end_time": current, **diag})
        if ok:
            detected_period = float(diag.get("detected_period", period))
            diagnostics["status"] = "limit_cycle_detected"
            return current, detected_period, diagnostics
    diagnostics["status"] = "periodic_fallback_after_minimum_detection_window"
    return max(current, min_transient), detected_period, diagnostics


def vtk_time_from_name(case_name: str, path: Path) -> float:
    match = re.match(rf"{re.escape(case_name)}_(.+)\.vtk$", path.name)
    if not match:
        raise ValueError(f"Cannot parse VTK time from {path.name}")
    return float(match.group(1))


def read_legacy_vtk(path: Path):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def convert_vtk_to_npz(
    root: Path,
    vtk_dir: Path,
    case_name: str,
    plan: CasePlan,
    retain_start_time: float,
    diagnostics: dict,
) -> dict:
    files_all = sorted(vtk_dir.glob(f"{case_name}_*.vtk"), key=lambda p: vtk_time_from_name(case_name, p))
    files = [p for p in files_all if vtk_time_from_name(case_name, p) >= retain_start_time - 1.0e-9]
    if not files:
        raise RuntimeError(f"No retained VTK files found in {vtk_dir}")

    first = read_legacy_vtk(files[0])
    n_points = first.GetNumberOfPoints()
    n_cells = first.GetNumberOfCells()
    points = vtk_to_numpy(first.GetPoints().GetData()).astype(np.float32, copy=True)
    times = np.asarray([vtk_time_from_name(case_name, p) for p in files], dtype=np.float64)
    u = np.empty((len(files), n_points), dtype=np.float32)
    v = np.empty_like(u)
    p = np.empty_like(u)

    for ti, file in enumerate(files):
        grid = read_legacy_vtk(file)
        if grid.GetNumberOfPoints() != n_points:
            raise RuntimeError(f"Point count changed in {file}")
        point_data = grid.GetPointData()
        u_arr = vtk_to_numpy(point_data.GetArray("U")).astype(np.float32, copy=False)
        p_arr = vtk_to_numpy(point_data.GetArray("p")).astype(np.float32, copy=False)
        u[ti] = u_arr[:, 0]
        v[ti] = u_arr[:, 1]
        p[ti] = p_arr

    metadata = {
        "Re": plan.Re,
        "label": plan.label,
        "regime": plan.regime,
        "D": D,
        "nu": NU,
        "U_inlet": plan.U,
        "retain_start_time": retain_start_time,
        "n_times": int(times.size),
        "n_points": int(n_points),
        "n_cells": int(n_cells),
        "diagnostics": diagnostics,
        "arrays": {
            "times": list(times.shape),
            "points": list(points.shape),
            "u": list(u.shape),
            "v": list(v.shape),
            "p": list(p.shape),
        },
        "units": {"times": "s", "points": "m", "u/v": "m/s", "p": "m^2/s^2"},
    }
    npz = root / f"{plan.label}_uvp_pointData.npz"
    np.savez_compressed(npz, times=times, points=points, u=u, v=v, p=p, metadata=json.dumps(metadata, indent=2))
    (root / f"{plan.label}_uvp_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (root / f"{plan.label}_uvp_dimensions.txt").write_text(
        "\n".join(
            [
                f"Re = {plan.Re:.12g}",
                f"regime = {plan.regime}",
                f"data_file = {npz.name}",
                f"times shape = {times.shape}",
                f"points shape = {points.shape}",
                f"u shape = {u.shape}",
                f"v shape = {v.shape}",
                f"p shape = {p.shape}",
                "indexing = u[t, i], v[t, i], p[t, i] at coordinate points[i]",
                "",
            ]
        )
    )
    size = npz.stat().st_size
    del u, v, p, points
    gc.collect()
    return {
        "Re": plan.Re,
        "label": plan.label,
        "regime": plan.regime,
        "npz": npz.name,
        "n_times": int(times.size),
        "n_points": int(n_points),
        "n_cells": int(n_cells),
        "size_bytes": int(size),
        "retain_start_time": float(retain_start_time),
    }


def run_one_case(root: Path, plan: CasePlan, nprocs: int) -> dict:
    npz_path = root / f"{plan.label}_uvp_pointData.npz"
    info_path = root / f"{plan.label}_info.txt"
    if npz_path.is_file() and info_path.is_file():
        print(f"Skipping {plan.label}: existing complete result.", flush=True)
        return {"Re": plan.Re, "label": plan.label, "regime": plan.regime, "status": "skipped", "npz": npz_path.name}

    require_free(root, MIN_FREE_RUN_GB, f"run {plan.label}")
    run_dir = BRANCH / "work" / f"run_pg_{plan.label}"
    log_dir = root / f"{plan.label}_logs"
    for path in (run_dir, log_dir):
        if path.exists():
            shutil.rmtree(path)
    for path in (
        npz_path,
        root / f"{plan.label}_uvp_metadata.json",
        root / f"{plan.label}_uvp_dimensions.txt",
        info_path,
    ):
        if path.exists():
            path.unlink()

    print(
        f"\n=== {plan.label} Re={plan.Re:.6f} regime={plan.regime} U={plan.U:.8g} "
        f"free={free_gb(root):.2f}GiB ===",
        flush=True,
    )
    shutil.copytree(BASE_CASE, run_dir, ignore=shutil.ignore_patterns("processor*", "postProcessing", "VTK", "log.*"))
    patch_initial_velocity(run_dir / "0" / "U", plan.U, periodic=plan.Re >= HOPF_RE)
    if nprocs > 1:
        set_decompose(run_dir, nprocs)

    try:
        run("blockMesh", run_dir, run_dir / "log.blockMesh")
        if nprocs > 1:
            run("decomposePar -force", run_dir, run_dir / "log.decomposePar")

        if plan.Re < HOPF_RE:
            retain_start, diagnostics = initialise_steady_wake(run_dir, plan, nprocs)
            save_window = plan.nominal_retained_time
            write_interval = plan.nominal_write_interval
            end_time = retain_start + save_window
        else:
            retain_start, detected_period, diagnostics = detect_start_for_periodic(run_dir, plan, nprocs)
            write_interval = detected_period / FRAMES_PER_CYCLE
            end_time = retain_start + PERIODIC_RETAINED_CYCLES * detected_period + 0.25 * write_interval
            diagnostics["retained_period"] = detected_period

        patch_control_dict(
            run_dir,
            plan.U,
            end_time=end_time,
            write_interval=write_interval,
            max_delta_t=min(MAX_DELTA_T, write_interval / 4.0),
            purge_write=0,
        )
        run(solver_command(nprocs), run_dir, run_dir / "log.retain.pimpleFoam")
        if nprocs > 1:
            run("reconstructPar", run_dir, run_dir / "log.reconstructPar")
        run("foamToVTK -useTimeName -noFaceZones", run_dir, run_dir / "log.foamToVTK")
        conversion = convert_vtk_to_npz(root, run_dir / "VTK", run_dir.name, plan, retain_start, diagnostics)
        shutil.rmtree(run_dir / "VTK", ignore_errors=True)

        log_dir.mkdir(parents=True, exist_ok=True)
        for log_file in run_dir.glob("log.*"):
            shutil.move(str(log_file), str(log_dir / log_file.name))
        if (run_dir / "postProcessing").exists():
            shutil.move(str(run_dir / "postProcessing"), str(log_dir / "postProcessing"))
        info_path.write_text(
            json.dumps(
                {
                    "plan": asdict(plan),
                    "conversion": conversion,
                    "diagnostics": diagnostics,
                    "nprocs": nprocs,
                    "free_gb_after_case": free_gb(root),
                },
                indent=2,
            )
            + "\n"
        )
        conversion["status"] = "completed"
        print(
            f"Completed {plan.label}: snapshots={conversion['n_times']} "
            f"npz={conversion['size_bytes'] / 1024**2:.1f}MiB free={free_gb(root):.2f}GiB",
            flush=True,
        )
        return conversion
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir)


def build_mesh_vtk() -> Path:
    if MESH_TMP_CASE.exists():
        shutil.rmtree(MESH_TMP_CASE)
    shutil.copytree(BASE_CASE, MESH_TMP_CASE, ignore=shutil.ignore_patterns("processor*", "postProcessing", "VTK", "log.*"))
    run(f"blockMesh -case {MESH_TMP_CASE}", BRANCH, None)
    run(f"foamToVTK -case {MESH_TMP_CASE} -constant -noZero -noFaceZones", BRANCH, None)
    vtk_path = MESH_TMP_CASE / "VTK" / f"{MESH_TMP_CASE.name}_0.vtk"
    if not vtk_path.is_file():
        raise RuntimeError(f"Mesh VTK was not created: {vtk_path}")
    return vtk_path


def extract_point_control_areas(reference_points: np.ndarray) -> dict:
    vtk_path = build_mesh_vtk()
    grid = read_legacy_vtk(vtk_path)
    points = vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float64)
    if points.shape != reference_points.shape:
        raise RuntimeError(f"Mesh/data point shape mismatch: {points.shape} vs {reference_points.shape}")
    max_delta = float(np.linalg.norm(points - reference_points.astype(np.float64), axis=1).max())
    if max_delta > 1.0e-6:
        raise RuntimeError(f"Mesh/data points do not match: max_delta={max_delta:g}")

    z_span = float(points[:, 2].max() - points[:, 2].min())
    thickness = z_span if z_span > 0 else DOMAIN_THICKNESS_FALLBACK

    size_filter = vtk.vtkCellSizeFilter()
    size_filter.SetInputData(grid)
    size_filter.SetComputeVolume(True)
    size_filter.SetComputeArea(False)
    size_filter.SetComputeLength(False)
    size_filter.SetComputeVertexCount(False)
    size_filter.Update()
    sized = size_filter.GetOutput()
    cell_volumes = vtk_to_numpy(sized.GetCellData().GetArray("Volume")).astype(np.float64)
    cell_areas = cell_volumes / thickness
    if np.any(cell_areas <= 0.0) or not np.all(np.isfinite(cell_areas)):
        raise RuntimeError("Invalid cell areas")

    point_areas = np.zeros(grid.GetNumberOfPoints(), dtype=np.float64)
    for ci in range(grid.GetNumberOfCells()):
        ids = grid.GetCell(ci).GetPointIds()
        share = cell_areas[ci] / ids.GetNumberOfIds()
        for local in range(ids.GetNumberOfIds()):
            point_areas[ids.GetId(local)] += share
    if np.any(point_areas <= 0.0) or not np.all(np.isfinite(point_areas)):
        raise RuntimeError("Invalid point control areas")
    return {
        "points": points.astype(np.float32),
        "cell_areas": cell_areas.astype(np.float32),
        "point_areas": point_areas.astype(np.float32),
        "sqrt_point_areas": np.sqrt(point_areas).astype(np.float32),
        "thickness": thickness,
        "total_area": float(cell_areas.sum()),
        "total_point_area": float(point_areas.sum()),
        "min_point_area": float(point_areas.min()),
        "max_point_area": float(point_areas.max()),
        "max_point_delta": max_delta,
    }


def load_npz(root: Path, re_value: float):
    return np.load(root / f"{re_label(re_value)}_uvp_pointData.npz")


def prepare_snapshot_metadata(root: Path) -> tuple[list[dict], dict[str, dict], int, int, np.ndarray]:
    first = load_npz(root, float(RE_VALUES[0]))
    points = first["points"].astype(np.float32)
    n_points = points.shape[0]
    del first
    rows: list[dict] = []
    per_re: dict[str, dict] = {}
    row0 = 0
    for re_value in RE_VALUES:
        plan = case_plan(float(re_value))
        data = load_npz(root, plan.Re)
        times = data["times"].astype(np.float64)
        n = int(times.size)
        mean_u = data["u"].mean(axis=0, dtype=np.float64).astype(np.float32)
        mean_v = data["v"].mean(axis=0, dtype=np.float64).astype(np.float32)
        mean_p = data["p"].mean(axis=0, dtype=np.float64).astype(np.float32)
        per_re[plan.label] = {
            "Re": plan.Re,
            "label": plan.label,
            "regime": plan.regime,
            "row_start": row0,
            "row_stop": row0 + n,
            "times": times,
            "mean_u": mean_u,
            "mean_v": mean_v,
            "mean_p": mean_p,
        }
        period = plan.estimated_period if plan.estimated_period else np.nan
        for i, t in enumerate(times):
            rows.append(
                {
                    "snapshot_id": row0 + i,
                    "Re": f"{plan.Re:.12g}",
                    "Re_label": plan.label,
                    "regime": plan.regime,
                    "time": float(t),
                    "estimated_period": float(period) if np.isfinite(period) else "",
                    "local_snapshot_index": i,
                }
            )
        row0 += n
        print(f"Prepared {plan.label}: snapshots={n}, rows={row0}", flush=True)
        del data
        gc.collect()
    return rows, per_re, row0, n_points, points


def make_weighted_feature_block(
    root: Path,
    field: str,
    per_re: dict[str, dict],
    n_snapshots: int,
    n_points: int,
    sqrt_area: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    block = np.empty((n_snapshots, stop - start), dtype=np.float32)
    feature_idx = np.arange(start, stop)
    for re_value in RE_VALUES:
        plan = case_plan(float(re_value))
        info = per_re[plan.label]
        rows = slice(info["row_start"], info["row_stop"])
        data = load_npz(root, plan.Re)
        if field == "uv":
            u_mask = feature_idx < n_points
            v_mask = ~u_mask
            if np.any(u_mask):
                cols = feature_idx[u_mask]
                block[rows, u_mask] = (data["u"][:, cols] - info["mean_u"][cols]) * sqrt_area[cols]
            if np.any(v_mask):
                cols = feature_idx[v_mask] - n_points
                block[rows, v_mask] = (data["v"][:, cols] - info["mean_v"][cols]) * sqrt_area[cols]
        elif field == "p":
            cols = feature_idx
            block[rows, :] = (data["p"][:, cols] - info["mean_p"][cols]) * sqrt_area[cols]
        else:
            raise ValueError(field)
        del data
    return block


def streaming_weighted_pod(
    root: Path,
    field: str,
    per_re: dict[str, dict],
    n_snapshots: int,
    n_points: int,
    sqrt_area: np.ndarray,
    max_modes: int,
    feature_block: int = 2048,
    row_block: int = 512,
) -> dict:
    n_features = 2 * n_points if field == "uv" else n_points
    label = "velocity [u,v]" if field == "uv" else "pressure p"
    print(f"Streaming covariance for {label}: snapshots={n_snapshots}, features={n_features}", flush=True)
    covariance = np.zeros((n_snapshots, n_snapshots), dtype=np.float64)
    for start in range(0, n_features, feature_block):
        stop = min(start + feature_block, n_features)
        xb = make_weighted_feature_block(root, field, per_re, n_snapshots, n_points, sqrt_area, start, stop)
        for r0 in range(0, n_snapshots, row_block):
            r1 = min(r0 + row_block, n_snapshots)
            covariance[r0:r1, :] += xb[r0:r1, :] @ xb.T
        print(f"  covariance block {start}:{stop}", flush=True)
        del xb
        gc.collect()
    covariance = 0.5 * (covariance + covariance.T)
    total_energy = float(np.trace(covariance))
    eigvals, eigvecs = eigsh(covariance, k=min(max_modes, n_snapshots - 2), which="LM", tol=1.0e-8)
    order = eigvals.argsort()[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    singular = np.sqrt(eigvals)
    energy = eigvals / total_energy
    cumulative = np.cumsum(energy)
    rank = int(eigvals.size)
    coeff = (eigvecs[:, :rank] * singular[:rank]).astype(np.float32)
    basis_t = (eigvecs[:, :rank].T / singular[:rank, None]).astype(np.float32)

    phi_weighted = np.empty((rank, n_features), dtype=np.float32)
    print(f"Streaming modal back-substitution for {label}", flush=True)
    for start in range(0, n_features, feature_block):
        stop = min(start + feature_block, n_features)
        xb = make_weighted_feature_block(root, field, per_re, n_snapshots, n_points, sqrt_area, start, stop)
        phi_weighted[:, start:stop] = basis_t @ xb
        print(f"  mode block {start}:{stop}", flush=True)
        del xb
        gc.collect()
    del covariance, eigvecs, basis_t
    gc.collect()
    return {
        "phi_weighted": phi_weighted,
        "coeff": coeff,
        "singular_values": singular.astype(np.float64),
        "energy": energy.astype(np.float64),
        "cumulative_energy": cumulative.astype(np.float64),
        "rank": rank,
        "total_energy": total_energy,
    }


def compute_area_weighted_pod(root: Path) -> None:
    require_free(root, MIN_FREE_POD_GB, "streaming area-weighted POD")
    out = root / "Global_POD_AreaWeighted_L2"
    out.mkdir(parents=True, exist_ok=True)
    snapshot_rows, per_re, n_snapshots, n_points, points = prepare_snapshot_metadata(root)
    weights = extract_point_control_areas(points)
    sqrt_area = weights["sqrt_point_areas"].astype(np.float32)
    sqrt_area_uv = np.concatenate([sqrt_area, sqrt_area])
    np.savez_compressed(
        out / "mesh_l2_point_area_weights.npz",
        points=points,
        cell_areas=weights["cell_areas"],
        point_areas=weights["point_areas"],
        sqrt_point_areas=weights["sqrt_point_areas"],
        thickness=np.array(weights["thickness"], dtype=np.float64),
        total_area=np.array(weights["total_area"], dtype=np.float64),
        total_point_area=np.array(weights["total_point_area"], dtype=np.float64),
    )

    uv_pod = streaming_weighted_pod(root, "uv", per_re, n_snapshots, n_points, sqrt_area, MAX_MODES)
    phi_uv_weighted = uv_pod["phi_weighted"]
    phi_uv = (phi_uv_weighted / sqrt_area_uv[None, :]).astype(np.float32)

    p_pod = streaming_weighted_pod(root, "p", per_re, n_snapshots, n_points, sqrt_area, MAX_MODES)
    phi_p_weighted = p_pod["phi_weighted"]
    phi_p = (phi_p_weighted / sqrt_area[None, :]).astype(np.float32)

    re_array = np.array(RE_VALUES, dtype=np.float64)
    labels = np.array([re_label(float(v)) for v in RE_VALUES])
    regimes = np.array([classify_regime(float(v)) for v in RE_VALUES])
    mean_uv = np.empty((len(RE_VALUES), 2 * n_points), dtype=np.float32)
    mean_p = np.empty((len(RE_VALUES), n_points), dtype=np.float32)
    for i, re_value in enumerate(RE_VALUES):
        info = per_re[re_label(float(re_value))]
        mean_uv[i, :n_points] = info["mean_u"]
        mean_uv[i, n_points:] = info["mean_v"]
        mean_p[i] = info["mean_p"]

    np.savez_compressed(
        out / "global_velocity_pod_area_weighted_l2.npz",
        phi_uv=phi_uv,
        phi_uv_weighted=phi_uv_weighted,
        coeff_uv=uv_pod["coeff"],
        mean_uv_by_Re=mean_uv,
        Re_values=re_array,
        Re_labels=labels,
        regimes=regimes,
        points=points,
        point_areas=weights["point_areas"],
        sqrt_point_areas=weights["sqrt_point_areas"],
        singular_values_uv=uv_pod["singular_values"],
        energy_uv=uv_pod["energy"],
        cumulative_energy_uv=uv_pod["cumulative_energy"],
        total_weighted_energy_uv=np.array(uv_pod["total_energy"], dtype=np.float64),
    )
    np.savez_compressed(
        out / "global_pressure_pod_area_weighted_l2.npz",
        phi_p=phi_p,
        phi_p_weighted=phi_p_weighted,
        coeff_p=p_pod["coeff"],
        mean_p_by_Re=mean_p,
        Re_values=re_array,
        Re_labels=labels,
        regimes=regimes,
        points=points,
        point_areas=weights["point_areas"],
        sqrt_point_areas=weights["sqrt_point_areas"],
        singular_values_p=p_pod["singular_values"],
        energy_p=p_pod["energy"],
        cumulative_energy_p=p_pod["cumulative_energy"],
        total_weighted_energy_p=np.array(p_pod["total_energy"], dtype=np.float64),
    )
    with (out / "pod_snapshot_index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(snapshot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(snapshot_rows)
    metadata = {
        "method": "streaming exact snapshot POD with lumped nodal 2D area weights",
        "total_snapshots": n_snapshots,
        "n_points": n_points,
        "velocity": {
            "phi_shape": list(phi_uv.shape),
            "coeff_shape": list(uv_pod["coeff"].shape),
            "energy_80": float(uv_pod["cumulative_energy"][-1]),
        },
        "pressure": {
            "phi_shape": list(phi_p.shape),
            "coeff_shape": list(p_pod["coeff"].shape),
            "energy_80": float(p_pod["cumulative_energy"][-1]),
        },
        "area_weights": {k: float(v) for k, v in weights.items() if isinstance(v, float)},
    }
    (out / "pod_area_weighted_l2_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if MESH_TMP_CASE.exists():
        shutil.rmtree(MESH_TMP_CASE)


def write_sampling_csv(root: Path) -> None:
    with (root / "Re_sampling_strategy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "Re",
                "label",
                "regime",
                "U",
                "estimated_St",
                "estimated_period",
                "nominal_transient_time",
                "nominal_retained_time",
                "nominal_write_interval",
            ],
        )
        writer.writeheader()
        for i, re_value in enumerate(RE_VALUES):
            p = asdict(case_plan(float(re_value)))
            p["index"] = i
            writer.writerow(p)


def write_design_report(root: Path, sim_summaries: list[dict] | None = None, final: bool = False) -> None:
    counts: dict[str, int] = {}
    for re_value in RE_VALUES:
        counts[classify_regime(float(re_value))] = counts.get(classify_regime(float(re_value)), 0) + 1
    re_lines = "\n".join(f"{i:03d}: {v:.6f}  {classify_regime(float(v))}" for i, v in enumerate(RE_VALUES))
    completed = len([s for s in (sim_summaries or []) if s.get("status") in {"completed", "skipped"}])
    md = root / "PHYSICS_GENERALIZABLE_DATABASE_REPORT.md"
    md.write_text(
        f"""# Physics-Generalizable 2D Cylinder ROM Database

Generated/updated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Goal

This independent data-generation branch targets physics-generalizable parametric ROM training, not only interpolation accuracy in one Reynolds-number interval.  The database covers `Re=20-200`, spanning stable steady wake, the Hopf onset near `Re≈46-47`, developing periodic vortex shedding, mature two-dimensional periodic shedding, and the upper two-dimensional range before the real cylinder wake develops Mode-A three-dimensional instability.

Existing data and scripts are not modified.  Branch path:

`{BRANCH}`

Dataset path:

`{root}`

## Physical Basis

- The circular-cylinder wake onset is treated as a supercritical Hopf bifurcation near `Re≈46-47`.
- Three-dimensional secondary instability appears near `Re≈188.5` with Mode A, and another branch appears near `Re≈259`; therefore this physics-generalizable 2D database stops at `Re=200`.
- References used for this design:
  - Williamson, C. H. K. (1996), "Vortex Dynamics in the Cylinder Wake", Annual Review of Fluid Mechanics, DOI: https://doi.org/10.1146/annurev.fl.28.010196.002401
  - Barkley & Henderson (1996), "Three-dimensional Floquet stability analysis of the wake of a circular cylinder", Journal of Fluid Mechanics, https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/threedimensional-floquet-stability-analysis-of-the-wake-of-a-circular-cylinder/61575FBF0BC45054592D46382DEF30BB
  - Noack & Eckelmann (1994), "A global stability analysis of the steady and periodic cylinder wake", Journal of Fluid Mechanics.

## Reynolds Sampling

The 100 Reynolds numbers are generated automatically from a smooth density function rather than a fixed hand-written list.  The density is highest in `40-60`, high in `20-40` and `60-100`, moderate-high in `100-150`, and lighter but still covering `150-200`.

Regime counts:

{json.dumps(counts, indent=2)}

Actual Re distribution:

```text
{re_lines}
```

The same values are stored in `Re_sampling_strategy.csv`.

## Steady And Periodic Criteria

For `Re < 47`, the workflow first solves the stable steady wake with `simpleFoam` using residual, force, and volume-statistic monitoring.  This avoids contaminating the steady-wake regime with non-physical startup oscillations from an impulsive transient run.  The retained data are still written by restarting `pimpleFoam` from the steady field, so the saved files remain consistent with the transient OpenFOAM data path.  A case enters its retained-save window after:

- lift coefficient standard deviation is small,
- drag coefficient relative variation and trend are small,
- recent solver final residuals are below tolerance,
- volume-averaged field statistics are stable when available, or the steady solver has completed its conservative iteration cap and the exact diagnostics are recorded.

For `Re >= 47`, the workflow monitors lift coefficient peaks.  A stable limit cycle is accepted when recent peak amplitudes and peak-to-peak periods have small coefficient of variation after at least five estimated shedding periods.  At least 20 detected shedding periods are retained.

## Storage Flow

Each case runs in a temporary branch-local work directory.  After `pimpleFoam`:

1. `reconstructPar`
2. `foamToVTK -useTimeName -noFaceZones`
3. convert retained VTK point fields to compressed `.npz`
4. verify the `.npz`
5. delete VTK and temporary OpenFOAM work directory

## Area-Weighted POD

The POD uses lumped nodal 2D control areas.  OpenFOAM mesh cell volumes are exported through VTK and divided by the mesh thickness to form cell areas.  Point areas are assembled by distributing each adjacent cell area equally to that cell's vertices:

```text
A_i = sum_{{cell c containing point i}} Area(c) / N_vertices(c)
```

Velocity and pressure snapshots are weighted by `sqrt(A_i)` before SVD:

```text
X_w = sqrt(A) * X
Y_w = sqrt(A) * Y
```

Thus Euclidean snapshot products correspond to the finite-volume discrete L2 inner products:

```text
<q_a, q_b> = sum_i A_i (u_a,i u_b,i + v_a,i v_b,i)
<p_a, p_b> = sum_i A_i p_a,i p_b,i
```

The POD implementation is streaming/blockwise: it does not write a full weighted snapshot matrix to disk, which prevents disk exhaustion and avoids the large dense temporary arrays that caused problems in older runs.

## Current Status

- Final dataset completed: `{final}`
- Completed/skipped case count recorded in this report run: `{completed}`
- Current free disk: `{free_gb(root):.2f} GiB`
"""
    )


def write_sim_summary(root: Path, summaries: list[dict]) -> None:
    with (root / "Simulation_Summary.csv").open("w", newline="") as f:
        fieldnames = [
            "Re",
            "label",
            "regime",
            "status",
            "npz",
            "n_times",
            "n_points",
            "n_cells",
            "size_bytes",
            "retain_start_time",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s.get(k, "") for k in fieldnames})
    (root / "uvp_npz_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


def collect_existing_summaries(root: Path) -> list[dict]:
    summaries = []
    for re_value in RE_VALUES:
        plan = case_plan(float(re_value))
        meta_path = root / f"{plan.label}_uvp_metadata.json"
        npz_path = root / f"{plan.label}_uvp_pointData.npz"
        info_path = root / f"{plan.label}_info.txt"
        if meta_path.is_file() and npz_path.is_file() and info_path.is_file():
            meta = json.loads(meta_path.read_text())
            summaries.append(
                {
                    "Re": plan.Re,
                    "label": plan.label,
                    "regime": plan.regime,
                    "status": "skipped",
                    "npz": npz_path.name,
                    "n_times": meta["n_times"],
                    "n_points": meta["n_points"],
                    "n_cells": meta["n_cells"],
                    "size_bytes": npz_path.stat().st_size,
                    "retain_start_time": meta.get("retain_start_time", ""),
                }
            )
    return summaries


def dry_run(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_sampling_csv(root)
    write_design_report(root, final=False)
    values = RE_VALUES
    print(f"Dry-run only. Result root: {root}")
    print(f"Free disk: {free_gb(root):.2f} GiB")
    print(f"Generated {values.size} smooth nonuniform Re points: first={values[0]:.6f}, last={values[-1]:.6f}")
    print("Regime counts:")
    counts: dict[str, int] = {}
    for v in values:
        counts[classify_regime(float(v))] = counts.get(classify_regime(float(v)), 0) + 1
    for key, val in counts.items():
        print(f"  {key}: {val}")
    print(f"Wrote {root / 'Re_sampling_strategy.csv'}")
    print(f"Wrote {root / 'PHYSICS_GENERALIZABLE_DATABASE_REPORT.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run", action="store_true", help="Run OpenFOAM simulations. Omit for dry-run.")
    parser.add_argument("--pod-only", action="store_true", help="Skip simulations and compute area-weighted POD from existing NPZ files.")
    parser.add_argument("--skip-pod", action="store_true", help="Run simulations but skip final POD.")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of Re cases for validation/resume.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--nprocs", type=int, default=NPROCS_DEFAULT)
    args = parser.parse_args()

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    if not BASE_CASE.is_dir():
        raise SystemExit(f"Missing independent base case: {BASE_CASE}")
    write_sampling_csv(root)

    if not args.run and not args.pod_only:
        dry_run(root)
        return

    require_free(root, MIN_FREE_START_GB if args.run else MIN_FREE_POD_GB, "start")
    started = time.time()
    summaries = collect_existing_summaries(root)
    if args.run:
        selected = list(RE_VALUES)[args.start_index :]
        if args.max_cases is not None:
            selected = selected[: args.max_cases]
        for re_value in selected:
            plan = case_plan(float(re_value))
            if any(s.get("label") == plan.label for s in summaries):
                print(f"Skipping {plan.label}: already summarized.", flush=True)
                continue
            result = run_one_case(root, plan, args.nprocs)
            summaries.append(result)
            write_sim_summary(root, summaries)
            write_design_report(root, summaries, final=False)

    write_sim_summary(root, summaries)
    expected = len(RE_VALUES)
    complete = len([s for s in collect_existing_summaries(root)])
    if not args.skip_pod:
        wait_flag = root / "AUTO_POD_WAIT_FOR_SIMULATIONS"
        while args.pod_only and wait_flag.exists() and complete != expected:
            print(
                f"POD waiting for simulations: complete={complete}/{expected}, "
                f"free={free_gb(root):.2f}GiB",
                flush=True,
            )
            time.sleep(300)
            summaries = collect_existing_summaries(root)
            write_sim_summary(root, summaries)
            write_design_report(root, summaries, final=False)
            complete = len(summaries)
        if complete != expected:
            raise SystemExit(f"POD requires all {expected} cases; found {complete} complete cases.")
        pod_free = free_gb(root)
        (root / "POD_SPACE_CHECK.txt").write_text(
            f"Completed cases: {complete}/{expected}\n"
            f"Free disk before POD: {pod_free:.2f} GiB\n"
            f"Required free disk for POD: {MIN_FREE_POD_GB:.2f} GiB\n"
            f"Decision: {'run POD' if pod_free >= MIN_FREE_POD_GB else 'do not run POD'}\n"
        )
        if pod_free < MIN_FREE_POD_GB:
            raise SystemExit(
                f"POD blocked by disk space: {pod_free:.2f} GiB free, "
                f"{MIN_FREE_POD_GB:.2f} GiB required."
            )
        compute_area_weighted_pod(root)
    write_design_report(root, summaries, final=(complete == expected and not args.skip_pod))
    (root / "RUN_COMPLETE.txt").write_text(
        f"Completed workflow in {(time.time() - started) / 3600:.3f} hours.\n"
        f"Complete cases: {complete}/{expected}\n"
        f"Final free disk: {free_gb(root):.2f} GiB\n"
    )
    print(f"Workflow finished. Complete cases: {complete}/{expected}. Free disk: {free_gb(root):.2f} GiB")


if __name__ == "__main__":
    main()
