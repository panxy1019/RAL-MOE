from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return ROOT


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config" / "sweep.yaml"
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Missing PyYAML. Install it with: python3 -m pip install pyyaml"
        ) from exc
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def ensure_project_dirs(root: Path | None = None) -> None:
    root = root or ROOT
    for rel in [
        "templates/bottom_square_obstacle/0",
        "templates/bottom_square_obstacle/constant",
        "templates/bottom_square_obstacle/system",
        "data/npz",
        "data/manifest",
        "data/pod",
        "data/logs",
        "data/logs/figures",
        "data/vtk_preview",
        "work/cases",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def mode_settings(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    modes = cfg.get("modes", {})
    if mode in modes:
        return modes[mode]
    if mode in cfg and isinstance(cfg[mode], dict):
        return cfg[mode]
    available = sorted(set(modes) | {k for k, v in cfg.items() if isinstance(v, dict)})
    raise ValueError(f"Unknown mode {mode!r}. Available modes: {available}")


def effective_parallel_config(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    effective = dict(cfg)
    parallel = dict(cfg.get("parallel", {}))
    if mode == "pilot_fast":
        parallel.update(cfg.get("parallel_pilot_fast", {}))
    if mode == "pilot_viz":
        parallel.update(cfg.get("parallel_pilot_viz", {}))
    if mode == "scan_long_viz":
        parallel.update(cfg.get("parallel_scan_long_viz", {}))
    effective["parallel"] = parallel
    return effective


def re_tag(reynolds: float | int) -> str:
    r = float(reynolds)
    if abs(r - round(r)) < 1e-12:
        return f"{int(round(r)):06d}"
    return f"{r:010.4f}".replace(".", "p").replace("-", "m")


def case_name(reynolds: float | int) -> str:
    return f"case_Re_{re_tag(reynolds)}"


def case_dir(reynolds: float | int) -> Path:
    return ROOT / "work" / "cases" / case_name(reynolds)


def npz_path(reynolds: float | int) -> Path:
    return ROOT / "data" / "npz" / f"Re_{re_tag(reynolds)}.npz"


def log_path(reynolds: float | int, name: str) -> Path:
    return ROOT / "data" / "logs" / f"Re_{re_tag(reynolds)}_{name}.log"


def detect_total_cores(value: Any = "auto") -> int:
    if value == "auto" or value is None:
        return os.cpu_count() or 1
    return max(1, int(value))


def resolve_parallel(
    cfg: dict[str, Any],
    *,
    nprocs_override: int | None = None,
    concurrent_override: int | None = None,
    serial: bool = False,
) -> dict[str, Any]:
    parallel = cfg.get("parallel", {})
    total = detect_total_cores(parallel.get("total_cores", "auto"))
    enabled = bool(parallel.get("enabled", True)) and not serial
    nprocs = int(nprocs_override or parallel.get("nProcs_per_case", 1))
    concurrent = int(concurrent_override or parallel.get("concurrent_cases", 1))
    warnings: list[str] = []

    if not enabled:
        nprocs = 1

    nprocs = max(1, nprocs)
    concurrent = max(1, concurrent)

    if nprocs * concurrent > total:
        original = (nprocs, concurrent)
        concurrent = max(1, total // nprocs)
        if nprocs * concurrent > total:
            nprocs = max(1, total // concurrent)
        warnings.append(
            "Requested nProcs_per_case * concurrent_cases exceeds CPU cores: "
            f"{original[0]} * {original[1]} > {total}. "
            f"Using {nprocs} * {concurrent}."
        )

    return {
        "enabled": enabled,
        "total_cores": total,
        "nprocs": nprocs,
        "concurrent_cases": concurrent,
        "warnings": warnings,
        "method": parallel.get("decomposition_method", "scotch"),
        "fallback_method": parallel.get("fallback_decomposition_method", "simple"),
    }


def require_openfoam(parallel: bool = True, vtk: bool = True) -> None:
    commands = [
        "blockMesh",
        "checkMesh",
        "simpleFoam",
        "pimpleFoam",
    ]
    if vtk:
        commands += ["reconstructPar", "foamToVTK"]
    if parallel:
        commands += ["decomposePar", "mpirun"]
    missing = [cmd for cmd in commands if shutil.which(cmd) is None]
    if missing:
        raise RuntimeError(
            "OpenFOAM environment is not ready. Missing commands: "
            + ", ".join(missing)
            + "\nRun first: source /opt/openfoam13/etc/bashrc"
        )


def run_logged(
    cmd: list[str],
    *,
    cwd: Path,
    log_file: Path,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_file.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"# command: {json.dumps(cmd)}\n")
        log.write(f"# cwd: {cwd}\n")
        log.write(f"# started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
        code = proc.wait()
        elapsed = time.time() - start
        log.write(f"\n# exit_code: {code}\n# elapsed_seconds: {elapsed:.3f}\n")
    if check and code != 0:
        raise RuntimeError(f"Command failed with exit {code}: {' '.join(cmd)}\nLog: {log_file}")
    return code


def openfoam_header(obj: str, location: str, cls: str = "dictionary") -> str:
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| OpenFOAM dictionary generated by bottom_obstacle_attractor_rom              |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {cls};
    location    "{location}";
    object      {obj};
}}
// ************************************************************************* //

"""


def write_decompose_par(case_path: Path, nprocs: int, method: str) -> None:
    coeffs = ""
    if method == "simple":
        coeffs = """
simpleCoeffs
{
    n               ({nprocs} 1 1);
    delta           0.001;
}
""".format(nprocs=nprocs)
    text = (
        openfoam_header("decomposeParDict", "system")
        + f"""numberOfSubdomains {nprocs};

method          {method};
{coeffs}
distributed     no;

roots           ();
"""
    )
    (case_path / "system" / "decomposeParDict").write_text(text, encoding="utf-8")


def _fmt(value: float | int) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:.12g}"


def control_dict_text(
    cfg: dict[str, Any],
    *,
    application: str,
    start_from: str,
    start_time: float,
    end_time: float,
    delta_t: float,
    write_control: str,
    write_interval: float,
    adjust_time_step: bool,
    max_co: float,
    max_delta_t: float,
    purge_write: int = 0,
    function_write_control: str = "timeStep",
    function_write_interval: float = 1,
) -> str:
    probes = cfg.get("function_objects", {}).get("probes", [])
    probe_lines = "\n".join(
        f"            ({_fmt(p[0])} {_fmt(p[1])} {_fmt(p[2])})" for p in probes
    )
    adjust = "yes" if adjust_time_step else "no"
    return (
        openfoam_header("controlDict", "system")
        + f"""application     {application};

startFrom       {start_from};
startTime       {_fmt(start_time)};
stopAt          endTime;
endTime         {_fmt(end_time)};
deltaT          {_fmt(delta_t)};

writeControl    {write_control};
writeInterval   {_fmt(write_interval)};
purgeWrite      {int(purge_write)};
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   10;
runTimeModifiable true;

adjustTimeStep  {adjust};
maxCo           {_fmt(max_co)};
maxDeltaT       {_fmt(max_delta_t)};

functions
{{
    forces
    {{
        type            forces;
        libs            ("libforces.so");
        patches         (obstacle);
        rho             rhoInf;
        rhoInf          1;
        CofR            (0.5 0.5 0.05);
        writeControl    {function_write_control};
        writeInterval   {_fmt(function_write_interval)};
    }}

    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        patches         (obstacle);
        rho             rhoInf;
        rhoInf          1;
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        pitchAxis       (0 0 1);
        magUInf         1;
        lRef            1;
        Aref            1;
        CofR            (0.5 0.5 0.05);
        writeControl    {function_write_control};
        writeInterval   {_fmt(function_write_interval)};
    }}

    probes
    {{
        type            probes;
        libs            ("libsampling.so");
        fields          (U p);
        probeLocations
        (
{probe_lines}
        );
        writeControl    {function_write_control};
        writeInterval   {_fmt(function_write_interval)};
    }}
}}
"""
    )


def write_control_dict(case_path: Path, cfg: dict[str, Any], **kwargs: Any) -> None:
    (case_path / "system" / "controlDict").write_text(
        control_dict_text(cfg, **kwargs), encoding="utf-8"
    )


def write_viscosity_files(case_path: Path, nu: float) -> None:
    transport = (
        openfoam_header("transportProperties", "constant")
        + f"""transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] {_fmt(nu)};
"""
    )
    physical = (
        openfoam_header("physicalProperties", "constant")
        + f"""viscosityModel  constant;

nu              {_fmt(nu)} [m^2/s];
"""
    )
    (case_path / "constant" / "transportProperties").write_text(transport, encoding="utf-8")
    (case_path / "constant" / "physicalProperties").write_text(physical, encoding="utf-8")


def numeric_time_dirs(path: Path) -> list[float]:
    values: list[float] = []
    if not path.exists():
        return values
    for child in path.iterdir():
        if child.is_dir():
            try:
                values.append(float(child.name))
            except ValueError:
                pass
    return sorted(values)


def latest_time(case_path: Path, decomposed: bool) -> float:
    base = case_path / "processor0" if decomposed else case_path
    times = numeric_time_dirs(base)
    if not times:
        return 0.0
    return max(times)


def validate_npz(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        import numpy as np

        with np.load(path, allow_pickle=True) as data:
            for key in ["coords", "times", "U", "p", "Re", "nu", "metadata_json"]:
                if key not in data:
                    return False, f"missing field {key}"
            if data["U"].ndim != 3 or data["p"].ndim != 2:
                return False, "unexpected U/p dimensions"
            if not np.isfinite(data["coords"]).all():
                return False, "coords contain NaN/Inf"
            if not np.isfinite(data["times"]).all():
                return False, "times contain NaN/Inf"
            if not np.isfinite(data["U"]).all():
                return False, "U contains NaN/Inf"
            if not np.isfinite(data["p"]).all():
                return False, "p contains NaN/Inf"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, "ok"


def _lock_file(handle: Any) -> None:
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except Exception:
        return


def _unlock_file(handle: Any) -> None:
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:
        return


def upsert_csv_row(path: Path, key_fields: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    with path.open("a+", encoding="utf-8", newline="") as fh:
        _lock_file(fh)
        fh.seek(0)
        rows = list(csv.DictReader(fh)) if path.stat().st_size else []
        for old in rows:
            for key in old:
                if key not in fieldnames:
                    fieldnames.append(key)
        rows = [
            old
            for old in rows
            if not all(str(old.get(k, "")) == str(row.get(k, "")) for k in key_fields)
        ]
        rows.append({k: "" if row.get(k) is None else row.get(k) for k in fieldnames})
        fh.seek(0)
        fh.truncate()
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        fh.flush()
        _unlock_file(fh)


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        _lock_file(fh)
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()
        _unlock_file(fh)


def parse_numeric_table(path: Path) -> tuple[list[str], list[list[float]]]:
    columns: list[str] = []
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            tokens = line.lstrip("#").split()
            if tokens and tokens[0].lower() == "time":
                columns = tokens
            continue
        try:
            rows.append([float(item) for item in line.split()])
        except ValueError:
            continue
    return columns, rows


def _last_window(values: "Any", fraction: float = 0.5) -> "Any":
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    start = max(0, int(math.floor(arr.shape[0] * (1.0 - fraction))))
    return arr[start:]


def _estimate_period(times: "Any", signal: "Any") -> tuple[float, float]:
    import numpy as np

    t = np.asarray(times, dtype=float)
    y = np.asarray(signal, dtype=float)
    if t.size < 8 or np.std(y) <= 0:
        return math.nan, math.nan
    y = y - np.mean(y)
    peaks = []
    for i in range(1, y.size - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            peaks.append(t[i])
    if len(peaks) >= 3:
        periods = np.diff(peaks)
        period = float(np.median(periods))
        return period, float(1.0 / period) if period > 0 else math.nan
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return math.nan, math.nan
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(y.size, d=dt)
    if freqs.size <= 1:
        return math.nan, math.nan
    idx = int(np.argmax(np.abs(spec[1:])) + 1)
    freq = float(freqs[idx])
    return (float(1.0 / freq), freq) if freq > 0 else (math.nan, math.nan)


def _find_first(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[-1] if matches else None


def _find_all(root: Path, *patterns: str) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    return sorted(set(paths))


def _parse_probe_u(path: Path) -> tuple[float, int]:
    import numpy as np

    samples: list[list[float]] = []
    vector_re = re.compile(
        r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)"
    )
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        vecs = vector_re.findall(line)
        if vecs:
            samples.append([float(v[1]) for v in vecs])
    if not samples:
        return math.nan, 0
    arr = np.asarray(samples, dtype=float)
    tail = _last_window(arr)
    return float(np.std(tail)), int(arr.shape[0])


def _parse_probe_u_series(paths: list[Path]) -> tuple["Any", "Any"]:
    import numpy as np

    rows: list[tuple[float, list[float]]] = []
    vector_re = re.compile(
        r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)"
    )
    for path in paths:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if not parts:
                continue
            try:
                t = float(parts[0])
            except ValueError:
                continue
            vecs = vector_re.findall(line)
            if vecs:
                rows.append((t, [float(v[1]) for v in vecs]))
    if not rows:
        return np.asarray([]), np.asarray([[]])
    rows.sort(key=lambda item: item[0])
    times = np.asarray([item[0] for item in rows], dtype=float)
    values = np.asarray([item[1] for item in rows], dtype=float)
    _, unique_idx = np.unique(times, return_index=True)
    unique_idx.sort()
    return times[unique_idx], values[unique_idx]


def _parse_force_coeff_series(paths: list[Path]) -> tuple[list[str], "Any"]:
    import numpy as np

    columns: list[str] = []
    rows: list[list[float]] = []
    for path in paths:
        cols, vals = parse_numeric_table(path)
        if cols:
            columns = cols
        rows.extend(vals)
    if not rows:
        return columns, np.asarray([])
    arr = np.asarray(rows, dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    _, unique_idx = np.unique(arr[:, 0], return_index=True)
    unique_idx.sort()
    return columns, arr[unique_idx]


def _estimate_period_details(times: "Any", signal: "Any") -> tuple[float, float, int, float]:
    import numpy as np

    t = np.asarray(times, dtype=float)
    y = np.asarray(signal, dtype=float)
    if t.size < 8 or np.std(y) <= 0:
        return math.nan, math.nan, 0, math.nan
    y = y - np.mean(y)
    peaks = []
    for i in range(1, y.size - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            peaks.append(t[i])
    if len(peaks) >= 3:
        periods = np.diff(peaks)
        period = float(np.median(periods))
        cv = float(np.std(periods) / period) if period > 0 else math.nan
        return period, float(1.0 / period) if period > 0 else math.nan, len(peaks), cv
    period, freq = _estimate_period(t, y)
    return period, freq, len(peaks), math.nan


def compute_pilot_fast_diagnostics(
    case_path: Path,
    reynolds: float | int,
    *,
    pimple_start_time: float | None = None,
) -> dict[str, Any]:
    import numpy as np

    force_paths = _find_all(
        case_path,
        "postProcessing/forceCoeffs*/**/coefficient.dat",
        "postProcessing/forceCoeffs*/**/forceCoeffs.dat",
        "processor0/postProcessing/forceCoeffs*/**/coefficient.dat",
        "processor0/postProcessing/forceCoeffs*/**/forceCoeffs.dat",
    )
    probe_paths = _find_all(
        case_path,
        "postProcessing/probes*/**/U",
        "processor0/postProcessing/probes*/**/U",
    )

    diag: dict[str, Any] = {
        "Re": float(reynolds),
        "pimple_time_min": math.nan,
        "pimple_time_max": math.nan,
        "Cd_mean": math.nan,
        "Cd_std": math.nan,
        "Cd_relative_std": math.nan,
        "Cl_mean": math.nan,
        "Cl_std": math.nan,
        "probe_v_std_max": math.nan,
        "periodic_hint": False,
        "estimated_period": math.nan,
        "estimated_frequency": math.nan,
        "Strouhal": math.nan,
        "status": "no-diagnostics",
    }

    columns, force_arr = _parse_force_coeff_series(force_paths)
    probe_times, probe_v = _parse_probe_u_series(probe_paths)
    start = -math.inf if pimple_start_time is None else float(pimple_start_time) - 1e-10

    time_sources: list[np.ndarray] = []
    if force_arr.size:
        force_arr = force_arr[force_arr[:, 0] >= start]
        if force_arr.size:
            time_sources.append(force_arr[:, 0])
    if probe_times.size:
        mask = probe_times >= start
        probe_times = probe_times[mask]
        probe_v = probe_v[mask]
        if probe_times.size:
            time_sources.append(probe_times)

    if time_sources:
        all_times = np.concatenate(time_sources)
        diag["pimple_time_min"] = float(np.min(all_times))
        diag["pimple_time_max"] = float(np.max(all_times))

    cl_series = None
    time_series = None
    if force_arr.size:
        cd_idx = columns.index("Cd") if "Cd" in columns else (2 if force_arr.shape[1] > 2 else None)
        cl_idx = columns.index("Cl") if "Cl" in columns else (3 if force_arr.shape[1] > 3 else None)
        tail = _last_window(force_arr)
        if cd_idx is not None:
            cd = tail[:, cd_idx]
            diag["Cd_mean"] = float(np.mean(cd))
            diag["Cd_std"] = float(np.std(cd))
            if abs(diag["Cd_mean"]) > 1e-14:
                diag["Cd_relative_std"] = float(diag["Cd_std"] / abs(diag["Cd_mean"]))
        if cl_idx is not None:
            cl = tail[:, cl_idx]
            diag["Cl_mean"] = float(np.mean(cl))
            diag["Cl_std"] = float(np.std(cl))
            cl_series = cl
            time_series = tail[:, 0]

    if probe_v.size:
        tail_probe = _last_window(probe_v)
        if tail_probe.ndim == 2 and tail_probe.shape[1] > 0:
            diag["probe_v_std_max"] = float(np.max(np.std(tail_probe, axis=0)))

    if cl_series is not None and time_series is not None:
        period, freq, peak_count, peak_cv = _estimate_period_details(time_series, cl_series)
        diag["estimated_period"] = period
        diag["estimated_frequency"] = freq
        diag["Strouhal"] = freq
    else:
        peak_count = 0
        peak_cv = math.nan

    cl_std = diag["Cl_std"]
    pv_std = diag["probe_v_std_max"]
    cl_small = (not math.isfinite(cl_std)) or cl_std < 1e-5
    pv_small = (not math.isfinite(pv_std)) or pv_std < 1e-5
    if not time_sources:
        diag["status"] = "no-diagnostics"
    elif cl_small and pv_small:
        diag["status"] = "steady-like"
    elif peak_count >= 3 and math.isfinite(peak_cv) and peak_cv <= 0.3:
        diag["status"] = "periodic-like"
        diag["periodic_hint"] = True
    else:
        diag["status"] = "possibly onset/transient"
    return diag


def compute_case_diagnostics(case_path: Path, reynolds: float | int) -> dict[str, Any]:
    import numpy as np

    diag: dict[str, Any] = {
        "Re": float(reynolds),
        "Cd_mean": math.nan,
        "Cd_std": math.nan,
        "Cl_mean": math.nan,
        "Cl_std": math.nan,
        "probe_v_std": math.nan,
        "periodic_hint": False,
        "estimated_period": math.nan,
        "Strouhal": math.nan,
        "force_samples": 0,
        "probe_samples": 0,
    }
    coeff_path = _find_first(case_path, "postProcessing/forceCoeffs*/**/coefficient.dat")
    if coeff_path is None:
        coeff_path = _find_first(case_path, "postProcessing/forceCoeffs*/**/forceCoeffs.dat")
    if coeff_path is None:
        coeff_path = _find_first(case_path, "processor0/postProcessing/forceCoeffs*/**/coefficient.dat")
    if coeff_path is None:
        coeff_path = _find_first(case_path, "processor0/postProcessing/forceCoeffs*/**/forceCoeffs.dat")
    if coeff_path is not None:
        columns, rows = parse_numeric_table(coeff_path)
        if rows:
            arr = np.asarray(rows, dtype=float)
            diag["force_samples"] = int(arr.shape[0])
            cd_idx = columns.index("Cd") if "Cd" in columns else (1 if arr.shape[1] > 1 else None)
            cl_idx = columns.index("Cl") if "Cl" in columns else (3 if arr.shape[1] > 3 else None)
            tail = _last_window(arr)
            if cd_idx is not None:
                diag["Cd_mean"] = float(np.mean(tail[:, cd_idx]))
                diag["Cd_std"] = float(np.std(tail[:, cd_idx]))
            if cl_idx is not None:
                diag["Cl_mean"] = float(np.mean(tail[:, cl_idx]))
                diag["Cl_std"] = float(np.std(tail[:, cl_idx]))
                period, freq = _estimate_period(tail[:, 0], tail[:, cl_idx])
                diag["estimated_period"] = period
                diag["Strouhal"] = freq

    probe_path = _find_first(case_path, "postProcessing/probes*/**/U")
    if probe_path is None:
        probe_path = _find_first(case_path, "processor0/postProcessing/probes*/**/U")
    if probe_path is not None:
        probe_std, n = _parse_probe_u(probe_path)
        diag["probe_v_std"] = probe_std
        diag["probe_samples"] = n

    cl_std = diag.get("Cl_std", math.nan)
    pv_std = diag.get("probe_v_std", math.nan)
    diag["periodic_hint"] = bool(
        (math.isfinite(cl_std) and cl_std > 1e-5)
        or (math.isfinite(pv_std) and pv_std > 1e-5)
    )
    return diag


def read_npz_summary(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=True) as data:
        times = data["times"]
        return {
            "snapshots": int(times.shape[0]),
            "time_min": float(np.min(times)) if times.size else math.nan,
            "time_max": float(np.max(times)) if times.size else math.nan,
            "N": int(data["coords"].shape[0]),
        }


def print_warning(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)
