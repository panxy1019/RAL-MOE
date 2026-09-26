#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from common import (
    ROOT,
    case_dir,
    case_name,
    compute_case_diagnostics,
    compute_pilot_fast_diagnostics,
    effective_parallel_config,
    ensure_project_dirs,
    latest_time,
    load_config,
    log_path,
    mode_settings,
    npz_path,
    read_npz_summary,
    re_tag,
    require_openfoam,
    resolve_parallel,
    run_logged,
    upsert_csv_row,
    validate_npz,
    write_control_dict,
    write_decompose_par,
    write_viscosity_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Reynolds-number OpenFOAM case.")
    parser.add_argument("--re", type=float, required=True)
    parser.add_argument(
        "--mode",
        choices=["test", "pilot", "pilot_fast", "pilot_viz", "scan_long_viz", "probe_long", "full"],
        default="test",
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "sweep.yaml"))
    parser.add_argument("--nprocs", type=int, help="MPI ranks for this case")
    parser.add_argument("--serial", action="store_true", help="Disable MPI for debugging")
    parser.add_argument("--keep-case", action="store_true", help="Keep OpenFOAM case directory")
    parser.add_argument("--cleanup-case", action="store_true", help="Delete case after verified npz")
    parser.add_argument("--keep-vtk", action="store_true", help="Keep VTK directory after npz conversion")
    parser.add_argument("--copy-vtk-preview", action="store_true", help="Copy VTK output to data/vtk_preview/Re_XXXXXX")
    parser.add_argument("--overwrite", action="store_true", help="Rerun even if a valid npz exists")
    return parser.parse_args()


def command_for_solver(name: str, parallel_enabled: bool, nprocs: int) -> list[str]:
    if not parallel_enabled:
        return [name]
    return ["mpirun", "-np", str(nprocs), name, "-parallel"]


def write_stage_control(
    case_path: Path,
    cfg: dict,
    mode_cfg: dict,
    *,
    application: str,
    start_from: str,
    start_time: float,
    end_time: float,
    delta_t: float,
    write_control: str,
    write_interval: float,
    adjust_time_step: bool,
    purge_write: int = 0,
    function_write_control: str = "timeStep",
    function_write_interval: float = 1,
) -> None:
    write_control_dict(
        case_path,
        cfg,
        application=application,
        start_from=start_from,
        start_time=start_time,
        end_time=end_time,
        delta_t=delta_t,
        write_control=write_control,
        write_interval=write_interval,
        adjust_time_step=adjust_time_step,
        max_co=float(mode_cfg.get("maxCo", 0.8)),
        max_delta_t=float(mode_cfg.get("maxDeltaT", delta_t)),
        purge_write=purge_write,
        function_write_control=function_write_control,
        function_write_interval=function_write_interval,
    )


def prepare_case(case_path: Path, template: Path, overwrite: bool) -> None:
    if case_path.exists():
        if overwrite:
            shutil.rmtree(case_path)
        else:
            shutil.rmtree(case_path)
    shutil.copytree(template, case_path)


def run_pilot_fast_pipeline(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    mode_cfg = mode_settings(cfg, args.mode)
    ensure_project_dirs(ROOT)

    nu = 1.0 / args.re
    parallel_cfg = effective_parallel_config(cfg, args.mode)
    parallel = resolve_parallel(
        parallel_cfg,
        nprocs_override=args.nprocs,
        concurrent_override=1,
        serial=args.serial,
    )
    for warning in parallel["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    parallel_enabled = bool(parallel["enabled"])
    nprocs = int(parallel["nprocs"])
    run_id = uuid.uuid4().hex[:12]

    require_openfoam(parallel=parallel_enabled, vtk=False)

    template = ROOT / cfg["project"]["template"]
    if not (template / "system" / "blockMeshDict").exists():
        raise RuntimeError("Template missing. Run: python3 scripts/00_make_template.py")

    case_path = case_dir(args.re)
    prepare_case(case_path, template, overwrite=True)
    write_viscosity_files(case_path, nu)
    write_decompose_par(case_path, nprocs, str(parallel["method"]))

    run_logged(["blockMesh"], cwd=case_path, log_file=log_path(args.re, "blockMesh"))
    run_logged(
        ["checkMesh", "-allTopology", "-allGeometry"],
        cwd=case_path,
        log_file=log_path(args.re, "checkMesh"),
    )

    if parallel_enabled:
        try:
            run_logged(
                ["decomposePar", "-force"],
                cwd=case_path,
                log_file=log_path(args.re, "decomposePar"),
            )
        except RuntimeError:
            fallback = str(parallel.get("fallback_method") or "simple")
            print(f"decomposePar failed with {parallel['method']}; retrying with {fallback}", file=sys.stderr)
            write_decompose_par(case_path, nprocs, fallback)
            run_logged(
                ["decomposePar", "-force"],
                cwd=case_path,
                log_file=log_path(args.re, "decomposePar_fallback"),
            )

    simple_end = float(mode_cfg.get("simpleFoam_endTime", 80))
    purge_write = int(mode_cfg.get("purgeWrite", 1))
    force_interval = float(mode_cfg.get("force_probe_writeInterval", 0.05))
    field_interval = float(mode_cfg.get("field_writeInterval", 1000000))

    write_stage_control(
        case_path,
        cfg,
        mode_cfg,
        application="simpleFoam",
        start_from="startTime",
        start_time=0.0,
        end_time=simple_end,
        delta_t=1.0,
        write_control="timeStep",
        write_interval=simple_end,
        adjust_time_step=False,
        purge_write=purge_write,
        function_write_control="timeStep",
        function_write_interval=1,
    )
    run_logged(
        command_for_solver("simpleFoam", parallel_enabled, nprocs),
        cwd=case_path,
        log_file=log_path(args.re, "simpleFoam_parallel" if parallel_enabled else "simpleFoam"),
    )

    pimple_start = latest_time(case_path, decomposed=parallel_enabled)
    pimple_end = pimple_start + float(mode_cfg.get("pimpleFoam_endTime", 5))
    write_stage_control(
        case_path,
        cfg,
        mode_cfg,
        application="pimpleFoam",
        start_from="latestTime",
        start_time=0.0,
        end_time=pimple_end,
        delta_t=float(mode_cfg.get("deltaT", 0.02)),
        write_control="timeStep",
        write_interval=field_interval,
        adjust_time_step=True,
        purge_write=purge_write,
        function_write_control="runTime",
        function_write_interval=force_interval,
    )
    run_logged(
        command_for_solver("pimpleFoam", parallel_enabled, nprocs),
        cwd=case_path,
        log_file=log_path(args.re, "pimpleFoam_pilot_fast_parallel" if parallel_enabled else "pimpleFoam_pilot_fast"),
    )

    diag = compute_pilot_fast_diagnostics(case_path, args.re, pimple_start_time=pimple_start)
    diag.update(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "nProcs": nprocs,
            "case_dir": str(case_path),
            "simpleFoam_log": str(log_path(args.re, "simpleFoam_parallel" if parallel_enabled else "simpleFoam")),
            "pimpleFoam_log": str(
                log_path(args.re, "pimpleFoam_pilot_fast_parallel" if parallel_enabled else "pimpleFoam_pilot_fast")
            ),
        }
    )
    diagnostics_name = "pilot_fast_diagnostics.csv" if args.mode == "pilot_fast" else "probe_long_diagnostics.csv"
    diagnostics_path = ROOT / "data" / "manifest" / diagnostics_name
    upsert_csv_row(diagnostics_path, ["Re"], diag)

    row = manifest_row(
        args,
        nu,
        nprocs,
        True,
        "pilot_fast_ok",
        summary={
            "snapshots": "",
            "N": "",
            "time_min": diag["pimple_time_min"],
            "time_max": diag["pimple_time_max"],
        },
        npz_file=Path(""),
        retained_range=(diag["pimple_time_min"], diag["pimple_time_max"]),
    )
    row["timestamp"] = diag["timestamp"]
    row["run_id"] = run_id
    row["npz_path"] = ""
    row["pimpleFoam_spinup_log"] = ""
    row["pimpleFoam_retain_log"] = ""
    row["pimpleFoam_pilot_fast_log"] = diag["pimpleFoam_log"]
    row["status"] = "pilot_fast_ok" if args.mode == "pilot_fast" else "probe_long_ok"
    row["diagnostics_path"] = str(diagnostics_path)
    upsert_csv_row(ROOT / "data" / "manifest" / "cases.csv", ["Re"], row)

    print(
        f"{args.mode} diagnostics: "
        f"Re={diag['Re']:g}, time=[{diag['pimple_time_min']}, {diag['pimple_time_max']}], "
        f"Cd_mean={diag['Cd_mean']}, Cd_std={diag['Cd_std']}, "
        f"Cl_mean={diag['Cl_mean']}, Cl_std={diag['Cl_std']}, "
        f"probe_v_std_max={diag['probe_v_std_max']}, "
        f"periodic_hint={diag['periodic_hint']}, status={diag['status']}"
    )
    return row


def run_pilot_viz_pipeline(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    mode_cfg = mode_settings(cfg, args.mode)
    ensure_project_dirs(ROOT)

    nu = 1.0 / args.re
    parallel_cfg = effective_parallel_config(cfg, args.mode)
    parallel = resolve_parallel(
        parallel_cfg,
        nprocs_override=args.nprocs,
        concurrent_override=1,
        serial=args.serial,
    )
    for warning in parallel["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    parallel_enabled = bool(parallel["enabled"])
    nprocs = int(parallel["nprocs"])
    run_id = uuid.uuid4().hex[:12]

    require_openfoam(parallel=parallel_enabled, vtk=True)
    template = ROOT / cfg["project"]["template"]
    if not (template / "system" / "blockMeshDict").exists():
        raise RuntimeError("Template missing. Run: python3 scripts/00_make_template.py")

    case_path = case_dir(args.re)
    prepare_case(case_path, template, overwrite=True)
    write_viscosity_files(case_path, nu)
    write_decompose_par(case_path, nprocs, str(parallel["method"]))

    run_logged(["blockMesh"], cwd=case_path, log_file=log_path(args.re, "blockMesh"))
    run_logged(
        ["checkMesh", "-allTopology", "-allGeometry"],
        cwd=case_path,
        log_file=log_path(args.re, "checkMesh"),
    )
    if parallel_enabled:
        try:
            run_logged(
                ["decomposePar", "-force"],
                cwd=case_path,
                log_file=log_path(args.re, "decomposePar"),
            )
        except RuntimeError:
            fallback = str(parallel.get("fallback_method") or "simple")
            print(f"decomposePar failed with {parallel['method']}; retrying with {fallback}", file=sys.stderr)
            write_decompose_par(case_path, nprocs, fallback)
            run_logged(
                ["decomposePar", "-force"],
                cwd=case_path,
                log_file=log_path(args.re, "decomposePar_fallback"),
            )

    simple_end = float(mode_cfg.get("simpleFoam_endTime", 80))
    purge_write = int(mode_cfg.get("purgeWrite", 0))
    force_interval = float(mode_cfg.get("force_probe_writeInterval", 0.05))
    field_interval = float(mode_cfg.get("field_writeInterval", 1.0))
    write_stage_control(
        case_path,
        cfg,
        mode_cfg,
        application="simpleFoam",
        start_from="startTime",
        start_time=0.0,
        end_time=simple_end,
        delta_t=1.0,
        write_control="timeStep",
        write_interval=simple_end,
        adjust_time_step=False,
        purge_write=purge_write,
        function_write_control="timeStep",
        function_write_interval=1,
    )
    run_logged(
        command_for_solver("simpleFoam", parallel_enabled, nprocs),
        cwd=case_path,
        log_file=log_path(args.re, "simpleFoam_parallel" if parallel_enabled else "simpleFoam"),
    )

    pimple_start = latest_time(case_path, decomposed=parallel_enabled)
    pimple_end = pimple_start + float(mode_cfg.get("pimpleFoam_endTime", 5))
    write_stage_control(
        case_path,
        cfg,
        mode_cfg,
        application="pimpleFoam",
        start_from="latestTime",
        start_time=0.0,
        end_time=pimple_end,
        delta_t=float(mode_cfg.get("deltaT", 0.02)),
        write_control="runTime",
        write_interval=field_interval,
        adjust_time_step=True,
        purge_write=purge_write,
        function_write_control="runTime",
        function_write_interval=force_interval,
    )
    run_logged(
        command_for_solver("pimpleFoam", parallel_enabled, nprocs),
        cwd=case_path,
        log_file=log_path(args.re, f"pimpleFoam_{args.mode}_parallel" if parallel_enabled else f"pimpleFoam_{args.mode}"),
    )
    pimple_end = latest_time(case_path, decomposed=parallel_enabled)

    if parallel_enabled:
        run_logged(
            ["reconstructPar", "-time", f"{pimple_start:.12g}:"],
            cwd=case_path,
            log_file=log_path(args.re, f"reconstructPar_{args.mode}"),
        )

    vtk_dir = case_path / "VTK"
    if vtk_dir.exists():
        shutil.rmtree(vtk_dir)
    run_logged(
        [
            "foamToVTK",
            "-ascii",
            "-useTimeName",
            "-fields",
            "(U p)",
            "-time",
            f"{pimple_start:.12g}:",
        ],
        cwd=case_path,
        log_file=log_path(args.re, f"foamToVTK_{args.mode}"),
    )
    vtk_files = sorted(vtk_dir.glob("*.vtk"))
    keep_vtk = bool(args.keep_vtk or mode_cfg.get("keep_vtk", True))
    preview_dir = ROOT / "data" / "vtk_preview" / f"Re_{re_tag(args.re)}"
    copied_preview = False
    copy_preview = bool(args.copy_vtk_preview or mode_cfg.get("copy_vtk_preview", False))
    if copy_preview:
        if preview_dir.exists():
            shutil.rmtree(preview_dir)
        shutil.copytree(vtk_dir, preview_dir)
        copied_preview = True
    if not keep_vtk and not copy_preview:
        shutil.rmtree(vtk_dir, ignore_errors=True)

    row = manifest_row(
        args,
        nu,
        nprocs,
        True,
        f"{args.mode}_ok",
        summary={"snapshots": len(vtk_files), "N": "", "time_min": pimple_start, "time_max": pimple_end},
        npz_file=Path(""),
        retained_range=(pimple_start, pimple_end),
    )
    row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    row["run_id"] = run_id
    row["npz_path"] = ""
    row["vtk_path"] = str(vtk_dir)
    row["vtk_preview_path"] = str(preview_dir) if copied_preview else ""
    row["vtk_time_steps"] = len(vtk_files)
    upsert_csv_row(ROOT / "data" / "manifest" / "cases.csv", ["Re"], row)

    print(f"ParaView VTK path: {vtk_dir}")
    print(f"VTK time steps: {len(vtk_files)}")
    print(f"OpenFOAM case path: {case_path}")
    print(f"Copied VTK preview: {copied_preview}")
    if copied_preview:
        print(f"VTK preview path: {preview_dir}")
    if vtk_files:
        print(f"ParaView open file example: {vtk_files[0]}")
    return row


def run_pipeline(args: argparse.Namespace) -> dict:
    if args.mode in {"pilot_fast", "probe_long"}:
        return run_pilot_fast_pipeline(args)
    if args.mode in {"pilot_viz", "scan_long_viz"}:
        return run_pilot_viz_pipeline(args)

    cfg = load_config(args.config)
    mode_cfg = mode_settings(cfg, args.mode)
    ensure_project_dirs(ROOT)

    nu = 1.0 / args.re
    parallel = resolve_parallel(
        cfg,
        nprocs_override=args.nprocs,
        concurrent_override=1,
        serial=args.serial,
    )
    for warning in parallel["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    parallel_enabled = bool(parallel["enabled"])
    nprocs = int(parallel["nprocs"])

    out_npz = npz_path(args.re)
    if not args.overwrite:
        ok, reason = validate_npz(out_npz)
        if ok:
            summary = read_npz_summary(out_npz)
            row = manifest_row(
                args,
                nu,
                nprocs,
                True,
                "skipped_existing_npz",
                summary=summary,
                npz_file=out_npz,
            )
            upsert_csv_row(ROOT / "data" / "manifest" / "cases.csv", ["Re"], row)
            print(f"Valid npz already exists, skipping: {out_npz}")
            return row
        if out_npz.exists():
            print(f"Existing npz is invalid ({reason}); rerunning.", file=sys.stderr)

    require_openfoam(parallel=parallel_enabled)

    template = ROOT / cfg["project"]["template"]
    if not (template / "system" / "blockMeshDict").exists():
        raise RuntimeError("Template missing. Run: python3 scripts/00_make_template.py")

    case_path = case_dir(args.re)
    prepare_case(case_path, template, overwrite=True)
    write_viscosity_files(case_path, nu)
    write_decompose_par(case_path, nprocs, str(parallel["method"]))

    run_logged(["blockMesh"], cwd=case_path, log_file=log_path(args.re, "blockMesh"))
    run_logged(
        ["checkMesh", "-allTopology", "-allGeometry"],
        cwd=case_path,
        log_file=log_path(args.re, "checkMesh"),
    )

    if parallel_enabled:
        try:
            run_logged(
                ["decomposePar", "-force"],
                cwd=case_path,
                log_file=log_path(args.re, "decomposePar"),
            )
        except RuntimeError:
            fallback = str(parallel.get("fallback_method") or "simple")
            print(f"decomposePar failed with {parallel['method']}; retrying with {fallback}", file=sys.stderr)
            write_decompose_par(case_path, nprocs, fallback)
            run_logged(
                ["decomposePar", "-force"],
                cwd=case_path,
                log_file=log_path(args.re, "decomposePar_fallback"),
            )

    simple_end = float(mode_cfg.get("simple_iterations", 100))
    write_stage_control(
        case_path,
        cfg,
        mode_cfg,
        application="simpleFoam",
        start_from="startTime",
        start_time=0.0,
        end_time=simple_end,
        delta_t=1.0,
        write_control="timeStep",
        write_interval=simple_end,
        adjust_time_step=False,
    )
    run_logged(
        command_for_solver("simpleFoam", parallel_enabled, nprocs),
        cwd=case_path,
        log_file=log_path(args.re, "simpleFoam_parallel" if parallel_enabled else "simpleFoam"),
    )

    latest_after_simple = latest_time(case_path, decomposed=parallel_enabled)
    spinup_end = latest_after_simple + float(mode_cfg["endTime_spinup"])
    write_stage_control(
        case_path,
        cfg,
        mode_cfg,
        application="pimpleFoam",
        start_from="latestTime",
        start_time=0.0,
        end_time=spinup_end,
        delta_t=float(mode_cfg["deltaT"]),
        write_control="runTime",
        write_interval=float(mode_cfg["writeInterval"]),
        adjust_time_step=True,
    )
    run_logged(
        command_for_solver("pimpleFoam", parallel_enabled, nprocs),
        cwd=case_path,
        log_file=log_path(args.re, "pimpleFoam_spinup_parallel" if parallel_enabled else "pimpleFoam_spinup"),
    )

    retain_start = latest_time(case_path, decomposed=parallel_enabled)
    retain_end = retain_start + float(mode_cfg["endTime_retain"])
    write_stage_control(
        case_path,
        cfg,
        mode_cfg,
        application="pimpleFoam",
        start_from="latestTime",
        start_time=0.0,
        end_time=retain_end,
        delta_t=float(mode_cfg["deltaT"]),
        write_control="runTime",
        write_interval=float(mode_cfg["writeInterval"]),
        adjust_time_step=True,
    )
    run_logged(
        command_for_solver("pimpleFoam", parallel_enabled, nprocs),
        cwd=case_path,
        log_file=log_path(args.re, "pimpleFoam_retain_parallel" if parallel_enabled else "pimpleFoam_retain"),
    )
    retain_end = latest_time(case_path, decomposed=parallel_enabled)

    if parallel_enabled:
        reconstruct_log = log_path(args.re, "reconstructPar")
        code = run_logged(
            ["reconstructPar", "-time", f"{retain_start:.12g}:"],
            cwd=case_path,
            log_file=reconstruct_log,
            check=False,
        )
        if code != 0:
            run_logged(
                ["reconstructPar", "-newTimes"],
                cwd=case_path,
                log_file=log_path(args.re, "reconstructPar_newTimes"),
            )

    vtk_dir = case_path / "VTK"
    run_logged(
        [
            "foamToVTK",
            "-ascii",
            "-useTimeName",
            "-fields",
            "(U p)",
            "-time",
            f"{retain_start:.12g}:",
        ],
        cwd=case_path,
        log_file=log_path(args.re, "foamToVTK"),
    )

    convert_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "03_vtk_to_npz.py"),
        "--case-dir",
        str(case_path),
        "--vtk-dir",
        str(vtk_dir),
        "--output",
        str(out_npz),
        "--re",
        str(args.re),
        "--nu",
        str(nu),
        "--nprocs",
        str(nprocs),
        "--mode",
        args.mode,
        "--retain-start",
        str(retain_start),
        "--retain-end",
        str(retain_end),
        "--config",
        str(args.config),
    ]
    if args.keep_vtk:
        convert_cmd.append("--keep-vtk")
    run_logged(convert_cmd, cwd=ROOT, log_file=log_path(args.re, "vtk_to_npz"))

    ok, reason = validate_npz(out_npz)
    if not ok:
        raise RuntimeError(f"npz validation failed after conversion: {reason}")
    summary = read_npz_summary(out_npz)

    if args.cleanup_case:
        shutil.rmtree(case_path, ignore_errors=True)

    row = manifest_row(
        args,
        nu,
        nprocs,
        True,
        "ok",
        summary=summary,
        npz_file=out_npz,
        retained_range=(retain_start, retain_end),
    )
    upsert_csv_row(ROOT / "data" / "manifest" / "cases.csv", ["Re"], row)
    print(f"Finished Re={args.re:g}: {out_npz}")
    return row


def manifest_row(
    args: argparse.Namespace,
    nu: float,
    nprocs: int,
    success: bool,
    status: str,
    *,
    summary: dict | None = None,
    npz_file: Path | None = None,
    retained_range: tuple[float, float] | None = None,
) -> dict:
    summary = summary or {}
    retained_range = retained_range or (
        summary.get("time_min", math.nan),
        summary.get("time_max", math.nan),
    )
    return {
        "Re": f"{args.re:g}",
        "nu": f"{nu:.12g}",
        "mode": args.mode,
        "nProcs": nprocs,
        "snapshots": summary.get("snapshots", ""),
        "N": summary.get("N", ""),
        "time_min": summary.get("time_min", retained_range[0]),
        "time_max": summary.get("time_max", retained_range[1]),
        "npz_path": str(npz_file or npz_path(args.re)),
        "success": success,
        "status": status,
        "case_dir": str(case_dir(args.re)),
        "blockMesh_log": str(log_path(args.re, "blockMesh")),
        "checkMesh_log": str(log_path(args.re, "checkMesh")),
        "simpleFoam_log": str(
            log_path(args.re, "simpleFoam_parallel" if not args.serial else "simpleFoam")
        ),
        "pimpleFoam_spinup_log": str(
            log_path(args.re, "pimpleFoam_spinup_parallel" if not args.serial else "pimpleFoam_spinup")
        ),
        "pimpleFoam_retain_log": str(
            log_path(args.re, "pimpleFoam_retain_parallel" if not args.serial else "pimpleFoam_retain")
        ),
    }


def main() -> None:
    args = parse_args()
    try:
        row = run_pipeline(args)
        if args.mode in {"pilot_fast", "probe_long", "pilot_viz", "scan_long_viz"}:
            return
        try:
            diag = compute_case_diagnostics(case_dir(args.re), args.re)
            if row.get("success") is True or str(row.get("success")).lower() == "true":
                print(
                    "Diagnostics: "
                    f"Cd_mean={diag['Cd_mean']}, Cd_std={diag['Cd_std']}, "
                    f"Cl_mean={diag['Cl_mean']}, Cl_std={diag['Cl_std']}, "
                    f"probe_v_std={diag['probe_v_std']}, periodic_hint={diag['periodic_hint']}"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: diagnostics failed: {exc}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        cfg = load_config(args.config)
        nu = 1.0 / args.re
        parallel_cfg = effective_parallel_config(cfg, args.mode)
        nprocs = args.nprocs or parallel_cfg.get("parallel", {}).get("nProcs_per_case", 1)
        fail_row = manifest_row(args, nu, int(nprocs), False, str(exc))
        fail_row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        upsert_csv_row(ROOT / "data" / "manifest" / "cases.csv", ["Re"], fail_row)
        print("\nERROR: single-Re pipeline failed", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
