#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import subprocess
import sys
from pathlib import Path

from common import (
    ROOT,
    append_csv_row,
    case_dir,
    compute_case_diagnostics,
    effective_parallel_config,
    ensure_project_dirs,
    load_config,
    log_path,
    mode_settings,
    npz_path,
    resolve_parallel,
    validate_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Reynolds-number sweep.")
    parser.add_argument("--config", default=str(ROOT / "config" / "sweep.yaml"))
    parser.add_argument(
        "--mode",
        choices=["test", "pilot", "pilot_fast", "pilot_viz", "scan_long_viz", "probe_long", "full"],
        required=True,
    )
    parser.add_argument("--concurrent-cases", type=int)
    parser.add_argument("--nprocs-per-case", type=int)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-vtk", action="store_true")
    parser.add_argument("--copy-vtk-preview", action="store_true")
    parser.add_argument("--cleanup-case", action="store_true")
    return parser.parse_args()


def runner_command(args: argparse.Namespace, reynolds: float, nprocs: int) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "01_run_single_re.py"),
        "--re",
        str(reynolds),
        "--mode",
        args.mode,
        "--config",
        str(args.config),
        "--nprocs",
        str(nprocs),
    ]
    if args.serial:
        cmd.append("--serial")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.keep_vtk:
        cmd.append("--keep-vtk")
    if args.copy_vtk_preview:
        cmd.append("--copy-vtk-preview")
    if args.cleanup_case:
        cmd.append("--cleanup-case")
    return cmd


def run_one(args: argparse.Namespace, reynolds: float, nprocs: int) -> tuple[float, int, Path]:
    log_file = log_path(reynolds, "runner")
    with log_file.open("w", encoding="utf-8", errors="replace") as log:
        cmd = runner_command(args, reynolds, nprocs)
        log.write(f"# command: {cmd}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
        code = proc.wait()
    return reynolds, code, log_file


def write_diagnostics(reynolds: float) -> None:
    diag = compute_case_diagnostics(case_dir(reynolds), reynolds)
    append_csv_row(ROOT / "data" / "manifest" / "pilot_diagnostics.csv", diag)


def pilot_fast_success_exists(reynolds: float) -> bool:
    path = ROOT / "data" / "manifest" / "pilot_fast_diagnostics.csv"
    if not path.exists() or path.stat().st_size == 0:
        return False
    good_status = {"steady-like", "possibly onset/transient", "periodic-like"}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                same_re = abs(float(row.get("Re", "nan")) - float(reynolds)) < 1e-12
            except ValueError:
                same_re = False
            if same_re and row.get("status") in good_status:
                return True
    return False


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    mode_cfg = mode_settings(cfg, args.mode)
    ensure_project_dirs(ROOT)

    parallel_cfg = effective_parallel_config(cfg, args.mode)
    parallel = resolve_parallel(
        parallel_cfg,
        nprocs_override=args.nprocs_per_case,
        concurrent_override=args.concurrent_cases,
        serial=args.serial,
    )
    for warning in parallel["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    nprocs = int(parallel["nprocs"])
    concurrent_cases = int(parallel["concurrent_cases"])
    reynolds_values = [float(x) for x in mode_cfg.get("reynolds", mode_cfg.get("re_list", []))]
    if not reynolds_values:
        raise ValueError(f"No Reynolds list found for mode {args.mode}")

    print(
        f"Running {args.mode} sweep with concurrent_cases={concurrent_cases}, "
        f"nprocs_per_case={nprocs}, total_cores={parallel['total_cores']}"
    )

    pending: list[float] = []
    for re_value in reynolds_values:
        if args.mode == "pilot_fast":
            if pilot_fast_success_exists(re_value) and not args.overwrite:
                print(f"Skipping Re={re_value:g}; pilot_fast diagnostics already exist.")
                continue
            pending.append(re_value)
            continue
        if args.mode in {"pilot_viz", "scan_long_viz", "probe_long"}:
            pending.append(re_value)
            continue
        ok, _ = validate_npz(npz_path(re_value))
        if ok and not args.overwrite:
            print(f"Skipping Re={re_value:g}; valid npz exists.")
            if args.mode in {"pilot", "test"}:
                try:
                    write_diagnostics(re_value)
                except Exception as exc:  # noqa: BLE001
                    print(f"WARNING: diagnostics failed for skipped Re={re_value:g}: {exc}", file=sys.stderr)
            continue
        pending.append(re_value)

    failures: list[tuple[float, Path]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_cases) as pool:
        futures = [pool.submit(run_one, args, re_value, nprocs) for re_value in pending]
        for future in concurrent.futures.as_completed(futures):
            re_value, code, log_file = future.result()
            if code == 0:
                print(f"Re={re_value:g} finished.")
                if args.mode in {"pilot", "test"}:
                    try:
                        write_diagnostics(re_value)
                    except Exception as exc:  # noqa: BLE001
                        print(f"WARNING: diagnostics failed for Re={re_value:g}: {exc}", file=sys.stderr)
            else:
                failures.append((re_value, log_file))
                print(f"Re={re_value:g} failed; see {log_file}", file=sys.stderr)

    if failures:
        print("Sweep completed with failures:", file=sys.stderr)
        for re_value, log_file in failures:
            print(f"  Re={re_value:g}: {log_file}", file=sys.stderr)
        raise SystemExit(1)
    print("Sweep completed successfully.")


if __name__ == "__main__":
    main()
