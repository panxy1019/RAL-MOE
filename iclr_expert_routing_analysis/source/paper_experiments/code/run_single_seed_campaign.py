#!/usr/bin/env python3
"""Launch three regime state machines and emit only objective terminal manifests."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--vanilla-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    run = args.run_root.resolve()
    if run.exists():
        raise FileExistsError(run)
    (run / "logs").mkdir(parents=True)
    code = root / "paper_experiments/code/run_single_seed_regime.py"
    processes: dict[str, tuple[subprocess.Popen, object]] = {}
    for regime in ("steady", "hopf", "periodic"):
        stream = (run / "logs" / f"{regime}.log").open("w", encoding="utf-8")
        command = [
            args.python,
            str(code),
            "--root",
            str(root),
            "--run-root",
            str(run),
            "--regime",
            regime,
            "--python",
            args.python,
        ]
        if args.vanilla_only:
            command.append("--vanilla-only")
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        processes[regime] = (process, stream)
    atomic_json(
        run / "CAMPAIGN_STARTED.json",
        {
            "status": "RUNNING",
            "pid": os.getpid(),
            "regime_pids": {
                regime: process.pid
                for regime, (process, _stream) in processes.items()
            },
            "single_seed": True,
            "started_unix": time.time(),
        },
    )
    if not args.vanilla_only:
        while True:
            terminal = [
                (run / regime / "VANILLA_TERMINAL.json").is_file()
                for regime in ("steady", "hopf", "periodic")
            ]
            if all(terminal):
                break
            if all(process.poll() is not None for process, _stream in processes.values()):
                break
            time.sleep(10)
        (run / "START_DATAONLY").write_text(
            "all three Vanilla tasks reached a terminal state\n", encoding="utf-8"
        )
    return_codes = {}
    for regime, (process, stream) in processes.items():
        return_codes[regime] = process.wait()
        stream.close()
    matrix = []
    for regime in ("steady", "hopf", "periodic"):
        regime_failed = read_json(run / regime / "REGIME_FAILED.json")
        methods = ("vanilla-fnn",) if args.vanilla_only else ("vanilla-fnn", "data-only")
        for method in methods:
            done = read_json(run / regime / method / "DONE.json")
            failed = read_json(run / regime / method / "FAILED.json")
            frozen = read_json(
                run / regime / method / "FROZEN_SELECTION.json"
            )
            validation_pass = (
                run / regime / method / "evaluation/validation/PASS.json"
            ).is_file()
            test_pass = (
                run / regime / method / "evaluation/test/PASS.json"
            ).is_file()
            if done:
                status = "DONE"
                reason = None
            elif failed:
                status = "FAILED"
                reason = failed.get("error")
            elif regime_failed:
                status = "BLOCKED"
                reason = regime_failed.get("error")
            else:
                status = "BLOCKED"
                reason = "missing terminal artifact"
            matrix.append(
                {
                    "regime": regime,
                    "method": method,
                    "status": status,
                    "checkpoint": frozen.get("checkpoint"),
                    "checkpoint_sha256": frozen.get("checkpoint_sha256"),
                    "validation_selection": frozen.get("selection"),
                    "validation_evaluation_complete": validation_pass,
                    "test_complete": test_pass,
                    "result_directory": str(run / regime / method),
                    "failure_reason": reason,
                }
            )
    atomic_json(run / "FINAL_EXECUTION_MATRIX.json", {"experiments": matrix})
    atomic_json(run / "FINAL_RETURN_CODES.json", return_codes)
    atomic_json(
        run / "READY_FOR_OBJECTIVE_REPORT.json",
        {
            "ready": True,
            "all_expected_terminal": all(
                item["status"] in {"DONE", "FAILED", "BLOCKED"} for item in matrix
            ),
            "expected_experiment_count": 3 if args.vanilla_only else 6,
            "automatic_paper_report_generated": False,
            "finished_unix": time.time(),
        },
    )
    return 0 if all(code == 0 for code in return_codes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
