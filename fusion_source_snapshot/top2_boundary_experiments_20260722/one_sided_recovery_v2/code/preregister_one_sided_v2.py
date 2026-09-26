"""Pre-register the one-sided native-supported adjacent Top-2 recovery."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time


def digest(path: Path) -> str:
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    return value


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-preregistration", type=Path, required=True)
    parser.add_argument("--v1-failure-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v1_configs = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in args.v1_preregistration.glob("*.json") if path.name != "PREREGISTRATION.json"
    }
    domains = {
        "Steady_core": "S only",
        "S_H_boundary": "S database-native trajectories only; S indexed phase plus phase-free H",
        "Hopf_middle": "H only; no S/P candidate",
        "H_P_boundary": "P database-native trajectories only; P indexed phase plus phase-free H",
        "Periodic_core": "P only",
    }
    hashes = {}
    for filename, config in v1_configs.items():
        config.update({
            "protocol_version": "ONE_SIDED_NATIVE_SUPPORTED_V2",
            "domains": domains,
            "source_trajectory_contract": {
                "Steady-Hopf": "Steady train/validation trajectories only during development",
                "Hopf-Periodic": "Periodic train/validation trajectories only during development",
            },
            "phase_contract": "S/P may read only their own existing indexed database phase; H remains phase-free; no synthetic/zero/horizon-normalized phase",
            "test_contract": "test physical/modal states and metrics remain sealed until all started routes and thresholds are validation-frozen",
            "gate_independence": "G_SH_native and G_HP_native authorize training independently",
        })
        path = args.output_dir / filename
        atomic_json(path, config)
        path.chmod(0o444)
        hashes[filename] = digest(path)
    manifest = {
        "status": "ONE_SIDED_V2_PREREGISTERED_PENDING_INDEPENDENT_GATES",
        "created_unix": time.time(),
        "v1_preregistration": str(args.v1_preregistration),
        "v1_preregistration_sha256": digest(args.v1_preregistration / "PREREGISTRATION.json"),
        "v1_failure_report": str(args.v1_failure_report),
        "v1_failure_report_sha256": digest(args.v1_failure_report),
        "old_artifacts_modified": False,
        "domains": domains,
        "config_sha256": hashes,
        "test_seal": "SEALED",
        "known_test_history_disclosure": "Legacy test metrics were seen before this recovery; recovery tuning is mechanically train/validation-only and will not be called blind.",
    }
    atomic_json(args.output_dir / "PREREGISTRATION_V2.json", manifest)
    (args.output_dir / "PREREGISTRATION_V2.json").chmod(0o444)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
