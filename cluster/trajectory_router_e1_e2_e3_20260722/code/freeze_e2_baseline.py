"""Freeze the selected E2 Router and its exact training/evaluation contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time


ROOT = Path("/root/panxy/particalMOE")
SOURCE = ROOT / "trajectory_router_e1_e2_e3_20260722/formal_E2_re_only_top1"
CODE = ROOT / "trajectory_router_e1_e2_e3_20260722/code/run_top1_experiment.py"
DESTINATION = ROOT / "trajectory_router_e1_e2_e3_20260722/frozen_E2_baseline_candidate"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    if DESTINATION.exists():
        raise RuntimeError(f"refusing to overwrite frozen destination: {DESTINATION}")
    required = {
        "best.pt": SOURCE / "best.pt",
        "STARTED.json": SOURCE / "STARTED.json",
        "DATA_AUDIT.json": SOURCE / "DATA_AUDIT.json",
        "metrics.json": SOURCE / "metrics.json",
        "run_top1_experiment.py": CODE,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    DESTINATION.mkdir(parents=True)
    inventory = {}
    for name, source in required.items():
        target = DESTINATION / name
        shutil.copy2(source, target)
        inventory[name] = {
            "source": str(source),
            "bytes": target.stat().st_size,
            "sha256": digest(target),
        }

    started = json.loads((DESTINATION / "STARTED.json").read_text(encoding="utf-8"))
    metrics = json.loads((DESTINATION / "metrics.json").read_text(encoding="utf-8"))
    config = {
        "experiment": "E2",
        "router": "Re-only trajectory-level fixed Top-1",
        "checkpoint": "best.pt",
        "checkpoint_sha256": inventory["best.pt"]["sha256"],
        "best_step": metrics["best_step"],
        "seed": started["seed"],
        "steps": started["steps"],
        "batch_size": started["batch_size"],
        "learning_rate": started["learning_rate"],
        "windows_per_Re": started["windows_per_Re"],
        "split": started["split"],
        "feature_contract": started["feature_contract"],
        "specialists": started["specialists"],
    }
    atomic_json(DESTINATION / "FROZEN_CONFIG.json", config)
    inventory["FROZEN_CONFIG.json"] = {
        "source": "generated from frozen STARTED.json and metrics.json",
        "bytes": (DESTINATION / "FROZEN_CONFIG.json").stat().st_size,
        "sha256": digest(DESTINATION / "FROZEN_CONFIG.json"),
    }
    manifest = {
        "status": "FROZEN_CANDIDATE_AWAITING_UNIFIED_WRAPPER_SMOKE",
        "created_unix": time.time(),
        "destination": str(DESTINATION),
        "inventory": inventory,
    }
    atomic_json(DESTINATION / "FREEZE_MANIFEST.json", manifest)
    for path in DESTINATION.iterdir():
        path.chmod(0o444)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
