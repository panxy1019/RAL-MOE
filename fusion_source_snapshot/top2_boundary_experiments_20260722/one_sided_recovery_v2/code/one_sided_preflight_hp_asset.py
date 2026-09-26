"""Metadata-only G_HP_native audit that preserves the Periodic test seal."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import zipfile


ROOT = Path("/root/panxy/particalMOE/periodic_specialist_r32")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pod_dir = ROOT / "assets/Global_POD_AreaWeighted_L2"
    index_path = pod_dir / "pod_snapshot_index.csv"
    bundles = [
        pod_dir / "global_velocity_pod_area_weighted_l2.npz",
        pod_dir / "global_pressure_pod_area_weighted_l2.npz",
    ]
    with index_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    split_column = next((name for name in columns if name.lower() in {"split", "subset", "partition"}), None)
    split_counts: dict[str, int] = {}
    if split_column:
        for row in rows:
            key = row[split_column]
            split_counts[key] = split_counts.get(key, 0) + 1
    archives = []
    coefficient_members = []
    for path in bundles:
        with zipfile.ZipFile(path) as archive:
            members = sorted(archive.namelist())
        coeff = [name for name in members if any(token in name.lower() for token in ("coeff", "modal", "score"))]
        coefficient_members.extend([f"{path.name}:{name}" for name in coeff])
        archives.append({"path": str(path), "sha256": sha256(path), "members": members})
    separate_trainval = sorted(str(path) for path in ROOT.rglob("*.npz") if "trainval" in path.name.lower() or "train_validation" in path.name.lower())
    result = {
        "gate": "G_HP_native",
        "decision": "BLOCKED_TEST_SEAL_ASSET",
        "protocol": "Periodic database-native phase/time plus phase-free Hopf",
        "audit_mode": "metadata/schema only; NPZ coefficient arrays were not loaded",
        "test_physical_modal_or_metric_data_loaded": False,
        "index_path": str(index_path),
        "index_sha256": sha256(index_path),
        "index_columns": columns,
        "index_split_counts": split_counts,
        "archives": archives,
        "coefficient_members_detected": coefficient_members,
        "separate_periodic_trainval_modal_assets": separate_trainval,
        "failure_reason": "Periodic native POD coefficient archives couple train/validation/test snapshots to one coefficient member and no independently materialized train/validation modal bundle exists. Loading the native rollout arrays before freeze would breach the recovery test seal.",
        "authorized_next_action": "Do not construct H-P cache or train H-P routes. A mechanically separated train/validation Periodic modal bundle may be materialized in a future audited data-preparation step without changing specialist checkpoints.",
        "independent_boundary_effect": "G_SH_native authorization is unaffected.",
    }
    atomic_json(args.output_dir / "G_HP_native.json", result)
    atomic_json(args.output_dir / "BLOCKED_TEST_SEAL_ASSET.json", {"gate": "G_HP_native", "decision": result["decision"]})
    print(json.dumps({"decision": result["decision"], "split_counts": split_counts, "trainval_assets": separate_trainval}, indent=2))


if __name__ == "__main__":
    main()
