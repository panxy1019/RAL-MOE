#!/usr/bin/env python3
"""Restore only periodic train assets from the private CenteredSquare dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "panxy1019/centeredSquare"
PERIODIC = "three_regime_overlap_v1/subsets/periodic"
FIXED_FILES = [
    f"{PERIODIC}/pod/weighted_pod_velocity.npz",
    f"{PERIODIC}/pod/weighted_pod_pressure.npz",
    f"{PERIODIC}/mesh/mesh_metadata.npz",
    "three_regime_overlap_v1/config/canonical_cases.json",
    "three_regime_overlap_v1/config/regime_split_config.json",
    "three_regime_overlap_v1/config/split_contract_resolved.json",
    "scripts/dataset_tools.py",
]


def fetch(path, output):
    return Path(hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=path, local_dir=output,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for path in FIXED_FILES:
        fetch(path, args.output)

    velocity_pod = args.output / FIXED_FILES[0]
    with np.load(velocity_pod, allow_pickle=False) as pod:
        train_tags = sorted(set(np.asarray(pod["snapshot_case_tags"]).astype(str).tolist()))
    if len(train_tags) != 27:
        raise RuntimeError(f"periodic POD contains {len(train_tags)} rather than 27 train cases")
    train_files = [f"{PERIODIC}/cases_npz/snapshots_{tag}.npz" for tag in train_tags]
    for path in train_files:
        fetch(path, args.output)

    # Restore one original OpenFOAM case only as a boundary/mesh template.
    api = HfApi()
    template_files = ["Re100/100/U", "Re100/100/p"]
    for directory in ("Re100/constant", "Re100/system"):
        template_files.extend(
            item.path for item in api.list_repo_tree(
                REPO_ID, repo_type="dataset", path_in_repo=directory, recursive=True,
            ) if item.__class__.__name__ == "RepoFile"
        )
    template_files = sorted(set(template_files))
    if not template_files or "Re100/100/U" not in template_files or "Re100/100/p" not in template_files:
        raise FileNotFoundError("Re100 OpenFOAM template is incomplete")
    for path in template_files:
        fetch(path, args.output)

    manifest = {
        "schema_version": 1,
        "repository": REPO_ID,
        "train_tags": train_tags,
        "train_case_count": len(train_tags),
        "template_file_count": len(template_files),
        "validation_snapshots_downloaded": False,
        "heldout_snapshots_downloaded": False,
    }
    (args.output / "PERIODIC_TRAIN_RESTORE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
