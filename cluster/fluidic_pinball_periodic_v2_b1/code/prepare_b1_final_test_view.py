#!/usr/bin/env python3
"""Create the sealed final-test coefficient view after the B1 validation gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe-assets", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    velocity = np.load(args.moe_assets / "global_velocity_pod_area_weighted_l2.npz")
    pressure = np.load(args.moe_assets / "global_pressure_pod_area_weighted_l2.npz")
    with (args.moe_assets / "pod_snapshot_index.csv").open(newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["snapshot_id"]))
    split = np.asarray(velocity["snapshot_splits"]).astype(str)
    keep = split == "final_test"
    re_values = np.asarray([float(row["Re"]) for row in rows], dtype=np.float64)[keep]
    times = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)[keep]
    expected = np.asarray(manifest["heldout_reynolds"], dtype=np.float64)
    if not np.allclose(np.sort(np.unique(re_values)), np.sort(expected), atol=5e-7):
        raise RuntimeError("Final-test view does not match the sealed heldout Re list.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    view = args.output_dir / "fluidic_pinball_periodic_finaltest_rank999.npz"
    tmp = view.with_name(view.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        coeff_uv=np.asarray(velocity["coeff_uv"][keep, :17], dtype=np.float32),
        coeff_p=np.asarray(pressure["coeff_p"][keep, :16], dtype=np.float32),
        Re=re_values,
        time=times,
        phi_uv=np.asarray(velocity["phi_uv"][:17], dtype=np.float32),
        phi_p=np.asarray(pressure["phi_p"][:16], dtype=np.float32),
        point_areas=np.asarray(velocity["point_areas"], dtype=np.float32),
        mean_uv_train=np.asarray(velocity["mean_uv_regime"], dtype=np.float32),
        mean_p_train=np.asarray(pressure["mean_p_regime"], dtype=np.float32),
    )
    os.replace(tmp, view)
    seal = {
        "scope": "one_time_final_test_rollout",
        "heldout_reynolds": expected.tolist(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "final_test_view": str(view),
        "final_test_view_sha256": sha256(view),
        "training_assets_unchanged": True,
        "hyperparameter_updates_after_unseal_forbidden": True,
    }
    (args.output_dir / "FINAL_TEST_UNSEAL_MANIFEST.json").write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(seal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
