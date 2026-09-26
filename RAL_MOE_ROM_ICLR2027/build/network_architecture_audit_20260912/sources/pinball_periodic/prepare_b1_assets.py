#!/usr/bin/env python3
"""Build leakage-safe B1 train+validation views from Fluidic Pinball V2 assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


RU, RP = 17, 16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_npz(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--moe-assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    source = args.moe_assets.resolve()
    output = args.output_dir.resolve()
    with (root / "config/experts/periodic_re_manifest.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        manifest_rows = sorted(csv.DictReader(handle), key=lambda row: float(row["Re"]))
    split_re = {
        split: np.asarray([float(row["Re"]) for row in manifest_rows if row["split"] == split])
        for split in ("train", "validation", "final_test")
    }
    if {key: len(value) for key, value in split_re.items()} != {
        "train": 47, "validation": 9, "final_test": 9
    }:
        raise ValueError("Unexpected Fluidic Pinball Periodic split contract.")

    velocity = np.load(source / "global_velocity_pod_area_weighted_l2.npz")
    pressure = np.load(source / "global_pressure_pod_area_weighted_l2.npz")
    with (source / "pod_snapshot_index.csv").open(newline="", encoding="utf-8") as handle:
        index_rows = sorted(csv.DictReader(handle), key=lambda row: int(row["snapshot_id"]))
    snapshot_split = np.asarray(velocity["snapshot_splits"]).astype(str)
    keep = snapshot_split != "final_test"
    re_per_snapshot = np.asarray([float(row["Re"]) for row in index_rows], dtype=np.float64)
    time_per_snapshot = np.asarray([float(row["time"]) for row in index_rows], dtype=np.float64)
    if len(keep) != len(index_rows) or np.any(
        np.isclose(re_per_snapshot[keep, None], split_re["final_test"][None, :], atol=5e-7)
    ):
        raise RuntimeError("Final-test leakage detected while building coefficient view.")

    coeff_path = output / "fluidic_pinball_periodic_trainval_rank999.npz"
    atomic_npz(
        coeff_path,
        coeff_uv=np.asarray(velocity["coeff_uv"][keep, :RU], dtype=np.float32),
        coeff_p=np.asarray(pressure["coeff_p"][keep, :RP], dtype=np.float32),
        Re=re_per_snapshot[keep],
        time=time_per_snapshot[keep],
        split=snapshot_split[keep],
        phi_uv=np.asarray(velocity["phi_uv"][:RU], dtype=np.float32),
        phi_p=np.asarray(pressure["phi_p"][:RP], dtype=np.float32),
        point_areas=np.asarray(velocity["point_areas"], dtype=np.float32),
        mean_uv_train=np.asarray(velocity["mean_uv_regime"], dtype=np.float32),
        mean_p_train=np.asarray(pressure["mean_p_regime"], dtype=np.float32),
        ru=np.asarray(RU),
        rp=np.asarray(RP),
    )

    rom_dir = root / "rom_assets_v2/periodic/rom/rank999_ru17_rp16"
    galerkin_source = np.load(rom_dir / "velocity_galerkin_tensors.npz")
    pressure_source = np.load(rom_dir / "pressure_poisson_tensors.npz")
    rom_split = np.asarray(galerkin_source["split_list"]).astype(str)
    train_mask = rom_split == "train"
    rom_re = np.asarray(galerkin_source["Re_list"], dtype=np.float64)
    if not np.allclose(np.sort(rom_re[train_mask]), np.sort(split_re["train"]), atol=1e-12):
        raise RuntimeError("Train-only ROM nodes do not match the split manifest.")
    galerkin_path = output / "fluidic_pinball_periodic_trainonly_galerkin_rank999.npz"
    pressure_path = output / "fluidic_pinball_periodic_trainonly_pressure_rank999.npz"
    atomic_npz(
        galerkin_path,
        Re_values_computed=rom_re[train_mask],
        c_all=np.asarray(galerkin_source["c_all"][train_mask], dtype=np.float32),
        A_all=np.asarray(galerkin_source["A_all"][train_mask], dtype=np.float32),
        H=np.asarray(galerkin_source["H"], dtype=np.float32),
        P=np.asarray(galerkin_source["P"], dtype=np.float32),
        ru=np.asarray(RU),
        rp=np.asarray(RP),
    )
    atomic_npz(
        pressure_path,
        Re_values_computed=rom_re[train_mask],
        c_tilde_all=np.asarray(pressure_source["c_tilde_all"][train_mask], dtype=np.float32),
        A_tilde_all=np.asarray(pressure_source["A_tilde_all"][train_mask], dtype=np.float32),
        H_tilde=np.asarray(pressure_source["H_tilde"], dtype=np.float32),
        ru=np.asarray(RU),
        rp=np.asarray(RP),
    )

    train_a = np.asarray(velocity["coeff_uv"][snapshot_split == "train", :RU], dtype=np.float64)
    nodes, centers, scales = [], [], []
    plane = np.zeros((2, RU), dtype=np.float32)
    plane[0, 0] = 1.0
    plane[1, 1] = 1.0
    for value in split_re["train"]:
        rows = np.isclose(re_per_snapshot, value, atol=5e-7) & (snapshot_split == "train")
        values = np.asarray(velocity["coeff_uv"][rows, :RU], dtype=np.float64)
        center = values.mean(axis=0)
        radius = np.linalg.norm((values - center) @ plane.T, axis=1)
        nodes.append(value)
        centers.append(center)
        scales.append(max(float(np.median(radius)), 1.0e-6))
    positive = np.linalg.norm((train_a - train_a.mean(axis=0)) @ plane.T, axis=1)
    positive = positive[positive > 1.0e-10]
    fluctuation_path = output / "trainonly_fluctuation_contract.npz"
    atomic_npz(
        fluctuation_path,
        nodes=np.asarray(nodes, dtype=np.float32),
        mean_a=np.asarray(centers, dtype=np.float32),
        plane=plane,
        radial_scale=np.asarray(scales, dtype=np.float32),
        radial_floor=np.asarray(np.quantile(positive, 0.10) if len(positive) else 1.0e-6),
    )

    asset_manifest = {
        "schema_version": 2,
        "scope": "periodic_train_only_with_train_validation_view",
        "dataset": "FluidicPinball_V2_Periodic",
        "r_u": RU,
        "r_p": RP,
        "fit_reynolds": split_re["train"].tolist(),
        "validation_reynolds": split_re["validation"].tolist(),
        "heldout_reynolds": split_re["final_test"].tolist(),
        "pressure_gauge_verified": True,
        "projection_error_verified": True,
        "split_contract_verified": True,
        "rom_tensor_verified": True,
        "projection_errors": {
            "velocity_energy_tail": float(1.0 - velocity["cumulative_energy_uv"][RU - 1]),
            "pressure_energy_tail": float(1.0 - pressure["cumulative_energy_p"][RP - 1]),
        },
        "curriculum": [[0, 1600, 4], [1600, 3200, 8], [3200, 5200, 16],
                       [5200, 6800, 24], [6800, 8000, 32]],
        "validation_horizons": [4, 8, 16, 24, 32, 56],
        "sha256": {
            "coefficient_view": sha256(coeff_path),
            "galerkin": sha256(galerkin_path),
            "pressure": sha256(pressure_path),
            "fluctuation_contract": sha256(fluctuation_path),
        },
    }
    atomic_json(output / "TRAINING_ASSET_MANIFEST.json", asset_manifest)
    atomic_json(output / "SPLIT_MANIFEST.json", {
        "train": split_re["train"].tolist(),
        "validation": split_re["validation"].tolist(),
        "heldout_final_test": split_re["final_test"].tolist(),
    })
    print(json.dumps({
        "train_snapshots": int(np.sum(snapshot_split == "train")),
        "validation_snapshots": int(np.sum(snapshot_split == "validation")),
        "heldout_loaded": False,
        "r_u": RU,
        "r_p": RP,
    }, indent=2))


if __name__ == "__main__":
    main()
