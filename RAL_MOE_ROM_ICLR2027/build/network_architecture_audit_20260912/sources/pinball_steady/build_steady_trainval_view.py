#!/usr/bin/env python3
"""Build a held-out-free steady rank999 coefficient view from Fluidic Pinball V2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--asset-manifest", type=Path, required=True)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_split(path: Path) -> tuple[list[float], list[float], list[float]]:
    groups: dict[str, list[float]] = {"train": [], "validation": [], "final_test": []}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            groups[row["split"]].append(float(row["Re"]))
    return groups["train"], groups["validation"], groups["final_test"]


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    pod = root / "rom_assets_v2" / "steady" / "pod"
    rom = root / "rom_assets_v2" / "steady" / "rom" / "rank999_ru4_rp3"
    split_path = root / "config" / "experts" / "steady_re_manifest.csv"
    train_re, validation_re, final_test_re = read_split(split_path)

    with np.load(pod / "weighted_pod_velocity.npz", allow_pickle=False) as u, \
         np.load(pod / "weighted_pod_pressure.npz", allow_pickle=False) as p, \
         np.load(rom / "pod_rank_pack.npz", allow_pickle=False) as pack:
        ru, rp = int(pack["ru"]), int(pack["rp"])
        if (ru, rp) != (4, 3):
            raise RuntimeError(f"expected steady rank999 (4, 3), got {(ru, rp)}")
        if not bool(u["train_only"]) or not bool(p["train_only"]):
            raise RuntimeError("POD coefficient sources are not marked train_only")
        u_offsets = json.loads(str(u["case_offsets"].item()))
        p_offsets = json.loads(str(p["case_offsets"].item()))
        if u_offsets != p_offsets:
            raise RuntimeError("velocity/pressure POD case offsets differ")
        a_parts = [u["coefficients"][:, :ru].astype(np.float32)]
        b_parts = [p["coefficients"][:, :rp].astype(np.float32)]
        t_parts = [u["snapshot_times"].astype(np.float64)]
        re_parts = [np.concatenate([
            np.full(int(row["count"]), float(row["Re"]), dtype=np.float64)
            for row in u_offsets
        ])]
        split_parts = [np.full(len(t_parts[0]), "train", dtype="U10")]
        source_parts = [u["snapshot_case_tags"].astype("U16")]
        if sorted(round(x, 8) for x in np.unique(re_parts[0])) != sorted(round(x, 8) for x in train_re):
            raise RuntimeError("POD coefficient Reynolds values do not match the frozen train split")
        phi_u = pack["Phi_u"].transpose(0, 2, 1).reshape(ru, -1).astype(np.float32)
        mean_u = pack["U_mean"].T.reshape(-1).astype(np.float32)
        phi_p = pack["Phi_p"].reshape(rp, -1).astype(np.float32)
        mean_p = pack["p_mean"].reshape(-1).astype(np.float32)
        cell_volumes = pack["cellVolumes"].reshape(-1).astype(np.float32)
        mesh_hash = str(u["mesh_hash"].item())

    validation_dir = root / "eval_pod_coefficients_v2" / "steady" / "validation"
    observed_validation: list[float] = []
    for path in sorted(validation_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            if str(z["split"].item()) != "validation" or str(z["expert"].item()) != "steady":
                raise RuntimeError(f"unexpected evaluation file contract: {path}")
            rv = float(z["Re"])
            observed_validation.append(rv)
            count = len(z["times"])
            a_parts.append(z["a_velocity_rank999"].astype(np.float32))
            b_parts.append(z["b_pressure_rank999"].astype(np.float32))
            t_parts.append(z["times"].astype(np.float64))
            re_parts.append(np.full(count, rv, dtype=np.float64))
            split_parts.append(np.full(count, "validation", dtype="U10"))
            source_parts.append(np.full(count, path.stem, dtype="U40"))
            if str(z["mesh_hash"].item()) != mesh_hash:
                raise RuntimeError(f"mesh hash mismatch: {path}")
    if sorted(round(x, 8) for x in observed_validation) != sorted(round(x, 8) for x in validation_re):
        raise RuntimeError("validation coefficient files do not match the frozen validation split")

    payload = {
        "a": np.concatenate(a_parts), "b": np.concatenate(b_parts),
        "re": np.concatenate(re_parts), "time": np.concatenate(t_parts),
        "split": np.concatenate(split_parts), "source_case": np.concatenate(source_parts),
        "phi_u": phi_u, "phi_p": phi_p, "cell_volumes": cell_volumes,
        "mean_u": mean_u, "mean_p": mean_p,
        "ru": np.asarray(ru), "rp": np.asarray(rp),
        "mesh_hash": np.asarray(mesh_hash),
    }
    for name, value in payload.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "f" and not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite output array: {name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)

    galerkin = rom / "velocity_galerkin_tensors.npz"
    pressure = rom / "pressure_poisson_tensors.npz"
    manifest = {
        "schema_version": 1,
        "scope": "fluidic_pinball_v2_steady_rank999_train_validation_only",
        "expert": "steady", "rank_label": "rank999", "r_u": ru, "r_p": rp,
        "mesh_hash": mesh_hash,
        "train_reynolds": train_re, "validation_reynolds": validation_re,
        "heldout_reynolds_hard_disabled": final_test_re,
        "train_snapshot_count": int(len(a_parts[0])),
        "validation_snapshot_count": int(sum(len(x) for x in a_parts[1:])),
        "split_contract_verified": True, "train_only_pod_verified": True,
        "rom_tensor_verified": True, "finite_verified": True,
        "sha256": {
            "coefficient_view": sha256(args.output),
            "galerkin": sha256(galerkin), "pressure": sha256(pressure),
            "split_manifest": sha256(split_path),
        },
    }
    args.asset_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.asset_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
