#!/usr/bin/env python3
"""Targeted acceptance checks for the expanded 34-Re Hopf POD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.split_config.read_text(encoding="utf-8"))
    expected = set(config["include_labels"]["hopf"])
    validation = set(config["validation_labels"]["hopf"])
    heldout = set(config["heldout_labels"]["hopf"])
    train = expected - validation - heldout
    if (len(expected), len(train), len(validation), len(heldout)) != (34, 29, 2, 3):
        raise ValueError("Split contract is not 34 = 29 train + 2 validation + 3 test")

    hopf = args.artifact_root / "hopf"
    manifest_path = hopf / "pod_artifact_manifest_hopf.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "PASS" or manifest["inputs"]["raw_case_count"] != 34:
        raise ValueError("POD manifest did not pass the 34-case contract")
    leakage = manifest["leakage_audit"]
    if any(leakage[key] for key in leakage if key != "fit_rows"):
        raise ValueError(f"Leakage audit failed: {leakage}")

    resolved = json.loads((args.artifact_root / "split_contract_resolved.json").read_text(encoding="utf-8"))
    rows = resolved["resolved"]["hopf"]
    by_split = {
        split: {row["Re_label"] for row in rows if row["split"] == split}
        for split in ("train", "validation", "heldout")
    }
    if by_split != {"train": train, "validation": validation, "heldout": heldout}:
        raise ValueError(f"Resolved split differs from preregistration: {by_split}")

    results: dict[str, object] = {
        "status": "PASS",
        "counts": {"all": 34, "train": 29, "validation": 2, "test": 3},
        "validation_labels": sorted(validation),
        "test_labels": sorted(heldout),
        "artifacts": {},
    }
    for variable, filename, basis_key, coeff_key in (
        ("velocity", "velocity_pod_hopf.npz", "phi_uv", "coeff_uv"),
        ("pressure", "pressure_pod_hopf.npz", "phi_p", "coeff_p"),
    ):
        path = hopf / filename
        with np.load(path, allow_pickle=False) as archive:
            labels_by_re = np.asarray(archive["Re_labels"]).astype(str)
            split_by_re = np.asarray(archive["split_by_Re"]).astype(str)
            labels = set(labels_by_re)
            fit_labels = set(labels_by_re[split_by_re == "train"])
            basis_shape = tuple(archive[basis_key].shape)
            coeff_shape = tuple(archive[coeff_key].shape)
            finite = np.isfinite(archive[basis_key]).all() and np.isfinite(archive[coeff_key]).all()
        if labels != expected or fit_labels != train:
            raise ValueError(f"{variable}: labels or train-only fit labels differ from contract")
        if basis_shape[0] != 80 or coeff_shape[1] != 80 or not finite:
            raise ValueError(f"{variable}: invalid rank/finite check: basis={basis_shape}, coeff={coeff_shape}")
        if {name: int(np.count_nonzero(split_by_re == name)) for name in ("train", "validation", "heldout")} != {
            "train": 29,
            "validation": 2,
            "heldout": 3,
        }:
            raise ValueError(f"{variable}: split_by_Re counts are wrong")
        results["artifacts"][filename] = {
            "sha256": sha256(path),
            "basis_shape": basis_shape,
            "coefficient_shape": coeff_shape,
        }

    for filename in ("velocity_rom_hopf.npz", "pressure_poisson_surrogate_hopf.npz"):
        path = hopf / filename
        if not path.is_file():
            raise FileNotFoundError(f"Consistent Hopf ROM tensor asset is missing: {path}")
        with np.load(path, allow_pickle=False) as archive:
            computed = set(np.asarray(archive["Re_labels_computed"]).astype(str))
            pod_labels = set(np.asarray(archive["pod_Re_labels"]).astype(str))
            finite = all(
                np.isfinite(archive[key]).all()
                for key in archive.files
                if np.issubdtype(archive[key].dtype, np.number)
            )
        if computed != expected or pod_labels != expected or not finite:
            raise ValueError(f"{filename}: Re coverage or finite tensor audit failed")
        results["artifacts"][filename] = {"sha256": sha256(path), "Re_count": len(computed)}

    output = args.artifact_root / "HOPF_EXPANDED_POD_ACCEPTANCE.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
