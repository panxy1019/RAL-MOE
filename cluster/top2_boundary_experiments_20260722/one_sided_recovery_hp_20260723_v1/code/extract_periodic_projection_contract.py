#!/usr/bin/env python3
"""Extract the frozen Periodic projection contract without coefficient members."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np


def load_member(path: Path, key: str) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        with archive.open(f"{key}.npy") as handle:
            return np.load(io.BytesIO(handle.read()), allow_pickle=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    velocity = args.pod_root / "global_velocity_pod_area_weighted_l2.npz"
    pressure = args.pod_root / "global_pressure_pod_area_weighted_l2.npz"
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = {
        "phi_uv": load_member(velocity, "phi_uv"),
        "mean_uv_regime": load_member(velocity, "mean_uv_regime"),
        "phi_p": load_member(pressure, "phi_p"),
        "mean_p_regime": load_member(pressure, "mean_p_regime"),
        "points": load_member(velocity, "points"),
        "point_areas": load_member(velocity, "point_areas"),
        "sqrt_point_areas": load_member(velocity, "sqrt_point_areas"),
        "Re_values": load_member(velocity, "Re_values"),
        "Re_labels": load_member(velocity, "Re_labels"),
        "regimes": load_member(velocity, "regimes"),
        "split_by_Re": load_member(velocity, "split_by_Re"),
        "fit_split": load_member(velocity, "fit_split"),
        "centering": load_member(velocity, "centering"),
        "pressure_gauge": load_member(pressure, "pressure_gauge"),
    }
    np.savez_compressed(args.output, **payload)
    manifest = {
        "schema": "periodic_projection_contract/v1",
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "source_velocity": str(velocity),
        "source_velocity_sha256": sha256(velocity),
        "source_pressure": str(pressure),
        "source_pressure_sha256": sha256(pressure),
        "included_members": sorted(payload),
        "excluded_members": ["coeff_uv", "coeff_p"],
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
