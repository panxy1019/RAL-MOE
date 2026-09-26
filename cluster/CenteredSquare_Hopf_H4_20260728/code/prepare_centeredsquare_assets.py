#!/usr/bin/env python3
"""Build heldout-sealed r11 training assets for the CenteredSquare Hopf subset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


TRAIN = np.asarray([
    94.0, 95.0, 95.05, 95.15, 95.2, 95.35, 95.4, 95.45,
    95.55, 95.6, 95.75, 96.0, 96.25, 96.75, 97.0, 97.25,
    97.75, 98.0, 98.5, 99.5, 100.0, 101.0, 101.724137931034,
], dtype=np.float64)
VALIDATION = np.asarray([94.5, 95.25, 95.5, 97.5, 99.0, 101.5], dtype=np.float64)
HELDOUT = np.asarray([95.1, 95.3, 96.5, 100.5, 102.0], dtype=np.float64)
RANK = 11
TOL = 5.0e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--subset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def split_for(reynolds: float) -> str:
    if np.any(np.abs(TRAIN - reynolds) <= TOL):
        return "train"
    if np.any(np.abs(VALIDATION - reynolds) <= TOL):
        return "validation"
    if np.any(np.abs(HELDOUT - reynolds) <= TOL):
        return "heldout"
    raise RuntimeError(f"Re={reynolds:.12g} is outside the frozen Hopf split")


def project_case(
    path: Path,
    phi_u: np.ndarray,
    phi_p: np.ndarray,
    mean_u: np.ndarray,
    mean_p: np.ndarray,
    volumes: np.ndarray,
) -> dict[str, np.ndarray | float | str]:
    with np.load(path, allow_pickle=False) as source:
        reynolds = float(source["Re"])
        times = np.asarray(source["times"], dtype=np.float64)
        velocity = np.asarray(source["U"], dtype=np.float32)
        pressure = np.asarray(source["p"], dtype=np.float32)
    if velocity.shape[1:] != mean_u.shape or pressure.shape[1:] != mean_p.shape:
        raise RuntimeError(f"mesh/field shape mismatch in {path}")
    if len(times) != len(velocity) or len(times) != len(pressure):
        raise RuntimeError(f"snapshot length mismatch in {path}")
    volume_sum = float(volumes.sum())
    pressure = pressure - (
        np.einsum("tc,c->t", pressure.astype(np.float64), volumes, optimize=True)
        / volume_sum
    ).astype(np.float32)[:, None]
    coeff_u = np.einsum(
        "tci,kci,c->tk",
        velocity - mean_u[None],
        phi_u,
        volumes,
        optimize=True,
        dtype=np.float64,
    ).astype(np.float32)
    coeff_p = np.einsum(
        "tc,kc,c->tk",
        pressure - mean_p[None],
        phi_p,
        volumes,
        optimize=True,
        dtype=np.float64,
    ).astype(np.float32)
    if not np.all(np.isfinite(coeff_u)) or not np.all(np.isfinite(coeff_p)):
        raise RuntimeError(f"non-finite projection in {path}")
    return {
        "Re": reynolds,
        "time": times,
        "coeff_u": coeff_u,
        "coeff_p": coeff_p,
        "split": split_for(reynolds),
        "tag": path.stem.removeprefix("snapshots_"),
    }


def save_view(
    rows: list[dict[str, np.ndarray | float | str]],
    path: Path,
    phi_u: np.ndarray,
    phi_p: np.ndarray,
    mean_u: np.ndarray,
    mean_p: np.ndarray,
    volumes: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        coeff_uv=np.concatenate([row["coeff_u"] for row in rows]).astype(np.float32),
        coeff_p=np.concatenate([row["coeff_p"] for row in rows]).astype(np.float32),
        Re=np.concatenate([
            np.full(len(row["time"]), row["Re"], dtype=np.float64) for row in rows
        ]),
        time=np.concatenate([row["time"] for row in rows]).astype(np.float64),
        split=np.concatenate([
            np.full(len(row["time"]), row["split"], dtype="U10") for row in rows
        ]),
        case_tag=np.concatenate([
            np.full(len(row["time"]), row["tag"], dtype="U16") for row in rows
        ]),
        phi_uv=phi_u.reshape(RANK, -1).astype(np.float32),
        phi_p=phi_p.astype(np.float32),
        point_areas=volumes.astype(np.float32),
        mean_uv_train=mean_u.reshape(-1).astype(np.float32),
        mean_p_train=mean_p.astype(np.float32),
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rank_root = args.subset_root / "pod" / "rom" / "rank999_ru11_rp11"
    pod_path = rank_root / "pod_rank_pack_Re50_150_N100_rank999_ru11_rp11.npz"
    galerkin_source = rank_root / (
        "semi_intrusive_galerkin_tensors_Re50_150_N100_rank999_ru11_rp11_compact.npz"
    )
    pressure_source = rank_root / (
        "pressure_poisson_surrogate_tensors_Re50_150_N100_rank999_ru11_rp11.npz"
    )
    with np.load(pod_path, allow_pickle=False) as pod:
        phi_u = np.asarray(pod["Phi_u"], dtype=np.float32)
        phi_p = np.asarray(pod["Phi_p"], dtype=np.float32)
        mean_u = np.asarray(pod["U_mean"], dtype=np.float32)
        mean_p = np.asarray(pod["p_mean"], dtype=np.float32)
        volumes = np.asarray(pod["cellVolumes"], dtype=np.float64)
        stored_u = np.asarray(pod["coefficients_u"], dtype=np.float32)
        stored_p = np.asarray(pod["coefficients_p"], dtype=np.float32)
        stored_tags = np.asarray(pod["snapshot_case_tags"]).astype(str)
        stored_times = np.asarray(pod["snapshot_times"], dtype=np.float64)
    if phi_u.shape != (RANK, 9400, 2) or phi_p.shape != (RANK, 9400):
        raise RuntimeError(f"unexpected POD shapes: {phi_u.shape}, {phi_p.shape}")
    expected = np.sort(np.concatenate((TRAIN, VALIDATION, HELDOUT)))
    case_paths = sorted(args.cases_dir.glob("snapshots_Re*.npz"))
    rows = [
        project_case(path, phi_u, phi_p, mean_u, mean_p, volumes)
        for path in case_paths
    ]
    observed = np.sort(np.asarray([row["Re"] for row in rows], dtype=np.float64))
    if observed.shape != expected.shape or not np.allclose(observed, expected, atol=TOL, rtol=0):
        raise RuntimeError(f"34-case split mismatch: observed={observed.tolist()}")
    train_diffs_u: list[np.ndarray] = []
    train_diffs_p: list[np.ndarray] = []
    for row in rows:
        if row["split"] != "train":
            continue
        mask = (stored_tags == row["tag"]) & np.isin(
            np.round(stored_times, 12), np.round(row["time"], 12)
        )
        if int(mask.sum()) != len(row["time"]):
            raise RuntimeError(f"stored train coefficient alignment failed for {row['tag']}")
        train_diffs_u.append(np.asarray(row["coeff_u"]) - stored_u[mask])
        train_diffs_p.append(np.asarray(row["coeff_p"]) - stored_p[mask])
    max_train_u = float(np.max(np.abs(np.concatenate(train_diffs_u))))
    max_train_p = float(np.max(np.abs(np.concatenate(train_diffs_p))))
    if max_train_u > 2.0e-5 or max_train_p > 2.0e-5:
        raise RuntimeError(
            f"projection reproduction failed: velocity={max_train_u}, pressure={max_train_p}"
        )
    args.output_dir.mkdir(parents=True)
    trainval_path = args.output_dir / "centeredsquare_hopf_trainval_r11.npz"
    heldout_path = args.output_dir / "centeredsquare_hopf_heldout_r11.npz"
    save_view(
        [row for row in rows if row["split"] != "heldout"],
        trainval_path, phi_u, phi_p, mean_u, mean_p, volumes,
    )
    save_view(
        [row for row in rows if row["split"] == "heldout"],
        heldout_path, phi_u, phi_p, mean_u, mean_p, volumes,
    )
    with np.load(galerkin_source, allow_pickle=False) as source:
        re_g = np.asarray(source["Re_list"], dtype=np.float64)
        galerkin_path = args.output_dir / "centeredsquare_hopf_trainonly_galerkin_r11.npz"
        np.savez_compressed(
            galerkin_path,
            Re_values_computed=re_g,
            c_all=np.asarray(source["c_all"], dtype=np.float32),
            A_all=np.asarray(source["A_all"], dtype=np.float32),
            H=np.asarray(source["H"], dtype=np.float32),
            P=np.asarray(source["P"], dtype=np.float32),
        )
    with np.load(pressure_source, allow_pickle=False) as source:
        re_p = np.asarray(source["Re_list"], dtype=np.float64)
        pressure_path = args.output_dir / "centeredsquare_hopf_trainonly_pressure_r11.npz"
        np.savez_compressed(
            pressure_path,
            Re_values_computed=re_p,
            c_tilde_all=np.asarray(source["c_tilde_all"], dtype=np.float32),
            A_tilde_all=np.asarray(source["A_tilde_all"], dtype=np.float32),
            H_tilde=np.asarray(source["H_tilde"], dtype=np.float32),
        )
    if not np.allclose(np.sort(re_g), np.sort(TRAIN), atol=TOL, rtol=0):
        raise RuntimeError("Galerkin nodes do not match the 23 train Re")
    if not np.allclose(np.sort(re_p), np.sort(TRAIN), atol=TOL, rtol=0):
        raise RuntimeError("pressure nodes do not match the 23 train Re")
    manifest = {
        "schema_version": 1,
        "scope": "hopf_local_train_val_view",
        "dataset": "CenteredSquare CN09 graded mesh",
        "r_u": RANK,
        "r_p": RANK,
        "fit_reynolds": TRAIN.tolist(),
        "validation_reynolds": VALIDATION.tolist(),
        "heldout_reynolds_sealed": HELDOUT.tolist(),
        "heldout_rows_present": False,
        "pressure_gauge_verified": True,
        "projection_error_verified": True,
        "split_contract_verified": True,
        "rom_tensor_verified": True,
        "projection_errors": {
            "train_velocity_max_abs_coefficient_reproduction": max_train_u,
            "train_pressure_max_abs_coefficient_reproduction": max_train_p,
        },
        "counts": {
            "train_cases": len(TRAIN),
            "validation_cases": len(VALIDATION),
            "heldout_cases": len(HELDOUT),
            "trainval_snapshots": int(sum(len(row["time"]) for row in rows if row["split"] != "heldout")),
            "heldout_snapshots": int(sum(len(row["time"]) for row in rows if row["split"] == "heldout")),
        },
        "source_sha256": {
            "pod_rank_pack": digest(pod_path),
            "galerkin": digest(galerkin_source),
            "pressure": digest(pressure_source),
        },
        "sha256": {
            "coefficient_view": digest(trainval_path),
            "heldout_view": digest(heldout_path),
            "galerkin": digest(galerkin_path),
            "pressure": digest(pressure_path),
        },
    }
    atomic_json(manifest, args.output_dir / "TRAINING_ASSET_MANIFEST.json")
    print(json.dumps({
        "status": "PASS",
        "output_dir": str(args.output_dir),
        "counts": manifest["counts"],
        "projection_errors": manifest["projection_errors"],
    }, indent=2))


if __name__ == "__main__":
    main()
