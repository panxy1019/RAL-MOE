#!/usr/bin/env python3
"""Build compact, trainer-compatible CenteredSquare steady specialist assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def re_tag(value: float) -> str:
    return ("Re" + f"{float(value):010.6f}").replace(".", "p")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pressure_gauge(values: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    means = values.astype(np.float64) @ volumes / volumes.sum()
    return values - means.astype(np.float32)[:, None]


def relative_weighted_error(
    centered: np.ndarray, coeff: np.ndarray, modes: np.ndarray, weights: np.ndarray
) -> float:
    residual = centered - coeff @ modes
    numerator = np.sum(residual.astype(np.float64) ** 2 * weights[None, :])
    denominator = np.sum(centered.astype(np.float64) ** 2 * weights[None, :])
    return float(np.sqrt(numerator / max(denominator, np.finfo(np.float64).eps)))


def affine_in_inverse_re(
    train_re: np.ndarray, train_values: np.ndarray, target_re: np.ndarray
) -> tuple[np.ndarray, float]:
    design = np.column_stack([np.ones_like(train_re), 1.0 / train_re])
    flat = train_values.reshape(len(train_re), -1)
    coefficients, *_ = np.linalg.lstsq(design, flat, rcond=None)
    fitted = design @ coefficients
    residual = float(
        np.linalg.norm(fitted - flat)
        / max(np.linalg.norm(flat), np.finfo(np.float64).eps)
    )
    target_design = np.column_stack([np.ones_like(target_re), 1.0 / target_re])
    values = (target_design @ coefficients).reshape((len(target_re),) + train_values.shape[1:])
    return values, residual


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    artifact_dir = output_root / "source_artifacts" / "steady"
    compat_dir = artifact_dir / "Global_POD_AreaWeighted_L2"
    compat_dir.mkdir(parents=True, exist_ok=True)

    config_root = dataset_root / "config"
    split_cfg = json.loads((config_root / "regime_split_config.json").read_text())
    records = json.loads((config_root / "canonical_cases.json").read_text())
    steady_records = [
        item
        for item in records
        if split_cfg["specialist_domains"]["steady"][0] - 1e-9
        <= float(item["Re"])
        <= split_cfg["specialist_domains"]["steady"][1] + 1e-9
    ]
    steady_records.sort(key=lambda item: float(item["Re"]))
    if len(steady_records) != 69:
        raise RuntimeError(f"expected 69 steady cases, found {len(steady_records)}")

    validation = {round(float(x), 12) for x in split_cfg["validation"]["steady"]}
    heldout = {round(float(x), 12) for x in split_cfg["heldout"]["steady"]}
    roles = []
    for record in steady_records:
        value = round(float(record["Re"]), 12)
        roles.append("validation" if value in validation else "heldout" if value in heldout else "train")
    if roles.count("train") != 60 or roles.count("validation") != 5 or roles.count("heldout") != 4:
        raise RuntimeError(f"invalid role counts: {roles}")

    steady_root = dataset_root / "subsets" / "steady"
    with np.load(steady_root / "mesh" / "mesh_metadata.npz") as mesh:
        centers = mesh["cellCenters"].astype(np.float32)
        volumes = mesh["cellVolumes"].astype(np.float64)
    with np.load(steady_root / "pod" / "weighted_pod_velocity.npz") as pod:
        u_modes = pod["modes"].astype(np.float32)
        u_weighted_modes = pod["weighted_modes"].astype(np.float32)
        u_mean = pod["mean"].astype(np.float32).reshape(-1)
        u_singular = pod["singular_values"].astype(np.float64)
        u_cumulative = pod["cumulative_energy"].astype(np.float64)
        u_train_coeff = pod["coefficients"].astype(np.float32)
        u_train_tags = pod["snapshot_case_tags"].astype(str)
        u_total_energy = float(pod["total_energy"])
    with np.load(steady_root / "pod" / "weighted_pod_pressure.npz") as pod:
        p_modes = pod["modes"].astype(np.float32)
        p_weighted_modes = pod["weighted_modes"].astype(np.float32)
        p_mean = pod["mean"].astype(np.float32).reshape(-1)
        p_singular = pod["singular_values"].astype(np.float64)
        p_cumulative = pod["cumulative_energy"].astype(np.float64)
        p_train_coeff = pod["coefficients"].astype(np.float32)
        p_train_tags = pod["snapshot_case_tags"].astype(str)
        p_total_energy = float(pod["total_energy"])

    if u_modes.shape != (5, 18800) or p_modes.shape != (4, 9400):
        raise RuntimeError(f"unexpected rank999 modes: {u_modes.shape}, {p_modes.shape}")

    u_mass = np.repeat(volumes, 2)
    all_u, all_p = [], []
    snapshot_rows: list[dict[str, object]] = []
    snapshot_splits, snapshot_labels = [], []
    projection_reports = []
    train_u_projected, train_p_projected, train_projected_tags = [], [], []
    snapshot_id = 0

    for record, role in zip(steady_records, roles):
        source = Path(record["path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as case:
            times = case["times"].astype(np.float64)
            velocity = case["U"].astype(np.float32)
            pressure = pressure_gauge(case["p"].astype(np.float32), volumes)
        if velocity.shape != (len(times), 9400, 2) or pressure.shape != (len(times), 9400):
            raise RuntimeError(f"shape mismatch for {source}")
        if not np.all(np.diff(times) > 0):
            raise RuntimeError(f"non-monotone time for {source}")

        centered_u = velocity.reshape(len(times), -1) - u_mean[None, :]
        centered_p = pressure - p_mean[None, :]
        coeff_u = ((centered_u.astype(np.float64) * u_mass[None, :]) @ u_modes.T).astype(np.float32)
        coeff_p = ((centered_p.astype(np.float64) * volumes[None, :]) @ p_modes.T).astype(np.float32)
        all_u.append(coeff_u)
        all_p.append(coeff_p)

        label = re_tag(float(record["Re"]))
        snapshot_splits.extend([role] * len(times))
        snapshot_labels.extend([label] * len(times))
        for local_index, time_value in enumerate(times):
            snapshot_rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "Re": f"{float(record['Re']):.12f}",
                    "Re_label": label,
                    "regime": "steady",
                    "target_regime": "steady",
                    "split": role,
                    "time": f"{float(time_value):.12g}",
                    "local_snapshot_index": local_index,
                    "phase": 0.0,
                }
            )
            snapshot_id += 1

        report = {
            "Re": float(record["Re"]),
            "label": label,
            "split": role,
            "snapshots": int(len(times)),
            "velocity_weighted_rel_l2": relative_weighted_error(centered_u, coeff_u, u_modes, u_mass),
            "pressure_weighted_rel_l2": relative_weighted_error(centered_p, coeff_p, p_modes, volumes),
            "source_sha256": record["sha256"],
        }
        projection_reports.append(report)
        if role == "train":
            train_u_projected.append(coeff_u)
            train_p_projected.append(coeff_p)
            train_projected_tags.extend([record["tag"]] * len(times))

    coeff_u = np.concatenate(all_u)
    coeff_p = np.concatenate(all_p)
    snapshot_splits_arr = np.asarray(snapshot_splits)
    snapshot_labels_arr = np.asarray(snapshot_labels)
    re_values = np.asarray([float(item["Re"]) for item in steady_records], dtype=np.float64)
    re_labels = np.asarray([re_tag(value) for value in re_values])
    split_by_re = np.asarray(roles)

    train_u_projected_arr = np.concatenate(train_u_projected)
    train_p_projected_arr = np.concatenate(train_p_projected)
    if train_projected_tags != u_train_tags.tolist() or train_projected_tags != p_train_tags.tolist():
        raise RuntimeError("train snapshot order does not match frozen POD coefficient order")
    u_coeff_rel = float(
        np.linalg.norm(train_u_projected_arr - u_train_coeff)
        / max(np.linalg.norm(u_train_coeff), np.finfo(np.float32).eps)
    )
    p_coeff_rel = float(
        np.linalg.norm(train_p_projected_arr - p_train_coeff)
        / max(np.linalg.norm(p_train_coeff), np.finfo(np.float32).eps)
    )

    train_mask = snapshot_splits_arr == "train"
    u_coeff_mean = coeff_u[train_mask].mean(axis=0, dtype=np.float64)
    p_coeff_mean = coeff_p[train_mask].mean(axis=0, dtype=np.float64)
    u_coeff_std = np.maximum(coeff_u[train_mask].std(axis=0, dtype=np.float64), 1e-7)
    p_coeff_std = np.maximum(coeff_p[train_mask].std(axis=0, dtype=np.float64), 1e-7)

    velocity_pod = artifact_dir / "velocity_pod_steady.npz"
    pressure_pod = artifact_dir / "pressure_pod_steady.npz"
    np.savez_compressed(
        velocity_pod,
        phi_uv=u_modes,
        phi_uv_weighted=u_weighted_modes,
        coeff_uv=coeff_u,
        mean_uv_regime=u_mean,
        Re_values=re_values,
        Re_labels=re_labels,
        regimes=np.asarray(["steady"] * len(re_values)),
        split_by_Re=split_by_re,
        snapshot_splits=snapshot_splits_arr,
        snapshot_Re_labels=snapshot_labels_arr,
        points=centers,
        point_areas=volumes.astype(np.float32),
        sqrt_point_areas=np.sqrt(volumes).astype(np.float32),
        singular_values_uv=u_singular,
        cumulative_energy_uv=u_cumulative,
        total_weighted_energy_uv=np.asarray(u_total_energy),
        coeff_train_mean=u_coeff_mean,
        coeff_train_std=u_coeff_std,
        fit_split=np.asarray("train"),
        centering=np.asarray("single_train_only_regime_mean"),
    )
    np.savez_compressed(
        pressure_pod,
        phi_p=p_modes,
        phi_p_weighted=p_weighted_modes,
        coeff_p=coeff_p,
        mean_p_regime=p_mean,
        Re_values=re_values,
        Re_labels=re_labels,
        regimes=np.asarray(["steady"] * len(re_values)),
        split_by_Re=split_by_re,
        snapshot_splits=snapshot_splits_arr,
        snapshot_Re_labels=snapshot_labels_arr,
        points=centers,
        point_areas=volumes.astype(np.float32),
        sqrt_point_areas=np.sqrt(volumes).astype(np.float32),
        singular_values_p=p_singular,
        cumulative_energy_p=p_cumulative,
        total_weighted_energy_p=np.asarray(p_total_energy),
        coeff_train_mean=p_coeff_mean,
        coeff_train_std=p_coeff_std,
        fit_split=np.asarray("train"),
        centering=np.asarray("single_train_only_regime_mean"),
        pressure_gauge=np.asarray("subtract_volume_mean_per_snapshot"),
    )
    np.savez_compressed(
        artifact_dir / "normalization_steady.npz",
        velocity_coeff_mean=u_coeff_mean,
        velocity_coeff_std=u_coeff_std,
        pressure_coeff_mean=p_coeff_mean,
        pressure_coeff_std=p_coeff_std,
        fit_split=np.asarray("train"),
        train_Re_labels=re_labels[split_by_re == "train"],
    )

    for source, target_name in [
        (velocity_pod, "global_velocity_pod_area_weighted_l2.npz"),
        (pressure_pod, "global_pressure_pod_area_weighted_l2.npz"),
    ]:
        target = compat_dir / target_name
        target.unlink(missing_ok=True)
        target.hardlink_to(source)
    with (compat_dir / "pod_snapshot_index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(snapshot_rows[0]))
        writer.writeheader()
        writer.writerows(snapshot_rows)
    np.savez_compressed(
        compat_dir / "mesh_l2_point_area_weights.npz",
        points=centers,
        point_areas=volumes.astype(np.float32),
        sqrt_point_areas=np.sqrt(volumes).astype(np.float32),
    )

    rom_dir = steady_root / "pod" / "rom" / "rank999_ru5_rp4"
    velocity_candidates = list(rom_dir.glob("semi_intrusive_galerkin_tensors_*compact.npz"))
    pressure_candidates = list(rom_dir.glob("pressure_poisson_surrogate_tensors_*.npz"))
    if len(velocity_candidates) != 1 or len(pressure_candidates) != 1:
        raise RuntimeError(f"ROM files not uniquely resolved in {rom_dir}")
    with np.load(velocity_candidates[0]) as rom:
        train_re = rom["Re_list"].astype(np.float64)
        c_all = np.stack(
            [
                np.linalg.solve(
                    rom["G_u"].astype(np.float64),
                    rom["c_raw_no_nu"].astype(np.float64)
                    + rom["c_raw_nu"].astype(np.float64) / value,
                )
                for value in re_values
            ]
        )
        a_all = np.stack(
            [
                np.linalg.solve(
                    rom["G_u"].astype(np.float64),
                    rom["A_raw_no_nu"].astype(np.float64)
                    + rom["A_raw_nu"].astype(np.float64) / value,
                )
                for value in re_values
            ]
        )
        velocity_payload = {
            "Re_values_computed": re_values,
            "Re_labels_computed": re_labels,
            "pod_Re_values": re_values,
            "pod_Re_labels": re_labels,
            "mass_weights": volumes,
            "r_u": np.asarray(5, dtype=np.int32),
            "r_p": np.asarray(4, dtype=np.int32),
            "G_u": rom["G_u"],
            "H": rom["H"],
            "P": rom["P"],
            "c_all": c_all,
            "A_all": a_all,
        }
        velocity_reference = rom["c_all"].astype(np.float64)
        velocity_reference_a = rom["A_all"].astype(np.float64)
        reference_rows = {round(float(value), 12): i for i, value in enumerate(train_re)}
        selected = np.asarray([reference_rows[round(float(value), 12)] for value in train_re])
        velocity_c_residual = float(
            np.linalg.norm(c_all[[np.where(np.isclose(re_values, x, atol=1e-10))[0][0] for x in train_re]] - velocity_reference)
            / max(np.linalg.norm(velocity_reference), np.finfo(np.float64).eps)
        )
        velocity_a_residual = float(
            np.linalg.norm(a_all[[np.where(np.isclose(re_values, x, atol=1e-10))[0][0] for x in train_re]] - velocity_reference_a)
            / max(np.linalg.norm(velocity_reference_a), np.finfo(np.float64).eps)
        )
    velocity_rom = artifact_dir / "velocity_rom_steady.npz"
    np.savez_compressed(velocity_rom, **velocity_payload)

    with np.load(pressure_candidates[0]) as rom:
        pressure_train_re = rom["Re_list"].astype(np.float64)
        c_tilde_all, pressure_c_residual = affine_in_inverse_re(
            pressure_train_re, rom["c_tilde_all"].astype(np.float64), re_values
        )
        a_tilde_all, pressure_a_residual = affine_in_inverse_re(
            pressure_train_re, rom["A_tilde_all"].astype(np.float64), re_values
        )
        pressure_payload: dict[str, np.ndarray] = {
            "Re_values_computed": re_values,
            "Re_labels_computed": re_labels,
            "pod_Re_values": re_values,
            "pod_Re_labels": re_labels,
            "mass_weights": volumes,
            "L": rom["L"],
            "L_pinv": rom["L_pinv"],
            "H_p": rom["H_p"],
            "H_tilde": rom["H_tilde"],
            "r_u": np.asarray(5, dtype=np.int32),
            "r_p": np.asarray(4, dtype=np.int32),
        }
        for index, label in enumerate(re_labels):
            pressure_payload[f"{label}_c_tilde"] = c_tilde_all[index]
            pressure_payload[f"{label}_A_tilde"] = a_tilde_all[index]
    pressure_rom = artifact_dir / "pressure_poisson_surrogate_steady.npz"
    np.savez_compressed(pressure_rom, **pressure_payload)

    split_sets = {
        role: {round(float(record["Re"]), 12) for record, item_role in zip(steady_records, roles) if item_role == role}
        for role in ("train", "validation", "heldout")
    }
    if any(split_sets[a] & split_sets[b] for a, b in [("train", "validation"), ("train", "heldout"), ("validation", "heldout")]):
        raise RuntimeError("Re leakage detected")
    audit = {
        "status": "PASS",
        "dataset_root": str(dataset_root),
        "rank": {"r_u": 5, "r_p": 4, "policy": "rank999"},
        "pressure_gauge": "subtract_volume_mean_per_snapshot",
        "case_counts": {role: roles.count(role) for role in ("train", "validation", "heldout")},
        "snapshot_counts": {
            role: int(np.sum(snapshot_splits_arr == role)) for role in ("train", "validation", "heldout")
        },
        "re_values": {role: sorted(split_sets[role]) for role in split_sets},
        "split_intersections": {
            "train_validation": sorted(split_sets["train"] & split_sets["validation"]),
            "train_heldout": sorted(split_sets["train"] & split_sets["heldout"]),
            "validation_heldout": sorted(split_sets["validation"] & split_sets["heldout"]),
        },
        "train_projection_vs_stored_coeff_relative": {"velocity": u_coeff_rel, "pressure": p_coeff_rel},
        "rom_reconstruction_relative": {
            "velocity_c": velocity_c_residual,
            "velocity_A": velocity_a_residual,
            "pressure_c_tilde_affine_fit": pressure_c_residual,
            "pressure_A_tilde_affine_fit": pressure_a_residual,
        },
        "projection_reports": projection_reports,
        "assets": {},
    }
    for path in [
        velocity_pod,
        pressure_pod,
        artifact_dir / "normalization_steady.npz",
        velocity_rom,
        pressure_rom,
        compat_dir / "pod_snapshot_index.csv",
    ]:
        audit["assets"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (output_root / "ASSET_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    archive = output_root / "centeredsquare_steady_trainer_assets_rank999.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(artifact_dir, arcname="source_artifacts/steady")
        tar.add(output_root / "ASSET_AUDIT.json", arcname="ASSET_AUDIT.json")
    print(json.dumps({"status": "PASS", "archive": str(archive), "sha256": sha256(archive), **audit["case_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
