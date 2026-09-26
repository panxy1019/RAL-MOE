#!/usr/bin/env python3
"""Build the CenteredSquare Periodic local-POD package expected by the trainer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def case_tag(re_value: float) -> str:
    return f"Re{re_value:010.6f}".replace(".", "p")


def find_npz_by_re(paths: list[Path], re_value: float) -> Path:
    matches: list[tuple[float, Path]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            matches.append((abs(float(data["Re"]) - re_value), path))
    error, path = min(matches, key=lambda item: item[0])
    if error > 1.0e-8:
        raise ValueError(f"No exact file for Re={re_value}; nearest={path} error={error}")
    return path


def estimate_period(probe_path: Path) -> float:
    with np.load(probe_path, allow_pickle=False) as data:
        time = data["probe_time"].astype(np.float64)
        velocity = data["probe_U"].astype(np.float64)
    late = time >= max(300.0, float(time.min()))
    signals = velocity[late, :, 1]
    signal = signals[:, int(np.argmax(np.std(signals, axis=0)))]
    signal = signal - np.mean(signal)
    dt = float(np.median(np.diff(time[late])))
    frequencies = np.fft.rfftfreq(signal.size, d=dt)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size))) ** 2
    valid = (frequencies >= 0.03) & (frequencies <= 0.40)
    peak = int(np.where(valid)[0][np.argmax(spectrum[valid])])
    frequency = float(frequencies[peak])
    if 0 < peak < len(spectrum) - 1:
        y0, y1, y2 = np.log(np.maximum(spectrum[peak - 1 : peak + 2], 1.0e-30))
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1.0e-12:
            frequency += float(0.5 * (y0 - y2) / denom) * float(frequencies[1])
    return 1.0 / frequency


def affine_in_inverse_re(re_train: np.ndarray, values: np.ndarray, re_all: np.ndarray) -> np.ndarray:
    x_train = np.column_stack([np.ones(len(re_train)), 1.0 / re_train])
    flat = values.reshape(len(re_train), -1)
    coefficients = np.linalg.lstsq(x_train, flat, rcond=None)[0]
    x_all = np.column_stack([np.ones(len(re_all)), 1.0 / re_all])
    return (x_all @ coefficients).reshape((len(re_all),) + values.shape[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--square-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--probe-formal-dir", type=Path, required=True)
    parser.add_argument("--probe-refined-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    periodic = args.square_root / "subsets" / "periodic"
    split_contract = json.loads(
        (args.square_root / "config" / "split_contract_resolved.json").read_text()
    )["config"]
    canonical = json.loads(
        (args.square_root / "config" / "canonical_cases.json").read_text()
    )
    domain = [
        item for item in canonical
        if 98.5 - 1.0e-9 <= float(item["Re"]) <= 150.0 + 1.0e-9
    ]
    domain.sort(key=lambda item: float(item["Re"]))
    if len(domain) != 37:
        raise ValueError(f"Expected 37 Periodic-domain cases, found {len(domain)}")

    heldout_values = np.asarray(split_contract["heldout"]["periodic"], dtype=np.float64)
    validation_values = np.asarray(split_contract["validation"]["periodic"], dtype=np.float64)
    re_values = np.asarray([float(item["Re"]) for item in domain], dtype=np.float64)
    tags = np.asarray([str(item["tag"]) for item in domain])
    roles = np.asarray([
        "heldout" if np.min(np.abs(heldout_values - re_value)) < 1.0e-8
        else "validation" if np.min(np.abs(validation_values - re_value)) < 1.0e-8
        else "train"
        for re_value in re_values
    ])
    if tuple(np.bincount([{"train": 0, "validation": 1, "heldout": 2}[r] for r in roles], minlength=3)) != (27, 6, 4):
        raise ValueError("Periodic split counts do not match 27/6/4")

    pod_dir = periodic / "pod"
    vel = np.load(pod_dir / "weighted_pod_velocity.npz")
    pre = np.load(pod_dir / "weighted_pod_pressure.npz")
    volumes = np.load(periodic / "mesh" / "mesh_metadata.npz")["cellVolumes"].astype(np.float64)
    sqrt_volumes = np.sqrt(volumes).astype(np.float32)
    phi_u = vel["modes"].astype(np.float32)
    phi_p = pre["modes"].astype(np.float32)
    mean_u = vel["mean"].astype(np.float32).reshape(-1)
    mean_p = pre["mean"].astype(np.float32).reshape(-1)

    train_coeff_u = {
        str(tag): (vel["snapshot_times"][vel["snapshot_case_tags"] == tag], vel["coefficients"][vel["snapshot_case_tags"] == tag])
        for tag in np.unique(vel["snapshot_case_tags"])
    }
    train_coeff_p = {
        str(tag): (pre["snapshot_times"][pre["snapshot_case_tags"] == tag], pre["coefficients"][pre["snapshot_case_tags"] == tag])
        for tag in np.unique(pre["snapshot_case_tags"])
    }
    raw_paths = list(args.raw_dir.glob("snapshots_Re*.npz"))
    formal_probes = list(args.probe_formal_dir.glob("probe_Re*.npz"))
    refined_probes = list(args.probe_refined_dir.glob("probe_Re*.npz"))

    all_u: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    index_rows: list[dict[str, object]] = []
    periods: dict[str, float] = {}
    snapshot_id = 0
    for item, re_value, tag, role in zip(domain, re_values, tags, roles):
        if role == "train":
            times_u, coeff_u = train_coeff_u[str(tag)]
            times_p, coeff_p = train_coeff_p[str(tag)]
            if not np.array_equal(times_u, times_p):
                raise ValueError(f"Velocity/pressure time mismatch for {tag}")
            times = times_u.astype(np.float64)
        else:
            raw_path = find_npz_by_re(raw_paths, float(re_value))
            with np.load(raw_path, allow_pickle=False) as raw:
                times = raw["times"].astype(np.float64)
                centered_u = raw["U"].astype(np.float64) - vel["mean"].astype(np.float64)[None, :, :]
                coeff_u = np.einsum(
                    "tci,mci,c->tm",
                    centered_u,
                    phi_u.reshape(phi_u.shape[0], -1, 2).astype(np.float64),
                    volumes,
                    optimize=True,
                ).astype(np.float32)
                pressure = raw["p"].astype(np.float64)
                gauge = np.sum(pressure * volumes[None, :], axis=1) / np.sum(volumes)
                centered_p = pressure - gauge[:, None] - mean_p.astype(np.float64)[None, :]
                coeff_p = np.einsum(
                    "tc,mc,c->tm",
                    centered_p,
                    phi_p.astype(np.float64),
                    volumes,
                    optimize=True,
                ).astype(np.float32)

        probe_candidates = refined_probes if "refined" in str(item["source_dataset"]) else formal_probes
        probe_path = find_npz_by_re(probe_candidates, float(re_value))
        period = estimate_period(probe_path)
        periods[str(tag)] = period
        phase = np.mod((times - times[0]) / period, 1.0)
        for local_index, (time_value, phase_value) in enumerate(zip(times, phase)):
            index_rows.append({
                "snapshot_id": snapshot_id,
                "Re": f"{re_value:.12f}",
                "Re_label": str(tag),
                "regime": "periodic",
                "target_regime": "periodic",
                "split": str(role),
                "time": f"{time_value:.12f}",
                "period": f"{period:.12f}",
                "phase": f"{phase_value:.12f}",
                "local_snapshot_index": local_index,
            })
            snapshot_id += 1
        all_u.append(np.asarray(coeff_u, dtype=np.float32))
        all_p.append(np.asarray(coeff_p, dtype=np.float32))

    coeff_u_all = np.concatenate(all_u)
    coeff_p_all = np.concatenate(all_p)
    snapshot_splits = np.concatenate([np.repeat(role, len(coeff)) for role, coeff in zip(roles, all_u)])
    snapshot_labels = np.concatenate([np.repeat(tag, len(coeff)) for tag, coeff in zip(tags, all_u)])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.output_dir / "global_velocity_pod_area_weighted_l2.npz",
        phi_uv=phi_u,
        phi_uv_weighted=vel["weighted_modes"].astype(np.float32),
        coeff_uv=coeff_u_all,
        mean_uv_regime=mean_u,
        mean_uv_by_Re=np.repeat(mean_u[None, :], len(re_values), axis=0),
        Re_values=re_values,
        Re_labels=tags,
        regimes=np.repeat("periodic", len(re_values)),
        split_by_Re=roles,
        snapshot_splits=snapshot_splits,
        snapshot_Re_labels=snapshot_labels,
        points=np.load(periodic / "mesh" / "mesh_metadata.npz")["cellCenters"].astype(np.float32),
        point_areas=volumes.astype(np.float32),
        sqrt_point_areas=sqrt_volumes,
        singular_values_uv=vel["singular_values"],
        cumulative_energy_uv=vel["cumulative_energy"],
        total_weighted_energy_uv=vel["total_energy"],
        coeff_train_mean=np.mean(coeff_u_all[snapshot_splits == "train"], axis=0),
        coeff_train_std=np.std(coeff_u_all[snapshot_splits == "train"], axis=0),
        fit_split="train",
        centering="single_train_only_regime_mean",
    )
    np.savez_compressed(
        args.output_dir / "global_pressure_pod_area_weighted_l2.npz",
        phi_p=phi_p,
        phi_p_weighted=pre["weighted_modes"].astype(np.float32),
        coeff_p=coeff_p_all,
        mean_p_regime=mean_p,
        mean_p_by_Re=np.repeat(mean_p[None, :], len(re_values), axis=0),
        Re_values=re_values,
        Re_labels=tags,
        regimes=np.repeat("periodic", len(re_values)),
        split_by_Re=roles,
        snapshot_splits=snapshot_splits,
        snapshot_Re_labels=snapshot_labels,
        points=np.load(periodic / "mesh" / "mesh_metadata.npz")["cellCenters"].astype(np.float32),
        point_areas=volumes.astype(np.float32),
        sqrt_point_areas=sqrt_volumes,
        singular_values_p=pre["singular_values"],
        cumulative_energy_p=pre["cumulative_energy"],
        total_weighted_energy_p=pre["total_energy"],
        coeff_train_mean=np.mean(coeff_p_all[snapshot_splits == "train"], axis=0),
        coeff_train_std=np.std(coeff_p_all[snapshot_splits == "train"], axis=0),
        fit_split="train",
        centering="single_train_only_regime_mean",
        pressure_gauge="subtract_volume_mean_per_snapshot",
    )
    with (args.output_dir / "pod_snapshot_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    rom_dir = pod_dir / "rom" / "rank999_ru28_rp26"
    velocity = np.load(rom_dir / "semi_intrusive_galerkin_tensors_Re50_150_N100_rank999_ru28_rp26_compact.npz")
    c_all = velocity["c_raw_no_nu"][None, :] + velocity["c_raw_nu"][None, :] / re_values[:, None]
    a_all = velocity["A_raw_no_nu"][None, :, :] + velocity["A_raw_nu"][None, :, :] / re_values[:, None, None]
    np.savez_compressed(
        args.output_dir / "velocity_rom_periodic.npz",
        Re_values_computed=re_values,
        Re_labels_computed=tags,
        pod_Re_values=re_values,
        pod_Re_labels=tags,
        mass_weights=volumes,
        points=np.load(periodic / "mesh" / "mesh_metadata.npz")["cellCenters"],
        r_u=np.int32(phi_u.shape[0]),
        r_p=np.int32(phi_p.shape[0]),
        G_u=velocity["G_u"],
        H=velocity["H"],
        P=velocity["P"],
        c_all=c_all,
        A_all=a_all,
    )

    pressure = np.load(rom_dir / "pressure_poisson_surrogate_tensors_Re50_150_N100_rank999_ru28_rp26.npz")
    c_tilde = affine_in_inverse_re(pressure["Re_list"], pressure["c_tilde_all"], re_values)
    a_tilde = affine_in_inverse_re(pressure["Re_list"], pressure["A_tilde_all"], re_values)
    pressure_payload: dict[str, np.ndarray] = {
        "Re_values_computed": re_values,
        "Re_labels_computed": tags,
        "H_tilde": pressure["H_tilde"],
        "L": pressure["L"],
        "L_pinv": pressure["L_pinv"],
    }
    for i, tag in enumerate(tags):
        pressure_payload[f"{tag}_c_tilde"] = c_tilde[i]
        pressure_payload[f"{tag}_A_tilde"] = a_tilde[i]
    np.savez_compressed(args.output_dir / "pressure_poisson_surrogate_periodic.npz", **pressure_payload)

    gram_u = (phi_u * np.repeat(volumes, 2)[None, :]) @ phi_u.T
    gram_p = (phi_p * volumes[None, :]) @ phi_p.T
    summary = {
        "cases": len(re_values),
        "snapshots": int(len(coeff_u_all)),
        "split_counts": {name: int(np.sum(roles == name)) for name in ("train", "validation", "heldout")},
        "r_u": int(phi_u.shape[0]),
        "r_p": int(phi_p.shape[0]),
        "max_velocity_gram_error": float(np.max(np.abs(gram_u - np.eye(phi_u.shape[0])))),
        "max_pressure_gram_error": float(np.max(np.abs(gram_p - np.eye(phi_p.shape[0])))),
        "all_finite": bool(np.isfinite(coeff_u_all).all() and np.isfinite(coeff_p_all).all()),
        "period_by_Re_label": periods,
    }
    (args.output_dir / "PREPARATION_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
