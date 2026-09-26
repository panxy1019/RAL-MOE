#!/usr/bin/env python3
"""Adapt Fluidic Pinball V2 periodic POD/ROM assets to the MoE trainer contract."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


RU = 17
RP = 16
EXPECTED_SPLITS = {"train": 47, "validation": 9, "final_test": 9}


def atomic_npz(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def re_label(value: float) -> str:
    return f"Re_{value:07.3f}"


def phase_from_coefficients(coeff: np.ndarray, times: np.ndarray) -> np.ndarray:
    if coeff.shape[1] >= 2 and len(coeff) > 1:
        pair = coeff[:, :2] - np.mean(coeff[:, :2], axis=0, keepdims=True)
        radius = np.linalg.norm(pair, axis=1)
        if float(np.max(radius)) > 1.0e-10:
            return (np.arctan2(pair[:, 1], pair[:, 0]) / (2.0 * np.pi) % 1.0).astype(np.float32)
    span = max(float(times[-1] - times[0]), np.finfo(np.float64).eps)
    return ((times - times[0]) / span % 1.0).astype(np.float32)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["Re"]))
    counts = {split: sum(row["split"] == split for row in rows) for split in EXPECTED_SPLITS}
    if counts != EXPECTED_SPLITS:
        raise ValueError(f"Unexpected periodic split counts: {counts}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    periodic = root / "rom_assets_v2/periodic"
    eval_root = root / "eval_pod_coefficients_v2/periodic"
    rows = load_manifest(root / "config/experts/periodic_re_manifest.csv")

    velocity = np.load(periodic / "pod/weighted_pod_velocity.npz", allow_pickle=True)
    pressure = np.load(periodic / "pod/weighted_pod_pressure.npz", allow_pickle=True)
    if not np.array_equal(velocity["snapshot_case_tags"], pressure["snapshot_case_tags"]):
        raise ValueError("Velocity and pressure POD case tags are not aligned.")
    if not np.allclose(velocity["snapshot_times"], pressure["snapshot_times"], rtol=0, atol=1e-12):
        raise ValueError("Velocity and pressure POD times are not aligned.")

    train_offsets = json.loads(str(velocity["case_offsets"].item()))
    train_by_label = {str(item["case_tag"]): item for item in train_offsets}
    all_a: list[np.ndarray] = []
    all_b: list[np.ndarray] = []
    index_rows: list[dict[str, object]] = []
    snapshot_splits: list[str] = []
    snapshot_labels: list[str] = []
    snapshot_id = 0

    for row in rows:
        value = float(row["Re"])
        label = re_label(value)
        split = row["split"]
        if split == "train":
            item = train_by_label.get(label)
            if item is None:
                raise ValueError(f"Missing train POD coefficients for {label}")
            start, stop = int(item["start"]), int(item["stop"])
            times = np.asarray(velocity["snapshot_times"][start:stop], dtype=np.float64)
            a = np.asarray(velocity["coefficients"][start:stop, :RU], dtype=np.float32)
            b = np.asarray(pressure["coefficients"][start:stop, :RP], dtype=np.float32)
        else:
            coeff_path = eval_root / split / f"{label}_pod_coefficients.npz"
            coeff = np.load(coeff_path, allow_pickle=False)
            if abs(float(coeff["Re"].item()) - value) > 1.0e-12 or str(coeff["split"].item()) != split:
                raise ValueError(f"Evaluation coefficient contract mismatch: {coeff_path}")
            times = np.asarray(coeff["times"], dtype=np.float64)
            a = np.asarray(coeff["a_velocity_rank999"][:, :RU], dtype=np.float32)
            b = np.asarray(coeff["b_pressure_rank999"][:, :RP], dtype=np.float32)
        order = np.argsort(times, kind="stable")
        times, a, b = times[order], a[order], b[order]
        if len(times) < 4 or np.any(np.diff(times) <= 0):
            raise ValueError(f"Invalid time sequence for {label}")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError(f"Non-finite coefficients for {label}")
        phase = phase_from_coefficients(a, times)
        for local, (time_value, phase_value) in enumerate(zip(times.tolist(), phase.tolist())):
            index_rows.append({
                "snapshot_id": snapshot_id,
                "Re": value,
                "Re_label": label,
                "time": time_value,
                "phase": phase_value,
                "regime": "periodic",
                "local_snapshot_index": local,
            })
            snapshot_id += 1
        all_a.append(a)
        all_b.append(b)
        snapshot_splits.extend([split] * len(times))
        snapshot_labels.extend([label] * len(times))

    a_all = np.concatenate(all_a, axis=0)
    b_all = np.concatenate(all_b, axis=0)
    labels = np.asarray([re_label(float(row["Re"])) for row in rows])
    re_values = np.asarray([float(row["Re"]) for row in rows], dtype=np.float64)
    split_by_re = np.asarray([row["split"] for row in rows])
    if len(index_rows) != len(a_all) or len(a_all) != len(b_all):
        raise ValueError("Combined coefficient/index lengths do not match.")

    fieldnames = ["snapshot_id", "Re", "Re_label", "time", "phase", "regime", "local_snapshot_index"]
    csv_lines = []
    from io import StringIO
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(index_rows)
    atomic_text(output / "pod_snapshot_index.csv", stream.getvalue())

    sqrt_cell_volume = np.asarray(pressure["weights"], dtype=np.float32)
    cell_volume = np.square(sqrt_cell_volume).astype(np.float32)
    velocity_modes = np.asarray(velocity["modes"][:RU], dtype=np.float32)
    pressure_modes = np.asarray(pressure["modes"][:RP], dtype=np.float32)
    common = {
        "Re_values": re_values,
        "Re_labels": labels,
        "regimes": np.asarray(["periodic"] * len(rows)),
        "split_by_Re": split_by_re,
        "snapshot_splits": np.asarray(snapshot_splits),
        "snapshot_Re_labels": np.asarray(snapshot_labels),
        "point_areas": cell_volume,
        "sqrt_point_areas": sqrt_cell_volume,
        "fit_split": np.asarray("train"),
    }
    atomic_npz(
        output / "global_velocity_pod_area_weighted_l2.npz",
        phi_uv=velocity_modes,
        phi_uv_weighted=np.asarray(velocity["weighted_modes"][:RU], dtype=np.float32),
        coeff_uv=a_all,
        mean_uv_regime=np.asarray(velocity["mean"], dtype=np.float32).reshape(-1),
        singular_values_uv=np.asarray(velocity["singular_values"][:RU], dtype=np.float64),
        cumulative_energy_uv=np.asarray(velocity["cumulative_energy"][:RU], dtype=np.float64),
        **common,
    )
    atomic_npz(
        output / "global_pressure_pod_area_weighted_l2.npz",
        phi_p=pressure_modes,
        phi_p_weighted=np.asarray(pressure["weighted_modes"][:RP], dtype=np.float32),
        coeff_p=b_all,
        mean_p_regime=np.asarray(pressure["mean"], dtype=np.float32),
        singular_values_p=np.asarray(pressure["singular_values"][:RP], dtype=np.float64),
        cumulative_energy_p=np.asarray(pressure["cumulative_energy"][:RP], dtype=np.float64),
        pressure_gauge=np.asarray("volume_weighted_zero_mean"),
        **common,
    )

    velocity_rom = np.load(periodic / "rom/rank999_ru17_rp16/velocity_galerkin_tensors.npz")
    pressure_rom = np.load(periodic / "rom/rank999_ru17_rp16/pressure_poisson_tensors.npz")
    rom_re = np.asarray(velocity_rom["Re_list"], dtype=np.float64)
    if not np.allclose(rom_re, re_values, rtol=0, atol=1.0e-12):
        raise ValueError("ROM Re_list does not match the periodic manifest.")
    atomic_npz(
        output / "velocity_rom_periodic.npz",
        Re_values_computed=rom_re,
        Re_labels_computed=labels,
        G_u=np.asarray(velocity_rom["G_u"], dtype=np.float32),
        H=np.asarray(velocity_rom["H"], dtype=np.float32),
        P=np.asarray(velocity_rom["P"], dtype=np.float32),
        c_all=np.asarray(velocity_rom["c_all"], dtype=np.float32),
        A_all=np.asarray(velocity_rom["A_all"], dtype=np.float32),
        r_u=np.asarray(RU),
        r_p=np.asarray(RP),
    )
    pressure_payload: dict[str, np.ndarray] = {
        "Re_values_computed": rom_re,
        "Re_labels_computed": labels,
        "H_tilde": np.asarray(pressure_rom["H_tilde"], dtype=np.float32),
        "L": np.asarray(pressure_rom["L"], dtype=np.float32),
        "L_pinv": np.asarray(pressure_rom["L_pinv"], dtype=np.float32),
    }
    for idx, label in enumerate(labels.tolist()):
        pressure_payload[f"{label}_c_tilde"] = np.asarray(pressure_rom["c_tilde_all"][idx], dtype=np.float32)
        pressure_payload[f"{label}_A_tilde"] = np.asarray(pressure_rom["A_tilde_all"][idx], dtype=np.float32)
    atomic_npz(output / "pressure_poisson_surrogate_periodic.npz", **pressure_payload)

    contract = {
        "dataset_root": str(root),
        "expert": "periodic",
        "rank": {"r_u": RU, "r_p": RP, "energy_target": 0.999},
        "split_counts": EXPECTED_SPLITS,
        "validation_re_values": [float(row["Re"]) for row in rows if row["split"] == "validation"],
        "test_re_values": [float(row["Re"]) for row in rows if row["split"] == "final_test"],
        "snapshot_counts": {split: int(sum(s == split for s in snapshot_splits)) for split in EXPECTED_SPLITS},
        "total_snapshots": int(len(index_rows)),
        "phase_definition": "atan2 of centered first two velocity POD coefficients modulo one",
        "pod_fit_data": "train only",
        "evaluation_coefficients_used_for_fit": False,
    }
    atomic_text(output / "dataset_contract.json", json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
