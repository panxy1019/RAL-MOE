#!/usr/bin/env python3
"""Build the common P-native H/P K56 development cache after formal preflight."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


RANK = 32
HORIZON = 56
WINDOWS_PER_RE = 5
BOUNDARY_UPPER_RE = 79.3811581027


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def evenly(ids: np.ndarray, count: int) -> np.ndarray:
    if len(ids) < count:
        raise RuntimeError(f"only {len(ids)} valid windows, need {count}")
    return ids[np.linspace(0, len(ids) - 1, count, dtype=np.int64)]


def quad_stats(
    first: np.ndarray,
    second: np.ndarray,
    truth: np.ndarray,
    first_phi: np.ndarray,
    first_mean: np.ndarray,
    second_phi: np.ndarray,
    second_mean: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Quadratic error coefficients for alpha*first+(1-alpha)*second."""
    g11 = (first_phi * weights) @ first_phi.T
    g22 = (second_phi * weights) @ second_phi.T
    cross = (first_phi * weights) @ second_phi.T
    mean_delta = first_mean - second_mean
    p1d = (first_phi * weights) @ mean_delta
    p2d = (second_phi * weights) @ mean_delta
    mean_delta_sq = float(np.dot(mean_delta * weights, mean_delta))
    p2m = (second_phi * weights) @ second_mean
    mean2_sq = float(np.dot(second_mean * weights, second_mean))
    first = first.astype(np.float64)
    second = second.astype(np.float64)
    truth = truth.astype(np.float64)

    # Truth uses the second (Periodic) chart in this one-sided protocol.
    second_error = second - truth
    q22 = np.einsum("ti,ij,tj->t", second_error, g22, second_error)
    q11 = (
        mean_delta_sq
        + np.einsum("ti,ij,tj->t", first, g11, first)
        + np.einsum("ti,ij,tj->t", truth, g22, truth)
        + 2.0 * first @ p1d
        - 2.0 * truth @ p2d
        - 2.0 * np.einsum("ti,ij,tj->t", first, cross, truth)
    )
    q12 = np.einsum(
        "ti,ti->t",
        second_error,
        p2d[None] + first @ cross - truth @ g22,
    )
    truth_norm = (
        mean2_sq
        + 2.0 * truth @ p2m
        + np.einsum("ti,ij,tj->t", truth, g22, truth)
    )
    return np.stack((q11, q22, q12, truth_norm), axis=-1).astype(np.float32)


def rollout_periodic(
    hp: object,
    trainer: object,
    model: torch.nn.Module,
    periodic_args: object,
    scalers: dict[str, object],
    arrays: dict[str, np.ndarray],
    tensors: object,
    pressure_tensors: object,
    device: torch.device,
    start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequence = [start]
    for _ in range(HORIZON):
        nxt = int(arrays["next_idx"][sequence[-1]])
        if nxt < 0:
            raise RuntimeError("window exceeded native Periodic trajectory")
        sequence.append(nxt)
    ids = np.asarray(sequence, dtype=np.int64)
    times = arrays["time"][ids].astype(np.float64)
    a_cur = arrays["a"][start].copy()
    b_cur = arrays["b"][start].copy()
    a_hist, b_hist, rhs_hist = trainer.init_history_states_np(start, arrays)
    pred_a, pred_b = [], []
    cur = start
    with torch.inference_mode():
        for dt in np.diff(times).tolist():
            an, bn, rhs_g = trainer.integrate_autonomous_step_np(
                model,
                a_cur,
                b_cur,
                cur,
                float(dt),
                a_hist,
                b_hist,
                rhs_hist,
                arrays,
                scalers,
                tensors,
                pressure_tensors,
                periodic_args,
                device,
            )
            pred_a.append(an)
            pred_b.append(bn)
            a_hist = np.concatenate((an[None, None], a_hist[:, :-1]), axis=1)
            b_hist = np.concatenate((bn[None, None], b_hist[:, :-1]), axis=1)
            rhs_hist = np.concatenate((rhs_g[None, None], rhs_hist[:, :-1]), axis=1)
            a_cur, b_cur = an, bn
            cur = int(arrays["next_idx"][cur])
    return (
        np.asarray(pred_a, dtype=np.float32),
        np.asarray(pred_b, dtype=np.float32),
        arrays["a"][ids[1:]].astype(np.float32),
        arrays["b"][ids[1:]].astype(np.float32),
        times,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    gate_path = (
        args.recovery_dir
        / "preflight/G_HP_native_FULL_K56/G_HP_native_FULL_K56.json"
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("decision") != "PASS" or len(gate.get("rows", [])) != 59:
        raise RuntimeError("formal 59-Re G_HP_native_FULL_K56 has not passed")
    args.output_dir.mkdir(parents=True)
    atomic_json(
        args.output_dir / "STARTED.json",
        {
            "mode": "development",
            "splits": ["train", "validation"],
            "windows_per_Re": WINDOWS_PER_RE,
            "horizon": HORIZON,
            "heldout_bundle_loaded": False,
        },
    )

    hp = load_module(
        "hp_cache_preflight",
        args.recovery_dir / "code/one_sided_preflight_hp_native.py",
    )
    top1 = load_module(
        "hp_cache_top1",
        args.root / "trajectory_router_e1_e2_e3_20260722/code/run_top1_experiment.py",
    )
    periodic_root = args.root / "periodic_specialist_r32"
    trainer = load_module("hp_cache_periodic", periodic_root / "code/train_periodic_moe.py")
    device = torch.device("cuda")
    checkpoint_path = periodic_root / "checkpoint/FINAL_PERIODIC_SPECIALIST.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, periodic_args, scalers = hp.instantiate_periodic(trainer, checkpoint, device)
    tensor_path = periodic_root / "assets/velocity_rom_periodic.npz"
    pressure_tensor_path = periodic_root / "assets/pressure_poisson_surrogate_periodic.npz"
    tensors = np.load(tensor_path)
    pressure_tensors = np.load(pressure_tensor_path)
    train_bundle = args.recovery_dir / "assets/periodic_train_modal_r32.npz"
    validation_bundle = args.recovery_dir / "assets/periodic_validation_modal_r32.npz"
    arrays = hp.load_bundle([train_bundle, validation_bundle], trainer, tensors)
    hopf = hp.HopfRuntime(args.root, device)

    records = []
    source_audit = []
    contract_path = args.recovery_dir / "assets/periodic_projection_contract_no_coeff.npz"
    with np.load(contract_path, allow_pickle=False) as periodic, np.load(
        args.root / "Hopf/artifacts/hopf/velocity_pod_hopf.npz", allow_pickle=False
    ) as hopf_velocity, np.load(
        args.root / "Hopf/artifacts/hopf/pressure_pod_hopf.npz", allow_pickle=False
    ) as hopf_pressure:
        p_phi_u = np.asarray(periodic["phi_uv"][:RANK], dtype=np.float64)
        p_mu_u = np.asarray(periodic["mean_uv_regime"], dtype=np.float64)
        p_phi_p = np.asarray(periodic["phi_p"][:RANK], dtype=np.float64)
        p_mu_p = np.asarray(periodic["mean_p_regime"], dtype=np.float64)
        h_phi_u = np.asarray(hopf_velocity["phi_uv"][:RANK], dtype=np.float64)
        h_mu_u = np.asarray(hopf_velocity["mean_uv_regime"], dtype=np.float64)
        h_phi_p = np.asarray(hopf_pressure["phi_p"][:RANK], dtype=np.float64)
        h_mu_p = np.asarray(hopf_pressure["mean_p_regime"], dtype=np.float64)
        areas = np.asarray(periodic["point_areas"], dtype=np.float64)
        uv_weights = np.concatenate((areas, areas))
        if not np.array_equal(periodic["points"], hopf_velocity["points"]):
            raise RuntimeError("P/H point ordering mismatch")
        if not np.array_equal(periodic["point_areas"], hopf_velocity["point_areas"]):
            raise RuntimeError("P/H area weights mismatch")
        if str(np.asarray(periodic["pressure_gauge"]).item()) != str(
            np.asarray(hopf_pressure["pressure_gauge"]).item()
        ):
            raise RuntimeError("P/H pressure gauge mismatch")
        p_to_h_u, p_to_h_u_offset = hp.affine_map(
            p_phi_u, p_mu_u, h_phi_u, h_mu_u, uv_weights
        )
        p_to_h_p, p_to_h_p_offset = hp.affine_map(
            p_phi_p, p_mu_p, h_phi_p, h_mu_p, areas
        )
        h_to_p_u, h_to_p_u_offset = hp.affine_map(
            h_phi_u, h_mu_u, p_phi_u, p_mu_u, uv_weights
        )
        h_to_p_p, h_to_p_p_offset = hp.affine_map(
            h_phi_p, h_mu_p, p_phi_p, p_mu_p, areas
        )
        velocity_geometry = top1.weighted_geometry(p_phi_u, p_mu_u, areas, True)
        pressure_geometry = top1.weighted_geometry(p_phi_p, p_mu_p, areas, False)

        validation_labels = {
            "Re_66p970112",
            "Re_91p792204",
            "Re_121p050171",
            "Re_139p642302",
            "Re_169p244893",
            "Re_196p160723",
        }
        for label in arrays["labels"].tolist():
            ids = np.where(arrays["Re_label"].astype(str) == label)[0]
            ids = ids[np.argsort(arrays["time"][ids])]
            valid = np.asarray(
                [
                    int(i)
                    for i in ids[2:-HORIZON]
                    if np.all(arrays["hist_idx"][i] >= 0)
                ],
                dtype=np.int64,
            )
            starts = evenly(valid, WINDOWS_PER_RE)
            split = "validation" if label in validation_labels else "train"
            re_value = float(arrays["re"][starts[0]])
            if re_value > BOUNDARY_UPPER_RE:
                continue
            source_audit.append(
                {
                    "split": split,
                    "Re_label": label,
                    "Re": re_value,
                    "valid_K56_windows": int(valid.size),
                    "selected_starts": starts.tolist(),
                }
            )
            for start in starts.tolist():
                p_a, p_b, true_a, true_b, times = rollout_periodic(
                    hp,
                    trainer,
                    model,
                    periodic_args,
                    scalers,
                    arrays,
                    tensors,
                    pressure_tensors,
                    device,
                    int(start),
                )
                history_ids = arrays["hist_idx"][start]
                history_times = arrays["time"][history_ids].astype(np.float64)
                order = np.argsort(history_times)
                history_a = arrays["a"][history_ids]
                history_b = arrays["b"][history_ids]
                descriptor = top1.physical_descriptors(
                    history_a[order][0],
                    history_a[order][1],
                    history_a[order][2],
                    history_b[order][0],
                    history_b[order][1],
                    history_b[order][2],
                    history_times[order][0],
                    history_times[order][1],
                    history_times[order][2],
                    velocity_geometry,
                    pressure_geometry,
                    float(areas.sum()),
                )
                h_a0 = p_to_h_u @ arrays["a"][start] + p_to_h_u_offset
                h_b0 = p_to_h_p @ arrays["b"][start] + p_to_h_p_offset
                h_a_hist = history_a @ p_to_h_u.T + p_to_h_u_offset
                h_b_hist = history_b @ p_to_h_p.T + p_to_h_p_offset
                h_a, h_b = hopf.rollout(
                    h_a0,
                    h_b0,
                    h_a_hist,
                    h_b_hist,
                    re_value,
                    np.diff(times).astype(np.float32),
                )
                if not all(
                    np.isfinite(value).all() for value in (p_a, p_b, h_a, h_b)
                ):
                    raise RuntimeError(f"non-finite native cache window {label}:{start}")
                roundtrip_a = h_to_p_u @ h_a0 + h_to_p_u_offset
                roundtrip_b = h_to_p_p @ h_b0 + h_to_p_p_offset
                representation_features = np.asarray(
                    [
                        np.linalg.norm(roundtrip_a - arrays["a"][start])
                        / max(np.linalg.norm(arrays["a"][start]), 1.0e-8),
                        np.linalg.norm(roundtrip_b - arrays["b"][start])
                        / max(np.linalg.norm(arrays["b"][start]), 1.0e-8),
                    ],
                    dtype=np.float32,
                )
                records.append(
                    (
                        split,
                        re_value,
                        label,
                        int(start),
                        times,
                        np.r_[re_value, descriptor].astype(np.float32),
                        representation_features,
                        h_a,
                        h_b,
                        p_a,
                        p_b,
                        true_a,
                        true_b,
                        h_a @ h_to_p_u.T + h_to_p_u_offset,
                        h_b @ h_to_p_p.T + h_to_p_p_offset,
                        quad_stats(
                            h_a,
                            p_a,
                            true_a,
                            h_phi_u,
                            h_mu_u,
                            p_phi_u,
                            p_mu_u,
                            uv_weights,
                        ),
                        quad_stats(
                            h_b,
                            p_b,
                            true_b,
                            h_phi_p,
                            h_mu_p,
                            p_phi_p,
                            p_mu_p,
                            areas,
                        ),
                    )
                )

    names = (
        "split",
        "re",
        "Re_label",
        "start",
        "times",
        "features",
        "representation_features",
        "h_a",
        "h_b",
        "p_a",
        "p_b",
        "true_a",
        "true_b",
        "h_to_p_a",
        "h_to_p_b",
        "quad_u",
        "quad_p",
    )
    payload = {
        name: np.asarray([record[i] for record in records])
        for i, name in enumerate(names)
    }
    expected = {"train": 10 * WINDOWS_PER_RE, "validation": 1 * WINDOWS_PER_RE}
    actual = {
        split: int(np.sum(payload["split"] == split)) for split in expected
    }
    if actual != expected:
        raise RuntimeError(f"cache count mismatch: {actual} != {expected}")
    for name, value in payload.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise RuntimeError(f"non-finite cache member: {name}")
    cache_path = args.output_dir / "G_HP_train_validation_cache_K56.npz"
    np.savez_compressed(cache_path, **payload)
    manifest = {
        "status": "CACHE_COMPLETE",
        "mode": "development",
        "cache": str(cache_path),
        "cache_sha256": hp.sha256(cache_path),
        "split_counts": actual,
        "complete_Re_counts": {"train": 10, "validation": 1},
        "boundary_upper_Re": BOUNDARY_UPPER_RE,
        "boundary_rule": "midpoint between the first two Periodic validation Re nodes; metadata-only and fixed before route training",
        "windows_per_Re": WINDOWS_PER_RE,
        "horizon": HORIZON,
        "source_audit": source_audit,
        "fusion_feedback": False,
        "timestamps_shared": True,
        "Periodic_phase_source": "native indexed time/estimated_period",
        "Hopf_phase_source": "phase-free",
        "heldout_bundle_loaded": False,
        "periodic_checkpoint_sha256": hp.sha256(checkpoint_path),
        "hopf_checkpoint_sha256": hp.sha256(hopf.checkpoint_path),
        "formal_preflight_sha256": hp.sha256(gate_path),
        "projection_contract_sha256": hp.sha256(contract_path),
        "train_bundle_sha256": hp.sha256(train_bundle),
        "validation_bundle_sha256": hp.sha256(validation_bundle),
    }
    atomic_json(args.output_dir / "CACHE_MANIFEST.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "split_counts", "cache_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
