#!/usr/bin/env python3
"""Read-only phase and adjacent-chart projection diagnostics.

This script never modifies specialist assets.  It evaluates the exact weighted
POD affine maps and a deliberately small, state/history-only atan2 phase probe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/root/panxy/particalMOE")
EPS = 1e-12


def load_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def weighted_dot_rows(phi_to: np.ndarray, phi_from: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return Phi_to M Phi_from^T without materializing M."""
    if phi_from.shape[1] == 2 * weights.size:
        w = np.concatenate((weights, weights))
    else:
        w = weights
    return (phi_to.astype(np.float64) * w[None, :]) @ phi_from.astype(np.float64).T


def weighted_project(phi: np.ndarray, vector: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if phi.shape[1] == 2 * weights.size:
        w = np.concatenate((weights, weights))
    else:
        w = weights
    return (phi.astype(np.float64) * w[None, :]) @ vector.astype(np.float64)


def weighted_norm2(vector: np.ndarray, weights: np.ndarray) -> float:
    if vector.size == 2 * weights.size:
        w = np.concatenate((weights, weights))
    else:
        w = weights
    return float(np.dot(vector.astype(np.float64) * w, vector.astype(np.float64)))


def quantiles(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "worst": float(np.max(x)),
    }


def directional_projection(
    name: str,
    phi_src: np.ndarray,
    mean_src: np.ndarray,
    coeff_src: np.ndarray,
    phi_dst: np.ndarray,
    mean_dst: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    # Affine state map: a_dst = c + T a_src.  RHS maps use T only.
    t = weighted_dot_rows(phi_dst, phi_src, weights)
    delta = mean_src.astype(np.float64) - mean_dst.astype(np.float64)
    c = weighted_project(phi_dst, delta, weights)
    gram_cross_delta = weighted_project(phi_src, delta, weights)
    delta_n2 = weighted_norm2(delta, weights)
    a = coeff_src.astype(np.float64)
    mapped = a @ t.T + c[None, :]

    # Since both bases are weighted-orthonormal, these modal contractions are
    # exact up to the measured Gram error and avoid huge field batches.
    centered_n2 = delta_n2 + 2.0 * (a @ gram_cross_delta) + np.sum(a * a, axis=1)
    floor_n2 = np.maximum(centered_n2 - np.sum(mapped * mapped, axis=1), 0.0)
    field_floor = np.sqrt(floor_n2 / np.maximum(centered_n2, EPS))

    fluc_map = a @ t.T
    fluc_res_n2 = np.maximum(np.sum(a * a, axis=1) - np.sum(fluc_map * fluc_map, axis=1), 0.0)
    fluctuation_error = np.sqrt(fluc_res_n2 / np.maximum(np.sum(a * a, axis=1), EPS))
    energy_error = np.abs(np.sum(fluc_map * fluc_map, axis=1) - np.sum(a * a, axis=1)) / np.maximum(np.sum(a * a, axis=1), EPS)

    t_back = weighted_dot_rows(phi_src, phi_dst, weights)
    c_back = weighted_project(phi_src, -delta, weights)
    roundtrip = mapped @ t_back.T + c_back[None, :]
    roundtrip_modal = np.linalg.norm(roundtrip - a, axis=1) / np.maximum(np.linalg.norm(a, axis=1), EPS)

    _u, s, vt = np.linalg.svd(t, full_matrices=False)
    angles = np.degrees(np.arccos(np.clip(s, 0.0, 1.0)))
    directional_energy = np.mean((a @ vt.T) ** 2, axis=0)
    total_directional_energy = float(np.sum(directional_energy))
    low = s < 0.5
    low_energy_fraction = float(np.sum(directional_energy[low]) / max(total_directional_energy, EPS))
    shared90 = s >= 0.9
    shared50 = s >= 0.5
    return {
        "direction": name,
        "samples": int(a.shape[0]),
        "affine_mean_offset_l2": float(np.linalg.norm(c)),
        "projection_floor_relative_field_l2": quantiles(field_floor),
        "adapter_field_error_relative_l2": quantiles(field_floor),
        "fluctuation_error_relative_l2": quantiles(fluctuation_error),
        "energy_relative_error": quantiles(energy_error),
        "roundtrip_relative_modal_l2": quantiles(roundtrip_modal),
        "singular_values": s.tolist(),
        "principal_angles_degrees": angles.tolist(),
        "sigma_min": float(s[-1]),
        "condition_number_report_only_no_online_inverse": float(s[0] / max(s[-1], EPS)),
        "shared_direction_count_cos_ge_0_9": int(np.sum(shared90)),
        "shared_direction_count_cos_ge_0_5": int(np.sum(shared50)),
        "source_energy_fraction_in_cos_lt_0_5_directions": low_energy_fraction,
    }


def load_charts() -> dict[str, dict[str, Any]]:
    sroot = ROOT / "steady_specialist_v1"
    with np.load(sroot / "data/steady_pod_runtime_r32.npz") as pod, np.load(sroot / "data/steady_trainval_modal_r32.npz") as dat:
        steady = {
            "u_phi": pod["velocity_basis"].copy(), "p_phi": pod["pressure_basis"].copy(),
            "u_mean": pod["velocity_mean"].copy(), "p_mean": pod["pressure_mean"].copy(),
            "areas": pod["point_areas"].copy(), "a": dat["a_raw"].copy(), "b": dat["b_raw"].copy(),
            "re": dat["Re"].copy(), "split": dat["split"].copy(),
        }
    hpath = ROOT / "Hopf/migrated_h4_expanded/assets_r32/expanded_h4_trainval_r32.npz"
    with np.load(hpath) as dat:
        hopf = {
            "u_phi": dat["phi_uv"].copy(), "p_phi": dat["phi_p"].copy(),
            "u_mean": dat["mean_uv_train"].copy(), "p_mean": dat["mean_p_train"].copy(),
            "areas": dat["point_areas"].copy(), "a": dat["coeff_uv"].copy(), "b": dat["coeff_p"].copy(),
            "re": dat["Re"].copy(), "split": dat["split"].copy(),
        }
    proot = ROOT / "periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2"
    with np.load(proot / "global_velocity_pod_area_weighted_l2.npz") as u, np.load(proot / "global_pressure_pod_area_weighted_l2.npz") as p:
        idx = load_csv(proot / "pod_snapshot_index.csv")
        periodic = {
            "u_phi": u["phi_uv"].copy(), "p_phi": p["phi_p"].copy(),
            "u_mean": u["mean_uv_regime"].copy(), "p_mean": p["mean_p_regime"].copy(),
            "areas": u["point_areas"].copy(), "a": u["coeff_uv"].copy(), "b": p["coeff_p"].copy(),
            "re": idx["Re"].copy(), "split": idx["split"].copy(), "time": idx["time"].copy(),
        }
    return {"S": steady, "H": hopf, "P": periodic}


def pair_report(left_name: str, right_name: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    common_lo = max(float(np.min(left["re"])), float(np.min(right["re"])))
    common_hi = min(float(np.max(left["re"])), float(np.max(right["re"])))
    exact_overlap = common_lo <= common_hi
    if exact_overlap:
        lm = (left["re"] >= common_lo) & (left["re"] <= common_hi)
        rm = (right["re"] >= common_lo) & (right["re"] <= common_hi)
        support = "exact_Re_overlap"
    else:
        # Boundary-support diagnostic only.  It must not be described as a
        # certified overlap and is never assembled into a pseudo-trajectory.
        lvals = np.unique(left["re"])[-3:]
        rvals = np.unique(right["re"])[:3]
        lm = np.isin(left["re"], lvals)
        rm = np.isin(right["re"], rvals)
        support = "nearest_three_Re_boundary_proxy_no_overlap"
    if not np.array_equal(left["areas"], right["areas"]):
        raise RuntimeError(f"{left_name}-{right_name}: point-area ordering mismatch")
    out: dict[str, Any] = {
        "pair": f"{left_name}-{right_name}",
        "exact_Re_overlap_exists": bool(exact_overlap),
        "support_type": support,
        "common_or_gap_interval": [common_lo, common_hi],
        "left_Re_support": np.unique(left["re"][lm]).tolist(),
        "right_Re_support": np.unique(right["re"][rm]).tolist(),
        "velocity": {}, "pressure": {},
    }
    for field, coeff in (("velocity", "a"), ("pressure", "b")):
        prefix = "u" if field == "velocity" else "p"
        out[field][f"{left_name}_to_{right_name}"] = directional_projection(
            f"{left_name}->{right_name}", left[f"{prefix}_phi"], left[f"{prefix}_mean"], left[coeff][lm],
            right[f"{prefix}_phi"], right[f"{prefix}_mean"], left["areas"],
        )
        out[field][f"{right_name}_to_{left_name}"] = directional_projection(
            f"{right_name}->{left_name}", right[f"{prefix}_phi"], right[f"{prefix}_mean"], right[coeff][rm],
            left[f"{prefix}_phi"], left[f"{prefix}_mean"], left["areas"],
        )
    metrics = [out[f][k] for f in ("velocity", "pressure") for k in out[f]]
    low_energy = max(m["source_energy_fraction_in_cos_lt_0_5_directions"] for m in metrics)
    shared_min = min(m["shared_direction_count_cos_ge_0_9"] for m in metrics)
    out["principal_angle_aware_overlap_mask_recommendation"] = {
        "recommended": bool(low_energy > 0.01 or shared_min < 32),
        "rule": "blend only SVD directions with cos(principal_angle)>=0.9; evolve anchor-private complement with anchor specialist",
        "reason": f"max source energy in cos<0.5 directions={low_energy:.6g}; minimum cos>=0.9 shared count={shared_min}/32",
    }
    return out


def wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def phase_audit(periodic: dict[str, Any]) -> dict[str, Any]:
    a = periodic["a"].astype(np.float64)
    re = periodic["re"].astype(np.float64)
    time = periodic["time"].astype(np.float64)
    split = periodic["split"]
    train = split == "train"

    # Train-only principal observation pair.  Runtime phase uses only current
    # state plus previous unwrapped phase for branch selection.
    center = np.mean(a[train], axis=0)
    _u, _s, vt = np.linalg.svd(a[train] - center, full_matrices=False)
    plane = vt[:2]
    obs = (a - center) @ plane.T
    raw = np.arctan2(obs[:, 1], obs[:, 0])

    db_phase = np.zeros_like(time)
    est_unwrapped = np.zeros_like(time)
    trajectory_rows: list[dict[str, Any]] = []
    # Fit one train-only circular offset/orientation, never using validation or heldout.
    train_raw: list[np.ndarray] = []
    train_target: list[np.ndarray] = []
    per_re_indices: dict[float, np.ndarray] = {}
    for rv in np.unique(re):
        ids = np.where(re == rv)[0]
        ids = ids[np.argsort(time[ids])]
        per_re_indices[float(rv)] = ids
        db_phase[ids] = 2.0 * np.pi * (time[ids] - time[ids[0]]) / max(time[ids[-1]] - time[ids[0]], EPS)
        if split[ids[0]] == "train":
            train_raw.append(raw[ids])
            train_target.append(db_phase[ids])
    rr = np.concatenate(train_raw)
    tt = np.concatenate(train_target)
    # Select orientation by maximum circular correlation.
    candidates = []
    for sign in (1.0, -1.0):
        offset = np.angle(np.mean(np.exp(1j * (tt - sign * rr))))
        err = np.mean(np.abs(wrap_angle(sign * rr + offset - tt)))
        candidates.append((err, sign, offset))
    _, sign, offset = min(candidates)

    for rv, ids in per_re_indices.items():
        # Online-equivalent unwrap: current atan2 plus previous unwrapped phase.
        wrapped = wrap_angle(sign * raw[ids] + offset)
        unwrapped = np.empty_like(wrapped)
        unwrapped[0] = wrapped[0]
        for j in range(1, len(ids)):
            unwrapped[j] = unwrapped[j - 1] + wrap_angle(wrapped[j] - wrapped[j - 1])
        # Only the initial current state fixes an integer branch; no endpoint or future state is used.
        unwrapped -= 2.0 * np.pi * np.round((unwrapped[0] - db_phase[ids[0]]) / (2.0 * np.pi))
        est_unwrapped[ids] = unwrapped
        dt = max(time[ids[-1]] - time[ids[0]], EPS)
        est_freq = (unwrapped[-1] - unwrapped[0]) / dt
        db_freq = 2.0 * np.pi / dt
        trajectory_rows.append({
            "Re": rv, "split": str(split[ids[0]]), "snapshots": int(len(ids)),
            "phase_circular_mae_rad": float(np.mean(np.abs(wrap_angle(unwrapped - db_phase[ids])))),
            "frequency_abs_error_rad_per_time": float(abs(est_freq - db_freq)),
            "frequency_relative_error": float(abs(est_freq - db_freq) / max(abs(db_freq), EPS)),
            "long_phase_drift_rad": float(abs((unwrapped[-1] - unwrapped[0]) - 2.0 * np.pi)),
        })

    aggregates: dict[str, Any] = {}
    for label in ("train", "validation", "heldout"):
        rows = [r for r in trajectory_rows if r["split"] == label]
        aggregates[label] = {
            "trajectories": len(rows),
            "phase_mae_rad_mean": float(np.mean([r["phase_circular_mae_rad"] for r in rows])) if rows else None,
            "frequency_relative_error_mean": float(np.mean([r["frequency_relative_error"] for r in rows])) if rows else None,
            "long_phase_drift_rad_mean": float(np.mean([r["long_phase_drift_rad"] for r in rows])) if rows else None,
            "long_phase_drift_rad_worst": float(np.max([r["long_phase_drift_rad"] for r in rows])) if rows else None,
        }
    return {
        "checkpoint_phase_harmonics": 4,
        "csv_fields_phase_or_period_present": False,
        "certified_checkpoint_feature_source": "database normalized trajectory progress: (t-t_first)/(t_last-t_first)",
        "certified_source_requires_future_trajectory_endpoint": True,
        "atan2_probe": {
            "observation": "train-only PCA pair of current velocity modal state",
            "runtime_inputs": "current state and previous unwrapped phase only",
            "uses_future_truth_at_runtime": False,
            "orientation": sign,
            "train_only_offset_rad": float(offset),
            "aggregates": aggregates,
            "per_trajectory": trajectory_rows,
        },
        "autonomous_phase_contract_certified": False,
        "failure_reason": "The frozen checkpoint was trained on a phase feature that needs each database trajectory's future endpoint. The state-only atan2 probe is diagnostic and cannot replace that input without re-certifying or fine-tuning the specialist.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    charts = load_charts()
    report = {
        "schema": "CONTINUOUS_TIME_PHASE_PROJECTION_AUDIT_V1",
        "read_only": True,
        "phase": phase_audit(charts["P"]),
        "adapters": {
            "S_H": pair_report("S", "H", charts["S"], charts["H"]),
            "H_P": pair_report("H", "P", charts["H"], charts["P"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "phase_certified": report["phase"]["autonomous_phase_contract_certified"],
        "S_H_exact_overlap": report["adapters"]["S_H"]["exact_Re_overlap_exists"],
        "H_P_exact_overlap": report["adapters"]["H_P"]["exact_Re_overlap_exists"],
    }))


if __name__ == "__main__":
    main()
