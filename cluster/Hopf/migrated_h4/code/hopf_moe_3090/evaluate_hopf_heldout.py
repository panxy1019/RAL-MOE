#!/usr/bin/env python3
"""Strict common-window heldout evaluation for frozen Hopf-local A/B checkpoints."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


HORIZONS = (1, 2, 4, 8, 16, 24, 48)
HELDOUT_RE = (47.081355, 49.022357, 51.786450)
EPS = 1.0e-12


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("hopf_trainer_final", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(payload: object, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(tmp, path)


def weighted_geometry(phi: np.ndarray, mean: np.ndarray, areas: np.ndarray, vector: bool):
    weights = np.concatenate((areas, areas)) if vector else areas
    gram = (phi * weights[None, :]) @ phi.T
    cross = (phi * weights[None, :]) @ mean
    mean_energy = float(np.sum(weights * mean * mean))
    return gram.astype(np.float64), cross.astype(np.float64), mean_energy


def field_energy(coeff: np.ndarray, geom) -> np.ndarray:
    gram, cross, mean_energy = geom
    c = np.asarray(coeff, dtype=np.float64)
    return mean_energy + 2.0 * (c @ cross) + np.einsum("...i,ij,...j->...", c, gram, c)


def field_error_energy(true: np.ndarray, pred: np.ndarray, geom) -> np.ndarray:
    delta = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return np.einsum("...i,ij,...j->...", delta, geom[0], delta)


def relative_l2(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + EPS))


def build_heldout_data(module, source_root: Path, history_len: int) -> dict[str, np.ndarray]:
    pod = source_root / "Global_POD_AreaWeighted_L2"
    velocity = np.load(pod / "global_velocity_pod_area_weighted_l2.npz", allow_pickle=True)
    pressure = np.load(pod / "global_pressure_pod_area_weighted_l2.npz", allow_pickle=True)
    with (pod / "pod_snapshot_index.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    re_all = np.asarray([float(row["Re"]) for row in rows], dtype=np.float64)
    time_all = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
    split = np.asarray([row["split"] for row in rows])
    keep = split == "heldout"
    observed = np.sort(np.unique(np.round(re_all[keep], 6)))
    if observed.shape != (3,) or not np.allclose(observed, HELDOUT_RE, atol=1.1e-6, rtol=0):
        raise RuntimeError(f"Heldout source does not contain exact sealed Re set: {observed}")
    a = np.asarray(velocity["coeff_uv"], dtype=np.float32)[keep, :32]
    b = np.asarray(pressure["coeff_p"], dtype=np.float32)[keep, :32]
    re = re_all[keep]
    time = time_all[keep]
    order = np.lexsort((time, re))
    a, b, re, time = a[order], b[order], re[order], time[order]
    next_idx = np.full(len(re), -1, dtype=np.int64)
    prev_idx = np.full(len(re), -1, dtype=np.int64)
    for value in np.unique(re):
        ids = np.flatnonzero(np.abs(re - value) <= 5e-7)
        next_idx[ids[:-1]] = ids[1:]
        prev_idx[ids[1:]] = ids[:-1]
    hist = module.v16.history_index_matrix(np.arange(len(re), dtype=np.int64), prev_idx, history_len)
    valid = np.flatnonzero((next_idx >= 0) & np.all(hist >= 0, axis=1))
    return {
        "a": a, "b": b, "re": re.astype(np.float32), "time": time.astype(np.float32),
        "next": next_idx, "prev": prev_idx, "hist": hist, "valid": valid,
        "phi_u": np.asarray(velocity["phi_uv"], dtype=np.float32)[:32],
        "phi_p": np.asarray(pressure["phi_p"], dtype=np.float32)[:32],
        "areas": np.asarray(velocity["point_areas"], dtype=np.float32),
        "mean_u": np.asarray(velocity["mean_uv_regime"], dtype=np.float32),
        "mean_p": np.asarray(pressure["mean_p_regime"], dtype=np.float32),
    }


def phase_metrics(true_a: np.ndarray, pred_a: np.ndarray, times: np.ndarray,
                  center: np.ndarray, plane: np.ndarray, orbit_scale: np.ndarray,
                  r_floor: float, growth_tol: float) -> dict[str, float | bool]:
    tz = (true_a - center) @ plane.T
    pz = (pred_a - center) @ plane.T
    tr = np.linalg.norm(tz, axis=-1)
    pr = np.linalg.norm(pz, axis=-1)
    true_rms, pred_rms = float(np.sqrt(np.mean(tr**2))), float(np.sqrt(np.mean(pr**2)))
    true_p2p, pred_p2p = float(np.ptp(tz[..., 0])), float(np.ptp(pz[..., 0]))
    tphase = np.unwrap(np.arctan2(tz[..., 1], tz[..., 0]), axis=1)
    pphase = np.unwrap(np.arctan2(pz[..., 1], pz[..., 0]), axis=1)
    dt = times - times[:, :1]
    denom = np.sum((dt - dt.mean(1, keepdims=True)) ** 2, axis=1) + EPS
    tslope = np.sum((dt - dt.mean(1, keepdims=True)) * (tphase - tphase.mean(1, keepdims=True)), axis=1) / denom
    pslope = np.sum((dt - dt.mean(1, keepdims=True)) * (pphase - pphase.mean(1, keepdims=True)), axis=1) / denom
    pslope = np.where(tslope * pslope < 0, -pslope, pslope)
    f_true = float(np.mean(np.abs(tslope)) / (2 * math.pi))
    f_pred = float(np.mean(np.abs(pslope)) / (2 * math.pi))
    delta = (pphase - tphase) - (pphase[:, :1] - tphase[:, :1])
    tlog = np.log(tr + r_floor)
    plog = np.log(pr + r_floor)
    tg, pg = np.diff(tlog, axis=1), np.diff(plog, axis=1)
    active = np.abs(tg) > growth_tol
    sign_accuracy = float(np.mean(np.sign(pg[active]) == np.sign(tg[active]))) if np.any(active) else float("nan")
    scale = np.asarray(orbit_scale, dtype=np.float64)
    orbit_dist = []
    for tw, pw in zip(tz / scale, pz / scale):
        distances = np.sqrt(np.sum((pw[:, None, :] - tw[None, :, :]) ** 2, axis=2))
        orbit_dist.append(float(np.mean(np.min(distances, axis=1))))
    true_growth, pred_growth = float(np.mean(tg)), float(np.mean(pg))
    return {
        "rms_amplitude_error": abs(pred_rms - true_rms) / (true_rms + EPS),
        "peak_to_peak_amplitude_error": abs(pred_p2p - true_p2p) / (abs(true_p2p) + EPS),
        "growth_sign_accuracy": sign_accuracy,
        "true_local_log_growth": true_growth,
        "predicted_local_log_growth": pred_growth,
        "local_growth_mae": float(np.mean(np.abs(pg - tg))),
        "false_growth": bool(true_growth <= growth_tol and pred_growth > growth_tol),
        "frequency_true": f_true,
        "frequency_pred": f_pred,
        "frequency_relative_error": abs(f_pred - f_true) / (abs(f_true) + EPS),
        "phase_rms_rad": float(np.sqrt(np.mean(delta**2))),
        "terminal_phase_drift_cycles": float(np.mean(delta[:, -1]) / (2 * math.pi)),
        "normalized_orbit_distance": float(np.mean(orbit_dist)),
    }


@torch.no_grad()
def evaluate_one(module, checkpoint_path: Path, source_root: Path, stride: int) -> dict[str, Any]:
    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = SimpleNamespace(**checkpoint["args"])
    args.device = "cuda"
    train_data = module.load_coefficients(args)
    rom_np = module.load_train_rom(args)
    norm_stats, scale_stats = module.fit_stats(train_data, rom_np, args)
    stats = {key: torch.as_tensor(value, device=device) for key, value in asdict(norm_stats).items()}
    rom = {key: torch.as_tensor(value, device=device) for key, value in rom_np.items()}
    probe = module.batch_from_ids(train_data, train_data["train_ids"][:2], device)
    prh = module.galerkin(
        probe["ah"].reshape(-1, 32), probe["bh"].reshape(-1, 32),
        probe["re"][:, None].expand(-1, args.history_len).reshape(-1), rom,
    ).reshape_as(probe["ah"])
    in_dim = module.state_features(
        probe["a"], probe["b"], probe["re"], probe["ah"], probe["bh"], prh, rom, stats
    )[0].shape[1]
    model = module.build_model(in_dim, args, stats, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    data = build_heldout_data(module, source_root, args.history_len)
    velocity_geom = weighted_geometry(data["phi_u"], data["mean_u"], data["areas"], True)
    pressure_geom = weighted_geometry(data["phi_p"], data["mean_p"], data["areas"], False)
    train_plane_values = (train_data["a"][train_data["train_ids"]] - scale_stats.center) @ scale_stats.plane.T
    orbit_scale = np.maximum(np.std(train_plane_values, axis=0), EPS)
    growth_tol = max(float(scale_stats.train_log_growth_mad) * 0.1, 1e-8)
    by_re: dict[str, Any] = {}

    for value in HELDOUT_RE:
        ids = data["valid"][np.abs(data["re"][data["valid"]] - value) <= 5e-6]
        starts = module.legal_starts(data, ids, max(HORIZONS))[::stride]
        if not len(starts):
            raise RuntimeError(f"No K48-capable heldout windows for Re={value:.6f}")
        output = module.rollout_batch(model, data, starts, max(HORIZONS), rom, stats, device)
        pa = torch.stack(output["pred_a"], 1).float().cpu().numpy()
        pb = torch.stack(output["pred_b"], 1).float().cpu().numpy()
        ta = torch.stack(output["true_a"], 1).float().cpu().numpy()
        tb = torch.stack(output["true_b"], 1).float().cpu().numpy()
        current = starts.copy()
        times = []
        for _ in range(max(HORIZONS)):
            current = data["next"][current]
            times.append(data["time"][current])
        times_np = np.stack(times, axis=1)
        finite_step = np.isfinite(pa).all(2) & np.isfinite(pb).all(2)
        true_u_max = np.maximum(np.max(np.linalg.norm(ta, axis=2), axis=1, keepdims=True), EPS)
        true_p_max = np.maximum(np.max(np.linalg.norm(tb, axis=2), axis=1, keepdims=True), EPS)
        norm_bad = (np.linalg.norm(pa, axis=2) > 10 * true_u_max) | (np.linalg.norm(pb, axis=2) > 10 * true_p_max)
        bad = (~finite_step) | norm_bad
        horizon_metrics: dict[str, Any] = {}
        for horizon in HORIZONS:
            hp, hb, ht, htb = pa[:, :horizon], pb[:, :horizon], ta[:, :horizon], tb[:, :horizon]
            flat_pa, flat_pb = hp.reshape(-1, 32), hb.reshape(-1, 32)
            flat_ta, flat_tb = ht.reshape(-1, 32), htb.reshape(-1, 32)
            window_bad = bad[:, :horizon].any(1)
            first = np.flatnonzero(bad[:, :horizon].any(0))
            u_energy_true, u_energy_pred = field_energy(flat_ta, velocity_geom), field_energy(flat_pa, velocity_geom)
            p_energy_true, p_energy_pred = field_energy(flat_tb, pressure_geom), field_energy(flat_pb, pressure_geom)
            horizon_metrics[str(horizon)] = {
                "num_windows": int(len(starts)),
                "velocity_area_weighted_physical_relative_l2": math.sqrt(float(np.nansum(field_error_energy(flat_ta, flat_pa, velocity_geom))) / (float(np.nansum(u_energy_true)) + EPS)),
                "pressure_area_weighted_physical_relative_l2": math.sqrt(float(np.nansum(field_error_energy(flat_tb, flat_pb, pressure_geom))) / (float(np.nansum(p_energy_true)) + EPS)),
                "velocity_modal_relative_l2": relative_l2(flat_ta, flat_pa),
                "pressure_modal_relative_l2": relative_l2(flat_tb, flat_pb),
                "finite_fraction": float(np.mean(finite_step[:, :horizon])),
                "divergent_windows": int(window_bad.sum()),
                "first_divergence_step": int(first[0] + 1) if len(first) else None,
                "velocity_energy_drift": abs(float(np.nanmean(u_energy_pred)) - float(np.mean(u_energy_true))) / (abs(float(np.mean(u_energy_true))) + EPS),
                "pressure_energy_drift": abs(float(np.nanmean(p_energy_pred)) - float(np.mean(p_energy_true))) / (abs(float(np.mean(p_energy_true))) + EPS),
            }
        diagnostics = phase_metrics(
            ta, pa, times_np, scale_stats.center, scale_stats.plane,
            orbit_scale, scale_stats.r_floor, growth_tol,
        )
        prhs = torch.stack(output["pred_rhs"], 1)[:, :1].float().cpu().numpy().reshape(-1, 32)
        trhs = torch.stack(output["true_rhs"], 1)[:, :1].float().cpu().numpy().reshape(-1, 32)
        diagnostics["one_step_rhs_relative_l2"] = relative_l2(trhs, prhs)
        diagnostics["one_step_pressure_closure_relative_l2"] = relative_l2(tb[:, :1].reshape(-1, 32), pb[:, :1].reshape(-1, 32))
        k48 = horizon_metrics["48"]
        preserved = (
            k48["finite_fraction"] == 1.0 and k48["divergent_windows"] == 0
            and k48["velocity_area_weighted_physical_relative_l2"] <= 0.05
            and k48["pressure_area_weighted_physical_relative_l2"] <= 0.05
            and diagnostics["rms_amplitude_error"] <= 0.10
            and diagnostics["peak_to_peak_amplitude_error"] <= 0.10
            and diagnostics["frequency_relative_error"] <= 0.05
            and abs(diagnostics["terminal_phase_drift_cycles"]) <= 0.25
            and diagnostics["normalized_orbit_distance"] <= 0.10
            and k48["velocity_energy_drift"] <= 0.10
            and k48["pressure_energy_drift"] <= 0.10
            and not diagnostics["false_growth"]
        )
        by_re[f"{value:.6f}"] = {
            "horizons": horizon_metrics,
            "hopf_diagnostics_k48": diagnostics,
            "hopf_attractor_preserved": bool(preserved),
        }
        print(json.dumps({"event": "heldout_re_complete", "experiment": checkpoint["experiment_name"],
                          "Re": value, "preserved": bool(preserved)}), flush=True)

    return {
        "experiment_name": checkpoint["experiment_name"],
        "variant": checkpoint["variant"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_optimizer_step": int(checkpoint["optimizer_step"]),
        "checkpoint_validation_score": float(checkpoint["best_score"]),
        "heldout_Re": list(HELDOUT_RE),
        "horizons": list(HORIZONS),
        "window_stride": stride,
        "by_re": by_re,
        "heldout_preserved_count": sum(int(row["hopf_attractor_preserved"]) for row in by_re.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--finalization-dir", type=Path, required=True)
    parser.add_argument("--window-stride", type=int, default=8)
    args = parser.parse_args()
    module = load_module(args.trainer)
    marker = args.finalization_dir / "HELDOUT_EVALUATION_STARTED.json"
    if marker.exists():
        raise FileExistsError("Heldout evaluation marker already exists; refusing silent rerun")
    atomic_json({"status": "STARTED", "selection_manifest": "checkpoint_selection_manifest.json"}, marker)
    results = {}
    for name in ("HopfLocal32_V16Common_K1248", "HopfLocal32_V16ScaleAware_K1248"):
        checkpoint = args.finalization_dir / name / "final.pt"
        results[name] = evaluate_one(module, checkpoint, args.source_root, args.window_stride)
        atomic_json(results[name], args.finalization_dir / name / "heldout_metrics.json")
        torch.cuda.empty_cache()
    output = {
        "schema_version": 1,
        "selection_split": "validation_only",
        "heldout_used_for_checkpoint_selection": False,
        "protocol": {
            "state_autonomous": True,
            "known_inputs": "fixed Re and observed dt only",
            "common_K48_capable_windows": True,
            "area_weighting": "exact train-only Hopf POD quadratic geometry",
            "critical_plane": "train-only local POD modes 2/3 (1-based)",
            "growth_floor_and_statistics": "12 training Re only",
            "divergence": "non-finite or velocity/pressure modal norm >10x true-window maximum",
            "preserved_thresholds": {
                "K48_u_p_physical_relative_l2_max": 0.05,
                "rms_and_peak_to_peak_amplitude_error_max": 0.10,
                "frequency_relative_error_max": 0.05,
                "terminal_phase_drift_abs_cycles_max": 0.25,
                "normalized_orbit_distance_max": 0.10,
                "velocity_pressure_energy_drift_max": 0.10,
                "finite_fraction": 1.0,
                "divergent_windows": 0,
                "false_growth": False,
            },
        },
        "experiments": results,
    }
    atomic_json(output, args.finalization_dir / "heldout_metrics.json")
    atomic_json({"status": "COMPLETE", "heldout_metrics": "heldout_metrics.json"},
                args.finalization_dir / "HELDOUT_EVALUATION_DONE.json")


if __name__ == "__main__":
    main()
