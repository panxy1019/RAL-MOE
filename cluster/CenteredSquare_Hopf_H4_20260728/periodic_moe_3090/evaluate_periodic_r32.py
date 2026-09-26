#!/usr/bin/env python3
"""Common-window heldout diagnostics for the Periodic r32/rp32 best checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


HORIZONS = (1, 4, 8, 16, 24, 48)
HELDOUT_RE = (70.314635, 100.352251, 149.059229, 189.862278)
EPS = 1.0e-12


def load_training_module(path: Path):
    spec = importlib.util.spec_from_file_location("periodic_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def analytic_phase(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    spectrum = np.fft.fft(x - np.mean(x))
    multiplier = np.zeros(n, dtype=np.float64)
    multiplier[0] = 1.0
    if n % 2 == 0:
        multiplier[n // 2] = 1.0
        multiplier[1 : n // 2] = 2.0
    else:
        multiplier[1 : (n + 1) // 2] = 2.0
    return np.unwrap(np.angle(np.fft.ifft(spectrum * multiplier)))


def dominant_frequency(signal: np.ndarray, dt: float) -> float:
    x = np.asarray(signal, dtype=np.float64)
    if len(x) < 4 or not np.all(np.isfinite(x)) or np.std(x) < EPS:
        return float("nan")
    power = np.abs(np.fft.rfft((x - np.mean(x)) * np.hanning(len(x)))) ** 2
    freq = np.fft.rfftfreq(len(x), d=float(dt))
    power[0] = 0.0
    k = int(np.argmax(power))
    if k <= 0 or power[k] <= EPS:
        return float("nan")
    # Parabolic peak interpolation reduces coarse K48 FFT-bin bias.
    delta = 0.0
    if 0 < k < len(power) - 1:
        a, b, c = np.log(power[k - 1 : k + 2] + EPS)
        denom = a - 2.0 * b + c
        if abs(denom) > EPS:
            delta = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    return float((k + delta) / (len(x) * dt))


def weighted_geometry(phi: np.ndarray, mean: np.ndarray, weights: np.ndarray, vector: bool):
    w = np.concatenate([weights, weights]) if vector else weights
    gram = (phi * w[None, :]) @ phi.T
    cross = (phi * w[None, :]) @ mean
    mean_energy = float(np.sum(w * mean * mean))
    return gram.astype(np.float64), cross.astype(np.float64), mean_energy


def field_energy(coeff: np.ndarray, geom) -> np.ndarray:
    gram, cross, mean_energy = geom
    c = np.asarray(coeff, dtype=np.float64)
    return mean_energy + 2.0 * (c @ cross) + np.einsum("...i,ij,...j->...", c, gram, c)


def field_error_energy(true: np.ndarray, pred: np.ndarray, geom) -> np.ndarray:
    gram = geom[0]
    d = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return np.einsum("...i,ij,...j->...", d, gram, d)


def relative_l2(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(pred) - np.asarray(true)) / (np.linalg.norm(true) + EPS))


def trajectory_metrics(
    true_a: np.ndarray,
    pred_a: np.ndarray,
    times: np.ndarray,
    orbit_center: np.ndarray,
    orbit_components: np.ndarray,
    orbit_scale: np.ndarray,
) -> dict:
    true_orbit = (np.asarray(true_a, dtype=np.float64) - orbit_center) @ orbit_components.T
    pred_orbit = (np.asarray(pred_a, dtype=np.float64) - orbit_center) @ orbit_components.T
    true_norm = true_orbit / orbit_scale
    pred_norm = pred_orbit / orbit_scale
    true_radius = np.sqrt(np.sum(true_norm**2, axis=1))
    pred_radius = np.sqrt(np.sum(pred_norm**2, axis=1))
    true_rms = float(np.sqrt(np.mean(true_radius**2)))
    pred_rms = float(np.sqrt(np.mean(pred_radius**2)))
    true_p2p = float(np.ptp(true_norm[:, 0]))
    pred_p2p = float(np.ptp(pred_norm[:, 0]))

    phase_true = np.unwrap(np.arctan2(true_norm[:, 1], true_norm[:, 0]))
    phase_pred = np.unwrap(np.arctan2(pred_norm[:, 1], pred_norm[:, 0]))
    # The PCA-plane orientation is arbitrary; select the sign matching true rotation.
    true_slope = float(np.polyfit(times, phase_true, 1)[0])
    pred_slope = float(np.polyfit(times, phase_pred, 1)[0])
    if true_slope * pred_slope < 0.0:
        phase_pred = -phase_pred
        pred_slope = -pred_slope
    f_true = abs(true_slope) / (2.0 * math.pi)
    f_pred = abs(pred_slope) / (2.0 * math.pi)
    phase_delta = (phase_pred - phase_true) - (phase_pred[0] - phase_true[0])
    distances = np.sqrt(np.sum((pred_norm[:, None, :] - true_norm[None, :, :]) ** 2, axis=2))

    return {
        "rms_amplitude_error": abs(pred_rms - true_rms) / (true_rms + EPS),
        "peak_to_peak_amplitude_error": abs(pred_p2p - true_p2p) / (true_p2p + EPS),
        "dominant_frequency_true": f_true,
        "dominant_frequency_pred": f_pred,
        "strouhal_error": abs(f_pred - f_true) / (abs(f_true) + EPS),
        "phase_rms_rad": float(np.sqrt(np.mean(phase_delta**2))),
        "terminal_cycle_drift": float(phase_delta[-1] / (2.0 * math.pi)),
        "normalized_orbit_distance": float(np.mean(np.min(distances, axis=1))),
    }


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def instantiate_model(module, args, arrays, state, device):
    model = module.OperatorSpaceMoEROM(
        in_dim=arrays["x"].shape[1], out_dim=args.r_u, pressure_dim=args.r_p,
        hidden_dim=args.hidden_dim, expert_hidden=args.expert_hidden,
        num_blocks=args.num_blocks, num_experts=args.num_experts,
        num_operator_spaces=args.num_shared_experts,
        num_regime_groups=args.num_regime_groups, experts_per_group=args.experts_per_group,
        top_k=args.top_k, group_top_k=args.group_top_k, dropout=args.dropout,
        temperature=args.temperature, gate_floor=args.gate_floor,
        group_temperature=args.group_temperature, group_gate_floor=args.group_gate_floor,
        shared_scale=args.shared_scale, routed_scale=args.routed_scale,
        expert_blocks=args.expert_blocks, quadratic_rank=args.quadratic_rank,
        quadratic_scale=args.quadratic_scale, phase_harmonics=args.phase_harmonics,
        closure_mode=args.closure_mode, pressure_base_mode=args.pressure_base_mode,
        film_base_hidden=args.film_base_hidden, film_base_scale=args.film_base_scale,
        attractor_conditioned=args.attractor_conditioned,
        attractor_adapter_dim=args.attractor_adapter_dim,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--existing-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-stride", type=int, default=8)
    cli = parser.parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    module = load_training_module(cli.trainer)
    device = torch.device("cuda")
    checkpoint = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    args = SimpleNamespace(**checkpoint["args"])
    args.device = "cuda"
    arrays, _ = module.build_arrays(args)
    tensors = np.load(args.tensor_path)
    pressure_tensors = np.load(args.pressure_surrogate_path)
    scalers = {
        key: module.Standardizer(
            mean=np.asarray(value["mean"], dtype=np.float32),
            scale=np.asarray(value["scale"], dtype=np.float32),
        )
        for key, value in checkpoint["scalers"].items()
    }
    model = instantiate_model(module, args, arrays, checkpoint["model_state"], device)

    pod_root = Path(args.data_root)
    vel = np.load(pod_root / "global_velocity_pod_area_weighted_l2.npz")
    pre = np.load(pod_root / "global_pressure_pod_area_weighted_l2.npz")
    velocity_geom = weighted_geometry(
        vel["phi_uv"][: args.r_u], vel["mean_uv_regime"], vel["point_areas"], True
    )
    pressure_geom = weighted_geometry(
        pre["phi_p"][: args.r_p], pre["mean_p_regime"], pre["point_areas"], False
    )

    old_metrics = json.loads(cli.existing_metrics.read_text())
    existing_by_re = {round(float(item["test_Re"]), 5): item for item in old_metrics["results"]}
    all_results = []
    valid_sample_set = set(arrays["sample_ids"].tolist())

    for requested_re in HELDOUT_RE:
        re_by_label = np.asarray([
            arrays["re"][np.where(arrays["label_id"] == i)[0][0]]
            for i in range(len(arrays["labels"]))
        ])
        label_id = int(np.argmin(np.abs(re_by_label - requested_re)))
        actual_re = float(re_by_label[label_id])
        idx = np.where(arrays["label_id"] == label_id)[0]
        idx = idx[np.argsort(arrays["time"][idx])]
        full_true_centered = np.asarray(arrays["a"][idx], dtype=np.float64)
        orbit_center = np.mean(full_true_centered, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(full_true_centered - orbit_center, full_matrices=False)
        orbit_components = vt[:2]
        full_orbit = (full_true_centered - orbit_center) @ orbit_components.T
        orbit_scale = np.maximum(np.std(full_orbit, axis=0, keepdims=True), EPS)
        starts = [
            p for p in range(1, len(idx) - max(HORIZONS) - 1, cli.window_stride)
            if int(idx[p]) in valid_sample_set
        ]
        windows = []
        for start_pos in starts:
            start = int(idx[start_pos])
            a_cur = arrays["a"][start].copy()
            b_cur = arrays["b"][start].copy()
            a_hist, b_hist, rhs_hist = module.init_history_states_np(start, arrays)
            pred_a, pred_b, true_a, true_b, times = [], [], [], [], []
            cur = start
            first_bad = None
            for step in range(1, max(HORIZONS) + 1):
                nxt = int(arrays["next_idx"][cur])
                dt = float(arrays["time"][nxt] - arrays["time"][cur])
                a_next, b_next, rhs_g = module.integrate_autonomous_step_np(
                    model, a_cur, b_cur, cur, dt, a_hist, b_hist, rhs_hist,
                    arrays, scalers, tensors, pressure_tensors, args, device,
                )
                finite = bool(np.all(np.isfinite(a_next)) and np.all(np.isfinite(b_next)))
                if not finite:
                    first_bad = step
                    break
                pred_a.append(a_next.copy()); pred_b.append(b_next.copy())
                true_a.append(arrays["a"][nxt].copy()); true_b.append(arrays["b"][nxt].copy())
                times.append(float(arrays["time"][nxt]))
                a_hist = np.concatenate([a_next[None, None, :], a_hist[:, :-1, :]], axis=1)
                b_hist = np.concatenate([b_next[None, None, :], b_hist[:, :-1, :]], axis=1)
                rhs_hist = np.concatenate([rhs_g[None, None, :], rhs_hist[:, :-1, :]], axis=1)
                a_cur, b_cur, cur = a_next, b_next, nxt
            windows.append({
                "pred_a": np.asarray(pred_a), "pred_b": np.asarray(pred_b),
                "true_a": np.asarray(true_a), "true_b": np.asarray(true_b),
                "times": np.asarray(times), "first_nonfinite": first_bad,
            })

        horizon_metrics = {}
        for horizon in HORIZONS:
            complete = [w for w in windows if len(w["pred_a"]) >= horizon]
            expected_steps = len(windows) * horizon
            finite_steps = sum(min(len(w["pred_a"]), horizon) for w in windows)
            divergent = 0
            first_divergence = None
            for w in windows:
                if len(w["pred_a"]) < horizon:
                    divergent += 1
                    bad = w["first_nonfinite"] or (len(w["pred_a"]) + 1)
                    first_divergence = bad if first_divergence is None else min(first_divergence, bad)
                    continue
                ta, pa = w["true_a"][:horizon], w["pred_a"][:horizon]
                tb, pb = w["true_b"][:horizon], w["pred_b"][:horizon]
                a_ratio = np.linalg.norm(pa, axis=1) / (np.max(np.linalg.norm(ta, axis=1)) + EPS)
                b_ratio = np.linalg.norm(pb, axis=1) / (np.max(np.linalg.norm(tb, axis=1)) + EPS)
                bad_idx = np.where((a_ratio > 10.0) | (b_ratio > 10.0))[0]
                if len(bad_idx):
                    divergent += 1
                    bad = int(bad_idx[0] + 1)
                    first_divergence = bad if first_divergence is None else min(first_divergence, bad)

            if complete:
                ta = np.concatenate([w["true_a"][:horizon] for w in complete])
                pa = np.concatenate([w["pred_a"][:horizon] for w in complete])
                tb = np.concatenate([w["true_b"][:horizon] for w in complete])
                pb = np.concatenate([w["pred_b"][:horizon] for w in complete])
                velocity_field_l2 = math.sqrt(
                    float(np.sum(field_error_energy(ta, pa, velocity_geom))) /
                    (float(np.sum(field_energy(ta, velocity_geom))) + EPS)
                )
                pressure_field_l2 = math.sqrt(
                    float(np.sum(field_error_energy(tb, pb, pressure_geom))) /
                    (float(np.sum(field_energy(tb, pressure_geom))) + EPS)
                )
                velocity_energy_drift = float(
                    abs(np.mean(field_energy(pa, velocity_geom)) - np.mean(field_energy(ta, velocity_geom))) /
                    (abs(np.mean(field_energy(ta, velocity_geom))) + EPS)
                )
                pressure_energy_drift = float(
                    abs(np.mean(field_energy(pb, pressure_geom)) - np.mean(field_energy(tb, pressure_geom))) /
                    (abs(np.mean(field_energy(tb, pressure_geom))) + EPS)
                )
                coeff_a = relative_l2(ta, pa)
                coeff_b = relative_l2(tb, pb)
            else:
                velocity_field_l2 = pressure_field_l2 = float("nan")
                velocity_energy_drift = pressure_energy_drift = float("nan")
                coeff_a = coeff_b = float("nan")
            horizon_metrics[str(horizon)] = {
                "num_windows": len(windows), "complete_windows": len(complete),
                "velocity_area_weighted_l2": velocity_field_l2,
                "pressure_area_weighted_l2": pressure_field_l2,
                "velocity_coefficient_relative_l2": coeff_a,
                "pressure_coefficient_relative_l2": coeff_b,
                "finite_fraction": finite_steps / max(1, expected_steps),
                "divergent_windows": divergent, "first_divergence_step": first_divergence,
                "velocity_energy_drift": velocity_energy_drift,
                "pressure_energy_drift": pressure_energy_drift,
            }

        trajectory = [
            trajectory_metrics(
                w["true_a"][:48], w["pred_a"][:48], w["times"][:48],
                orbit_center, orbit_components, orbit_scale,
            )
            for w in windows if len(w["pred_a"]) >= 48
        ]
        trajectory_mean = {
            key: finite_mean([item[key] for item in trajectory])
            for key in trajectory[0]
        } if trajectory else {}
        existing = existing_by_re[round(actual_re, 5)]
        result = {
            "Re": actual_re, "label": str(arrays["labels"][label_id]),
            "common_window_stride": cli.window_stride,
            "horizons": horizon_metrics,
            "k48_cycle_diagnostics": trajectory_mean,
            "rhs_relative_l2": float(existing["deep_moe"]["rhs_relative_l2"]),
            "pressure_closure_relative_l2": float(existing["deep_moe"]["pressure_head_relative_l2"]),
            "best_epoch": int(existing["best_epoch"]),
            "best_val_score": float(existing["best_val_score"]),
        }
        k48 = horizon_metrics["48"]
        preserved = (
            k48["finite_fraction"] == 1.0 and k48["divergent_windows"] == 0
            and k48["velocity_area_weighted_l2"] <= 0.05
            and k48["pressure_area_weighted_l2"] <= 0.05
            and trajectory_mean.get("rms_amplitude_error", math.inf) <= 0.10
            and trajectory_mean.get("peak_to_peak_amplitude_error", math.inf) <= 0.10
            and trajectory_mean.get("strouhal_error", math.inf) <= 0.05
            and abs(trajectory_mean.get("terminal_cycle_drift", math.inf)) <= 0.25
            and trajectory_mean.get("normalized_orbit_distance", math.inf) <= 0.10
            and k48["velocity_energy_drift"] <= 0.10
            and k48["pressure_energy_drift"] <= 0.10
        )
        result["preserved"] = bool(preserved)
        all_results.append(result)
        print(json.dumps({"event": "re_complete", "Re": actual_re, "preserved": preserved}), flush=True)

    output = {
        "checkpoint": str(cli.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_score": float(checkpoint["best_val_score"]),
        "horizons": list(HORIZONS),
        "heldout_Re": list(HELDOUT_RE),
        "protocol": {
            "autonomy": "predicted a/b/history; known Re and dataset-clock phase/time retained",
            "windows": "common K48-capable starts with fixed stride; all horizons use identical starts",
            "divergence": "non-finite state or velocity/pressure modal norm >10x window true maximum",
            "area_weighting": "exact POD field quadratic form with point areas and regime mean",
            "cycle_signal": "train-independent heldout trajectory PCA plane fitted to true modal orbit; phase from atan2 and frequency from phase slope",
            "preserved_thresholds": {
                "K48_velocity_pressure_area_l2": 0.05, "amplitude_error": 0.10,
                "strouhal_error": 0.05, "terminal_cycle_drift_abs": 0.25,
                "normalized_orbit_distance": 0.10, "energy_drift": 0.10,
                "finite_fraction": 1.0, "divergent_windows": 0,
            },
        },
        "results": all_results,
        "heldout_preserved_count": int(sum(item["preserved"] for item in all_results)),
    }
    out = cli.output_dir / "periodic_r32_multihorizon_evaluation.json"
    out.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({"event": "evaluation_complete", "output": str(out),
                      "heldout_preserved_count": output["heldout_preserved_count"]}), flush=True)


if __name__ == "__main__":
    main()
