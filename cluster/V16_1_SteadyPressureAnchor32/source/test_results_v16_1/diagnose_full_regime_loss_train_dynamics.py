#!/usr/bin/env python3
"""Train-Re dynamics diagnostics for V16 FullRegimeLoss.

Diagnostics:
- Hopf near-onset per-Re 24-step amplitude/frequency/energy rollout behavior.
- Steady train-Re pressure drift and pressure-closure component attribution.
- Top-2 expert load, including shared-vs-routed usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

import evaluate_full_regime_loss_train_set as base_eval
import train_v16_attractor_moe_rom as v16

EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "/root/moe/V16_AttractorMoEROM/test_results_v16/results/"
            "V16_AttractorMoEROM_FullRegimeLoss32/"
            "V16_AttractorMoEROM_FullRegimeLoss32_ru32_rp32"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--window-stride", type=int, default=24)
    parser.add_argument("--start-offset", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=768)
    return parser.parse_args()


def finite_stats(values: Iterable[float]) -> Dict[str, float]:
    return base_eval.finite_stats(values)


def scalar(value: np.ndarray | float) -> float:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return math.nan
    return float(arr.reshape(-1)[0])


def l2_per_row(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(x, dtype=np.float64), axis=-1)


def wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def make_setup(run_dir: Path, checkpoint: Path | None) -> Dict[str, object]:
    metrics_path = run_dir / f"{run_dir.name}_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    args = base_eval.make_v16_args(metrics, run_dir)
    ckpt_path = checkpoint or (run_dir / f"{run_dir.name}_Re_24p630436_checkpoint.pt")
    args.rollout_steps = int(args.rollout_steps)
    if getattr(args, "allow_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays, _meta = v16.build_arrays(args)
    args.test_re_indices = v16.resolve_test_re_indices(args, arrays)
    train_ids, val_ids, rollout_pool_ids, split_stats = v16.build_data_ablation_split(
        args, arrays, [int(v) for v in args.test_re_indices]
    )
    tensors = np.load(args.tensor_path)
    pressure_tensors = np.load(args.pressure_surrogate_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    scalers = base_eval.scalers_from_checkpoint(ckpt)
    model = base_eval.make_model(args, arrays["x"].shape[1], device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    arrays_t = base_eval.arrays_to_torch(arrays, device)
    scalers_t = base_eval.make_scalers_t(scalers, args, device)
    gal = v16.build_galerkin_torch(tensors, arrays["labels"], args.r_u, args.r_p, device)
    sur = v16.build_pressure_surrogate_torch(
        pressure_tensors, arrays["labels"], args.r_u, args.r_p, device
    )
    base_bundle = v16.build_base_bundle(args, arrays, device)
    return {
        "args": args,
        "arrays": arrays,
        "arrays_t": arrays_t,
        "scalers": scalers,
        "scalers_t": scalers_t,
        "model": model,
        "device": device,
        "tensors": tensors,
        "pressure_tensors": pressure_tensors,
        "gal": gal,
        "sur": sur,
        "base_bundle": base_bundle,
        "ckpt_path": ckpt_path,
        "ckpt": ckpt,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "rollout_pool_ids": rollout_pool_ids,
        "split_stats": split_stats,
    }


def integrate_step_with_components_torch(
    model: v16.OperatorSpaceMoEROM,
    a_state: torch.Tensor,
    b_state: torch.Tensor,
    current: torch.Tensor,
    dt: torch.Tensor,
    a_hist: torch.Tensor,
    b_hist: torch.Tensor,
    rhs_hist: torch.Tensor,
    arrays_t: Dict[str, torch.Tensor],
    scalers_t: Dict[str, torch.Tensor],
    gal: Dict[int, Dict[str, torch.Tensor]],
    sur: Dict[int, Dict[str, torch.Tensor]],
    base_bundle: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    k1, pressure_op, rhs_g, closure_params = v16.model_outputs_from_states_torch(
        model,
        a_state,
        b_state,
        current,
        a_hist,
        b_hist,
        rhs_hist,
        arrays_t,
        scalers_t,
        gal,
        base_bundle,
        args,
    )
    if args.integrator == "rk4":
        k2, _, _, _ = v16.model_outputs_from_states_torch(
            model,
            a_state + 0.5 * dt * k1,
            b_state,
            current,
            a_hist,
            b_hist,
            rhs_hist,
            arrays_t,
            scalers_t,
            gal,
            base_bundle,
            args,
        )
        k3, _, _, _ = v16.model_outputs_from_states_torch(
            model,
            a_state + 0.5 * dt * k2,
            b_state,
            current,
            a_hist,
            b_hist,
            rhs_hist,
            arrays_t,
            scalers_t,
            gal,
            base_bundle,
            args,
        )
        k4, _, _, _ = v16.model_outputs_from_states_torch(
            model,
            a_state + dt * k3,
            b_state,
            current,
            a_hist,
            b_hist,
            rhs_hist,
            arrays_t,
            scalers_t,
            gal,
            base_bundle,
            args,
        )
        a_next = a_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    else:
        a_next = a_state + dt * k1

    b_base = v16.pressure_base_from_sample_torch(
        model, a_next, current, arrays_t, sur, base_bundle, closure_params
    )
    if args.pressure_input_mode != "pressure_only":
        pressure_state_override = v16.pressure_input_override_torch(
            args.pressure_input_mode, a_next, b_base, scalers_t
        )
        _, pressure_op, _, closure_params = v16.model_outputs_from_states_torch(
            model,
            a_state,
            b_state,
            current,
            a_hist,
            b_hist,
            rhs_hist,
            arrays_t,
            scalers_t,
            gal,
            base_bundle,
            args,
            pressure_state_override=pressure_state_override,
        )
    if args.pressure_target == "state":
        b_next = pressure_op
        base_component = b_base
        residual_component = b_next - b_base
        base_scale = torch.ones_like(b_base)
        residual_scale = torch.ones_like(b_base)
    else:
        b_next, base_component, residual_component, base_scale, residual_scale = (
            v16.closure_components_torch(args.closure_mode, b_base, pressure_op, closure_params)
        )
    return {
        "a_next": a_next,
        "b_next": b_next,
        "rhs_g": rhs_g,
        "b_base": b_base,
        "pressure_residual": pressure_op,
        "base_component": base_component,
        "residual_component": residual_component,
        "base_scale": base_scale,
        "residual_scale": residual_scale,
        "alpha": closure_params["alpha"],
        "beta": closure_params["beta"],
    }


def start_ids_for_label(
    label_id: int,
    arrays: Dict[str, np.ndarray],
    train_ids: np.ndarray,
    rollout_pool_ids: np.ndarray,
    steps: int,
    stride: int,
    offset: int,
) -> np.ndarray:
    label_train = train_ids[arrays["label_id"][train_ids] == label_id]
    label_train = label_train[np.argsort(arrays["time"][label_train])]
    label_pool = rollout_pool_ids[arrays["label_id"][rollout_pool_ids] == label_id]
    starts = v16.sequence_start_ids(
        label_train,
        arrays["label_id"],
        arrays["next_idx"],
        steps,
        sequence_pool_ids=label_pool,
    )
    if len(starts) == 0:
        return starts
    return starts[int(offset) :: max(1, int(stride))]


def collect_rollout(
    label_id: int,
    starts: np.ndarray,
    setup: Dict[str, object],
    steps: int,
) -> Dict[str, np.ndarray]:
    model = setup["model"]
    arrays_t = setup["arrays_t"]
    scalers_t = setup["scalers_t"]
    args = setup["args"]
    device = setup["device"]
    current = torch.tensor(starts, dtype=torch.long, device=device)
    a_cur = arrays_t["a"][current].clone()
    b_cur = arrays_t["b"][current].clone()
    a_hist, b_hist, rhs_hist = v16.init_history_states_torch(current, arrays_t)
    init_a = a_cur.detach().cpu().numpy()
    init_b = b_cur.detach().cpu().numpy()
    out: Dict[str, List[np.ndarray]] = {
        "pred_a": [],
        "true_a": [],
        "pred_b": [],
        "true_b": [],
        "b_base": [],
        "pressure_residual": [],
        "base_component": [],
        "residual_component": [],
        "base_scale": [],
        "residual_scale": [],
        "alpha": [],
        "beta": [],
        "time": [],
    }
    with torch.no_grad():
        for _ in range(steps):
            nxt = arrays_t["next_idx"][current]
            dt = arrays_t["dt_next"][current][:, None]
            step = integrate_step_with_components_torch(
                model,
                a_cur,
                b_cur,
                current,
                dt,
                a_hist,
                b_hist,
                rhs_hist,
                arrays_t,
                scalers_t,
                setup["gal"],
                setup["sur"],
                setup["base_bundle"],
                args,
            )
            a_next = step["a_next"]
            b_next = step["b_next"]
            out["pred_a"].append(a_next.detach().cpu().numpy())
            out["true_a"].append(arrays_t["a"][nxt].detach().cpu().numpy())
            out["pred_b"].append(b_next.detach().cpu().numpy())
            out["true_b"].append(arrays_t["b"][nxt].detach().cpu().numpy())
            for key in [
                "b_base",
                "pressure_residual",
                "base_component",
                "residual_component",
                "base_scale",
                "residual_scale",
                "alpha",
                "beta",
            ]:
                out[key].append(step[key].detach().cpu().numpy())
            out["time"].append(arrays_t["time"][nxt].detach().cpu().numpy())
            a_hist = torch.cat([a_next[:, None, :], a_hist[:, :-1, :]], dim=1)
            b_hist = torch.cat([b_next[:, None, :], b_hist[:, :-1, :]], dim=1)
            rhs_hist = torch.cat([step["rhs_g"][:, None, :], rhs_hist[:, :-1, :]], dim=1)
            a_cur = a_next
            b_cur = b_next
            current = nxt
    stacked = {key: np.stack(values, axis=1) for key, values in out.items()}
    stacked["init_a"] = init_a
    stacked["init_b"] = init_b
    stacked["starts"] = starts.astype(np.int64)
    stacked["label_id"] = np.asarray([label_id], dtype=np.int64)
    return stacked


def hopf_metrics_for_rollout(
    rollout: Dict[str, np.ndarray],
    arrays: Dict[str, np.ndarray],
    label_id: int,
    args: argparse.Namespace,
    train_ids: np.ndarray,
) -> Dict[str, float]:
    true_a = rollout["true_a"]
    pred_a = rollout["pred_a"]
    init_a = rollout["init_a"]
    times = rollout["time"]
    i0, i1 = v16.parse_hopf_pair(args)
    true_pair = true_a[..., [i0, i1]].astype(np.float64)
    pred_pair = pred_a[..., [i0, i1]].astype(np.float64)
    r_true = np.sqrt(np.sum(true_pair * true_pair, axis=-1) + EPS)
    r_pred = np.sqrt(np.sum(pred_pair * pred_pair, axis=-1) + EPS)
    theta_true = np.unwrap(np.arctan2(true_pair[..., 1], true_pair[..., 0]), axis=1)
    theta_pred = np.unwrap(np.arctan2(pred_pair[..., 1], pred_pair[..., 0]), axis=1)
    phase_error = np.abs(wrap_angle(np.arctan2(pred_pair[..., 1], pred_pair[..., 0]) - np.arctan2(true_pair[..., 1], true_pair[..., 0])))
    dt = np.diff(times, axis=1)
    freq_true = np.diff(theta_true, axis=1) / np.maximum(dt, EPS)
    freq_pred = np.diff(theta_pred, axis=1) / np.maximum(dt, EPS)
    diff = pred_a.astype(np.float64) - true_a.astype(np.float64)
    true_energy_sum = float(np.sum(true_a.astype(np.float64) ** 2))
    diff_energy_sum = float(np.sum(diff * diff))
    label_train = train_ids[arrays["label_id"][train_ids] == label_id]
    global_floor = float(
        np.percentile(np.sum(arrays["a_next"][train_ids] ** 2, axis=1), 10)
        * float(args.relative_floor_frac)
        + EPS
    )
    local_floor = float(
        np.percentile(np.sum(arrays["a_next"][label_train] ** 2, axis=1), 10)
        * float(args.relative_floor_frac)
        + EPS
    ) if len(label_train) else global_floor
    n_vec = int(np.prod(true_a.shape[:2]))
    init_energy = np.sum(init_a.astype(np.float64) ** 2, axis=1)
    true_final_energy = np.sum(true_a[:, -1, :].astype(np.float64) ** 2, axis=1)
    pred_final_energy = np.sum(pred_a[:, -1, :].astype(np.float64) ** 2, axis=1)
    horizon_time = np.maximum(times[:, -1] - times[:, 0], EPS)
    true_growth = (np.log(true_final_energy + EPS) - np.log(init_energy + EPS)) / horizon_time
    pred_growth = (np.log(pred_final_energy + EPS) - np.log(init_energy + EPS)) / horizon_time
    return {
        "label_id": int(label_id),
        "label": str(arrays["labels"][label_id]),
        "Re": float(np.mean(arrays["re"][label_train])) if len(label_train) else float("nan"),
        "regime": str(arrays["regime"][label_train[0]]) if len(label_train) else "unknown",
        "attractor": str(arrays["attractor"][label_train[0]]) if len(label_train) else "unknown",
        "num_windows": int(true_a.shape[0]),
        "r_true_mean": float(np.mean(r_true)),
        "r_pred_mean": float(np.mean(r_pred)),
        "r_true_max": float(np.max(r_true)),
        "r_pred_max": float(np.max(r_pred)),
        "r_pred_over_true_mean": float(np.mean(r_pred / np.maximum(r_true, EPS))),
        "a_error_abs_l2": float(math.sqrt(diff_energy_sum)),
        "a_error_norm_mean": float(np.mean(l2_per_row(diff.reshape(-1, diff.shape[-1])))),
        "a_error_relative_l2": float(math.sqrt(diff_energy_sum) / (math.sqrt(true_energy_sum) + EPS)),
        "a_error_energy_floor_normalized_global": float(
            math.sqrt(diff_energy_sum) / math.sqrt(true_energy_sum + global_floor * n_vec)
        ),
        "a_error_energy_floor_normalized_local": float(
            math.sqrt(diff_energy_sum) / math.sqrt(true_energy_sum + local_floor * n_vec)
        ),
        "denominator_true_norm": float(math.sqrt(true_energy_sum)),
        "denominator_global_floor_norm": float(math.sqrt(true_energy_sum + global_floor * n_vec)),
        "denominator_local_floor_norm": float(math.sqrt(true_energy_sum + local_floor * n_vec)),
        "phase_error_mean": float(np.mean(phase_error)),
        "phase_error_median": float(np.median(phase_error)),
        "frequency_error_mean": float(np.mean(np.abs(freq_pred - freq_true))),
        "frequency_true_mean": float(np.mean(freq_true)),
        "frequency_pred_mean": float(np.mean(freq_pred)),
        "rollout_energy_growth_rate_true_mean": float(np.mean(true_growth)),
        "rollout_energy_growth_rate_pred_mean": float(np.mean(pred_growth)),
        "rollout_energy_growth_rate_excess_mean": float(np.mean(pred_growth - true_growth)),
        "final_energy_pred_over_true_mean": float(
            np.mean(pred_final_energy / np.maximum(true_final_energy, EPS))
        ),
    }


def steady_pressure_metrics_for_rollout(
    rollout: Dict[str, np.ndarray],
    arrays: Dict[str, np.ndarray],
    label_id: int,
    train_ids: np.ndarray,
) -> Dict[str, float]:
    true_b = rollout["true_b"].astype(np.float64)
    pred_b = rollout["pred_b"].astype(np.float64)
    b_base = rollout["b_base"].astype(np.float64)
    base_component = rollout["base_component"].astype(np.float64)
    residual_component = rollout["residual_component"].astype(np.float64)
    residual_raw = rollout["pressure_residual"].astype(np.float64)
    alpha = rollout["alpha"].astype(np.float64)
    pred_norm = l2_per_row(pred_b.reshape(-1, pred_b.shape[-1])) + EPS
    true_norm = l2_per_row(true_b.reshape(-1, true_b.shape[-1])) + EPS
    label_train = train_ids[arrays["label_id"][train_ids] == label_id]
    b_mean = np.mean(arrays["b"][label_train].astype(np.float64), axis=0)
    b_error = pred_b - true_b
    pred_minus_mean = pred_b - b_mean.reshape(1, 1, -1)
    true_energy = np.sum(true_b * true_b, axis=-1)
    pred_energy = np.sum(pred_b * pred_b, axis=-1)
    energy_drift = (pred_energy - true_energy) / np.maximum(true_energy, EPS)
    base_contrib = l2_per_row(base_component.reshape(-1, base_component.shape[-1])) / pred_norm
    residual_contrib = l2_per_row(residual_component.reshape(-1, residual_component.shape[-1])) / pred_norm
    raw_base_error = l2_per_row((b_base - true_b).reshape(-1, b_base.shape[-1])) / true_norm
    effective_base_error = l2_per_row(
        (base_component - true_b).reshape(-1, base_component.shape[-1])
    ) / true_norm
    residual_magnitude = l2_per_row(
        residual_component.reshape(-1, residual_component.shape[-1])
    ) / true_norm
    alpha_step_mean = alpha.mean(axis=(0, 2))
    residual_step = l2_per_row(residual_component.reshape(-1, residual_component.shape[-1]))
    base_step = l2_per_row(base_component.reshape(-1, base_component.shape[-1]))
    return {
        "label_id": int(label_id),
        "label": str(arrays["labels"][label_id]),
        "Re": float(np.mean(arrays["re"][label_train])) if len(label_train) else float("nan"),
        "regime": str(arrays["regime"][label_train[0]]) if len(label_train) else "unknown",
        "attractor": str(arrays["attractor"][label_train[0]]) if len(label_train) else "unknown",
        "num_windows": int(true_b.shape[0]),
        "b_pred_minus_true_norm_mean": float(np.mean(l2_per_row(b_error.reshape(-1, b_error.shape[-1])))),
        "b_pred_minus_true_relative_l2": v16.relative_l2_np(true_b, pred_b),
        "b_pred_minus_true_norm_final_mean": float(np.mean(l2_per_row(b_error[:, -1, :]))),
        "b_pred_minus_b_mean_norm_mean": float(
            np.mean(l2_per_row(pred_minus_mean.reshape(-1, pred_minus_mean.shape[-1])))
        ),
        "b_pred_minus_b_mean_norm_final_mean": float(np.mean(l2_per_row(pred_minus_mean[:, -1, :]))),
        "pressure_energy_drift_abs_mean": float(np.mean(np.abs(energy_drift))),
        "pressure_energy_drift_signed_mean": float(np.mean(energy_drift)),
        "pressure_energy_drift_final_signed_mean": float(np.mean(energy_drift[:, -1])),
        "adaptive_gate_alpha_mean": float(np.mean(alpha)),
        "adaptive_gate_alpha_std": float(np.std(alpha)),
        "adaptive_gate_alpha_time_std": float(np.std(alpha_step_mean)),
        "adaptive_gate_alpha_step_min": float(np.min(alpha_step_mean)),
        "adaptive_gate_alpha_step_max": float(np.max(alpha_step_mean)),
        "raw_base_error_relative_mean": float(np.mean(raw_base_error)),
        "effective_base_error_relative_mean": float(np.mean(effective_base_error)),
        "base_contribution_ratio_mean": float(np.mean(base_contrib)),
        "residual_contribution_ratio_mean": float(np.mean(residual_contrib)),
        "residual_magnitude_over_true_mean": float(np.mean(residual_magnitude)),
        "base_component_norm_initial_mean": float(np.mean(base_step[: true_b.shape[0]])),
        "base_component_norm_final_mean": float(np.mean(l2_per_row(base_component[:, -1, :]))),
        "residual_component_norm_initial_mean": float(np.mean(residual_step[: true_b.shape[0]])),
        "residual_component_norm_final_mean": float(np.mean(l2_per_row(residual_component[:, -1, :]))),
        "raw_residual_norm_mean": float(np.mean(l2_per_row(residual_raw.reshape(-1, residual_raw.shape[-1])))),
    }


def hopf_time_series_rows(
    rollout: Dict[str, np.ndarray],
    arrays: Dict[str, np.ndarray],
    label_id: int,
    args: argparse.Namespace,
    train_ids: np.ndarray,
) -> List[Dict[str, float]]:
    true_a = rollout["true_a"].astype(np.float64)
    pred_a = rollout["pred_a"].astype(np.float64)
    times = rollout["time"].astype(np.float64)
    i0, i1 = v16.parse_hopf_pair(args)
    true_pair = true_a[..., [i0, i1]]
    pred_pair = pred_a[..., [i0, i1]]
    r_true = np.sqrt(np.sum(true_pair * true_pair, axis=-1) + EPS)
    r_pred = np.sqrt(np.sum(pred_pair * pred_pair, axis=-1) + EPS)
    theta_true = np.arctan2(true_pair[..., 1], true_pair[..., 0])
    theta_pred = np.arctan2(pred_pair[..., 1], pred_pair[..., 0])
    phase_error = np.abs(wrap_angle(theta_pred - theta_true))
    global_floor = float(
        np.percentile(np.sum(arrays["a_next"][train_ids] ** 2, axis=1), 10)
        * float(args.relative_floor_frac)
        + EPS
    )
    label_train = train_ids[arrays["label_id"][train_ids] == label_id]
    rows: List[Dict[str, float]] = []
    for step in range(true_a.shape[1]):
        diff = pred_a[:, step, :] - true_a[:, step, :]
        true_energy = np.sum(true_a[:, step, :] ** 2)
        diff_energy = np.sum(diff * diff)
        n_vec = true_a.shape[0]
        pred_energy_per_window = np.sum(pred_a[:, step, :] ** 2, axis=1)
        true_energy_per_window = np.sum(true_a[:, step, :] ** 2, axis=1)
        rows.append(
            {
                "label_id": int(label_id),
                "label": str(arrays["labels"][label_id]),
                "Re": float(np.mean(arrays["re"][label_train])) if len(label_train) else float("nan"),
                "step": int(step + 1),
                "time_mean": float(np.mean(times[:, step])),
                "r_true_mean": float(np.mean(r_true[:, step])),
                "r_pred_mean": float(np.mean(r_pred[:, step])),
                "r_pred_over_true_mean": float(
                    np.mean(r_pred[:, step] / np.maximum(r_true[:, step], EPS))
                ),
                "a_error_relative_l2_step": float(
                    math.sqrt(diff_energy) / (math.sqrt(true_energy) + EPS)
                ),
                "a_error_floor_normalized_step": float(
                    math.sqrt(diff_energy) / math.sqrt(true_energy + global_floor * n_vec)
                ),
                "phase_error_mean": float(np.mean(phase_error[:, step])),
                "pred_energy_mean": float(np.mean(pred_energy_per_window)),
                "true_energy_mean": float(np.mean(true_energy_per_window)),
                "pred_over_true_energy_mean": float(
                    np.mean(pred_energy_per_window / np.maximum(true_energy_per_window, EPS))
                ),
            }
        )
    return rows


def steady_pressure_time_series_rows(
    rollout: Dict[str, np.ndarray],
    arrays: Dict[str, np.ndarray],
    label_id: int,
    train_ids: np.ndarray,
) -> List[Dict[str, float]]:
    true_b = rollout["true_b"].astype(np.float64)
    pred_b = rollout["pred_b"].astype(np.float64)
    b_base = rollout["b_base"].astype(np.float64)
    base_component = rollout["base_component"].astype(np.float64)
    residual_component = rollout["residual_component"].astype(np.float64)
    alpha = rollout["alpha"].astype(np.float64)
    times = rollout["time"].astype(np.float64)
    label_train = train_ids[arrays["label_id"][train_ids] == label_id]
    b_mean = np.mean(arrays["b"][label_train].astype(np.float64), axis=0)
    rows: List[Dict[str, float]] = []
    for step in range(true_b.shape[1]):
        true_step = true_b[:, step, :]
        pred_step = pred_b[:, step, :]
        pred_norm = l2_per_row(pred_step) + EPS
        true_norm = l2_per_row(true_step) + EPS
        b_error = pred_step - true_step
        pred_minus_mean = pred_step - b_mean.reshape(1, -1)
        energy_drift = (
            np.sum(pred_step * pred_step, axis=1) - np.sum(true_step * true_step, axis=1)
        ) / np.maximum(np.sum(true_step * true_step, axis=1), EPS)
        base_comp = base_component[:, step, :]
        resid_comp = residual_component[:, step, :]
        base_contrib = l2_per_row(base_comp) / pred_norm
        resid_contrib = l2_per_row(resid_comp) / pred_norm
        raw_base_error = l2_per_row(b_base[:, step, :] - true_step) / true_norm
        residual_magnitude = l2_per_row(resid_comp) / true_norm
        rows.append(
            {
                "label_id": int(label_id),
                "label": str(arrays["labels"][label_id]),
                "Re": float(np.mean(arrays["re"][label_train])) if len(label_train) else float("nan"),
                "step": int(step + 1),
                "time_mean": float(np.mean(times[:, step])),
                "b_pred_minus_true_norm_mean": float(np.mean(l2_per_row(b_error))),
                "b_pred_minus_true_relative_l2_step": v16.relative_l2_np(true_step, pred_step),
                "b_pred_minus_b_mean_norm_mean": float(np.mean(l2_per_row(pred_minus_mean))),
                "pressure_energy_drift_signed_mean": float(np.mean(energy_drift)),
                "pressure_energy_drift_abs_mean": float(np.mean(np.abs(energy_drift))),
                "adaptive_gate_alpha_mean": float(np.mean(alpha[:, step, :])),
                "adaptive_gate_alpha_std": float(np.std(alpha[:, step, :])),
                "base_contribution_ratio_mean": float(np.mean(base_contrib)),
                "residual_contribution_ratio_mean": float(np.mean(resid_contrib)),
                "raw_base_error_relative_mean": float(np.mean(raw_base_error)),
                "residual_magnitude_over_true_mean": float(np.mean(residual_magnitude)),
            }
        )
    return rows


def top2_load_for_gate(gate: np.ndarray, shared_indices: List[int]) -> Dict[str, object]:
    n, e = gate.shape
    order = np.argsort(-gate, axis=1)
    top1 = order[:, 0]
    top2 = order[:, :2]
    top1_count = np.bincount(top1, minlength=e).astype(np.float64)
    top2_count = np.zeros(e, dtype=np.float64)
    for row in top2:
        top2_count[row] += 1.0
    top2_mass = np.zeros(e, dtype=np.float64)
    for i in range(n):
        top2_mass[top2[i]] += gate[i, top2[i]]
    top2_mass /= max(1, n)
    shared_mask = np.zeros(e, dtype=bool)
    shared_mask[shared_indices] = True
    return {
        "num_samples": int(n),
        "mean_load": [float(v) for v in gate.mean(axis=0).tolist()],
        "top1_fraction": [float(v) for v in (top1_count / max(1, n)).tolist()],
        "top2_presence_fraction": [float(v) for v in (top2_count / max(1, n)).tolist()],
        "top2_mass_mean": [float(v) for v in top2_mass.tolist()],
        "top2_active_experts_gt1pct": int(np.sum((top2_count / max(1, n)) > 0.01)),
        "top2_active_routed_experts_gt1pct": int(
            np.sum(((top2_count / max(1, n)) > 0.01) & (~shared_mask))
        ),
        "top2_shared_presence_fraction": float(np.sum(top2_count[shared_mask]) / max(1, n)),
        "top2_routed_presence_fraction": float(np.sum(top2_count[~shared_mask]) / max(1, n)),
        "top2_shared_mass_mean": float(np.sum(top2_mass[shared_mask])),
        "top2_routed_mass_mean": float(np.sum(top2_mass[~shared_mask])),
    }


def top2_expert_diagnostics(
    model: v16.OperatorSpaceMoEROM,
    arrays: Dict[str, np.ndarray],
    scalers: Dict[str, v16.Standardizer],
    sample_ids: np.ndarray,
    device: torch.device,
) -> Dict[str, object]:
    x = torch.tensor(
        scalers["x"].transform(arrays["x"][sample_ids]),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        _, _, gates, _ = model(x, return_expert_stack=False)
    gate_arrays = [g.detach().cpu().numpy() for g in gates]
    shared_indices = [
        group * (model.experts_per_group + model.shared_per_group)
        for group in range(model.num_regime_groups)
    ]
    out: Dict[str, object] = {
        "num_samples": int(len(sample_ids)),
        "shared_indices": [int(v) for v in shared_indices],
        "combined": top2_load_for_gate(np.mean(np.stack(gate_arrays, axis=0), axis=0), shared_indices),
        "velocity": top2_load_for_gate(gate_arrays[0], shared_indices),
        "pressure": top2_load_for_gate(gate_arrays[1], shared_indices),
        "by_attractor": {},
    }
    for aid, name in enumerate(arrays["attractor_vocab"].tolist()):
        mask = arrays["attractor_id"][sample_ids] == aid
        if not np.any(mask):
            out["by_attractor"][str(name)] = {"num_samples": 0}
            continue
        sub_ids = sample_ids[mask]
        x_sub = torch.tensor(
            scalers["x"].transform(arrays["x"][sub_ids]),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            _, _, sub_gates, _ = model(x_sub, return_expert_stack=False)
        sub_gate_arrays = [g.detach().cpu().numpy() for g in sub_gates]
        out["by_attractor"][str(name)] = {
            "num_samples": int(len(sub_ids)),
            "combined": top2_load_for_gate(
                np.mean(np.stack(sub_gate_arrays, axis=0), axis=0),
                shared_indices,
            ),
            "velocity": top2_load_for_gate(sub_gate_arrays[0], shared_indices),
            "pressure": top2_load_for_gate(sub_gate_arrays[1], shared_indices),
        }
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, result: Dict[str, object]) -> None:
    hopf_rows = sorted(result["hopf_near_onset"], key=lambda r: float(r["Re"]))
    steady_rows = sorted(result["steady_pressure"], key=lambda r: float(r["Re"]))
    top2 = result["top2_expert_load"]["combined"]
    worst_hopf = max(hopf_rows, key=lambda r: float(r["a_error_relative_l2"])) if hopf_rows else None
    worst_steady = max(
        steady_rows,
        key=lambda r: float(r["b_pred_minus_true_relative_l2"]),
    ) if steady_rows else None
    lines = [
        "# V16 FullRegimeLoss Train Dynamics Diagnostics",
        "",
        f"Checkpoint: `{result['checkpoint_path']}`",
        (
            f"Rollout horizon: {result['rollout_steps']} steps, "
            f"window stride: {result['window_stride']}."
        ),
        "",
        "## Hopf Near-Onset",
        "",
        "| Re | windows | r true | r pred | rel a | floor rel a | phase | freq err | pred growth | true growth |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in hopf_rows:
        lines.append(
            f"| {row['Re']:.6g} | {row['num_windows']} | {row['r_true_mean']:.6g} | "
            f"{row['r_pred_mean']:.6g} | {row['a_error_relative_l2']:.6g} | "
            f"{row['a_error_energy_floor_normalized_global']:.6g} | "
            f"{row['phase_error_mean']:.6g} | {row['frequency_error_mean']:.6g} | "
            f"{row['rollout_energy_growth_rate_pred_mean']:.6g} | "
            f"{row['rollout_energy_growth_rate_true_mean']:.6g} |"
        )
    if worst_hopf:
        floor_ratio = (
            worst_hopf["a_error_energy_floor_normalized_global"]
            / max(worst_hopf["a_error_relative_l2"], EPS)
        )
        lines.extend(
            [
                "",
                (
                    f"Worst Hopf Re is {worst_hopf['Re']:.6g}: standard relative a-error "
                    f"{worst_hopf['a_error_relative_l2']:.6g}, global floor-normalized "
                    f"{worst_hopf['a_error_energy_floor_normalized_global']:.6g} "
                    f"(floor/standard ratio {floor_ratio:.3g})."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Steady Pressure Drift",
            "",
            "| Re | windows | b pred-true rel | pred-mean norm | energy drift | alpha mean | alpha std | base contrib | residual contrib | raw base err |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in steady_rows:
        lines.append(
            f"| {row['Re']:.6g} | {row['num_windows']} | "
            f"{row['b_pred_minus_true_relative_l2']:.6g} | "
            f"{row['b_pred_minus_b_mean_norm_mean']:.6g} | "
            f"{row['pressure_energy_drift_abs_mean']:.6g} | "
            f"{row['adaptive_gate_alpha_mean']:.6g} | "
            f"{row['adaptive_gate_alpha_std']:.6g} | "
            f"{row['base_contribution_ratio_mean']:.6g} | "
            f"{row['residual_contribution_ratio_mean']:.6g} | "
            f"{row['raw_base_error_relative_mean']:.6g} |"
        )
    if worst_steady:
        lines.extend(
            [
                "",
                (
                    f"Worst steady Re is {worst_steady['Re']:.6g}: pressure relative drift "
                    f"{worst_steady['b_pred_minus_true_relative_l2']:.6g}, "
                    f"base contribution {worst_steady['base_contribution_ratio_mean']:.6g}, "
                    f"residual contribution {worst_steady['residual_contribution_ratio_mean']:.6g}."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Top-2 Expert Load",
            "",
            f"Shared expert indices: `{result['top2_expert_load']['shared_indices']}`",
            f"Combined top-2 active experts >1%: `{top2['top2_active_experts_gt1pct']}`",
            f"Combined routed top-2 active experts >1%: `{top2['top2_active_routed_experts_gt1pct']}`",
            f"Combined top-2 shared presence per sample: `{top2['top2_shared_presence_fraction']:.6g}`",
            f"Combined top-2 routed presence per sample: `{top2['top2_routed_presence_fraction']:.6g}`",
            "",
            "Top-2 presence fraction by expert:",
            "",
            "`" + json.dumps(top2["top2_presence_fraction"]) + "`",
            "",
            "Additional per-step curves are saved in `hopf_near_onset_time_series.csv` "
            "and `steady_pressure_drift_time_series.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cli = parse_args()
    setup = make_setup(cli.run_dir, cli.checkpoint)
    args = setup["args"]
    args.rollout_steps = int(cli.rollout_steps)
    arrays = setup["arrays"]
    train_ids = setup["train_ids"]
    rollout_pool_ids = setup["rollout_pool_ids"]
    out_dir = cli.run_dir / "train_set_eval" / "dynamics_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "start",
                "device": str(setup["device"]),
                "rollout_steps": args.rollout_steps,
                "window_stride": cli.window_stride,
            }
        ),
        flush=True,
    )

    hopf_rows: List[Dict[str, float]] = []
    steady_rows: List[Dict[str, float]] = []
    hopf_time_rows: List[Dict[str, float]] = []
    steady_time_rows: List[Dict[str, float]] = []
    labels = sorted(set(int(v) for v in arrays["label_id"][train_ids].tolist()))
    for label_id in labels:
        label_train = train_ids[arrays["label_id"][train_ids] == label_id]
        if len(label_train) == 0:
            continue
        attractor = str(arrays["attractor"][label_train[0]])
        if attractor not in {"hopf", "steady"}:
            continue
        starts = start_ids_for_label(
            label_id,
            arrays,
            train_ids,
            rollout_pool_ids,
            args.rollout_steps,
            cli.window_stride,
            cli.start_offset,
        )
        if len(starts) == 0:
            continue
        rollout = collect_rollout(label_id, starts, setup, args.rollout_steps)
        if attractor == "hopf":
            hopf_rows.append(
                hopf_metrics_for_rollout(rollout, arrays, label_id, args, train_ids)
            )
            hopf_time_rows.extend(
                hopf_time_series_rows(rollout, arrays, label_id, args, train_ids)
            )
        elif attractor == "steady":
            steady_rows.append(steady_pressure_metrics_for_rollout(rollout, arrays, label_id, train_ids))
            steady_time_rows.extend(
                steady_pressure_time_series_rows(rollout, arrays, label_id, train_ids)
            )

    print(
        json.dumps(
            {
                "event": "rollout_diagnostics_done",
                "hopf_re": len(hopf_rows),
                "steady_re": len(steady_rows),
            }
        ),
        flush=True,
    )
    top2 = top2_expert_diagnostics(
        setup["model"], arrays, setup["scalers"], train_ids, setup["device"]
    )
    result = {
        "checkpoint_path": str(setup["ckpt_path"]),
        "best_epoch": int(setup["ckpt"].get("best_epoch", -1)),
        "best_val_score": float(setup["ckpt"].get("best_val_score", math.nan)),
        "rollout_steps": int(args.rollout_steps),
        "window_stride": int(cli.window_stride),
        "start_offset": int(cli.start_offset),
        "split": {
            "train_re_count": int(setup["split_stats"]["train_re_count"]),
            "train_samples": int(len(train_ids)),
            "rollout_pool_samples": int(len(rollout_pool_ids)),
            "attractor_sample_counts": setup["split_stats"].get("attractor_sample_counts", {}),
        },
        "hopf_near_onset": hopf_rows,
        "steady_pressure": steady_rows,
        "hopf_near_onset_time_series": hopf_time_rows,
        "steady_pressure_time_series": steady_time_rows,
        "top2_expert_load": top2,
    }
    json_path = out_dir / "full_regime_loss_train_dynamics_diagnostics.json"
    hopf_csv = out_dir / "hopf_near_onset_amplitude_diagnostics.csv"
    hopf_time_csv = out_dir / "hopf_near_onset_time_series.csv"
    steady_csv = out_dir / "steady_pressure_drift_diagnostics.csv"
    steady_time_csv = out_dir / "steady_pressure_drift_time_series.csv"
    md_path = out_dir / "FULL_REGIME_LOSS_TRAIN_DYNAMICS_DIAGNOSTICS.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(hopf_csv, hopf_rows)
    write_csv(hopf_time_csv, hopf_time_rows)
    write_csv(steady_csv, steady_rows)
    write_csv(steady_time_csv, steady_time_rows)
    write_summary_md(md_path, result)
    print(
        json.dumps(
            {
                "event": "done",
                "json": str(json_path),
                "hopf_csv": str(hopf_csv),
                "hopf_time_csv": str(hopf_time_csv),
                "steady_csv": str(steady_csv),
                "steady_time_csv": str(steady_time_csv),
                "md": str(md_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
