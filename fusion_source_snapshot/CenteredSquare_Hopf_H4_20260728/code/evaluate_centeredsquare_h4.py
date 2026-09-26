#!/usr/bin/env python3
"""Evaluate a frozen CenteredSquare Hopf H4 checkpoint on the five held-out Re."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


RANK = 11
HELDOUT = np.asarray([95.1, 95.3, 96.5, 100.5, 102.0], dtype=np.float64)
HORIZONS = (1, 2, 4, 8, 16, 24, 48)
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trainer", type=Path, required=True)
    parser.add_argument("--h4-trainer", type=Path, required=True)
    parser.add_argument("--coefficient-view", type=Path, required=True)
    parser.add_argument("--heldout-view", type=Path, required=True)
    parser.add_argument("--galerkin-path", type=Path, required=True)
    parser.add_argument("--pressure-path", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows-per-re", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
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


def load_heldout(path: Path, history_len: int, v16) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    a = np.asarray(data["coeff_uv"], dtype=np.float32)
    b = np.asarray(data["coeff_p"], dtype=np.float32)
    re = np.asarray(data["Re"], dtype=np.float64)
    time = np.asarray(data["time"], dtype=np.float64)
    observed = np.sort(np.unique(np.round(re, 6)))
    if not np.allclose(observed, HELDOUT, atol=1.1e-6, rtol=0):
        raise RuntimeError(f"heldout view mismatch: {observed.tolist()}")
    order = np.lexsort((time, re))
    a, b, re, time = a[order], b[order], re[order], time[order]
    next_idx = np.full(len(re), -1, dtype=np.int64)
    prev_idx = np.full(len(re), -1, dtype=np.int64)
    for value in np.unique(re):
        ids = np.flatnonzero(np.abs(re - value) <= 5.0e-7)
        next_idx[ids[:-1]] = ids[1:]
        prev_idx[ids[1:]] = ids[:-1]
    hist = v16.history_index_matrix(np.arange(len(re), dtype=np.int64), prev_idx, history_len)
    valid = np.flatnonzero((next_idx >= 0) & np.all(hist >= 0, axis=1))
    return {
        "a": a,
        "b": b,
        "re": re.astype(np.float32),
        "time": time.astype(np.float32),
        "next": next_idx,
        "prev": prev_idx,
        "hist": hist,
        "valid": valid,
        "heldout_ids": valid,
        "phi_u": np.asarray(data["phi_uv"], dtype=np.float32),
        "phi_p": np.asarray(data["phi_p"], dtype=np.float32),
        "areas": np.asarray(data["point_areas"], dtype=np.float32),
        "mean_u": np.asarray(data["mean_uv_train"], dtype=np.float32),
        "mean_p": np.asarray(data["mean_p_train"], dtype=np.float32),
    }


def relative_l2(predicted: torch.Tensor, truth: torch.Tensor) -> float:
    numerator = torch.sum((predicted.float() - truth.float()).square())
    denominator = torch.sum(truth.float().square()).clamp_min(EPS)
    return float(torch.sqrt(numerator / denominator))


def phase_frequency(z: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    angles = torch.atan2(z[..., 1], z[..., 0])
    delta = angles[:, 1:] - angles[:, :-1]
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))
    return torch.mean(delta / dt[:, 1:].clamp_min(1.0e-8), dim=1)


def evaluate_re(B, H4, fc, model, roll, data, rom, stats, device, value, windows):
    ids = data["heldout_ids"][np.abs(data["re"][data["heldout_ids"]] - value) <= 5.0e-6]
    result: dict[str, dict] = {}
    for horizon in HORIZONS:
        starts = B.legal_starts(data, ids, horizon)
        if not len(starts):
            raise RuntimeError(f"no legal heldout windows for Re={value}, K={horizon}")
        starts = starts[np.linspace(0, len(starts) - 1, min(len(starts), windows), dtype=int)]
        out = roll(model, data, starts, horizon, rom, stats, device)
        pred_a = torch.stack(out["pred_a"], dim=1)
        pred_b = torch.stack(out["pred_b"], dim=1)
        true_a = torch.stack(out["true_a"], dim=1)
        true_b = torch.stack(out["true_b"], dim=1)
        n = pred_a.shape[0]
        flat_re = out["re"][:, None].expand(-1, horizon).reshape(-1)
        pa = pred_a.reshape(-1, RANK)
        pb = pred_b.reshape(-1, RANK)
        ta = true_a.reshape(-1, RANK)
        tb = true_b.reshape(-1, RANK)
        _, _, pred_z, pred_r, _, _, _, _ = fc.components(pa, pb, flat_re)
        _, _, true_z, true_r, _, _, _, _ = fc.components(ta, tb, flat_re)
        pred_z = pred_z.reshape(n, horizon, 2)
        true_z = true_z.reshape(n, horizon, 2)
        pred_r = pred_r.reshape(n, horizon)
        true_r = true_r.reshape(n, horizon)
        final_pa, final_pb = pred_a[:, -1], pred_b[:, -1]
        final_ta, final_tb = true_a[:, -1], true_b[:, -1]
        finite = torch.isfinite(final_pa).all(1) & torch.isfinite(final_pb).all(1)
        velocity_norm_ratio = torch.linalg.norm(final_pa, dim=1) / torch.linalg.norm(final_ta, dim=1).clamp_min(EPS)
        pressure_norm_ratio = torch.linalg.norm(final_pb, dim=1) / torch.linalg.norm(final_tb, dim=1).clamp_min(EPS)
        divergent = (~finite) | (velocity_norm_ratio > 10.0) | (pressure_norm_ratio > 10.0)
        weights = torch.sqrt(torch.as_tensor(data["areas"], device=device))
        physical_u = B.physical_relative(
            final_pa, final_ta,
            torch.as_tensor(data["phi_u"], device=device),
            torch.as_tensor(data["mean_u"], device=device),
            torch.cat((weights, weights)),
        )
        physical_p = B.physical_relative(
            final_pb, final_tb,
            torch.as_tensor(data["phi_p"], device=device),
            torch.as_tensor(data["mean_p"], device=device),
            weights,
        )
        rms_pred = torch.sqrt(torch.mean(pred_r.square()))
        rms_true = torch.sqrt(torch.mean(true_r.square())).clamp_min(EPS)
        p2p_pred = (pred_r.max(dim=1).values - pred_r.min(dim=1).values).mean()
        p2p_true = (true_r.max(dim=1).values - true_r.min(dim=1).values).mean()
        radial_rms = torch.sqrt(torch.mean((pred_r - true_r).square()) / torch.mean(true_r.square()).clamp_min(EPS))
        frequency_error = None
        true_frequency = None
        predicted_frequency = None
        if horizon > 1:
            dt = torch.stack(out["dt"], dim=1).squeeze(-1)
            pf = phase_frequency(pred_z, dt)
            tf = phase_frequency(true_z, dt)
            predicted_frequency = float(torch.mean(pf))
            true_frequency = float(torch.mean(tf))
            frequency_error = float(torch.mean(torch.abs(pf - tf)) / torch.mean(torch.abs(tf)).clamp_min(1.0e-8))
            pred_growth = torch.diff(torch.log(pred_r + fc.rf), dim=1)
            true_growth = torch.diff(torch.log(true_r + fc.rf), dim=1)
            mask = torch.abs(true_growth) > fc.gt
            growth_sign = float(
                (torch.sign(pred_growth[mask]) == torch.sign(true_growth[mask])).float().mean()
            ) if mask.any() else 1.0
        else:
            growth_sign = 1.0
        result[str(horizon)] = {
            "windows": int(n),
            "velocity_modal_relative_l2": relative_l2(final_pa, final_ta),
            "pressure_modal_relative_l2": relative_l2(final_pb, final_tb),
            "velocity_physical_relative_l2_mean": float(torch.nanmean(physical_u)),
            "pressure_physical_relative_l2_mean": float(torch.nanmean(physical_p)),
            "radial_rms_relative_error": float(radial_rms),
            "rms_amplitude_relative_error": float(torch.abs(rms_pred - rms_true) / rms_true),
            "p2p_amplitude_relative_error": float(torch.abs(p2p_pred - p2p_true) / torch.abs(p2p_true).clamp_min(EPS)),
            "growth_sign_accuracy": growth_sign,
            "true_angular_frequency": true_frequency,
            "predicted_angular_frequency": predicted_frequency,
            "frequency_relative_error": frequency_error,
            "finite_fraction": float(finite.float().mean()),
            "divergent_windows": int(divergent.sum()),
            "max_norm_ratio": float(torch.maximum(velocity_norm_ratio, pressure_norm_ratio).max()),
        }
    return result


def main() -> None:
    args = parse_args()
    B = load_module(args.baseline_trainer, "centeredsquare_hopf_base_eval")
    H4 = load_module(args.h4_trainer, "centeredsquare_h4_eval")
    device = torch.device(args.device)
    train_data = B.load_coefficients(SimpleNamespace(
        coefficient_view=args.coefficient_view, history_len=3,
    ))
    heldout_data = load_heldout(args.heldout_view, 3, B.v16)
    rom_np = B.load_train_rom(SimpleNamespace(
        galerkin_path=args.galerkin_path, pressure_path=args.pressure_path,
    ))
    rom = {key: torch.as_tensor(value, device=device) for key, value in rom_np.items()}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = {
        key: torch.as_tensor(value, device=device)
        for key, value in checkpoint["norm_stats"].items()
    }
    model_args = SimpleNamespace(
        hidden_dim=256, expert_hidden=1024, num_blocks=3, experts=6, top_k=2,
        expert_blocks=4, quadratic_rank=4, dropout=0.04, temperature=0.8,
        adaptive_gate_initial_logit=6.0,
    )
    probe = B.batch_from_ids(train_data, train_data["train_ids"][:2], device)
    history_rhs = B.galerkin(
        probe["ah"].reshape(-1, RANK),
        probe["bh"].reshape(-1, RANK),
        probe["re"][:, None].expand(-1, 3).reshape(-1),
        rom,
    ).reshape_as(probe["ah"])
    input_dim = B.state_features(
        probe["a"], probe["b"], probe["re"], probe["ah"], probe["bh"],
        history_rhs, rom, stats,
    )[0].shape[1]
    model = B.build_model(input_dim, model_args, stats, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    fc = H4.FluctuationContract(args.contract, device)
    _, rollout = H4.make_ops(B, fc, B.state_features)
    per_re = {}
    with torch.no_grad():
        for value in HELDOUT:
            per_re[f"{value:.6f}"] = evaluate_re(
                B, H4, fc, model, rollout, heldout_data, rom, stats,
                device, float(value), args.windows_per_re,
            )
    k48 = [metrics["48"] for metrics in per_re.values()]
    summary = {
        "schema_version": 1,
        "dataset": "CenteredSquare Hopf34",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "optimizer_step": int(checkpoint["optimizer_step"]),
        "best_validation_score": float(checkpoint["best_score"]),
        "best_validation_step": int(checkpoint["best_step"]),
        "heldout_Re": HELDOUT.tolist(),
        "horizons": list(HORIZONS),
        "k48_aggregate": {
            "velocity_physical_relative_l2_mean": float(np.mean([
                row["velocity_physical_relative_l2_mean"] for row in k48
            ])),
            "pressure_physical_relative_l2_mean": float(np.mean([
                row["pressure_physical_relative_l2_mean"] for row in k48
            ])),
            "radial_rms_relative_error_mean": float(np.mean([
                row["radial_rms_relative_error"] for row in k48
            ])),
            "finite_fraction_min": float(np.min([
                row["finite_fraction"] for row in k48
            ])),
            "divergent_windows_total": int(np.sum([
                row["divergent_windows"] for row in k48
            ])),
        },
        "by_re": per_re,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(summary, args.output_dir / "heldout_metrics.json")
    print(json.dumps(summary["k48_aggregate"], indent=2))


if __name__ == "__main__":
    main()
