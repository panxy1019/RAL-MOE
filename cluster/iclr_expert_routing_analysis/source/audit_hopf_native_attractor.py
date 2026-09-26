#!/usr/bin/env python3
"""Read-only native-domain attractor audit for the frozen expanded H4 Hopf model."""

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


PRIMARY_HORIZON = 56
DIAGNOSTIC_HORIZON = 128
PRIMARY_HORIZONS = (1, 4, 8, 16, 48, 56)
EPS = 1.0e-12


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_all_split_data(
    trainer: Any, source_root: Path, history_len: int
) -> dict[str, np.ndarray]:
    velocity_path = source_root / "velocity_pod_hopf.npz"
    pressure_path = source_root / "pressure_pod_hopf.npz"
    velocity = np.load(velocity_path, allow_pickle=False)
    pressure = np.load(pressure_path, allow_pickle=False)
    index_path = source_root / "projection_snapshots_velocity_hopf.csv"
    with index_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    re_values = np.asarray([float(row["Re"]) for row in rows], dtype=np.float64)
    times = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
    split = np.asarray(
        [
            "heldout"
            if row["split"].strip().lower() in {"heldout", "test"}
            else row["split"].strip().lower()
            for row in rows
        ]
    )
    if set(split.tolist()) != {"train", "validation", "heldout"}:
        raise RuntimeError(f"unexpected Hopf split labels: {sorted(set(split.tolist()))}")
    a = np.asarray(velocity["coeff_uv"], dtype=np.float32)[:, :32]
    b = np.asarray(pressure["coeff_p"], dtype=np.float32)[:, :32]
    if not (len(rows) == len(a) == len(b)):
        raise RuntimeError("Hopf full coefficient/index length mismatch")
    order = np.lexsort((times, re_values))
    a, b = a[order], b[order]
    re_values, times, split = re_values[order], times[order], split[order]
    next_idx = np.full(len(re_values), -1, dtype=np.int64)
    prev_idx = np.full(len(re_values), -1, dtype=np.int64)
    for value in np.unique(re_values):
        ids = np.flatnonzero(np.abs(re_values - value) <= 5.0e-7)
        next_idx[ids[:-1]] = ids[1:]
        prev_idx[ids[1:]] = ids[:-1]
    hist = trainer.v16.history_index_matrix(
        np.arange(len(re_values), dtype=np.int64), prev_idx, history_len
    )
    valid = np.flatnonzero((next_idx >= 0) & np.all(hist >= 0, axis=1))
    return {
        "a": a,
        "b": b,
        "re": re_values.astype(np.float32),
        "time": times.astype(np.float32),
        "split": split,
        "next": next_idx,
        "prev": prev_idx,
        "hist": hist,
        "valid": valid,
        "phi_u": np.asarray(velocity["phi_uv"], dtype=np.float32)[:32],
        "phi_p": np.asarray(pressure["phi_p"], dtype=np.float32)[:32],
        "areas": np.asarray(velocity["point_areas"], dtype=np.float32),
        "mean_u": np.asarray(velocity["mean_uv_regime"], dtype=np.float32),
        "mean_p": np.asarray(pressure["mean_p_regime"], dtype=np.float32),
        "velocity_asset": np.asarray(str(velocity_path)),
        "pressure_asset": np.asarray(str(pressure_path)),
        "index_asset": np.asarray(str(index_path)),
    }


def evenly(starts: np.ndarray, count: int) -> np.ndarray:
    if len(starts) <= count:
        return starts
    return starts[np.linspace(0, len(starts) - 1, count, dtype=np.int64)]


def rollout_arrays(
    trainer: Any,
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    starts: np.ndarray,
    horizon: int,
    rom: dict[str, torch.Tensor],
    stats: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, np.ndarray]:
    output = trainer.rollout_batch(model, data, starts, horizon, rom, stats, device)
    result = {
        "pred_a": torch.stack(output["pred_a"], 1).float().cpu().numpy(),
        "pred_b": torch.stack(output["pred_b"], 1).float().cpu().numpy(),
        "true_a": torch.stack(output["true_a"], 1).float().cpu().numpy(),
        "true_b": torch.stack(output["true_b"], 1).float().cpu().numpy(),
        "pred_rhs": torch.stack(output["pred_rhs"], 1).float().cpu().numpy(),
        "true_rhs": torch.stack(output["true_rhs"], 1).float().cpu().numpy(),
    }
    current = starts.copy()
    times = []
    for _ in range(horizon):
        current = data["next"][current]
        times.append(data["time"][current])
    result["times"] = np.stack(times, axis=1)
    return result


def horizon_metrics(
    evaluator: Any,
    arrays: dict[str, np.ndarray],
    horizon: int,
    velocity_geometry: Any,
    pressure_geometry: Any,
) -> dict[str, Any]:
    pa, pb = arrays["pred_a"][:, :horizon], arrays["pred_b"][:, :horizon]
    ta, tb = arrays["true_a"][:, :horizon], arrays["true_b"][:, :horizon]
    finite = np.isfinite(pa).all(2) & np.isfinite(pb).all(2)
    true_u_max = np.maximum(
        np.max(np.linalg.norm(ta, axis=2), axis=1, keepdims=True), EPS
    )
    true_p_max = np.maximum(
        np.max(np.linalg.norm(tb, axis=2), axis=1, keepdims=True), EPS
    )
    bad = (~finite) | (np.linalg.norm(pa, axis=2) > 10.0 * true_u_max) | (
        np.linalg.norm(pb, axis=2) > 10.0 * true_p_max
    )
    flat_pa, flat_pb = pa.reshape(-1, 32), pb.reshape(-1, 32)
    flat_ta, flat_tb = ta.reshape(-1, 32), tb.reshape(-1, 32)
    u_true = evaluator.field_energy(flat_ta, velocity_geometry)
    u_pred = evaluator.field_energy(flat_pa, velocity_geometry)
    p_true = evaluator.field_energy(flat_tb, pressure_geometry)
    p_pred = evaluator.field_energy(flat_pb, pressure_geometry)
    first = np.flatnonzero(bad.any(axis=0))
    return {
        "num_windows": int(len(pa)),
        "velocity_area_weighted_physical_relative_l2": math.sqrt(
            float(np.nansum(evaluator.field_error_energy(flat_ta, flat_pa, velocity_geometry)))
            / (float(np.nansum(u_true)) + EPS)
        ),
        "pressure_area_weighted_physical_relative_l2": math.sqrt(
            float(np.nansum(evaluator.field_error_energy(flat_tb, flat_pb, pressure_geometry)))
            / (float(np.nansum(p_true)) + EPS)
        ),
        "velocity_modal_relative_l2": evaluator.relative_l2(flat_ta, flat_pa),
        "pressure_modal_relative_l2": evaluator.relative_l2(flat_tb, flat_pb),
        "finite_fraction": float(np.mean(finite)),
        "divergent_windows": int(bad.any(axis=1).sum()),
        "first_divergence_step": int(first[0] + 1) if len(first) else None,
        "velocity_energy_drift": abs(float(np.nanmean(u_pred)) - float(np.mean(u_true)))
        / (abs(float(np.mean(u_true))) + EPS),
        "pressure_energy_drift": abs(float(np.nanmean(p_pred)) - float(np.mean(p_true)))
        / (abs(float(np.mean(p_true))) + EPS),
    }


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows-per-re", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    trainer = load_module("hopf_native_audit_trainer", args.trainer)
    evaluator = load_module("hopf_native_audit_evaluator", args.evaluator)
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = SimpleNamespace(**checkpoint["args"])
    defaults = {
        "history_len": 3,
        "hidden_dim": 256,
        "expert_hidden": 1024,
        "num_blocks": 3,
        "experts": 6,
        "top_k": 2,
        "expert_blocks": 4,
        "quadratic_rank": 4,
        "dropout": 0.04,
        "temperature": 0.8,
        "adaptive_gate_initial_logit": 6.0,
        "lambda_scale_amplitude": 1.0,
        "lambda_scale_growth": 0.5,
        "lambda_scale_sign": 0.1,
        "scale_floor_quantile": 0.1,
    }
    for key, value in defaults.items():
        if not hasattr(model_args, key):
            setattr(model_args, key, value)
    model_args.device = "cuda"
    train_view = trainer.load_coefficients(model_args)
    rom_np = trainer.load_train_rom(model_args)
    norm_stats, _scale_stats = trainer.fit_stats(train_view, rom_np, model_args)
    stats = {
        key: torch.as_tensor(value, device=device)
        for key, value in asdict(norm_stats).items()
    }
    rom = {key: torch.as_tensor(value, device=device) for key, value in rom_np.items()}
    probe = trainer.batch_from_ids(train_view, train_view["train_ids"][:2], device)
    probe_rhs = trainer.galerkin(
        probe["ah"].reshape(-1, 32),
        probe["bh"].reshape(-1, 32),
        probe["re"][:, None].expand(-1, model_args.history_len).reshape(-1),
        rom,
    ).reshape_as(probe["ah"])
    in_dim = trainer.state_features(
        probe["a"],
        probe["b"],
        probe["re"],
        probe["ah"],
        probe["bh"],
        probe_rhs,
        rom,
        stats,
    )[0].shape[1]
    model = trainer.build_model(in_dim, model_args, stats, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    data = build_all_split_data(trainer, args.source_root, model_args.history_len)
    velocity_geometry = evaluator.weighted_geometry(
        data["phi_u"], data["mean_u"], data["areas"], True
    )
    pressure_geometry = evaluator.weighted_geometry(
        data["phi_p"], data["mean_p"], data["areas"], False
    )
    contract = np.load(args.contract, allow_pickle=False)
    nodes = contract["nodes"].astype(np.float64)
    means = contract["mean_a"].astype(np.float64)
    plane = contract["plane"].astype(np.float64)
    radial_scale = contract["radial_scale"].astype(np.float64)
    growth_tolerance = float(contract["growth_tolerance"])
    radial_floor = float(contract["radial_floor"])

    per_re: list[dict[str, Any]] = []
    for split in ("train", "validation", "heldout"):
        split_re = sorted(
            np.unique(np.round(data["re"][data["split"] == split], 6)).tolist()
        )
        for re_value in split_re:
            ids = data["valid"][
                (data["split"][data["valid"]] == split)
                & (np.abs(data["re"][data["valid"]] - re_value) <= 5.0e-6)
            ]
            legal_56 = trainer.legal_starts(data, ids, PRIMARY_HORIZON)
            if not len(legal_56):
                raise RuntimeError(f"No native K56 window for {split} Re={re_value}")
            starts_56 = evenly(legal_56, args.windows_per_re)
            arrays_56 = rollout_arrays(
                trainer,
                model,
                data,
                starts_56,
                PRIMARY_HORIZON,
                rom,
                stats,
                device,
            )
            metrics = {
                str(horizon): horizon_metrics(
                    evaluator,
                    arrays_56,
                    horizon,
                    velocity_geometry,
                    pressure_geometry,
                )
                for horizon in PRIMARY_HORIZONS
            }
            center = evaluator.interp_np(nodes, means, float(re_value))
            scale = float(evaluator.interp_np(nodes, radial_scale, float(re_value)))
            attractor = evaluator.phase_metrics(
                arrays_56["true_a"],
                arrays_56["pred_a"],
                arrays_56["times"],
                center,
                plane / max(scale, EPS),
                np.ones(2, dtype=np.float64),
                radial_floor,
                growth_tolerance,
            )
            attractor["one_step_rhs_relative_l2"] = evaluator.relative_l2(
                arrays_56["true_rhs"][:, :1].reshape(-1, 32),
                arrays_56["pred_rhs"][:, :1].reshape(-1, 32),
            )
            attractor["one_step_pressure_closure_relative_l2"] = evaluator.relative_l2(
                arrays_56["true_b"][:, :1].reshape(-1, 32),
                arrays_56["pred_b"][:, :1].reshape(-1, 32),
            )
            k56 = metrics["56"]
            preserved = (
                k56["finite_fraction"] == 1.0
                and k56["divergent_windows"] == 0
                and k56["velocity_area_weighted_physical_relative_l2"] <= 0.05
                and k56["pressure_area_weighted_physical_relative_l2"] <= 0.05
                and attractor["rms_amplitude_error"] <= 0.10
                and attractor["peak_to_peak_amplitude_error"] <= 0.10
                and attractor["frequency_relative_error"] <= 0.05
                and abs(attractor["terminal_phase_drift_cycles"]) <= 0.25
                and attractor["normalized_orbit_distance"] <= 0.10
                and k56["velocity_energy_drift"] <= 0.10
                and k56["pressure_energy_drift"] <= 0.10
                and not attractor["false_growth"]
            )

            legal_128 = trainer.legal_starts(data, ids, DIAGNOSTIC_HORIZON)
            diagnostic_128 = {"available": bool(len(legal_128))}
            if len(legal_128):
                starts_128 = evenly(legal_128, 1)
                arrays_128 = rollout_arrays(
                    trainer,
                    model,
                    data,
                    starts_128,
                    DIAGNOSTIC_HORIZON,
                    rom,
                    stats,
                    device,
                )
                diagnostic_128.update(
                    horizon_metrics(
                        evaluator,
                        arrays_128,
                        DIAGNOSTIC_HORIZON,
                        velocity_geometry,
                        pressure_geometry,
                    )
                )
            row = {
                "split": split,
                "Re": float(re_value),
                "K56_windows": int(len(starts_56)),
                "horizons": metrics,
                "attractor_K56": attractor,
                "K128_diagnostic": diagnostic_128,
                "attractor_preserved_K56": bool(preserved),
            }
            per_re.append(row)
            print(
                json.dumps(
                    {
                        "split": split,
                        "Re": re_value,
                        "preserved": bool(preserved),
                        "K56_finite": k56["finite_fraction"],
                        "K56_divergent": k56["divergent_windows"],
                    }
                ),
                flush=True,
            )

    by_split = {}
    for split in ("train", "validation", "heldout"):
        rows = [row for row in per_re if row["split"] == split]
        by_split[split] = {
            "Re_count": len(rows),
            "attractor_preserved_count": sum(
                row["attractor_preserved_K56"] for row in rows
            ),
            "K56_all_finite_count": sum(
                row["horizons"]["56"]["finite_fraction"] == 1.0 for row in rows
            ),
            "K56_zero_divergence_count": sum(
                row["horizons"]["56"]["divergent_windows"] == 0 for row in rows
            ),
            "K128_available_count": sum(row["K128_diagnostic"]["available"] for row in rows),
            "K128_all_finite_count": sum(
                row["K128_diagnostic"].get("finite_fraction") == 1.0 for row in rows
            ),
            "K128_zero_divergence_count": sum(
                row["K128_diagnostic"].get("divergent_windows") == 0 for row in rows
            ),
        }
    output = {
        "schema": "hopf_native_attractor_audit/v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_optimizer_step": int(checkpoint["optimizer_step"]),
        "selection_split": "validation_only",
        "evaluation": {
            "native_chart_only": True,
            "state_autonomous": True,
            "known_inputs": "fixed Re and observed native dt only",
            "future_truth_in_rollout": False,
            "K56_windows_per_Re": args.windows_per_re,
            "K128_windows_per_available_Re": 1,
        },
        "thresholds": {
            "K56_velocity_pressure_physical_relative_l2_max": 0.05,
            "rms_peak_to_peak_amplitude_error_max": 0.10,
            "frequency_relative_error_max": 0.05,
            "terminal_phase_drift_abs_cycles_max": 0.25,
            "normalized_orbit_distance_max": 0.10,
            "velocity_pressure_energy_drift_max": 0.10,
            "finite_fraction": 1.0,
            "divergent_windows": 0,
            "false_growth": False,
        },
        "by_split": by_split,
        "per_Re": per_re,
        "interpretation_contract": {
            "train": "in-sample preservation diagnostic only",
            "validation": "checkpoint-selected development support, not independent test",
            "heldout": "independent post-freeze generalization evidence",
        },
    }
    output_path = args.output_dir / "HOPF_NATIVE_ATTRACTOR_AUDIT.json"
    atomic_json(output_path, output)
    print(json.dumps({"status": "COMPLETE", "by_split": by_split}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
