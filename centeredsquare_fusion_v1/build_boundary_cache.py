#!/usr/bin/env python3
"""Build a sealed development cache from sequential frozen native rollouts."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from common import (
    affine_project,
    atomic_json,
    chain,
    evenly_spaced,
    history_matrix,
    load_module,
    physical_descriptors,
    pressure_gauge,
    quadratic_statistics,
    reconstruct,
    sha256,
)


def invert_next(next_idx: np.ndarray) -> np.ndarray:
    previous = np.full(len(next_idx), -1, dtype=np.int64)
    valid = np.flatnonzero(np.asarray(next_idx) >= 0)
    previous[np.asarray(next_idx)[valid].astype(np.int64)] = valid
    return previous


def hopf_split_re(root: Path, mode: str) -> dict[str, set[float]]:
    assets = root / "CenteredSquare_Hopf_H4_20260728/assets_r11"
    result: dict[str, set[float]] = {}
    if mode == "development":
        with np.load(assets / "centeredsquare_hopf_trainval_r11.npz", allow_pickle=False) as source:
            for split in ("train", "validation"):
                result[split] = set(np.round(np.unique(source["Re"][source["split"] == split]), 8))
    else:
        with np.load(assets / "centeredsquare_hopf_heldout_r11.npz", allow_pickle=False) as source:
            result["heldout"] = set(np.round(np.unique(source["Re"]), 8))
    return result


def load_hopf_assets(root: Path) -> dict[str, np.ndarray]:
    path = root / "CenteredSquare_Hopf_H4_20260728/assets_r11/centeredsquare_hopf_trainval_r11.npz"
    with np.load(path, allow_pickle=False) as source:
        return {
            "phi_u": np.asarray(source["phi_uv"], dtype=np.float64),
            "phi_p": np.asarray(source["phi_p"], dtype=np.float64),
            "mean_u": np.asarray(source["mean_uv_train"], dtype=np.float64),
            "mean_p": np.asarray(source["mean_p_train"], dtype=np.float64),
            "areas": np.asarray(source["point_areas"], dtype=np.float64),
        }


def build_hopf_runtime(root: Path, data: dict[str, np.ndarray], device: torch.device):
    hopf_root = root / "CenteredSquare_Hopf_H4_20260728"
    baseline = load_module("fusion_hopf_baseline", hopf_root / "code/train_centeredsquare_hopf_base.py")
    h4 = load_module("fusion_hopf_h4", hopf_root / "code/train_centeredsquare_h4.py")
    checkpoint_path = hopf_root / "final_evaluation_mb46_dataonly/selected_validation_step7200.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    rom_np = baseline.load_train_rom(
        SimpleNamespace(
            galerkin_path=hopf_root / "assets_r11/centeredsquare_hopf_trainonly_galerkin_r11.npz",
            pressure_path=hopf_root / "assets_r11/centeredsquare_hopf_trainonly_pressure_r11.npz",
        )
    )
    rom = {key: torch.as_tensor(value, device=device) for key, value in rom_np.items()}
    stats = {key: torch.as_tensor(value, device=device) for key, value in checkpoint["norm_stats"].items()}
    model_args = SimpleNamespace(
        hidden_dim=256,
        expert_hidden=1024,
        num_blocks=3,
        experts=6,
        top_k=2,
        expert_blocks=4,
        quadratic_rank=4,
        dropout=0.04,
        temperature=0.8,
        adaptive_gate_initial_logit=6.0,
    )
    probe_ids = np.asarray([int(data["starts"][0]), int(data["starts"][0])], dtype=np.int64)
    probe = baseline.batch_from_ids(data, probe_ids, device)
    history_rhs = baseline.galerkin(
        probe["ah"].reshape(-1, 11),
        probe["bh"].reshape(-1, 11),
        probe["re"][:, None].expand(-1, 3).reshape(-1),
        rom,
    ).reshape_as(probe["ah"])
    input_dim = baseline.state_features(
        probe["a"], probe["b"], probe["re"], probe["ah"], probe["bh"], history_rhs, rom, stats
    )[0].shape[1]
    model = baseline.build_model(input_dim, model_args, stats, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    contract = h4.FluctuationContract(
        hopf_root / "trainonly_contract_mb46/trainonly_fluctuation_contract.npz",
        device,
    )
    _, rollout = h4.make_ops(baseline, contract, baseline.state_features)
    return baseline, model, rollout, rom, stats, checkpoint_path


def target_hopf_rollout(
    root: Path,
    data: dict[str, np.ndarray],
    horizon: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, Path]:
    baseline, model, rollout, rom, stats, checkpoint_path = build_hopf_runtime(root, data, device)
    predicted_a, predicted_b = [], []
    with torch.inference_mode():
        for offset in range(0, len(data["starts"]), 16):
            starts = np.asarray(data["starts"][offset : offset + 16], dtype=np.int64)
            output = rollout(model, data, starts, horizon, rom, stats, device)
            predicted_a.append(torch.stack(output["pred_a"], dim=1).float().cpu().numpy())
            predicted_b.append(torch.stack(output["pred_b"], dim=1).float().cpu().numpy())
    del model, rom, stats, baseline
    gc.collect()
    torch.cuda.empty_cache()
    return np.concatenate(predicted_a), np.concatenate(predicted_b), checkpoint_path


def steady_source(
    root: Path,
    horizon: int,
    windows_per_re: int,
    device: torch.device,
    mode: str,
) -> dict[str, Any]:
    specialist = root / "centeredsquare_steady_specialist_v1"
    config = json.loads((specialist / "code/training_centeredsquare_steady_rank999.json").read_text())
    trainer = load_module("fusion_steady_source", specialist / "code/train_s2b_3090.py")
    exp = trainer.Experiment(SimpleNamespace(run_dir=str(specialist / "fusion_runtime"), resume=None), config)
    checkpoint_path = specialist / "checkpoint/frozen_centeredsquare_steady_step3200.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    exp.model.load_state_dict(checkpoint["model"], strict=True)
    exp.model.eval()
    del checkpoint
    hopf_re = hopf_split_re(root, mode)
    windows = exp.build_windows(horizon)
    starts, splits, reynolds = [], [], []
    selected_splits = ("train", "validation") if mode == "development" else ("heldout",)
    for split in selected_splits:
        allowed = hopf_re[split]
        for label_id, candidates in windows[split].items():
            value = float(exp.a["labels"][label_id].removeprefix("Re").replace("p", "."))
            if round(value, 8) not in allowed:
                continue
            chosen = evenly_spaced(candidates, windows_per_re)
            starts.extend(chosen.tolist())
            splits.extend([split] * len(chosen))
            reynolds.extend([value] * len(chosen))
    order = np.lexsort((np.asarray(starts), np.asarray(reynolds), np.asarray(splits)))
    starts = np.asarray(starts, dtype=np.int64)[order]
    splits = np.asarray(splits)[order]
    reynolds = np.asarray(reynolds, dtype=np.float64)[order]
    pred_a, pred_b, truth_a, truth_b, target_ids = [], [], [], [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=exp.amp_dtype):
        for offset in range(0, len(starts), 32):
            selected = torch.as_tensor(starts[offset : offset + 32], dtype=torch.long, device=device)
            output = exp.rollout(selected, horizon)
            pred_a.append(output["pa"].permute(1, 0, 2).float().cpu().numpy())
            pred_b.append(output["pb"].permute(1, 0, 2).float().cpu().numpy())
            truth_a.append(output["ta"].permute(1, 0, 2).float().cpu().numpy())
            truth_b.append(output["tb"].permute(1, 0, 2).float().cpu().numpy())
            target_ids.extend([exp.indices(selected, horizon)[i, 1:].cpu().numpy() for i in range(len(selected))])
    with np.load(exp.paths["velocity_pod"], allow_pickle=False) as pod:
        phi_u = np.asarray(pod["phi_uv"][: exp.ru], dtype=np.float64)
        mean_u = np.asarray(pod["mean_uv_regime"], dtype=np.float64)
        areas = np.asarray(pod["point_areas"], dtype=np.float64)
    with np.load(exp.paths["pressure_pod"], allow_pickle=False) as pod:
        phi_p = np.asarray(pod["phi_p"][: exp.rp], dtype=np.float64)
        mean_p = np.asarray(pod["mean_p_regime"], dtype=np.float64)
    next_idx = np.asarray(exp.a["next_idx"], dtype=np.int64)
    previous = invert_next(next_idx)
    time_values = np.asarray(exp.a["time"], dtype=np.float64)
    hist = np.asarray(exp.a["hist_idx"], dtype=np.int64)
    descriptors = []
    for start in starts:
        history = hist[start]
        velocity = reconstruct(exp.a["a"][history, : exp.ru], phi_u, mean_u).reshape(3, -1, 2)
        pressure = reconstruct(exp.a["b"][history, : exp.rp], phi_p, mean_p)
        descriptors.append(physical_descriptors(velocity, pressure, time_values[history], areas))
    result = {
        "starts": starts,
        "split": splits,
        "re": reynolds,
        "pred_a": np.concatenate(pred_a),
        "pred_b": np.concatenate(pred_b),
        "truth_a": np.concatenate(truth_a),
        "truth_b": np.concatenate(truth_b),
        "target_ids": np.asarray(target_ids, dtype=np.int64),
        "all_a": np.asarray(exp.a["a"][:, : exp.ru], dtype=np.float32),
        "all_b": np.asarray(exp.a["b"][:, : exp.rp], dtype=np.float32),
        "all_re": np.asarray(exp.a["re"], dtype=np.float32),
        "all_time": time_values.astype(np.float32),
        "next": next_idx,
        "prev": previous,
        "hist": hist,
        "phi_u": phi_u,
        "phi_p": phi_p,
        "mean_u": mean_u,
        "mean_p": mean_p,
        "areas": areas,
        "descriptors": np.stack(descriptors),
        "checkpoint": checkpoint_path,
        "source_name": "Steady",
        "pair_indices": np.asarray([0, 1], dtype=np.int64),
    }
    del exp
    gc.collect()
    torch.cuda.empty_cache()
    return result


def periodic_source(
    root: Path,
    horizon: int,
    windows_per_re: int,
    device: torch.device,
    mode: str,
) -> dict[str, Any]:
    specialist = root / "centered_square_periodic_v1"
    trainer = load_module("fusion_periodic_source", specialist / "code/train_square_periodic_moe_optimized.py")
    checkpoint_path = specialist / "runs/optimized_long_seed1600/best_validation.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**checkpoint["args"])
    args.data_root = specialist / "assets"
    args.tensor_path = specialist / "assets/velocity_rom_periodic.npz"
    args.pressure_surrogate_path = specialist / "assets/pressure_poisson_surrogate_periodic.npz"
    args.max_integrator_dt = 0.5
    arrays, _ = trainer.build_arrays(args)
    model = trainer.OperatorSpaceMoEROM(
        in_dim=arrays["x"].shape[1],
        out_dim=args.r_u,
        pressure_dim=args.r_p,
        hidden_dim=args.hidden_dim,
        expert_hidden=args.expert_hidden,
        num_blocks=args.num_blocks,
        num_experts=args.num_experts,
        num_operator_spaces=args.num_shared_experts,
        num_regime_groups=args.num_regime_groups,
        experts_per_group=args.experts_per_group,
        top_k=args.top_k,
        group_top_k=args.group_top_k,
        dropout=args.dropout,
        temperature=args.temperature,
        gate_floor=args.gate_floor,
        group_temperature=args.group_temperature,
        group_gate_floor=args.group_gate_floor,
        shared_scale=args.shared_scale,
        routed_scale=args.routed_scale,
        expert_blocks=args.expert_blocks,
        quadratic_rank=args.quadratic_rank,
        quadratic_scale=args.quadratic_scale,
        phase_harmonics=args.phase_harmonics,
        closure_mode=args.closure_mode,
        pressure_base_mode=args.pressure_base_mode,
        film_base_hidden=args.film_base_hidden,
        film_base_scale=args.film_base_scale,
        attractor_conditioned=args.attractor_conditioned,
        attractor_adapter_dim=args.attractor_adapter_dim,
        batched_experts=bool(getattr(args, "batched_experts", False)),
        fixed_regime_group=int(getattr(args, "fixed_regime_group", -1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    scalers = {
        key: trainer.Standardizer(mean=np.asarray(value["mean"], dtype=np.float32), scale=np.asarray(value["scale"], dtype=np.float32))
        for key, value in checkpoint["scalers"].items()
    }
    tensors = np.load(args.tensor_path)
    pressure_tensors = np.load(args.pressure_surrogate_path)
    with np.load(specialist / "assets/global_velocity_pod_area_weighted_l2.npz", allow_pickle=False) as pod:
        phi_u = np.asarray(pod["phi_uv"][: args.r_u], dtype=np.float64)
        mean_u = np.asarray(pod["mean_uv_regime"], dtype=np.float64)
        areas = np.asarray(pod["point_areas"], dtype=np.float64)
        snapshot_split = np.asarray(pod["snapshot_splits"]).astype(str)
    with np.load(specialist / "assets/global_pressure_pod_area_weighted_l2.npz", allow_pickle=False) as pod:
        phi_p = np.asarray(pod["phi_p"][: args.r_p], dtype=np.float64)
        mean_p = np.asarray(pod["mean_p_regime"], dtype=np.float64)
    if len(snapshot_split) != len(arrays["a"]):
        raise RuntimeError("Periodic split/POD alignment mismatch")
    hopf_re = hopf_split_re(root, mode)
    valid_samples = set(np.asarray(arrays["sample_ids"], dtype=np.int64).tolist())
    starts, splits, reynolds, chains = [], [], [], []
    selected_splits = ("train", "validation") if mode == "development" else ("heldout",)
    for split in selected_splits:
        for value in sorted(hopf_re[split]):
            ids = np.flatnonzero(
                (snapshot_split == split)
                & np.isclose(np.asarray(arrays["re"], dtype=np.float64), value, atol=5.0e-7, rtol=0.0)
            )
            candidates = []
            for start in ids:
                ids_chain = chain(arrays["next_idx"], int(start), horizon)
                if ids_chain is None or int(start) not in valid_samples:
                    continue
                if np.any(np.asarray(arrays["hist_idx"])[start] < 0):
                    continue
                candidates.append(int(start))
            for start in evenly_spaced(np.asarray(candidates), windows_per_re):
                starts.append(int(start))
                splits.append(split)
                reynolds.append(float(value))
                chains.append(chain(arrays["next_idx"], int(start), horizon))
    pred_a, pred_b, truth_a, truth_b, descriptors = [], [], [], [], []
    for start, ids_chain in zip(starts, chains):
        assert ids_chain is not None
        a_state = arrays["a"][start].copy()
        b_state = arrays["b"][start].copy()
        a_hist, b_hist, rhs_hist = trainer.init_history_states_np(start, arrays)
        pa, pb = [], []
        for step in range(horizon):
            current, nxt = int(ids_chain[step]), int(ids_chain[step + 1])
            dt = float(arrays["time"][nxt] - arrays["time"][current])
            a_state, b_state, rhs = trainer.integrate_autonomous_step_np(
                model,
                a_state,
                b_state,
                current,
                dt,
                a_hist,
                b_hist,
                rhs_hist,
                arrays,
                scalers,
                tensors,
                pressure_tensors,
                args,
                device,
            )
            pa.append(a_state.copy())
            pb.append(b_state.copy())
            a_hist = np.concatenate([a_state[None, None, :], a_hist[:, :-1, :]], axis=1)
            b_hist = np.concatenate([b_state[None, None, :], b_hist[:, :-1, :]], axis=1)
            rhs_hist = np.concatenate([rhs[None, None, :], rhs_hist[:, :-1, :]], axis=1)
        pred_a.append(pa)
        pred_b.append(pb)
        truth_a.append(arrays["a"][ids_chain[1:]])
        truth_b.append(arrays["b"][ids_chain[1:]])
        history = np.asarray(arrays["hist_idx"])[start]
        velocity = reconstruct(arrays["a"][history], phi_u, mean_u).reshape(3, -1, 2)
        pressure = reconstruct(arrays["b"][history], phi_p, mean_p)
        descriptors.append(physical_descriptors(velocity, pressure, arrays["time"][history], areas))
    result = {
        "starts": np.asarray(starts, dtype=np.int64),
        "split": np.asarray(splits),
        "re": np.asarray(reynolds, dtype=np.float64),
        "pred_a": np.asarray(pred_a, dtype=np.float32),
        "pred_b": np.asarray(pred_b, dtype=np.float32),
        "truth_a": np.asarray(truth_a, dtype=np.float32),
        "truth_b": np.asarray(truth_b, dtype=np.float32),
        "target_ids": np.asarray([ids[1:] for ids in chains], dtype=np.int64),
        "all_a": np.asarray(arrays["a"], dtype=np.float32),
        "all_b": np.asarray(arrays["b"], dtype=np.float32),
        "all_re": np.asarray(arrays["re"], dtype=np.float32),
        "all_time": np.asarray(arrays["time"], dtype=np.float32),
        "next": np.asarray(arrays["next_idx"], dtype=np.int64),
        "prev": invert_next(np.asarray(arrays["next_idx"], dtype=np.int64)),
        "hist": np.asarray(arrays["hist_idx"], dtype=np.int64),
        "phi_u": phi_u,
        "phi_p": phi_p,
        "mean_u": mean_u,
        "mean_p": mean_p,
        "areas": areas,
        "descriptors": np.stack(descriptors),
        "checkpoint": checkpoint_path,
        "source_name": "Periodic",
        # Candidate 1 is Periodic and candidate 2 is Hopf. The pair probability
        # order must match that candidate order so zero correction reproduces E2.
        "pair_indices": np.asarray([2, 1], dtype=np.int64),
    }
    del model, arrays, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--boundary", choices=("SH", "HP"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--windows-per-re", type=int, default=8)
    parser.add_argument("--mode", choices=("development", "heldout"), default="development")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    started = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source = (
        steady_source(args.root, args.horizon, args.windows_per_re, device, args.mode)
        if args.boundary == "SH"
        else periodic_source(args.root, args.horizon, args.windows_per_re, device, args.mode)
    )
    target = load_hopf_assets(args.root)
    if source["areas"].shape != target["areas"].shape or not np.allclose(source["areas"], target["areas"], atol=1.0e-8, rtol=1.0e-6):
        raise RuntimeError("common-grid area weights do not match")
    velocity_weights = np.repeat(source["areas"], 2)
    target_all_a = affine_project(
        source["all_a"], source["phi_u"], source["mean_u"], target["phi_u"], target["mean_u"], velocity_weights
    )
    target_all_b = affine_project(
        source["all_b"], source["phi_p"], source["mean_p"], target["phi_p"], target["mean_p"], source["areas"]
    )
    target_data = {
        "a": target_all_a,
        "b": target_all_b,
        "re": source["all_re"],
        "time": source["all_time"],
        "next": source["next"],
        "prev": source["prev"],
        "hist": source["hist"],
        "valid": source["starts"],
        "starts": source["starts"],
        "phi_u": target["phi_u"].astype(np.float32),
        "phi_p": target["phi_p"].astype(np.float32),
        "areas": target["areas"].astype(np.float32),
        "mean_u": target["mean_u"].astype(np.float32),
        "mean_p": target["mean_p"].astype(np.float32),
    }
    target_pred_a, target_pred_b, target_checkpoint = target_hopf_rollout(
        args.root, target_data, args.horizon, device
    )
    source_pred_u = reconstruct(source["pred_a"], source["phi_u"], source["mean_u"]).reshape(
        len(source["starts"]), args.horizon, -1, 2
    )
    target_pred_u = reconstruct(target_pred_a, target["phi_u"], target["mean_u"]).reshape(
        len(source["starts"]), args.horizon, -1, 2
    )
    truth_u = reconstruct(source["truth_a"], source["phi_u"], source["mean_u"]).reshape(
        len(source["starts"]), args.horizon, -1, 2
    )
    source_pred_p = pressure_gauge(reconstruct(source["pred_b"], source["phi_p"], source["mean_p"]), source["areas"])
    target_pred_p = pressure_gauge(reconstruct(target_pred_b, target["phi_p"], target["mean_p"]), source["areas"])
    truth_p = pressure_gauge(reconstruct(source["truth_b"], source["phi_p"], source["mean_p"]), source["areas"])
    quad_u = quadratic_statistics(source_pred_u, target_pred_u, truth_u, source["areas"][:, None])
    quad_p = quadratic_statistics(source_pred_p, target_pred_p, truth_p, source["areas"])
    first_finite = np.isfinite(source["pred_a"]).all((1, 2)) & np.isfinite(source["pred_b"]).all((1, 2))
    second_finite = np.isfinite(target_pred_a).all((1, 2)) & np.isfinite(target_pred_b).all((1, 2))
    truth_u_norm = np.sqrt(np.sum(truth_u * truth_u * source["areas"][None, None, :, None], axis=(2, 3))).clip(1.0e-8)
    truth_p_norm = np.sqrt(np.sum(truth_p * truth_p * source["areas"][None, None, :], axis=2)).clip(1.0e-8)
    first_ratio = np.maximum(
        np.sqrt(np.sum(source_pred_u * source_pred_u * source["areas"][None, None, :, None], axis=(2, 3))) / truth_u_norm,
        np.sqrt(np.sum(source_pred_p * source_pred_p * source["areas"][None, None, :], axis=2)) / truth_p_norm,
    )
    second_ratio = np.maximum(
        np.sqrt(np.sum(target_pred_u * target_pred_u * source["areas"][None, None, :, None], axis=(2, 3))) / truth_u_norm,
        np.sqrt(np.sum(target_pred_p * target_pred_p * source["areas"][None, None, :], axis=2)) / truth_p_norm,
    )
    first_divergent = np.any(first_ratio > 20.0, axis=1)
    second_divergent = np.any(second_ratio > 20.0, axis=1)
    report = {
        "schema_version": 1,
        "boundary": args.boundary,
        "native_source": source["source_name"],
        "horizon": args.horizon,
        "windows_per_Re": args.windows_per_re,
        "samples": int(len(source["starts"])),
        "mode": args.mode,
        "split_counts": {
            split: int(np.sum(source["split"] == split))
            for split in np.unique(source["split"])
        },
        "Re_by_split": {
            split: np.unique(source["re"][source["split"] == split]).astype(float).tolist()
            for split in np.unique(source["split"])
        },
        "candidate_1_finite_fraction": float(np.mean(first_finite)),
        "candidate_2_finite_fraction": float(np.mean(second_finite)),
        "candidate_1_divergent_windows": int(np.sum(first_divergent)),
        "candidate_2_divergent_windows": int(np.sum(second_divergent)),
        "candidate_1_max_norm_ratio": float(np.max(first_ratio)),
        "candidate_2_max_norm_ratio": float(np.max(second_ratio)),
        "common_grid_area_match": True,
        "history_length": 3,
        "common_physical_timestamps": True,
        "fusion_feedback": False,
        "heldout_loaded": args.mode == "heldout",
        "source_checkpoint": str(source["checkpoint"]),
        "source_checkpoint_sha256": sha256(source["checkpoint"]),
        "target_checkpoint": str(target_checkpoint),
        "target_checkpoint_sha256": sha256(target_checkpoint),
        "elapsed_seconds": time.time() - started,
    }
    passed = (
        bool(np.all(first_finite))
        and bool(np.all(second_finite))
        and not bool(np.any(first_divergent))
        and not bool(np.any(second_divergent))
    )
    report["status"] = "PASS" if passed else "FAIL_CLOSED"
    atomic_json(args.output_dir / "PREFLIGHT.json", report)
    if not passed:
        raise RuntimeError(f"{args.boundary} preflight failed: {report}")
    features = np.concatenate((source["re"][:, None].astype(np.float32), source["descriptors"]), axis=1)
    cache_path = args.output_dir / f"{args.boundary.lower()}_{args.mode}_cache.npz"
    temporary = cache_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(1),
            boundary=np.asarray(args.boundary),
            source_name=np.asarray(source["source_name"]),
            horizon=np.asarray(args.horizon),
            split=source["split"],
            re=source["re"],
            starts=source["starts"],
            target_ids=source["target_ids"],
            timestamps=source["all_time"][source["target_ids"]],
            features=features,
            pair_indices=np.repeat(source["pair_indices"][None, :], len(source["starts"]), axis=0),
            quad_u=quad_u,
            quad_p=quad_p,
            candidate_1_a=source["pred_a"],
            candidate_1_b=source["pred_b"],
            candidate_2_a=target_pred_a,
            candidate_2_b=target_pred_b,
            truth_a=source["truth_a"],
            truth_b=source["truth_b"],
            candidate_1_finite=first_finite,
            candidate_2_finite=second_finite,
            candidate_1_divergent=first_divergent,
            candidate_2_divergent=second_divergent,
        )
    os.replace(temporary, cache_path)
    report["cache_path"] = str(cache_path)
    report["cache_sha256"] = sha256(cache_path)
    atomic_json(args.output_dir / "PREFLIGHT.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
