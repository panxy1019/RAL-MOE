#!/usr/bin/env python3
"""Build a leak-free Pinball SH/HP cache from sequential frozen B1 rollouts."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
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
    load_module,
    physical_descriptors,
    pressure_gauge,
    quadratic_statistics,
    reconstruct,
    sha256,
)


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    trainer: Path
    baseline: Path
    checkpoint: Path
    coefficient_view: Path
    galerkin: Path
    pressure: Path
    asset_manifest: Path
    split_manifest: Path | None
    pair_indices: tuple[int, int]


@dataclass
class Runtime:
    spec: ExpertSpec
    trainer: Any
    baseline: Any
    model: torch.nn.Module
    data: dict[str, np.ndarray]
    rom: dict[str, torch.Tensor]
    stats: dict[str, torch.Tensor]
    checkpoint: dict[str, Any]


def specs(root: Path) -> dict[str, ExpertSpec]:
    dataset = root / "fluidicPinball_v2"
    steady = root / "steady_b1_rank999_20260806"
    hopf = root / "hopf_b1_rank999_20260806"
    periodic = root.parent / "particalMOE/fluidic_pinball_periodic_v2_b1"
    return {
        "Steady": ExpertSpec(
            "Steady",
            steady / "code/train_b1_steady.py",
            steady / "code/pinball_steady_base.py",
            steady / "runs/formal/FluidicPinballV2_Steady_rank999_B1_seed1248/best_validation.pt",
            steady / "assets/steady_rank999_trainval.npz",
            dataset / "rom_assets_v2/steady/rom/rank999_ru4_rp3/velocity_galerkin_tensors.npz",
            dataset / "rom_assets_v2/steady/rom/rank999_ru4_rp3/pressure_poisson_tensors.npz",
            steady / "assets/TRAINING_ASSET_MANIFEST.json",
            dataset / "config/experts/steady_re_manifest.csv",
            (0, 1),
        ),
        "Hopf": ExpertSpec(
            "Hopf",
            hopf / "code/train_kda_pr_fnn_rom.py",
            hopf / "code/pinball_hopf_base.py",
            hopf / "runs/formal/FluidicPinballV2_Hopf_rank999_B1_seed1248/best_validation.pt",
            hopf / "assets/hopf_rank999_trainval.npz",
            dataset / "rom_assets_v2/hopf/rom/rank999_ru5_rp5/velocity_galerkin_tensors.npz",
            dataset / "rom_assets_v2/hopf/rom/rank999_ru5_rp5/pressure_poisson_tensors.npz",
            hopf / "assets/TRAINING_ASSET_MANIFEST.json",
            dataset / "config/experts/hopf_re_manifest.csv",
            (1, 1),
        ),
        "Periodic": ExpertSpec(
            "Periodic",
            periodic / "code/train_b1_fluidic_pinball.py",
            periodic / "code/b1_data_contract.py",
            periodic / "runs/FluidicPinballV2_B1_Deep_FNN_H3_seed1248/best_validation.pt",
            periodic / "assets/fluidic_pinball_periodic_trainval_rank999.npz",
            periodic / "assets/fluidic_pinball_periodic_trainonly_galerkin_rank999.npz",
            periodic / "assets/fluidic_pinball_periodic_trainonly_pressure_rank999.npz",
            periodic / "assets/TRAINING_ASSET_MANIFEST.json",
            None,
            (2, 1),
        ),
    }


def namespace(spec: ExpertSpec) -> SimpleNamespace:
    return SimpleNamespace(
        variant="b1",
        baseline_trainer=spec.baseline,
        coefficient_view=spec.coefficient_view,
        galerkin_path=spec.galerkin,
        pressure_path=spec.pressure,
        asset_manifest=spec.asset_manifest,
        split_manifest=spec.split_manifest,
        fluctuation_contract=None,
        history_len=3,
        scale_floor_quantile=0.10,
        lambda_pressure_rollout=0.25,
        tbptt_steps=0,
    )


def load_runtime(spec: ExpertSpec, device: torch.device) -> Runtime:
    for path in (
        spec.trainer,
        spec.baseline,
        spec.checkpoint,
        spec.coefficient_view,
        spec.galerkin,
        spec.pressure,
        spec.asset_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    args = namespace(spec)
    trainer = load_module(f"fusion_{spec.name.lower()}_trainer", spec.trainer)
    baseline = load_module(f"fusion_{spec.name.lower()}_baseline", spec.baseline)
    if hasattr(trainer, "configure_shape_contract"):
        shape = trainer.configure_shape_contract(args)
    else:
        manifest = trainer.configure_contract(spec.asset_manifest)
        shape = {
            "ru": int(manifest["r_u"]),
            "rp": int(manifest["r_p"]),
            "h3_dim": int(trainer.H3_DIM),
        }
    checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "b1":
        raise RuntimeError(f"{spec.name} checkpoint is not B1")
    model = trainer.DeepFNNH3().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError(f"{spec.name} freeze failed")
    expected = {"Steady": 1684370, "Hopf": 1695679, "Periodic": 1792959}[spec.name]
    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != expected:
        raise RuntimeError(f"{spec.name} parameter count {observed} != {expected}")
    data = baseline.load_coefficients(trainer.base_args(args))
    rom_np = baseline.load_train_rom(trainer.base_args(args))
    rom = {key: torch.as_tensor(value, device=device) for key, value in rom_np.items()}
    stats = {key: torch.as_tensor(value, device=device) for key, value in checkpoint["norm_stats"].items()}
    if int(data["a"].shape[1]) != int(shape["ru"]) or int(data["b"].shape[1]) != int(shape["rp"]):
        raise RuntimeError(f"{spec.name} coefficient rank mismatch")
    return Runtime(spec, trainer, baseline, model, data, rom, stats, checkpoint)


def release(runtime: Runtime) -> None:
    del runtime.model, runtime.rom, runtime.stats, runtime.data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def canonical_velocity_assets(runtime: Runtime) -> tuple[np.ndarray, np.ndarray]:
    """Return velocity POD assets in component-major [all-u, all-v] layout."""
    phi = np.asarray(runtime.data["phi_u"], dtype=np.float64)
    mean = np.asarray(runtime.data["mean_u"], dtype=np.float64)
    if runtime.spec.name == "Periodic":
        cells = len(runtime.data["areas"])
        phi = phi.reshape(len(phi), cells, 2).transpose(0, 2, 1).reshape(len(phi), -1)
        mean = mean.reshape(cells, 2).T.reshape(-1)
    return phi, mean


def velocity_field(coefficients: np.ndarray, runtime: Runtime) -> np.ndarray:
    phi, mean = canonical_velocity_assets(runtime)
    flat = reconstruct(coefficients, phi, mean)
    cells = len(runtime.data["areas"])
    return flat.reshape(*flat.shape[:-1], 2, cells).swapaxes(-2, -1)


def overlap_re(data: dict[str, np.ndarray], boundary: str, split: str) -> np.ndarray:
    ranges = {
        "SH": (16.6, 19.0),
        "HP": (21.25, 25.0),
    }
    lo, hi = ranges[boundary]
    ids = data["train_ids"] if split == "train" else data["val_ids"]
    values = np.asarray(data["re"])[ids]
    return np.unique(values[(values >= lo - 5.0e-7) & (values <= hi + 5.0e-7)])


def select_starts(runtime: Runtime, boundary: str, horizon: int, windows_per_re: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts: list[int] = []
    splits: list[str] = []
    reynolds: list[float] = []
    for split in ("train", "validation"):
        pool = runtime.data["train_ids"] if split == "train" else runtime.data["val_ids"]
        for value in overlap_re(runtime.data, boundary, split):
            ids = pool[np.isclose(runtime.data["re"][pool], value, atol=5.0e-7, rtol=0.0)]
            legal = runtime.trainer.legal_starts(runtime.baseline, runtime.data, ids, horizon)
            chosen = evenly_spaced(legal, windows_per_re)
            if len(chosen) != windows_per_re:
                raise RuntimeError(
                    f"{runtime.spec.name} Re={float(value)} split={split} has only {len(chosen)} K{horizon} windows"
                )
            starts.extend(chosen.tolist())
            splits.extend([split] * len(chosen))
            reynolds.extend([float(value)] * len(chosen))
    order = np.lexsort((np.asarray(starts), np.asarray(reynolds), np.asarray(splits)))
    return (
        np.asarray(starts, dtype=np.int64)[order],
        np.asarray(splits)[order],
        np.asarray(reynolds, dtype=np.float64)[order],
    )


def run_rollout(runtime: Runtime, starts: np.ndarray, horizon: int, batch_size: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_a, pred_b, truth_a, truth_b = [], [], [], []
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            batch = starts[offset : offset + batch_size]
            output = runtime.trainer.rollout(
                runtime.model,
                "b1",
                runtime.baseline,
                runtime.data,
                batch,
                horizon,
                runtime.rom,
                runtime.stats,
                next(runtime.model.parameters()).device,
                0,
            )
            pred_a.append(torch.stack(output["pred_a"], dim=1).float().cpu().numpy())
            pred_b.append(torch.stack(output["pred_b"], dim=1).float().cpu().numpy())
            truth_a.append(torch.stack(output["true_a"], dim=1).float().cpu().numpy())
            truth_b.append(torch.stack(output["true_b"], dim=1).float().cpu().numpy())
    return tuple(np.concatenate(values) for values in (pred_a, pred_b, truth_a, truth_b))


def target_data(source: Runtime, target: Runtime, starts: np.ndarray) -> dict[str, np.ndarray]:
    source_phi_u, source_mean_u = canonical_velocity_assets(source)
    target_phi_u, target_mean_u = canonical_velocity_assets(target)
    velocity_weights = np.tile(np.asarray(source.data["areas"], dtype=np.float64), 2)
    projected_a = affine_project(
        source.data["a"], source_phi_u, source_mean_u,
        target_phi_u, target_mean_u, velocity_weights,
    )
    projected_b = affine_project(
        source.data["b"], source.data["phi_p"], source.data["mean_p"],
        target.data["phi_p"], target.data["mean_p"], source.data["areas"],
    )
    return {
        "a": projected_a,
        "b": projected_b,
        "re": source.data["re"],
        "time": source.data["time"],
        "next": source.data["next"],
        "prev": source.data["prev"],
        "hist": source.data["hist"],
        "valid": starts,
        "train_ids": starts,
        "val_ids": starts,
        "phi_u": target.data["phi_u"],
        "phi_p": target.data["phi_p"],
        "areas": target.data["areas"],
        "mean_u": target.data["mean_u"],
        "mean_p": target.data["mean_p"],
    }


def field_statistics(
    source_pred_a: np.ndarray,
    source_pred_b: np.ndarray,
    target_pred_a: np.ndarray,
    target_pred_b: np.ndarray,
    truth_a: np.ndarray,
    truth_b: np.ndarray,
    source: Runtime,
    target: Runtime,
    chunk: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    quad_u, quad_p, ratio_1, ratio_2 = [], [], [], []
    areas = np.asarray(source.data["areas"], dtype=np.float64)
    for offset in range(0, len(source_pred_a), chunk):
        sl = slice(offset, offset + chunk)
        u1 = velocity_field(source_pred_a[sl], source)
        u2 = velocity_field(target_pred_a[sl], target)
        ut = velocity_field(truth_a[sl], source)
        p1 = pressure_gauge(reconstruct(source_pred_b[sl], source.data["phi_p"], source.data["mean_p"]), areas)
        p2 = pressure_gauge(reconstruct(target_pred_b[sl], target.data["phi_p"], target.data["mean_p"]), areas)
        pt = pressure_gauge(reconstruct(truth_b[sl], source.data["phi_p"], source.data["mean_p"]), areas)
        quad_u.append(quadratic_statistics(u1, u2, ut, areas[:, None]))
        quad_p.append(quadratic_statistics(p1, p2, pt, areas))
        un = np.sqrt(np.sum(ut * ut * areas[None, None, :, None], axis=(2, 3))).clip(1.0e-8)
        pn = np.sqrt(np.sum(pt * pt * areas[None, None, :], axis=2)).clip(1.0e-8)
        ratio_1.append(np.maximum(
            np.sqrt(np.sum(u1 * u1 * areas[None, None, :, None], axis=(2, 3))) / un,
            np.sqrt(np.sum(p1 * p1 * areas[None, None, :], axis=2)) / pn,
        ))
        ratio_2.append(np.maximum(
            np.sqrt(np.sum(u2 * u2 * areas[None, None, :, None], axis=(2, 3))) / un,
            np.sqrt(np.sum(p2 * p2 * areas[None, None, :], axis=2)) / pn,
        ))
        del u1, u2, ut, p1, p2, pt
    return tuple(np.concatenate(values) for values in (quad_u, quad_p, ratio_1, ratio_2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Remote Pinball root containing dataset and steady/hopf experts")
    parser.add_argument("--boundary", choices=("SH", "HP"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--windows-per-re", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    if args.horizon != 24:
        raise RuntimeError("the certified fusion contract requires K=24")
    started = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_specs = specs(args.root)
    source_name = "Steady" if args.boundary == "SH" else "Periodic"
    source = load_runtime(all_specs[source_name], device)
    starts, split, reynolds = select_starts(source, args.boundary, args.horizon, args.windows_per_re)
    source_pred_a, source_pred_b, truth_a, truth_b = run_rollout(source, starts, args.horizon)
    target = load_runtime(all_specs["Hopf"], device)
    if source.data["areas"].shape != target.data["areas"].shape or not np.allclose(
        source.data["areas"], target.data["areas"], atol=1.0e-8, rtol=1.0e-6
    ):
        raise RuntimeError("common-grid area weights do not match")
    original_target_data = target.data
    target.data = target_data(source, target, starts)
    target_pred_a, target_pred_b, _, _ = run_rollout(target, starts, args.horizon)
    target.data = original_target_data
    quad_u, quad_p, first_ratio, second_ratio = field_statistics(
        source_pred_a, source_pred_b, target_pred_a, target_pred_b, truth_a, truth_b, source, target
    )
    descriptors = []
    for start in starts:
        history = np.asarray(source.data["hist"][start], dtype=np.int64)[::-1]
        velocity = velocity_field(source.data["a"][history], source)
        pressure = reconstruct(source.data["b"][history], source.data["phi_p"], source.data["mean_p"])
        descriptors.append(physical_descriptors(velocity, pressure, source.data["time"][history], source.data["areas"]))
    target_ids = np.asarray([chain(source.data["next"], int(start), args.horizon)[1:] for start in starts], dtype=np.int64)
    first_finite = np.isfinite(source_pred_a).all((1, 2)) & np.isfinite(source_pred_b).all((1, 2))
    second_finite = np.isfinite(target_pred_a).all((1, 2)) & np.isfinite(target_pred_b).all((1, 2))
    first_divergent = np.any(first_ratio > 20.0, axis=1)
    second_divergent = np.any(second_ratio > 20.0, axis=1)
    report = {
        "schema_version": 1,
        "boundary": args.boundary,
        "candidate_order": [source_name, "Hopf"],
        "horizon": args.horizon,
        "windows_per_Re": args.windows_per_re,
        "samples": int(len(starts)),
        "split_counts": {value: int(np.sum(split == value)) for value in np.unique(split)},
        "Re_by_split": {value: np.unique(reynolds[split == value]).astype(float).tolist() for value in np.unique(split)},
        "candidate_1_finite_fraction": float(np.mean(first_finite)),
        "candidate_2_finite_fraction": float(np.mean(second_finite)),
        "candidate_1_divergent_windows": int(np.sum(first_divergent)),
        "candidate_2_divergent_windows": int(np.sum(second_divergent)),
        "candidate_1_max_norm_ratio": float(np.max(first_ratio)),
        "candidate_2_max_norm_ratio": float(np.max(second_ratio)),
        "common_grid_area_match": True,
        "history_length": 3,
        "history_order_for_features": "chronological_t-2_t-1_t",
        "common_physical_timestamps": True,
        "fusion_feedback": False,
        "heldout_loaded": False,
        "source_checkpoint": str(source.spec.checkpoint),
        "source_checkpoint_sha256": sha256(source.spec.checkpoint),
        "target_checkpoint": str(target.spec.checkpoint),
        "target_checkpoint_sha256": sha256(target.spec.checkpoint),
        "elapsed_seconds": time.time() - started,
    }
    passed = bool(np.all(first_finite) and np.all(second_finite) and not np.any(first_divergent) and not np.any(second_divergent))
    report["status"] = "PASS" if passed else "FAIL_CLOSED"
    atomic_json(args.output_dir / "PREFLIGHT.json", report)
    if not passed:
        raise RuntimeError(f"{args.boundary} preflight failed: {report}")
    features = np.concatenate((reynolds[:, None].astype(np.float32), np.stack(descriptors)), axis=1)
    cache_path = args.output_dir / f"{args.boundary.lower()}_development_cache.npz"
    temporary = cache_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(1),
            boundary=np.asarray(args.boundary),
            source_name=np.asarray(source_name),
            horizon=np.asarray(args.horizon),
            split=split,
            re=reynolds,
            starts=starts,
            target_ids=target_ids,
            timestamps=source.data["time"][target_ids],
            features=features,
            pair_indices=np.repeat(np.asarray(source.spec.pair_indices)[None, :], len(starts), axis=0),
            quad_u=quad_u,
            quad_p=quad_p,
            candidate_1_a=source_pred_a,
            candidate_1_b=source_pred_b,
            candidate_2_a=target_pred_a,
            candidate_2_b=target_pred_b,
            truth_a=truth_a,
            truth_b=truth_b,
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
    release(source)
    release(target)


if __name__ == "__main__":
    main()
