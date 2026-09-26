#!/usr/bin/env python3
"""Evaluate a Fluidic Pinball V2 Hopf checkpoint on validation trajectories only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trainer", type=Path, required=True)
    p.add_argument("--baseline-trainer", type=Path, required=True)
    p.add_argument("--coefficient-view", type=Path, required=True)
    p.add_argument("--galerkin-path", type=Path, required=True)
    p.add_argument("--pressure-path", type=Path, required=True)
    p.add_argument("--asset-manifest", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--windows-per-re", type=int, default=64)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def percentile(value: torch.Tensor, q: float) -> float:
    return float(torch.quantile(value.float().reshape(-1), q).cpu())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("formal rollout evaluation requires CUDA")
    T = load_module("pinball_rollout_trainer", args.trainer)
    B = load_module("pinball_rollout_base", args.baseline_trainer)
    base_args = SimpleNamespace(
        coefficient_view=args.coefficient_view, galerkin_path=args.galerkin_path,
        pressure_path=args.pressure_path, asset_manifest=args.asset_manifest,
        split_manifest=args.split_manifest, history_len=3,
        scale_floor_quantile=0.10, lambda_pressure_rollout=0.25,
    )
    shape = T.configure_shape_contract(base_args)
    audit = B.audit_assets(base_args)
    data = B.load_coefficients(base_args)
    if np.any(data["split"] == "final_test"):
        raise RuntimeError("HELDOUT HARD-GATE: final_test rows loaded")
    T.TRAIN_RE, T.VAL_RE, T.HELDOUT_RE = (
        data["train_re"], data["validation_re"], data["heldout_re"])
    rom_np = B.load_train_rom(base_args)
    rom = {name: torch.as_tensor(value, device=device) for name, value in rom_np.items()}

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 2:
        raise RuntimeError("checkpoint schema_version mismatch")
    if checkpoint.get("contract") != "PINBALL_V2_HOPF_TRAIN_VALIDATION_ONLY":
        raise RuntimeError("checkpoint contract mismatch")
    if checkpoint.get("variant") != "b1":
        raise RuntimeError("this evaluator currently accepts the B1 checkpoint only")
    if checkpoint.get("heldout_evaluation_performed") is not False:
        raise RuntimeError("checkpoint held-out state is not sealed")
    if checkpoint.get("shape_contract") != shape:
        raise RuntimeError("checkpoint/runtime shape contract mismatch")
    if checkpoint.get("asset_hashes") != audit["observed_sha256"]:
        raise RuntimeError("checkpoint/runtime asset hash mismatch")
    if int(checkpoint["optimizer_step"]) != 8000:
        raise RuntimeError(f"checkpoint is not final-budget trained: {checkpoint['optimizer_step']}")

    model = T.DeepFNNH3().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    stats = {name: torch.as_tensor(value, device=device)
             for name, value in checkpoint["norm_stats"].items()}
    horizons = (1, 2, 4, 8, 16, 32, 56)
    rows: list[dict[str, float | int]] = []
    curves: list[dict[str, float | int]] = []
    total_window_steps = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for rv in data["validation_re"]:
            ids = data["val_ids"][np.isclose(data["re"][data["val_ids"]], rv, atol=5e-6)]
            for horizon in horizons:
                starts = T.legal_starts(B, data, ids, horizon)
                if len(starts) > args.windows_per_re:
                    starts = starts[np.linspace(0, len(starts) - 1,
                                                args.windows_per_re, dtype=int)]
                if not len(starts):
                    raise RuntimeError(f"no legal windows Re={rv}, K={horizon}")
                t0 = time.perf_counter()
                out = T.rollout(model, "b1", B, data, starts, horizon, rom, stats,
                                device, 0)
                u, p = T.physical_series(B, data, out, device)
                pa, pb = torch.stack(out["pred_a"], 1), torch.stack(out["pred_b"], 1)
                ta, tb = torch.stack(out["true_a"], 1), torch.stack(out["true_b"], 1)
                finite = torch.isfinite(pa).all((1, 2)) & torch.isfinite(pb).all((1, 2))
                ratio_u = torch.linalg.vector_norm(pa, dim=2) / torch.linalg.vector_norm(
                    ta, dim=2).clamp_min(T.EPS)
                ratio_p = torch.linalg.vector_norm(pb, dim=2) / torch.linalg.vector_norm(
                    tb, dim=2).clamp_min(T.EPS)
                divergent = (~finite) | (torch.maximum(ratio_u, ratio_p).max(1).values > 10)
                joint = u + p
                coeff_u = torch.linalg.vector_norm(pa - ta, dim=2) / torch.linalg.vector_norm(
                    ta, dim=2).clamp_min(T.EPS)
                coeff_p = torch.linalg.vector_norm(pb - tb, dim=2) / torch.linalg.vector_norm(
                    tb, dim=2).clamp_min(T.EPS)
                pressure_drift = (torch.abs(pb.square().sum(2) - tb.square().sum(2)) /
                                  tb.square().sum(2).clamp_min(T.EPS))
                elapsed = time.perf_counter() - t0
                total_window_steps += len(starts) * horizon
                rows.append({
                    "Re": float(rv), "horizon": horizon, "windows": int(len(starts)),
                    "velocity_mean": float(u.mean().cpu()),
                    "velocity_median": percentile(u, .5), "velocity_p95": percentile(u, .95),
                    "pressure_mean": float(p.mean().cpu()),
                    "pressure_median": percentile(p, .5), "pressure_p95": percentile(p, .95),
                    "joint_mean": float(joint.mean().cpu()),
                    "terminal_joint": float(joint[:, -1].mean().cpu()),
                    "worst_window": float(joint.max().cpu()),
                    "coefficient_u_mean": float(coeff_u.mean().cpu()),
                    "coefficient_p_mean": float(coeff_p.mean().cpu()),
                    "pressure_drift": float(pressure_drift.mean().cpu()),
                    "finite_fraction": float(finite.float().mean().cpu()),
                    "divergent_windows": int(divergent.sum().cpu()),
                    "elapsed_seconds": elapsed,
                })
                for step in range(horizon):
                    curves.append({"Re": float(rv), "horizon": horizon, "step": step + 1,
                                   "velocity_mean": float(u[:, step].mean().cpu()),
                                   "pressure_mean": float(p[:, step].mean().cpu()),
                                   "joint_mean": float(joint[:, step].mean().cpu())})
                del out, u, p, pa, pb, ta, tb
    torch.cuda.synchronize()
    elapsed_total = time.perf_counter() - started

    summary_horizons = {}
    for horizon in horizons:
        selected = [row for row in rows if row["horizon"] == horizon]
        summary_horizons[str(horizon)] = {
            "velocity_mean": float(np.mean([row["velocity_mean"] for row in selected])),
            "pressure_mean": float(np.mean([row["pressure_mean"] for row in selected])),
            "joint_mean": float(np.mean([row["joint_mean"] for row in selected])),
            "terminal_joint": float(np.mean([row["terminal_joint"] for row in selected])),
            "worst_window": float(np.max([row["worst_window"] for row in selected])),
            "pressure_drift": float(np.mean([row["pressure_drift"] for row in selected])),
            "finite_fraction": float(np.min([row["finite_fraction"] for row in selected])),
            "divergent_windows": int(np.sum([row["divergent_windows"] for row in selected])),
            "windows": int(np.sum([row["windows"] for row in selected])),
        }
    k56 = summary_horizons["56"]
    summary = {
        "schema_version": 1, "status": "PASS" if k56["finite_fraction"] == 1.0 and
        k56["divergent_windows"] == 0 else "FAIL",
        "scope": "validation_only", "heldout_loaded": False,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["optimizer_step"]),
        "checkpoint_best_step": int(checkpoint["best_step"]),
        "shape_contract": shape, "asset_hashes": audit["observed_sha256"],
        "validation_re": data["validation_re"].tolist(), "horizons": summary_horizons,
        "performance": {"elapsed_seconds": elapsed_total,
                        "window_steps": total_window_steps,
                        "window_steps_per_second": total_window_steps / max(elapsed_total, 1e-12),
                        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2**30},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "ROLLOUT_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "ERROR_GROWTH_CURVES.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0]))
        writer.writeheader(); writer.writerows(curves)
    atomic_json(summary, args.output_dir / "ROLLOUT_SUMMARY.json")
    manifest = {path.name: sha256(path) for path in sorted(args.output_dir.glob("*")) if path.is_file()}
    atomic_json(manifest, args.output_dir / "ROLLOUT_ARTIFACT_MANIFEST.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
