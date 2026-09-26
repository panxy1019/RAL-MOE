#!/usr/bin/env python3
"""Validation analysis and gated held-out evaluation for B1--B4 checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch


EPS = 1.0e-12
HELDOUT_RE = np.asarray([47.081355, 49.022357, 51.786450])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--baseline-trainer", type=Path, required=True)
    parser.add_argument("--h4-trainer", type=Path, required=True)
    parser.add_argument("--coefficient-view", type=Path, required=True)
    parser.add_argument("--galerkin-path", type=Path, required=True)
    parser.add_argument("--pressure-path", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--fluctuation-contract", type=Path, required=True)
    parser.add_argument("--heldout-source-root", type=Path)
    parser.add_argument("--gate-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows-per-re", type=int, default=16)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_heldout_data(B: Any, source_root: Path) -> dict[str, np.ndarray]:
    velocity = np.load(source_root / "velocity_pod_hopf.npz", allow_pickle=False)
    pressure = np.load(source_root / "pressure_pod_hopf.npz", allow_pickle=False)
    with (
        source_root / "projection_snapshots_velocity_hopf.csv"
    ).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    reynolds_all = np.asarray([float(row["Re"]) for row in rows], dtype=np.float64)
    time_all = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
    split = np.asarray([row["split"].lower() for row in rows])
    keep = np.isin(split, ("heldout", "test"))
    observed = np.sort(np.unique(np.round(reynolds_all[keep], 6)))
    if observed.shape != (3,) or not np.allclose(
        observed, HELDOUT_RE, atol=1.1e-6, rtol=0.0
    ):
        raise RuntimeError(f"unexpected heldout Re: {observed}")
    a = np.asarray(velocity["coeff_uv"], dtype=np.float32)[keep, :32]
    b = np.asarray(pressure["coeff_p"], dtype=np.float32)[keep, :32]
    reynolds = reynolds_all[keep]
    times = time_all[keep]
    order = np.lexsort((times, reynolds))
    a, b, reynolds, times = (
        a[order],
        b[order],
        reynolds[order],
        times[order],
    )
    next_indices = np.full(len(reynolds), -1, dtype=np.int64)
    previous_indices = np.full(len(reynolds), -1, dtype=np.int64)
    for value in np.unique(reynolds):
        ids = np.flatnonzero(np.isclose(reynolds, value, atol=5.0e-7))
        next_indices[ids[:-1]] = ids[1:]
        previous_indices[ids[1:]] = ids[:-1]
    history = B.v16.history_index_matrix(
        np.arange(len(reynolds), dtype=np.int64), previous_indices, 3
    )
    valid = np.flatnonzero(
        (next_indices >= 0) & np.all(history >= 0, axis=1)
    )
    return {
        "a": a,
        "b": b,
        "re": reynolds.astype(np.float32),
        "time": times.astype(np.float32),
        "next": next_indices,
        "prev": previous_indices,
        "hist": history,
        "valid": valid,
        "phi_u": np.asarray(velocity["phi_uv"], dtype=np.float32)[:32],
        "phi_p": np.asarray(pressure["phi_p"], dtype=np.float32)[:32],
        "areas": np.asarray(velocity["point_areas"], dtype=np.float32),
        "mean_u": np.asarray(velocity["mean_uv_regime"], dtype=np.float32),
        "mean_p": np.asarray(pressure["mean_p_regime"], dtype=np.float32),
    }


def load_runtime(args: argparse.Namespace):
    code_dir = str(args.trainer.resolve().parent)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    T = load_module("ctdm_analysis_trainer", args.trainer)
    B = load_module("ctdm_analysis_baseline", args.baseline_trainer)
    H = load_module("ctdm_analysis_h4", args.h4_trainer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = SimpleNamespace(**checkpoint["args"])
    variant = checkpoint["variant"]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    train_data = B.load_coefficients(
        SimpleNamespace(coefficient_view=args.coefficient_view, history_len=3)
    )
    rom_np = B.load_train_rom(
        SimpleNamespace(
            galerkin_path=args.galerkin_path,
            pressure_path=args.pressure_path,
        )
    )
    fitted_stats, _ = B.fit_stats(
        train_data, rom_np, SimpleNamespace(history_len=3, scale_floor_quantile=0.1)
    )
    stats = {
        key: torch.as_tensor(value, device=device)
        for key, value in asdict(fitted_stats).items()
    }
    rom = {
        key: torch.as_tensor(value, device=device) for key, value in rom_np.items()
    }
    fc = H.FluctuationContract(args.fluctuation_contract, device)
    model = T.build_model(variant, saved_args).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return T, B, H, checkpoint, saved_args, variant, device, train_data, rom, stats, fc, model


def evaluate_dataset(
    T: Any,
    B: Any,
    model: torch.nn.Module,
    variant: str,
    fc: Any,
    data: Mapping[str, np.ndarray],
    ids_name: str,
    re_values: np.ndarray,
    rom: Mapping[str, torch.Tensor],
    stats: Mapping[str, torch.Tensor],
    device: torch.device,
    windows_per_re: int,
    warmup_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    modes_u = torch.as_tensor(data["phi_u"], device=device)
    modes_p = torch.as_tensor(data["phi_p"], device=device)
    mean_u = torch.as_tensor(data["mean_u"], device=device)
    mean_p = torch.as_tensor(data["mean_p"], device=device)
    area = torch.sqrt(torch.as_tensor(data["areas"], device=device))
    weight_u = torch.cat((area, area))
    rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    source_ids = data[ids_name]
    for re_value in re_values:
        ids = source_ids[
            np.isclose(data["re"][source_ids], re_value, atol=5.0e-6)
        ]
        starts = T.legal_starts(data, ids, 56, warmup_length)
        if not len(starts):
            rows.append(
                {
                    "Re": float(re_value),
                    "warmup_length": warmup_length,
                    "windows": 0,
                    "available": False,
                    "reason": "trajectory too short for warmup plus K56",
                }
            )
            continue
        if len(starts) > windows_per_re:
            starts = starts[
                np.linspace(0, len(starts) - 1, windows_per_re, dtype=int)
            ]
        output = T.rollout(
            model,
            variant,
            B,
            fc,
            data,
            starts,
            56,
            rom,
            stats,
            device,
            warmup_length=warmup_length,
        )
        eu_steps: list[torch.Tensor] = []
        ep_steps: list[torch.Tensor] = []
        ea_steps: list[torch.Tensor] = []
        eb_steps: list[torch.Tensor] = []
        norm_ratios: list[torch.Tensor] = []
        pressure_drifts: list[torch.Tensor] = []
        curve: dict[str, float] = {}
        for index, (pa, pb, ta, tb) in enumerate(
            zip(
                output["pred_a"],
                output["pred_b"],
                output["true_a"],
                output["true_b"],
            ),
            start=1,
        ):
            eu = T.physical_relative(pa, ta, modes_u, mean_u, weight_u)
            ep = T.physical_relative(pb, tb, modes_p, mean_p, area)
            eu_steps.append(eu)
            ep_steps.append(ep)
            ea_steps.append(
                torch.sqrt(torch.mean(((pa - ta) / stats["a_scale"]).square(), 1))
            )
            eb_steps.append(
                torch.sqrt(
                    torch.mean(((pb - tb) / stats["b_state_scale"]).square(), 1)
                )
            )
            ratio_u = torch.linalg.vector_norm(pa, dim=1) / torch.linalg.vector_norm(
                ta, dim=1
            ).clamp_min(EPS)
            ratio_p = torch.linalg.vector_norm(pb, dim=1) / torch.linalg.vector_norm(
                tb, dim=1
            ).clamp_min(EPS)
            norm_ratios.append(torch.maximum(ratio_u, ratio_p))
            true_energy = torch.sum(tb.square(), 1)
            pressure_drifts.append(
                torch.abs(torch.sum(pb.square(), 1) - true_energy)
                / true_energy.clamp_min(stats["b_rel_floor"])
            )
            if index in (1, 4, 8, 16, 32, 56):
                curve[str(index)] = float(torch.mean(eu + ep))
        eu_tensor = torch.stack(eu_steps, 1)
        ep_tensor = torch.stack(ep_steps, 1)
        ea_tensor = torch.stack(ea_steps, 1)
        eb_tensor = torch.stack(eb_steps, 1)
        ratio_tensor = torch.stack(norm_ratios, 1)
        finite = (
            torch.isfinite(eu_tensor).all(1)
            & torch.isfinite(ep_tensor).all(1)
            & torch.isfinite(ratio_tensor).all(1)
        )
        divergent = (~finite) | (ratio_tensor.max(1).values > 10.0)
        joint_by_window = (eu_tensor + ep_tensor).mean(1)
        row = {
            "Re": float(re_value),
            "warmup_length": warmup_length,
            "windows": int(len(starts)),
            "available": True,
            "Eu": float(torch.nanmean(eu_tensor)),
            "Ep": float(torch.nanmean(ep_tensor)),
            "Ea": float(torch.nanmean(ea_tensor)),
            "Eb": float(torch.nanmean(eb_tensor)),
            "terminal_Eu": float(torch.nanmean(eu_tensor[:, -1])),
            "terminal_Ep": float(torch.nanmean(ep_tensor[:, -1])),
            "worst_window_joint": float(
                torch.nan_to_num(joint_by_window, nan=1.0e9).max()
            ),
            "pressure_drift": float(
                torch.nanmean(torch.stack(pressure_drifts, 1))
            ),
            "finite_fraction": float(finite.float().mean()),
            "divergent_windows": int(divergent.sum()),
            "max_norm_ratio": float(
                torch.nan_to_num(ratio_tensor, nan=1.0e9, posinf=1.0e9).max()
            ),
            **{f"joint_K{key}": value for key, value in curve.items()},
        }
        rows.append(row)
        memory = T.memory_summary(output, float(re_value))
        if memory:
            memory["warmup_length"] = warmup_length
            memory_rows.append(memory)
    available_rows = [row for row in rows if row.get("available")]
    if not available_rows:
        raise RuntimeError("no trajectories support the requested warmup plus K56")
    aggregate = {
        "Eu": float(np.mean([row["Eu"] for row in available_rows])),
        "Ep": float(np.mean([row["Ep"] for row in available_rows])),
        "Ea": float(np.mean([row["Ea"] for row in available_rows])),
        "Eb": float(np.mean([row["Eb"] for row in available_rows])),
        "terminal_Eu": float(np.mean([row["terminal_Eu"] for row in available_rows])),
        "terminal_Ep": float(np.mean([row["terminal_Ep"] for row in available_rows])),
        "worst_window_joint": float(
            np.max([row["worst_window_joint"] for row in available_rows])
        ),
        "pressure_drift": float(np.mean([row["pressure_drift"] for row in available_rows])),
        "finite_fraction_min": float(
            np.min([row["finite_fraction"] for row in available_rows])
        ),
        "divergent_windows": int(
            np.sum([row["divergent_windows"] for row in available_rows])
        ),
        "available_re_count": len(available_rows),
        "requested_re_count": len(rows),
    }
    return {"aggregate": aggregate, "by_re": rows}, memory_rows


def timestep_consistency(
    T: Any,
    B: Any,
    model: torch.nn.Module,
    variant: str,
    fc: Any,
    data: Mapping[str, np.ndarray],
    ids_name: str,
    re_values: np.ndarray,
    rom: Mapping[str, torch.Tensor],
    stats: Mapping[str, torch.Tensor],
    device: torch.device,
    windows_per_re: int,
) -> list[dict[str, Any]]:
    modes_u = torch.as_tensor(data["phi_u"], device=device)
    modes_p = torch.as_tensor(data["phi_p"], device=device)
    mean_u = torch.as_tensor(data["mean_u"], device=device)
    mean_p = torch.as_tensor(data["mean_p"], device=device)
    area = torch.sqrt(torch.as_tensor(data["areas"], device=device))
    weight_u = torch.cat((area, area))
    rows: list[dict[str, Any]] = []
    source_ids = data[ids_name]
    for re_value in re_values:
        ids = source_ids[
            np.isclose(data["re"][source_ids], re_value, atol=5.0e-6)
        ]
        starts = T.legal_starts(data, ids, 56, 3)
        if len(starts) > windows_per_re:
            starts = starts[
                np.linspace(0, len(starts) - 1, windows_per_re, dtype=int)
            ]
        predictions: dict[int, Any] = {}
        for substeps in (1, 2, 4):
            predictions[substeps] = T.rollout(
                model,
                variant,
                B,
                fc,
                data,
                starts,
                56,
                rom,
                stats,
                device,
                warmup_length=3,
                substeps=substeps,
            )
        for coarse, fine in ((1, 2), (2, 4)):
            coarse_a = predictions[coarse]["pred_a"][-1]
            fine_a = predictions[fine]["pred_a"][-1]
            coarse_b = predictions[coarse]["pred_b"][-1]
            fine_b = predictions[fine]["pred_b"][-1]
            cu = T.physical_relative(
                coarse_a, fine_a, modes_u, mean_u, weight_u
            )
            cp = T.physical_relative(
                coarse_b, fine_b, modes_p, mean_p, area
            )
            cs = float("nan")
            if predictions[coarse]["memory"] is not None:
                delta = torch.linalg.vector_norm(
                    (
                        predictions[coarse]["memory"]
                        - predictions[fine]["memory"]
                    ).reshape(len(starts), -1),
                    dim=1,
                )
                denominator = torch.linalg.vector_norm(
                    predictions[fine]["memory"].reshape(len(starts), -1), dim=1
                ).clamp_min(EPS)
                cs = float(torch.mean(delta / denominator))
            rows.append(
                {
                    "Re": float(re_value),
                    "coarse_substeps": coarse,
                    "fine_substeps": fine,
                    "C_u": float(torch.mean(cu)),
                    "C_p": float(torch.mean(cp)),
                    "C_S": cs,
                    "windows": int(len(starts)),
                }
            )
    return rows


def verify_test_gate(args: argparse.Namespace) -> dict[str, Any]:
    if args.gate_manifest is None or not args.gate_manifest.is_file():
        raise RuntimeError("test evaluation requires a frozen gate manifest")
    gate = json.loads(args.gate_manifest.read_text(encoding="utf-8"))
    if gate.get("validation_frozen") is not True:
        raise RuntimeError("validation decision is not frozen")
    if gate.get("test_access_authorized") is not True:
        raise RuntimeError("pre-registered test gate did not authorize access")
    actual = sha256(args.checkpoint)
    expected_values = gate.get("authorized_checkpoint_sha256s")
    if expected_values is None:
        expected_values = [gate.get("authorized_checkpoint_sha256")]
    if actual not in expected_values:
        raise RuntimeError(f"authorized checkpoint hash mismatch: {actual}")
    return gate


def main() -> None:
    args = parse_args()
    gate = None
    if args.split == "test":
        gate = verify_test_gate(args)
        if args.heldout_source_root is None:
            raise ValueError("--heldout-source-root is required for test")
    (
        T,
        B,
        H,
        checkpoint,
        saved_args,
        variant,
        device,
        train_data,
        rom,
        stats,
        fc,
        model,
    ) = load_runtime(args)
    if args.split == "validation":
        data = train_data
        ids_name = "val_ids"
        re_values = np.asarray(H.VAL)
    else:
        # This is the first line that opens held-out tensor assets, and it is
        # reached only after verify_test_gate succeeds.
        data = build_heldout_data(B, args.heldout_source_root)
        ids_name = "valid"
        re_values = HELDOUT_RE
    started = time.perf_counter()
    main_result, memory_rows = evaluate_dataset(
        T,
        B,
        model,
        variant,
        fc,
        data,
        ids_name,
        re_values,
        rom,
        stats,
        device,
        args.windows_per_re,
        warmup_length=3,
    )
    warmup_results: dict[str, Any] = {}
    if variant in ("b3", "b4"):
        for length in (8, 16):
            capacity, capacity_memory = evaluate_dataset(
                T,
                B,
                model,
                variant,
                fc,
                data,
                ids_name,
                re_values,
                rom,
                stats,
                device,
                args.windows_per_re,
                warmup_length=length,
            )
            warmup_results[str(length)] = capacity
            memory_rows.extend(capacity_memory)
    consistency = timestep_consistency(
        T,
        B,
        model,
        variant,
        fc,
        data,
        ids_name,
        re_values,
        rom,
        stats,
        device,
        min(args.windows_per_re, 4),
    )
    payload = {
        "schema_version": 1,
        "split": args.split,
        "variant": variant,
        "model_name": T.MODEL_NAMES[variant],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["optimizer_step"]),
        "seed": int(saved_args.seed),
        "main_warmup_length": 3,
        "main": main_result,
        "capacity_warmup_diagnostics": warmup_results,
        "timestep_consistency": consistency,
        "memory_diagnostics": memory_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "training_performance": (
            json.loads(
                (args.checkpoint.parent / "THROUGHPUT.json").read_text(
                    encoding="utf-8"
                )
            )
            if (args.checkpoint.parent / "THROUGHPUT.json").is_file()
            else None
        ),
        "gate_manifest": gate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(payload, args.output_dir / f"{args.split.upper()}_RESULTS.json")
    write_rows(
        main_result["by_re"], args.output_dir / f"{args.split.upper()}_RESULTS.csv"
    )
    write_rows(
        consistency, args.output_dir / "TIMESTEP_CONSISTENCY.csv"
    )
    write_rows(memory_rows, args.output_dir / "MEMORY_DIAGNOSTICS.csv")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
