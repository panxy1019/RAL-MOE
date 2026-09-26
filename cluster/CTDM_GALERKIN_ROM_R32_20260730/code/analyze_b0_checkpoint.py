#!/usr/bin/env python3
"""Evaluate the frozen B0 HPRS-MoE-ROM with the matched K56 metrics."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    parser.add_argument("--baseline-trainer", type=Path, required=True)
    parser.add_argument("--analysis-helper", type=Path, required=True)
    parser.add_argument("--coefficient-view", type=Path, required=True)
    parser.add_argument("--galerkin-path", type=Path, required=True)
    parser.add_argument("--pressure-path", type=Path, required=True)
    parser.add_argument("--training-throughput", type=Path)
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


def saved_args(checkpoint: Mapping[str, Any], args: argparse.Namespace) -> SimpleNamespace:
    values = dict(checkpoint["args"])
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
        values.setdefault(key, value)
    values.update(
        {
            "coefficient_view": args.coefficient_view,
            "galerkin_path": args.galerkin_path,
            "pressure_path": args.pressure_path,
            "device": args.device,
        }
    )
    return SimpleNamespace(**values)


def verify_test_gate(args: argparse.Namespace, helper: Any) -> dict[str, Any]:
    if args.gate_manifest is None or not args.gate_manifest.is_file():
        raise RuntimeError("B0 test evaluation requires a frozen gate manifest")
    gate = json.loads(args.gate_manifest.read_text(encoding="utf-8"))
    if gate.get("validation_frozen") is not True:
        raise RuntimeError("validation decision is not frozen")
    if gate.get("test_access_authorized") is not True:
        raise RuntimeError("pre-registered test gate did not authorize access")
    actual = helper.sha256(args.checkpoint)
    if actual not in gate.get("authorized_checkpoint_sha256s", []):
        raise RuntimeError(f"authorized B0 checkpoint hash mismatch: {actual}")
    return gate


def load_runtime(args: argparse.Namespace):
    helper = load_module("b0_analysis_helper", args.analysis_helper)
    gate = None
    if args.split == "test":
        gate = verify_test_gate(args, helper)
        if args.heldout_source_root is None:
            raise ValueError("--heldout-source-root is required for test")
    baseline = load_module("b0_frozen_baseline", args.baseline_trainer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "h4":
        raise RuntimeError(f"unexpected B0 variant: {checkpoint.get('variant')}")
    model_args = saved_args(checkpoint, args)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    train_data = baseline.load_coefficients(model_args)
    rom_np = baseline.load_train_rom(model_args)
    norm_stats, _ = baseline.fit_stats(train_data, rom_np, model_args)
    stats = {
        key: torch.as_tensor(value, device=device)
        for key, value in asdict(norm_stats).items()
    }
    rom = {
        key: torch.as_tensor(value, device=device)
        for key, value in rom_np.items()
    }
    probe = baseline.batch_from_ids(
        train_data, train_data["train_ids"][:2], device
    )
    previous_rhs = baseline.galerkin(
        probe["ah"].reshape(-1, 32),
        probe["bh"].reshape(-1, 32),
        probe["re"][:, None]
        .expand(-1, model_args.history_len)
        .reshape(-1),
        rom,
    ).reshape_as(probe["ah"])
    input_dim = baseline.state_features(
        probe["a"],
        probe["b"],
        probe["re"],
        probe["ah"],
        probe["bh"],
        previous_rhs,
        rom,
        stats,
    )[0].shape[1]
    if input_dim != 493:
        raise RuntimeError(f"frozen B0 runtime input dimension {input_dim} != 493")
    first_weight = checkpoint["model_state"]["encoder.net.0.weight"]
    if tuple(first_weight.shape) != (256, 493):
        raise RuntimeError(
            f"frozen B0 checkpoint input shape {tuple(first_weight.shape)}"
        )
    model = baseline.build_model(input_dim, model_args, stats, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return (
        helper,
        baseline,
        checkpoint,
        model_args,
        device,
        train_data,
        rom,
        stats,
        model,
        gate,
    )


@torch.no_grad()
def evaluate_dataset(
    baseline: Any,
    model: torch.nn.Module,
    data: Mapping[str, np.ndarray],
    ids_name: str,
    re_values: np.ndarray,
    rom: Mapping[str, torch.Tensor],
    stats: Mapping[str, torch.Tensor],
    device: torch.device,
    windows_per_re: int,
) -> dict[str, Any]:
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
        starts = baseline.legal_starts(data, ids, 56)
        if not len(starts):
            raise RuntimeError(f"B0 has no K56 windows for Re={re_value}")
        if len(starts) > windows_per_re:
            starts = starts[
                np.linspace(0, len(starts) - 1, windows_per_re, dtype=int)
            ]
        output = baseline.rollout_batch(
            model, data, starts, 56, rom, stats, device
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
            eu = baseline.physical_relative(pa, ta, modes_u, mean_u, weight_u)
            ep = baseline.physical_relative(pb, tb, modes_p, mean_p, area)
            eu_steps.append(eu)
            ep_steps.append(ep)
            ea_steps.append(
                torch.sqrt(
                    torch.mean(((pa - ta) / stats["a_scale"]).square(), 1)
                )
            )
            eb_steps.append(
                torch.sqrt(
                    torch.mean(
                        ((pb - tb) / stats["b_state_scale"]).square(), 1
                    )
                )
            )
            ratio_u = torch.linalg.vector_norm(pa, dim=1) / (
                torch.linalg.vector_norm(ta, dim=1).clamp_min(EPS)
            )
            ratio_p = torch.linalg.vector_norm(pb, dim=1) / (
                torch.linalg.vector_norm(tb, dim=1).clamp_min(EPS)
            )
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
        rows.append(
            {
                "Re": float(re_value),
                "warmup_length": 3,
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
                    torch.nan_to_num(
                        ratio_tensor, nan=1.0e9, posinf=1.0e9
                    ).max()
                ),
                **{f"joint_K{key}": value for key, value in curve.items()},
            }
        )
    aggregate = {
        "Eu": float(np.mean([row["Eu"] for row in rows])),
        "Ep": float(np.mean([row["Ep"] for row in rows])),
        "Ea": float(np.mean([row["Ea"] for row in rows])),
        "Eb": float(np.mean([row["Eb"] for row in rows])),
        "terminal_Eu": float(
            np.mean([row["terminal_Eu"] for row in rows])
        ),
        "terminal_Ep": float(
            np.mean([row["terminal_Ep"] for row in rows])
        ),
        "worst_window_joint": float(
            np.max([row["worst_window_joint"] for row in rows])
        ),
        "pressure_drift": float(
            np.mean([row["pressure_drift"] for row in rows])
        ),
        "finite_fraction_min": float(
            np.min([row["finite_fraction"] for row in rows])
        ),
        "divergent_windows": int(
            np.sum([row["divergent_windows"] for row in rows])
        ),
        "available_re_count": len(rows),
        "requested_re_count": len(rows),
    }
    return {"aggregate": aggregate, "by_re": rows}


def main() -> None:
    args = parse_args()
    (
        helper,
        baseline,
        checkpoint,
        model_args,
        device,
        train_data,
        rom,
        stats,
        model,
        gate,
    ) = load_runtime(args)
    if args.split == "validation":
        data = train_data
        ids_name = "val_ids"
        re_values = np.asarray([46.7, 56.543246])
    else:
        # The held-out tensor files are first opened here, after gate verification.
        data = helper.build_heldout_data(baseline, args.heldout_source_root)
        ids_name = "valid"
        re_values = HELDOUT_RE
    started = time.perf_counter()
    main_result = evaluate_dataset(
        baseline,
        model,
        data,
        ids_name,
        re_values,
        rom,
        stats,
        device,
        args.windows_per_re,
    )
    payload = {
        "schema_version": 1,
        "split": args.split,
        "variant": "b0",
        "model_name": "B0_frozen_HPRS_MoE_ROM",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": helper.sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["optimizer_step"]),
        "seed": int(model_args.seed),
        "main_warmup_length": 3,
        "main": main_result,
        "capacity_warmup_diagnostics": {},
        "timestep_consistency": [],
        "memory_diagnostics": [],
        "elapsed_seconds": time.perf_counter() - started,
        "training_performance": (
            json.loads(args.training_throughput.read_text(encoding="utf-8"))
            if args.training_throughput and args.training_throughput.is_file()
            else None
        ),
        "gate_manifest": gate,
        "input_contract": {
            "runtime_dimension": 493,
            "checkpoint_first_weight_shape": [256, 493],
            "hopf_augmentation_role": "loss_only",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    helper.atomic_json(
        payload, args.output_dir / f"{args.split.upper()}_RESULTS.json"
    )
    helper.write_rows(
        main_result["by_re"],
        args.output_dir / f"{args.split.upper()}_RESULTS.csv",
    )
    helper.atomic_json(
        {
            "variant": "b0",
            "model_name": payload["model_name"],
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
            "total_parameters": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
            "runtime_memory_state_per_trajectory": 0,
        },
        args.output_dir / "PARAMETER_COUNT.json",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
