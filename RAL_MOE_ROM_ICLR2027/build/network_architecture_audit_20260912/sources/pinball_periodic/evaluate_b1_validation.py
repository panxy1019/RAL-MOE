#!/usr/bin/env python3
"""Split-aware rollout evaluation for Fluidic Pinball V2 B1."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


HORIZONS = (1, 2, 4, 8, 16, 24, 32, 56)
EPS = 1.0e-12


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--baseline-trainer", type=Path, required=True)
    parser.add_argument("--coefficient-view", type=Path, required=True)
    parser.add_argument("--galerkin-path", type=Path, required=True)
    parser.add_argument("--pressure-path", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-view", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "final_test"),
                        default="validation")
    parser.add_argument("--re-values", type=float, nargs="*")
    parser.add_argument("--horizons", type=int, nargs="*", default=list(HORIZONS))
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--all-temporal-offsets", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows-per-re", type=int, default=64)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    K = load_module("fluidic_b1_trainer_eval", args.trainer)
    B = load_module("fluidic_b1_data_eval", args.baseline_trainer)
    contract = K.configure_contract(args.asset_manifest)
    base_args = SimpleNamespace(
        coefficient_view=args.coefficient_view,
        galerkin_path=args.galerkin_path,
        pressure_path=args.pressure_path,
        asset_manifest=args.asset_manifest,
        history_len=3,
        scale_floor_quantile=0.10,
        lambda_pressure_rollout=0.25,
    )
    audit = B.audit_assets(base_args)
    if args.split != "final_test":
        data = B.load_coefficients(base_args)
    else:
        z = np.load(args.evaluation_view, allow_pickle=False)
        a = np.asarray(z["coeff_uv"], dtype=np.float32)
        b = np.asarray(z["coeff_p"], dtype=np.float32)
        re = np.asarray(z["Re"], dtype=np.float64)
        time_values = np.asarray(z["time"], dtype=np.float64)
        order = np.lexsort((time_values, re))
        a, b, re, time_values = a[order], b[order], re[order], time_values[order]
        nxt = np.full(len(re), -1, dtype=np.int64)
        prev = np.full(len(re), -1, dtype=np.int64)
        for value in np.unique(re):
            ids = np.flatnonzero(np.isclose(re, value, atol=5e-7))
            nxt[ids[:-1]], prev[ids[1:]] = ids[1:], ids[:-1]
        hist = np.full((len(re), 3), -1, dtype=np.int64)
        for row in range(len(re)):
            current = row
            for offset in range(3):
                if current < 0:
                    break
                hist[row, offset] = current
                current = int(prev[current])
        valid = np.flatnonzero((nxt >= 0) & np.all(hist >= 0, axis=1))
        data = {
            "a": a, "b": b, "re": re.astype(np.float32),
            "time": time_values.astype(np.float32), "next": nxt, "prev": prev,
            "hist": hist, "valid": valid, "train_ids": np.empty(0, dtype=np.int64),
            "val_ids": valid, "phi_u": np.asarray(z["phi_uv"], dtype=np.float32),
            "phi_p": np.asarray(z["phi_p"], dtype=np.float32),
            "areas": np.asarray(z["point_areas"], dtype=np.float32),
            "mean_u": np.asarray(z["mean_uv_train"], dtype=np.float32),
            "mean_p": np.asarray(z["mean_p_train"], dtype=np.float32),
        }
    heldout = np.asarray(contract["heldout_reynolds"], dtype=np.float64)
    if args.split != "final_test" and np.any(
        np.isclose(data["re"][:, None], heldout[None, :], atol=5e-7)
    ):
        raise RuntimeError("Heldout leakage in B1 train/validation evaluator.")
    if args.split == "final_test" and not np.allclose(
        np.sort(np.unique(data["re"])), np.sort(heldout), atol=5e-7
    ):
        raise RuntimeError("Final-test evaluator did not load exactly the sealed heldout Re list.")
    if args.temporal_stride <= 0:
        raise ValueError("Temporal stride must be a positive integer.")
    if args.temporal_stride > 1:
        selected = []
        chains = []
        for value in np.unique(data["re"]):
            ids = np.flatnonzero(np.isclose(data["re"], value, atol=5e-7))
            offsets = (range(args.temporal_stride) if args.all_temporal_offsets else (0,))
            for offset in offsets:
                chain = ids[offset::args.temporal_stride]
                if len(chain):
                    chains.append(chain)
                    selected.extend(chain.tolist())
        nxt = np.full(len(data["re"]), -1, dtype=np.int64)
        prev = np.full(len(data["re"]), -1, dtype=np.int64)
        for chain in chains:
            nxt[chain[:-1]], prev[chain[1:]] = chain[1:], chain[:-1]
        hist = np.full((len(data["re"]), 3), -1, dtype=np.int64)
        for row in selected:
            current = row
            for offset in range(3):
                if current < 0:
                    break
                hist[row, offset] = current
                current = int(prev[current])
        valid = np.asarray([
            row for row in selected if nxt[row] >= 0 and np.all(hist[row] >= 0)
        ], dtype=np.int64)
        train_re = np.asarray(contract["fit_reynolds"], dtype=np.float64)
        val_re = np.asarray(contract["validation_reynolds"], dtype=np.float64)
        data["next"], data["prev"], data["hist"], data["valid"] = nxt, prev, hist, valid
        data["train_ids"] = valid[np.any(
            np.isclose(data["re"][valid, None], train_re[None, :], atol=5e-7), axis=1
        )]
        data["val_ids"] = valid[np.any(
            np.isclose(data["re"][valid, None], val_re[None, :], atol=5e-7), axis=1
        )] if args.split != "final_test" else valid
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    rom_np = B.load_train_rom(base_args)
    rom = {key: torch.as_tensor(value, device=device) for key, value in rom_np.items()}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["variant"] != "b1" or checkpoint.get("heldout_evaluation_performed") is not False:
        raise RuntimeError("Checkpoint is not a validation-only B1 checkpoint.")
    model = K.DeepFNNH3().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    stats = {key: torch.as_tensor(value, device=device)
             for key, value in checkpoint["norm_stats"].items()}
    rows = []
    split_re_key = {
        "train": "fit_reynolds",
        "validation": "validation_reynolds",
        "final_test": "heldout_reynolds",
    }[args.split]
    available_re = np.asarray(contract[split_re_key], dtype=np.float64)
    evaluation_re = (available_re if not args.re_values
                     else np.asarray(args.re_values, dtype=np.float64))
    if not all(np.any(np.isclose(available_re, value, atol=5e-7))
               for value in evaluation_re):
        raise RuntimeError(f"Requested Re values are not in the {args.split} split.")
    horizons = tuple(dict.fromkeys(args.horizons))
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("Evaluation horizons must be positive integers.")
    split_ids = data["train_ids" if args.split == "train" else "val_ids"]
    for value in evaluation_re:
        ids = split_ids[np.isclose(data["re"][split_ids], value, atol=5e-7)]
        for horizon in horizons:
            starts = B.legal_starts(data, ids, horizon)
            if not len(starts):
                raise RuntimeError(
                    f"No legal {args.split} windows for Re={value}, K={horizon}"
                )
            if len(starts) > args.windows_per_re:
                starts = starts[np.linspace(0, len(starts) - 1, args.windows_per_re, dtype=int)]
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                out = K.rollout(model, "b1", B, data, starts, horizon, rom, stats, device, 16)
                velocity, pressure = K.physical_series(B, data, out, device)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            pred_a = torch.stack(out["pred_a"], dim=1)
            pred_b = torch.stack(out["pred_b"], dim=1)
            true_a = torch.stack(out["true_a"], dim=1)
            true_b = torch.stack(out["true_b"], dim=1)
            finite = torch.isfinite(pred_a).all((1, 2)) & torch.isfinite(pred_b).all((1, 2))
            ratio = torch.maximum(
                torch.linalg.vector_norm(pred_a, dim=2)
                / torch.linalg.vector_norm(true_a, dim=2).clamp_min(EPS),
                torch.linalg.vector_norm(pred_b, dim=2)
                / torch.linalg.vector_norm(true_b, dim=2).clamp_min(EPS),
            )
            joint = velocity + pressure
            row = {
                "Re": float(value), "horizon": horizon, "windows": len(starts),
                "velocity_field_mean": float(velocity.mean()),
                "pressure_field_mean": float(pressure.mean()),
                "joint_field_mean": float(joint.mean()),
                "joint_terminal": float(joint[:, -1].mean()),
                "joint_worst_window": float(joint.max()),
                "finite_fraction": float(finite.float().mean()),
                "divergent_windows": int(((~finite) | (ratio.max(1).values > 10)).sum()),
                "inference_seconds": elapsed,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    aggregate = {}
    for horizon in horizons:
        selected = [row for row in rows if row["horizon"] == horizon]
        aggregate[str(horizon)] = {
            "velocity_field_mean": float(np.mean([row["velocity_field_mean"] for row in selected])),
            "pressure_field_mean": float(np.mean([row["pressure_field_mean"] for row in selected])),
            "joint_field_mean": float(np.mean([row["joint_field_mean"] for row in selected])),
            "joint_terminal": float(np.mean([row["joint_terminal"] for row in selected])),
            "joint_worst_window": float(np.max([row["joint_worst_window"] for row in selected])),
            "finite_fraction_min": float(np.min([row["finite_fraction"] for row in selected])),
            "divergent_windows": int(np.sum([row["divergent_windows"] for row in selected])),
        }
    with (args.output_dir / "ROLLOUT_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "scope": ({
            "train": "frozen_checkpoint_train_diagnostic_no_heldout_loaded",
            "validation": "validation_only_no_heldout_loaded",
            "final_test": "one_time_frozen_final_test",
        }[args.split]),
        "checkpoint": str(args.checkpoint),
        "optimizer_step": int(checkpoint["optimizer_step"]),
        "best_step": int(checkpoint["best_step"]),
        "split": args.split,
        "temporal_stride": args.temporal_stride,
        "all_temporal_offsets": args.all_temporal_offsets,
        "evaluation_re": evaluation_re.tolist(),
        "heldout_hard_disabled": heldout.tolist(),
        "horizons": list(horizons),
        "asset_audit": audit,
        "aggregate": aggregate,
    }
    (args.output_dir / "ROLLOUT_SUMMARY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "evaluation_complete", "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
