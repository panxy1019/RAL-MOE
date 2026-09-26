#!/usr/bin/env python3
"""Evaluate frozen T2-C gates against non-fused heldout rollouts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from common import ConvexGate, atomic_json, router_probabilities, sha256

HORIZONS = (1, 4, 8, 16, 24)


def errors(alpha: np.ndarray, quadratic: np.ndarray) -> np.ndarray:
    weight = alpha[:, None]
    numerator = (
        weight * weight * quadratic[:, :, 0]
        + (1 - weight) ** 2 * quadratic[:, :, 1]
        + 2 * weight * (1 - weight) * quadratic[:, :, 2]
    )
    return np.sqrt(np.maximum(numerator / np.maximum(quadratic[:, :, 3], 1.0e-12), 0.0))


def metrics(
    alpha: np.ndarray,
    quad_u: np.ndarray,
    quad_p: np.ndarray,
    reynolds: np.ndarray,
) -> dict:
    velocity = errors(alpha, quad_u)
    pressure = errors(alpha, quad_p)
    joint = velocity + pressure
    by_horizon = {}
    for horizon in HORIZONS:
        index = horizon - 1
        by_horizon[f"K{horizon}"] = {
            "velocity_mean": float(np.mean(velocity[:, index])),
            "velocity_worst": float(np.max(velocity[:, index])),
            "pressure_mean": float(np.mean(pressure[:, index])),
            "pressure_worst": float(np.max(pressure[:, index])),
            "joint_mean": float(np.mean(joint[:, index])),
            "joint_worst": float(np.max(joint[:, index])),
        }
    per_re = {}
    for value in np.unique(reynolds):
        mask = reynolds == value
        per_re[str(float(value))] = {
            "K24_velocity_mean": float(np.mean(velocity[mask, 23])),
            "K24_pressure_mean": float(np.mean(pressure[mask, 23])),
            "K24_joint_mean": float(np.mean(joint[mask, 23])),
            "K24_joint_worst": float(np.max(joint[mask, 23])),
            "windows": int(np.sum(mask)),
            "alpha_mean": float(np.mean(alpha[mask])),
            "alpha_min": float(np.min(alpha[mask])),
            "alpha_max": float(np.max(alpha[mask])),
        }
    return {
        "by_horizon": by_horizon,
        "per_Re": per_re,
        "all_step_joint_mean": float(np.mean(joint)),
        "all_step_joint_worst": float(np.max(joint)),
        "alpha_mean": float(np.mean(alpha)),
        "alpha_min": float(np.min(alpha)),
        "alpha_max": float(np.max(alpha)),
    }


def oracle_weights(quad_u: np.ndarray, quad_p: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, 10001, dtype=np.float64)
    output = []
    for velocity, pressure in zip(quad_u, quad_p):
        weight = grid[:, None]
        eu = (
            weight * weight * velocity[None, :, 0]
            + (1 - weight) ** 2 * velocity[None, :, 1]
            + 2 * weight * (1 - weight) * velocity[None, :, 2]
        ) / np.maximum(velocity[None, :, 3], 1.0e-12)
        ep = (
            weight * weight * pressure[None, :, 0]
            + (1 - weight) ** 2 * pressure[None, :, 1]
            + 2 * weight * (1 - weight) * pressure[None, :, 2]
        ) / np.maximum(pressure[None, :, 3], 1.0e-12)
        score = np.mean(np.sqrt(np.maximum(eu, 0)) + np.sqrt(np.maximum(ep, 0)), axis=1)
        output.append(grid[int(np.argmin(score))])
    return np.asarray(output, dtype=np.float32)


def evaluate_boundary(run_root: Path, cache_path: Path, boundary: str) -> dict:
    lower = boundary.lower()
    checkpoint_path = run_root / f"gate_{lower}/best.pt"
    router_path = run_root / "e2_router/best.pt"
    with np.load(cache_path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    if set(np.unique(data["split"])) != {"heldout"}:
        raise RuntimeError(f"{boundary} cache is not heldout-only")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if sha256(router_path) != checkpoint["router_checkpoint_sha256"]:
        raise RuntimeError(f"{boundary} gate/router hash mismatch")
    features = ((data["features"] - checkpoint["feature_mean"]) / checkpoint["feature_std"]).astype(np.float32)
    probabilities = router_probabilities(router_path, data["re"])
    pair_indices = np.asarray(checkpoint["pair_indices"], dtype=int)
    observed_pair = np.asarray(data["pair_indices"][0], dtype=int)
    if not np.array_equal(pair_indices, observed_pair):
        raise RuntimeError(f"{boundary} pair order mismatch: checkpoint={pair_indices}, cache={observed_pair}")
    pair = probabilities[:, pair_indices]
    pair /= pair.sum(axis=1, keepdims=True)
    base = np.clip(pair[:, 0], 1.0e-6, 1 - 1.0e-6)
    base_logit = np.log(base / (1 - base)).astype(np.float32)
    gate = ConvexGate(features.shape[1])
    gate.load_state_dict(checkpoint["gate_state"], strict=True)
    gate.eval()
    with torch.inference_mode():
        learned = gate(torch.as_tensor(features), torch.as_tensor(base_logit)).numpy()
    weights = {
        "candidate_1_only": np.ones(len(data["re"]), dtype=np.float32),
        "candidate_2_only": np.zeros(len(data["re"]), dtype=np.float32),
        "E2_Top1_nonfused": (base >= 0.5).astype(np.float32),
        "fixed_0.5": np.full(len(data["re"]), 0.5, dtype=np.float32),
        "E2_pair_probability_blend": base.astype(np.float32),
        "T2-C": learned,
        "per_window_convex_oracle": oracle_weights(data["quad_u"], data["quad_p"]),
    }
    method_metrics = {
        name: metrics(alpha, data["quad_u"], data["quad_p"], data["re"])
        for name, alpha in weights.items()
    }
    candidate_names = [str(data["source_name"].item()), "Hopf"]
    k24 = {name: row["by_horizon"]["K24"]["joint_mean"] for name, row in method_metrics.items()}
    best_single_name = min(("candidate_1_only", "candidate_2_only"), key=lambda name: k24[name])
    top1 = k24["E2_Top1_nonfused"]
    t2c = k24["T2-C"]
    return {
        "boundary": boundary,
        "candidate_order": candidate_names,
        "heldout_Re": np.unique(data["re"]).astype(float).tolist(),
        "windows": int(len(data["re"])),
        "horizon": int(data["horizon"].item()),
        "cache_path": str(cache_path),
        "cache_sha256": sha256(cache_path),
        "gate_checkpoint": str(checkpoint_path),
        "gate_checkpoint_sha256": sha256(checkpoint_path),
        "router_checkpoint_sha256": sha256(router_path),
        "finite_fraction": {
            candidate_names[0]: float(np.mean(data["candidate_1_finite"])),
            candidate_names[1]: float(np.mean(data["candidate_2_finite"])),
        },
        "divergent_windows": {
            candidate_names[0]: int(np.sum(data["candidate_1_divergent"])),
            candidate_names[1]: int(np.sum(data["candidate_2_divergent"])),
        },
        "methods": method_metrics,
        "K24_summary": {
            "best_single_method": best_single_name,
            "best_single_joint_mean": k24[best_single_name],
            "E2_Top1_joint_mean": top1,
            "T2-C_joint_mean": t2c,
            "T2-C_vs_best_single_relative_change": (t2c - k24[best_single_name]) / k24[best_single_name],
            "T2-C_vs_E2_Top1_relative_change": (t2c - top1) / top1,
            "oracle_joint_mean": k24["per_window_convex_oracle"],
        },
    }


def write_report(result: dict, output: Path) -> None:
    lines = [
        "# CenteredSquare heldout rollout：融合与非融合对比",
        "",
        "- 冻结后机械评估；未重新训练或调参。",
        "- 每个 window 的 T2-C 权重只由初始三帧历史计算一次，K24 内固定。",
        "- 融合只发生在物理输出场，不反馈给任何专家。",
        "",
    ]
    for boundary, row in result["boundaries"].items():
        methods = row["methods"]
        lines.extend(
            [
                f"## {boundary}",
                "",
                f"- heldout Re：`{row['heldout_Re']}`；窗口数：`{row['windows']}`。",
                f"- 候选顺序：candidate 1 = `{row['candidate_order'][0]}`，candidate 2 = `{row['candidate_order'][1]}`。",
                f"- finite：`{row['finite_fraction']}`；divergent：`{row['divergent_windows']}`。",
                "",
                "| 方法 | K1 U | K1 p | K1 joint | K4 joint | K8 joint | K16 joint | K24 U | K24 p | K24 joint |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        order = (
            "candidate_1_only",
            "candidate_2_only",
            "E2_Top1_nonfused",
            "fixed_0.5",
            "E2_pair_probability_blend",
            "T2-C",
            "per_window_convex_oracle",
        )
        for name in order:
            horizon = methods[name]["by_horizon"]
            lines.append(
                f"| {name} | {horizon['K1']['velocity_mean']:.8f} | "
                f"{horizon['K1']['pressure_mean']:.8f} | {horizon['K1']['joint_mean']:.8f} | "
                f"{horizon['K4']['joint_mean']:.8f} | {horizon['K8']['joint_mean']:.8f} | "
                f"{horizon['K16']['joint_mean']:.8f} | {horizon['K24']['velocity_mean']:.8f} | "
                f"{horizon['K24']['pressure_mean']:.8f} | {horizon['K24']['joint_mean']:.8f} |"
            )
        summary = row["K24_summary"]
        lines.extend(
            [
                "",
                f"- T2-C 相对最佳单专家 K24 变化：`{100 * summary['T2-C_vs_best_single_relative_change']:.3f}%`。",
                f"- T2-C 相对 E2 Top-1 K24 变化：`{100 * summary['T2-C_vs_E2_Top1_relative_change']:.3f}%`。",
                "",
                "### K24 逐 Re",
                "",
                "| Re | 方法 | U | p | joint | α mean [min,max] |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for reynolds in row["heldout_Re"]:
            key = str(float(reynolds))
            for name in ("E2_Top1_nonfused", "T2-C"):
                item = methods[name]["per_Re"][key]
                lines.append(
                    f"| {reynolds:g} | {name} | {item['K24_velocity_mean']:.8f} | "
                    f"{item['K24_pressure_mean']:.8f} | {item['K24_joint_mean']:.8f} | "
                    f"{item['alpha_mean']:.4f} [{item['alpha_min']:.4f},{item['alpha_max']:.4f}] |"
                )
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sh-cache", type=Path, required=True)
    parser.add_argument("--hp-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {args.output_dir}")
    result = {
        "schema_version": 1,
        "status": "FROZEN_HELDOUT_EVALUATION_COMPLETE",
        "selection_or_training_after_heldout": False,
        "fusion_feedback": False,
        "boundaries": {
            "SH": evaluate_boundary(args.run_root, args.sh_cache, "SH"),
            "HP": evaluate_boundary(args.run_root, args.hp_cache, "HP"),
        },
    }
    atomic_json(args.output_dir / "HELDOUT_ROLLOUT_COMPARISON.json", result)
    write_report(result, args.output_dir / "HELDOUT_ROLLOUT_COMPARISON.md")
    print(json.dumps({key: value["K24_summary"] for key, value in result["boundaries"].items()}, indent=2))


if __name__ == "__main__":
    main()
