#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from common import ConvexGate, atomic_json, router_probabilities, sha256
from train_t2c_gate import summarize


def oracle_weights(quad_u: np.ndarray, quad_p: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, 10001, dtype=np.float64)
    output = []
    for velocity, pressure in zip(quad_u, quad_p):
        weight = grid[:, None]
        velocity_error = (
            weight * weight * velocity[None, :, 0]
            + (1 - weight) ** 2 * velocity[None, :, 1]
            + 2 * weight * (1 - weight) * velocity[None, :, 2]
        ) / np.maximum(velocity[None, :, 3], 1.0e-12)
        pressure_error = (
            weight * weight * pressure[None, :, 0]
            + (1 - weight) ** 2 * pressure[None, :, 1]
            + 2 * weight * (1 - weight) * pressure[None, :, 2]
        ) / np.maximum(pressure[None, :, 3], 1.0e-12)
        score = np.mean(np.sqrt(np.maximum(velocity_error, 0)) + np.sqrt(np.maximum(pressure_error, 0)), axis=1)
        output.append(grid[int(np.argmin(score))])
    return np.asarray(output, dtype=np.float32)


def boundary_comparison(run_root: Path, boundary: str) -> dict:
    lower = boundary.lower()
    cache_path = run_root / f"cache_{lower}/{lower}_development_cache.npz"
    checkpoint_path = run_root / f"gate_{lower}/best.pt"
    router_path = run_root / "e2_router/best.pt"
    with np.load(cache_path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validation = data["split"] == "validation"
    features = ((data["features"] - checkpoint["feature_mean"]) / checkpoint["feature_std"]).astype(np.float32)
    probabilities = router_probabilities(router_path, data["re"])
    pair = probabilities[:, checkpoint["pair_indices"]]
    pair /= pair.sum(axis=1, keepdims=True)
    base = np.clip(pair[:, 0], 1.0e-6, 1 - 1.0e-6)
    logit = np.log(base / (1 - base)).astype(np.float32)
    gate = ConvexGate(features.shape[1])
    gate.load_state_dict(checkpoint["gate_state"], strict=True)
    gate.eval()
    with torch.inference_mode():
        gate_alpha = gate(torch.as_tensor(features), torch.as_tensor(logit)).numpy()
    weights = {
        "candidate_1_only": np.ones(len(data["re"]), dtype=np.float32),
        "candidate_2_only": np.zeros(len(data["re"]), dtype=np.float32),
        "fixed_0.5": np.full(len(data["re"]), 0.5, dtype=np.float32),
        "E2_pair_probability_blend": base.astype(np.float32),
        "T2-C": gate_alpha,
    }
    comparison = {
        name: summarize(
            alpha[validation],
            data["quad_u"][validation],
            data["quad_p"][validation],
            data["re"][validation],
        )
        for name, alpha in weights.items()
    }
    oracle = oracle_weights(data["quad_u"][validation], data["quad_p"][validation])
    comparison["per_window_convex_oracle"] = summarize(
        oracle,
        data["quad_u"][validation],
        data["quad_p"][validation],
        data["re"][validation],
    )
    result = {
        "boundary": boundary,
        "selection_data": "validation_only",
        "heldout_fields_or_metrics_access": False,
        "cache_sha256": sha256(cache_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "methods": comparison,
    }
    atomic_json(run_root / f"gate_{lower}/VALIDATION_COMPARISON.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pipeline-log", type=Path)
    args = parser.parse_args()
    e2_status = json.loads((args.run_root / "e2_router/runtime_status.json").read_text())
    sh_preflight = json.loads((args.run_root / "cache_sh/PREFLIGHT.json").read_text())
    hp_preflight = json.loads((args.run_root / "cache_hp/PREFLIGHT.json").read_text())
    sh_status = json.loads((args.run_root / "gate_sh/runtime_status.json").read_text())
    hp_status = json.loads((args.run_root / "gate_hp/runtime_status.json").read_text())
    required = [
        e2_status["status"] == "complete",
        sh_preflight["status"] == "PASS",
        hp_preflight["status"] == "PASS",
        sh_status["status"] == "complete",
        hp_status["status"] == "complete",
    ]
    if not all(required):
        raise RuntimeError("run is not complete")
    comparisons = {
        "SH": boundary_comparison(args.run_root, "SH"),
        "HP": boundary_comparison(args.run_root, "HP"),
    }
    urls = []
    if args.pipeline_log is not None:
        urls.extend(re.findall(r"https://swanlab\.cn/[^\s]+/runs/[a-zA-Z0-9]+", args.pipeline_log.read_text()))
    for run_dir in args.run_root.glob("*/swanlog/run-*"):
        run_id = run_dir.name.rsplit("-", 1)[-1]
        urls.append(f"https://swanlab.cn/@panxy1019/FluidicPinball_Fusion_E2_T2C/runs/{run_id}")
    urls = sorted(set(urls))
    inventory = {}
    for relative in (
        "e2_router/best.pt",
        "cache_sh/sh_development_cache.npz",
        "gate_sh/best.pt",
        "cache_hp/hp_development_cache.npz",
        "gate_hp/best.pt",
    ):
        path = args.run_root / relative
        inventory[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    summary = {
        "schema_version": 1,
        "status": "DEVELOPMENT_TRAINING_COMPLETE",
        "method": "E2_ReOnly_TrajectoryTop1 + pair-specific T2-C",
        "horizon": 24,
        "heldout_fields_or_metrics_access": False,
        "specialists_frozen": True,
        "fusion_feedback": False,
        "e2": e2_status,
        "preflight": {"SH": sh_preflight, "HP": hp_preflight},
        "gates": {"SH": sh_status, "HP": hp_status},
        "validation_comparison": comparisons,
        "swanlab_runs": urls,
        "artifact_inventory": inventory,
        "known_limitation": (
            "Steady validation includes a known low-Re local failure and Hopf includes a known Re=19.9 "
            "pressure failure; this fusion run reports boundary performance without hiding those limits."
        ),
        "known_prior_disclosure": (
            "Heldout Re and prior specialist metrics were already disclosed in the user-provided reports. "
            "This strict run itself did not load heldout physical fields or compute heldout metrics."
        ),
    }
    atomic_json(args.run_root / "FINAL_FUSION_SUMMARY.json", summary)
    lines = [
        "# Fluidic Pinball E2 + T2-C 融合训练报告",
        "",
        "- 状态：development 训练完成；heldout 物理场与指标未访问。",
        "- 合同：trajectory/query-level E2；相邻边界独立；K24 固定窗口权重；只做物理输出凸融合；无反馈。",
        f"- E2：完成，validation 选择 step `{e2_status['best_step']}`。",
        f"- S–H preflight：`{sh_preflight['samples']}` 窗口，finite=100%，divergent=0。",
        f"- S–H gate：best step `{sh_status['best_step']}`，validation joint mean `{sh_status['joint_mean']:.8f}`。",
        f"- H–P preflight：`{hp_preflight['samples']}` 窗口，finite=100%，divergent=0。",
        f"- H–P gate：best step `{hp_status['best_step']}`，validation joint mean `{hp_status['joint_mean']:.8f}`。",
        "",
        "## Validation 对比",
        "",
        "| 边界 | candidate 1 | candidate 2 | fixed 0.5 | E2 blend | T2-C | per-window oracle |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for boundary in ("SH", "HP"):
        methods = comparisons[boundary]["methods"]
        lines.append(
            f"| {boundary} | {methods['candidate_1_only']['joint_mean']:.8f} | "
            f"{methods['candidate_2_only']['joint_mean']:.8f} | "
            f"{methods['fixed_0.5']['joint_mean']:.8f} | "
            f"{methods['E2_pair_probability_blend']['joint_mean']:.8f} | "
            f"{methods['T2-C']['joint_mean']:.8f} | "
            f"{methods['per_window_convex_oracle']['joint_mean']:.8f} |"
        )
    lines.extend(["", "## SwanLab", ""])
    lines.extend(f"- {url}" for url in urls)
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 本报告只冻结 development 训练与 validation 选择，不包含 final-test 结论。",
            "- Steady 的低 Re 局部失效与 Hopf 的 Re=19.9 压力失效仍按专家报告保留，不由边界均值掩盖。",
            "- S–H 与 H–P 都是本 Pinball 数据集上的独立认证执行，不继承 CenteredSquare 的数值结论。",
            "- 用户提供的既有报告已披露 heldout Re 与专家指标；本 strict run 自身未加载 heldout 物理场或计算 heldout 指标。",
            "",
        ]
    )
    (args.run_root / "FINAL_FUSION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
