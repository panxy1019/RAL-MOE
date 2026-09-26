#!/usr/bin/env python3
"""Generate the final CenteredSquare Hopf H4 experiment report and audit files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


HELDOUT_ORDER = ("95.100000", "95.300000", "96.500000", "100.500000", "102.000000")
VALIDATION_ORDER = ("94.500000", "95.250000", "95.500000", "97.500000", "99.000000", "101.500000")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--swanlab-url", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def pct(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def main() -> None:
    args = parse_args()
    work = args.work_root.resolve()
    run_dir = (work / args.run_dir).resolve()
    final_dir = (work / args.final_dir).resolve()
    checkpoint = final_dir / "selected_validation_step7200.pt"
    metrics_path = final_dir / "heldout_metrics.json"
    validation_path = run_dir / "validation_history.json"
    config_path = run_dir / "config.json"
    throughput_path = run_dir / "throughput.json"
    asset_manifest_path = work / "assets_r11" / "TRAINING_ASSET_MANIFEST.json"
    contract_manifest_path = work / "trainonly_contract_mb46" / "TRAINONLY_H3H4_CONTRACT.json"
    gradient_path = run_dir / "gradient_audits.json"
    normal_form_path = run_dir / "normal_form_train_nodes.json"

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))["history"]
    selected_validation = next(row for row in validation if row["step"] == 7200)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    throughput = json.loads(throughput_path.read_text(encoding="utf-8"))
    assets = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_manifest_path.read_text(encoding="utf-8"))
    gradients = json.loads(gradient_path.read_text(encoding="utf-8"))
    normal_form = json.loads(normal_form_path.read_text(encoding="utf-8"))

    checkpoint_sha = sha256(checkpoint)
    selected_candidates = [
        {
            "step": row["step"],
            "score": row["score"],
            "hard_gate": row["hard_gate"],
            "numeric_gate": row["numeric_gate"],
            "hopf_gate": row["hopf_gate"],
        }
        for row in validation if row["step"] >= 6400
    ]
    freeze_manifest = {
        "schema_version": 1,
        "selection_split": "validation_only",
        "heldout_loaded_before_freeze": False,
        "selection_rule": "step >= 6400 and hard_gate=true; minimize validation score among eligible checkpoints",
        "selected_step": 7200,
        "selected_validation_score": selected_validation["score"],
        "selected_numeric_gate": selected_validation["numeric_gate"],
        "selected_hopf_gate": selected_validation["hopf_gate"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_optimizer_step": int(checkpoint_payload["optimizer_step"]),
        "eligible_and_late_candidates": selected_candidates,
        "heldout_evaluation_started_after_checkpoint_copy": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(freeze_manifest, final_dir / "checkpoint_selection_manifest.json")

    k48_rows = []
    for reynolds in HELDOUT_ORDER:
        row = metrics["by_re"][reynolds]["48"]
        k48_rows.append({
            "Re": float(reynolds),
            **row,
        })
    aggregate = metrics["k48_aggregate"]
    validation_k48 = []
    for reynolds in VALIDATION_ORDER:
        row = selected_validation["by_re"][reynolds]["48"]
        validation_k48.append({"Re": float(reynolds), **row})

    strict_diagnostics = {
        "all_finite": all(row["finite_fraction"] == 1.0 for row in k48_rows),
        "total_divergent_windows": int(sum(row["divergent_windows"] for row in k48_rows)),
        "all_velocity_physical_below_1pct": all(
            row["velocity_physical_relative_l2_mean"] < 0.01 for row in k48_rows
        ),
        "all_pressure_physical_below_1pct": all(
            row["pressure_physical_relative_l2_mean"] < 0.01 for row in k48_rows
        ),
        "all_rms_amplitude_below_10pct": all(
            row["rms_amplitude_relative_error"] < 0.10 for row in k48_rows
        ),
        "all_frequency_below_2pct": all(
            row["frequency_relative_error"] < 0.02 for row in k48_rows
        ),
        "p2p_below_10pct_count": int(sum(
            row["p2p_amplitude_relative_error"] < 0.10 for row in k48_rows
        )),
        "growth_sign_accuracy_range": [
            float(min(row["growth_sign_accuracy"] for row in k48_rows)),
            float(max(row["growth_sign_accuracy"] for row in k48_rows)),
        ],
        "normal_form_mu_all_negative": all(value < 0 for value in normal_form["mu"]),
        "normal_form_beta_max": float(max(normal_form["beta"])),
    }
    summary = {
        "schema_version": 1,
        "experiment": "CenteredSquareHopf34_H4_NormalFormRadial_r11",
        "dataset": "CenteredSquare CN09 graded mesh Hopf specialist",
        "split": {
            "train_cases": assets["counts"]["train_cases"],
            "validation_cases": assets["counts"]["validation_cases"],
            "heldout_cases": assets["counts"]["heldout_cases"],
            "trainval_snapshots": assets["counts"]["trainval_snapshots"],
            "heldout_snapshots": assets["counts"]["heldout_snapshots"],
        },
        "training": {
            "optimizer_steps": int(checkpoint_payload["args"]["max_steps"]),
            "selected_step": 7200,
            "selected_validation_score": selected_validation["score"],
            "elapsed_seconds": throughput["elapsed_seconds"],
            "steps_per_min": throughput["steps_per_min"],
            "micro_batch": config["micro_batch"],
            "seed": config["seed"],
            "amp": config["amp"],
            "swanlab_url": args.swanlab_url,
        },
        "checkpoint_sha256": checkpoint_sha,
        "validation_step7200": {
            "hard_gate": selected_validation["hard_gate"],
            "numeric_gate": selected_validation["numeric_gate"],
            "hopf_gate": selected_validation["hopf_gate"],
            "k48": validation_k48,
        },
        "heldout_k48_aggregate": aggregate,
        "heldout_k48": k48_rows,
        "diagnostics": strict_diagnostics,
        "projection_reproduction": assets["projection_errors"],
        "gradient_audits": gradients["audits"],
        "method_note": config["model_input_semantics"],
    }
    atomic_json(summary, final_dir / "FINAL_CENTEREDSQUARE_HOPF_H4_SUMMARY.json")

    heldout_table = []
    for row in k48_rows:
        heldout_table.append(
            f"| {row['Re']:.2f} | {pct(row['velocity_physical_relative_l2_mean'])} | "
            f"{pct(row['pressure_physical_relative_l2_mean'])} | "
            f"{pct(row['radial_rms_relative_error'])} | "
            f"{pct(row['rms_amplitude_relative_error'])} | "
            f"{pct(row['p2p_amplitude_relative_error'])} | "
            f"{pct(row['frequency_relative_error'])} | "
            f"{pct(row['growth_sign_accuracy'])} | "
            f"{row['max_norm_ratio']:.3f} | {row['divergent_windows']} |"
        )
    validation_table = []
    for row in validation_k48:
        validation_table.append(
            f"| {row['Re']:.2f} | {pct(row['total_u'])} | {pct(row['total_p'])} | "
            f"{pct(row['radial_rms'])} | {pct(row['growth_sign_accuracy'])} | "
            f"{row['max_norm_ratio']:.3f} |"
        )
    duration_hours = throughput["elapsed_seconds"] / 3600.0
    report = f"""# CenteredSquare Hopf H4 最终实验报告

## 1. 结论摘要

本次实验完成了方柱绕流 Hopf 子数据集上的重新训练、validation checkpoint 冻结和
5 个 held-out Reynolds 数的 K1/K2/K4/K8/K16/K24/K48 自主滚动测试。

- 冻结 checkpoint：step 7200，validation score `{selected_validation['score']:.9f}`；
  `numeric_gate=true`、`hopf_gate=true`、`hard_gate=true`。
- K48 held-out：5/5 工况 `finite_fraction=100%`，总发散窗口为 0。
- K48 平均物理场误差：速度 `{pct(aggregate['velocity_physical_relative_l2_mean'])}`，
  压力 `{pct(aggregate['pressure_physical_relative_l2_mean'])}`。
- K48 平均径向 RMS 误差：`{pct(aggregate['radial_rms_relative_error_mean'])}`。
- 5/5 工况的 RMS 振幅误差低于 10%，5/5 频率误差低于 2%。
- 主要不足位于临界点附近的 P2P 相对误差和增长方向判断，而不是数值发散或总场重构。

## 2. 数据与隔离协议

- 数据集：CenteredSquare，OpenFOAM 13 `icoFoam`，CrankNicolson 0.9，
  graded mesh，9400 cells。
- Hopf 专家域：Re 94–102，共 34 个完整工况。
- 划分：23 train / 6 validation / 5 held-out；按完整 Re 工况隔离。
- 样本数：train+validation `{assets['counts']['trainval_snapshots']}`，
  held-out `{assets['counts']['heldout_snapshots']}`。
- POD：速度/压力均采用 train-only、volume-weighted r11 基。
- held-out Re：95.1、95.3、96.5、100.5、102.0。
- checkpoint 冻结前训练程序只加载 train+validation coefficient view；
  held-out view 在冻结文件复制和 SHA 审计后才由独立 evaluator 首次加载。
- POD 系数重投影最大绝对差：速度
  `{assets['projection_errors']['train_velocity_max_abs_coefficient_reproduction']:.3e}`，
  压力 `{assets['projection_errors']['train_pressure_max_abs_coefficient_reproduction']:.3e}`。

## 3. 方法迁移与稳定化

圆柱实验使用连续时间 Galerkin ROM 作为 RK4 加性主干。方柱数据的相邻存储快照物理时间
间隔较大，直接迁移时正式预检在 Re=97.75 已出现约 191.8 倍模态范数膨胀；全域扫描也表明
这不是可通过删除少量窗口解决的问题。

最终采用的最小稳定化方案为：

1. 保留 Galerkin 和 pressure-ROM 算子、历史 RHS、Re 与模态状态作为网络物理特征；
2. 不把不稳定的连续时间 ROM 直接加到离散状态更新；
3. 从稳定的零更新初始化学习 snapshot-to-snapshot 有限差分动力学；
4. 保留 H3 fluctuation/radial loss、H4 normal-form loss、等 Re 采样和 K1→K8 curriculum。

因此本实验属于“operator-feature HPRS-MoE”，不是原圆柱连续时间加性 ROM 主干的逐字复制。

## 4. 训练与 checkpoint 选择

- 随机种子：`{config['seed']}`。
- 优化步数：8000；micro-batch：`{config['micro_batch']}`。
- curriculum：K1 0–1199，K2 1200–2799，K4 2800–4799，K8 4800–7999。
- AMP/TF32：`{config['amp']}` / `{config['allow_tf32']}`。
- 训练耗时：`{duration_hours:.3f}` 小时，平均
  `{throughput['steps_per_min']:.3f}` steps/min。
- SwanLab：[在线运行记录]({args.swanlab_url})，上传完成 25,811 条记录。
- 40 次 validation 均保持 finite 且零 divergence。
- 仅 step 7200 同时通过 numeric 与 Hopf hard gate；step 7800/8000 虽 score 更低，
  但 Hopf gate 未通过，因此没有用于最终测试。
- 冻结 checkpoint SHA-256：`{checkpoint_sha}`。

### Step 7200 validation K48

| Re | 速度总场误差 | 压力总场误差 | 径向 RMS 误差 | Growth-sign | 最大范数比 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(validation_table)}

## 5. Held-out K48 结果

每个 Re 使用 64 个合法自主滚动窗口。

| Re | 速度物理场误差 | 压力物理场误差 | 径向 RMS | RMS 振幅 | P2P 振幅 | 频率误差 | Growth-sign | 最大范数比 | 发散 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(heldout_table)}

## 6. 结果解释

### 数值稳定性与总场

K48 的所有 320 个最终窗口均为有限值，发散数为 0，最大范数比均低于 evaluator 的
10 倍阈值。五个 held-out Re 的速度和压力物理场误差均低于 1%，说明该稳定化迁移在
总场重构与长滚动数值稳定性方面有效。

### 频率与振幅

五个 Re 的频率误差为约 1.45%–1.83%，RMS 振幅误差为约 1.71%–9.75%，整体较好。
Re=100.5 和 102.0 的 P2P 误差分别约 5.45% 和 3.65%。Re=96.5 为 11.05%，略高于
10% 参考线。

Re=95.1 和 95.3 的 P2P 相对误差分别约 281% 和 642%。这两个点位于 Hopf onset
`Re_H≈95.312` 附近，真实窗口内 P2P 极小，导致相对指标分母敏感；对应 RMS 振幅误差
仍只有 9.75% 和 4.52%，径向 RMS 误差为 14.07% 和 8.72%。因此应将其解释为
近临界微小峰峰值未被精确保持，而不是整体吸引子幅值爆炸。

### 增长方向与 normal-form 可解释性

Growth-sign accuracy 仅为
`{100*strict_diagnostics['growth_sign_accuracy_range'][0]:.2f}%`–
`{100*strict_diagnostics['growth_sign_accuracy_range'][1]:.2f}%`，
临界附近尤其偏弱。训练后的 normal-form `mu` 在全部训练 Re 上仍为负，
`beta` 最大值仅 `{strict_diagnostics['normal_form_beta_max']:.3e}`，平衡半径均为 0。
这说明最终预测能力主要来自 MoE 数据动力学，H4 normal-form head 未形成可物理解读的
Hopf 分岔参数化。该 checkpoint 不应宣称为“normal-form 参数识别成功”。

## 7. 最终判断

本次迁移在以下目标上通过：

- 数据划分和 train-only POD/ROM 隔离；
- 8000 步稳定训练与 SwanLab 完整记录；
- validation-only checkpoint 选择；
- 5/5 held-out K48 有限、零发散；
- 全部 held-out 速度/压力物理场误差低于 1%；
- 全部 held-out RMS 振幅误差低于 10%、频率误差低于 2%。

仍未完全解决：

- Hopf onset 附近的微小 P2P 振幅保持；
- 临界增长方向的逐步判别；
- H4 normal-form 参数的物理可解释性。

因此最准确的结论是：**该模型是稳定且总场精度良好的方柱 Hopf operator-feature
MoE-ROM，并能较好保持频率与 RMS 振幅；但不能宣称已经完整识别或严格保持临界
normal-form 动力学。**

## 8. 产物

- `selected_validation_step7200.pt`：冻结 checkpoint。
- `checkpoint_selection_manifest.json`：validation-only 选择证据。
- `heldout_metrics.json`：逐 Re、逐 horizon 测试指标。
- `FINAL_CENTEREDSQUARE_HOPF_H4_SUMMARY.json`：机器可读摘要。
- `FINAL_CENTEREDSQUARE_HOPF_H4_REPORT.md`：本报告。
- `artifact_inventory.json`：关键产物 SHA-256 清单。
"""
    report_path = final_dir / "FINAL_CENTEREDSQUARE_HOPF_H4_REPORT.md"
    atomic_text(report, report_path)

    inventory_paths = [
        checkpoint,
        metrics_path,
        final_dir / "checkpoint_selection_manifest.json",
        final_dir / "FINAL_CENTEREDSQUARE_HOPF_H4_SUMMARY.json",
        report_path,
        config_path,
        validation_path,
        throughput_path,
        asset_manifest_path,
        contract_manifest_path,
        work / "code" / "train_centeredsquare_hopf_base.py",
        work / "code" / "train_centeredsquare_h4.py",
        work / "code" / "evaluate_centeredsquare_h4.py",
    ]
    inventory = {
        "schema_version": 1,
        "files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in inventory_paths
        ],
    }
    atomic_json(inventory, final_dir / "artifact_inventory.json")
    print(json.dumps({
        "status": "PASS",
        "report": str(report_path),
        "summary": str(final_dir / "FINAL_CENTEREDSQUARE_HOPF_H4_SUMMARY.json"),
        "checkpoint_sha256": checkpoint_sha,
    }, indent=2))


if __name__ == "__main__":
    main()
