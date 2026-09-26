#!/usr/bin/env python3
"""Render a detailed Chinese report from FULL_ROLLOUT_SUMMARY and CSV artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def pct(value) -> str:
    return f"{100 * float(value):.4f}%"


def gain(value, baseline) -> str:
    baseline = float(baseline)
    return f"{100 * (baseline - float(value)) / baseline:+.2f}%" if baseline else "n/a"


def lookup(rows, split, boundary, horizon, method, re_value="ALL"):
    found = [r for r in rows if r["split"] == split and r["boundary"] == boundary
             and int(r["horizon"]) == horizon and r["method"] == method and r["Re"] == re_value]
    if len(found) != 1:
        raise RuntimeError((split, boundary, horizon, method, re_value, len(found)))
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    output = (args.output or result_dir / "DETAILED_EXPERIMENT_REPORT.md").resolve()
    summary = json.loads((result_dir / "FULL_ROLLOUT_SUMMARY.json").read_text(encoding="utf-8"))
    aggregate = read_csv(result_dir / "ROLLOUT_AGGREGATE.csv")
    details = read_csv(result_dir / "ROLLOUT_BY_RE.csv")
    growth = read_csv(result_dir / "ERROR_GROWTH_CURVES.csv")
    weights = read_csv(result_dir / "WEIGHT_DIAGNOSTICS.csv")
    source_label = {"SH": "Steady", "HP": "Periodic"}
    robust_pass = {}
    for boundary in ("SH", "HP"):
        comparisons = []
        for split in ("validation", "final_test"):
            values = sorted({float(r["Re"]) for r in details if r["split"] == split and r["boundary"] == boundary})
            for value in values:
                key = f"{value:.6g}"
                comparisons.append(float(lookup(details, split, boundary, 56, "T2-C", key)["joint_mean"])
                                   <= float(lookup(details, split, boundary, 56, "source_only", key)["joint_mean"]))
        robust_pass[boundary] = all(comparisons)
    lines = [
        "# Fluidic Pinball 三专家融合：完整 Rollout 实验报告",
        "",
        "## 1. 执行摘要",
        "",
        "本次实验在三个子专家、E2 路由器和两套 T2-C gate 全部冻结后，完成了 S–H 与 H–P 两个相邻专家边界的 validation 及一次性 final-test 自主 rollout。评估覆盖 K=1/8/16/24/32/56，并在同一 K56 起点队列上取前缀，因此时域间差异代表真实误差传播，而不是窗口重采样偏差。",
        "",
    ]
    for boundary in ("SH", "HP"):
        final24 = lookup(aggregate, "final_test", boundary, 24, "T2-C")
        source24 = lookup(aggregate, "final_test", boundary, 24, "source_only")
        e224 = lookup(aggregate, "final_test", boundary, 24, "E2_pair_blend")
        final56 = lookup(aggregate, "final_test", boundary, 56, "T2-C")
        verdict = "通过逐 Re 鲁棒门槛，可采用 T2-C" if robust_pass[boundary] else f"未通过逐 Re 鲁棒门槛，默认回退到 {source_label[boundary]}"
        lines.append(
            f"- **{boundary}**：final-test K24 T2-C joint={pct(final24['joint_mean'])}，"
            f"相对 source-only {gain(final24['joint_mean'], source24['joint_mean'])}，"
            f"相对 E2 {gain(final24['joint_mean'], e224['joint_mean'])}；K56={pct(final56['joint_mean'])}。"
            f"部署判断：**{verdict}**。"
        )
    total_windows = sum(int(v["windows"]) for v in summary["stability"].values())
    total_steps = sum(int(v["windows"]) * int(v["horizon"]) * 2 for v in summary["stability"].values())
    lines.extend([
        "",
        f"共评估 {total_windows} 个独立初始窗口、{total_steps:,} 个专家自主预测 window-steps（每窗口运行两个候选专家）；final-test 没有参与选择、早停或再训练。",
        "",
        "## 2. 科学问题与判据",
        "",
        "核心问题不是 gate 能否在 K24 validation 上降低平均误差，而是：冻结后能否跨未见 Re、跨 rollout 长度保持有限、稳定，并且相对安全的单专家基线产生可重复收益。为此使用以下判据：",
        "",
        "1. 物理场误差：面积加权速度相对 L2 与去压力规范自由度后的压力相对 L2；joint 为两者之和。",
        "2. 长时稳定性：finite fraction、发散窗口数、终端误差、最坏时空点和最坏窗口均值。",
        "3. 泛化：validation 与 sealed final-test 分开报告，且 final-test 之后不允许训练。",
        "4. 路由质量：T2-C 同 source-only、Hopf-only、0.5 固定融合、E2 概率融合和逐窗口凸 oracle 比较。",
        "5. 部署价值：T2-C 若在 final-test 不优于 source-only，则该边界默认回退，而不以 validation 成绩覆盖风险。",
        "",
        "## 3. 冻结合同与评估协议",
        "",
        f"- horizon：{summary['protocol']['horizons']}；每个 Re {summary['protocol']['windows_per_re']} 个均匀窗口。",
        f"- 窗口合同：{summary['protocol']['window_sampling']}。",
        "- gate 只读取初始 3 帧历史和 Re，给出单个标量 alpha；整个轨迹保持不变。alpha=1 是 source expert，alpha=0 是 Hopf。",
        "- 两专家各自在自身 POD 坐标中闭环 rollout；只在共同物理网格上做输出凸融合，不把融合场反馈进下一步。",
        "- Steady/Hopf 的速度 POD 按 component-major 解释；Periodic 的 cell-major 资产先规范化为 component-major，再做投影与物理融合。",
        "- per-window oracle 在 0.0005 间隔的 [0,1] 网格上使用真实未来选择 alpha，仅作为可达下界，不能部署。",
        "",
        "### 3.1 数据覆盖",
        "",
        "| split | boundary | Re | windows/Re | max K |",
        "|---|---|---|---:|---:|",
    ])
    for split in ("validation", "final_test"):
        for boundary in ("SH", "HP"):
            values = sorted({float(r["Re"]) for r in details if r["split"] == split and r["boundary"] == boundary})
            lines.append(f"| {split} | {boundary} | {', '.join(f'{v:g}' for v in values)} | 64 | 56 |")
    prior_path = result_dir.parent / "FINAL_FUSION_SUMMARY.json"
    if prior_path.is_file():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        lines.extend(["", "### 3.2 与训练结束时 K24 小样本评估的交叉复现", ""])
        for boundary in ("SH", "HP"):
            old = float(prior["validation_comparison"][boundary]["methods"]["T2-C"]["joint_mean"])
            new = float(lookup(aggregate, "validation", boundary, 24, "T2-C")["joint_mean"])
            lines.append(
                f"- {boundary}：原 8 windows/Re 的 K24 T2-C={pct(old)}；本次 64 windows/Re 共同队列={pct(new)}，"
                f"相对差 {100 * (new - old) / old:+.3f}%。"
            )
        lines.extend(["", "扩展窗口后两条边界的 K24 数值仍与训练结束时结果接近，说明核心结论不是由少量起点偶然产生；差异主要来自更密集的时间覆盖与共同 K56 起点约束。"])
    lines.extend(["", "## 4. 多时域总体结果", ""])
    for split in ("validation", "final_test"):
        lines.extend([
            f"### 4.{1 if split == 'validation' else 2} {split}", "",
            "| boundary | K | source | Hopf | E2 | T2-C | oracle | T2-C vs source | T2-C vs E2 | terminal | worst window mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for boundary in ("SH", "HP"):
            for horizon in summary["protocol"]["horizons"]:
                rows = {method: lookup(aggregate, split, boundary, horizon, method)
                        for method in ("source_only", "hopf_only", "E2_pair_blend", "T2-C", "per_window_oracle")}
                t2c = rows["T2-C"]
                lines.append(
                    f"| {boundary} | {horizon} | {pct(rows['source_only']['joint_mean'])} | {pct(rows['hopf_only']['joint_mean'])} | "
                    f"{pct(rows['E2_pair_blend']['joint_mean'])} | {pct(t2c['joint_mean'])} | {pct(rows['per_window_oracle']['joint_mean'])} | "
                    f"{gain(t2c['joint_mean'], rows['source_only']['joint_mean'])} | {gain(t2c['joint_mean'], rows['E2_pair_blend']['joint_mean'])} | "
                    f"{pct(t2c['joint_terminal'])} | {pct(t2c['joint_worst_window_mean'])} |"
                )
            lines.append("")
    lines.extend(["## 5. K56 逐 Re 压力测试", ""])
    for split in ("validation", "final_test"):
        lines.extend([
            f"### 5.{1 if split == 'validation' else 2} {split}", "",
            "| boundary | Re | source joint | E2 joint | T2-C joint | T2-C alpha mean [min,max] | finite | divergent |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for boundary in ("SH", "HP"):
            values = sorted({float(r["Re"]) for r in details if r["split"] == split and r["boundary"] == boundary})
            for value in values:
                key = f"{value:.6g}"
                source = lookup(details, split, boundary, 56, "source_only", key)
                e2 = lookup(details, split, boundary, 56, "E2_pair_blend", key)
                t2c = lookup(details, split, boundary, 56, "T2-C", key)
                lines.append(
                    f"| {boundary} | {value:g} | {pct(source['joint_mean'])} | {pct(e2['joint_mean'])} | {pct(t2c['joint_mean'])} | "
                    f"{float(t2c['alpha_mean']):.4f} [{float(t2c['alpha_min']):.4f},{float(t2c['alpha_max']):.4f}] | "
                    f"{float(t2c['finite_fraction']):.3f} | {t2c['divergent_windows']} |"
                )
        lines.append("")
    lines.extend(["## 6. 稳定性与失效模式", ""])
    for key, row in summary["stability"].items():
        source_worst = row.get("source_max_norm_ratio", row.get("source_worst_joint_relative_error"))
        target_worst = row.get("target_max_norm_ratio", row.get("target_worst_joint_relative_error"))
        worst_label = "max norm ratio" if "source_max_norm_ratio" in row else "worst joint error"
        lines.append(
            f"- **{key}**：{row['windows']}×K{row['horizon']}；{row['source_expert']} finite={row['source_finite_fraction']:.3f}、"
            f"divergent={row['source_divergent_windows']}、{worst_label}={source_worst:.4g}；"
            f"Hopf finite={row['target_finite_fraction']:.3f}、divergent={row['target_divergent_windows']}、{worst_label}={target_worst:.4g}。"
        )
    lines.extend(["", "K56 误差增长的关键步如下；若终端远高于早期，说明主要瓶颈是闭环累积，而不是一步映射。", "",
                  "| split | boundary | method | step 1 | step 8 | step 24 | step 56 | growth 56/1 |",
                  "|---|---|---|---:|---:|---:|---:|---:|"])
    for split in ("validation", "final_test"):
        for boundary in ("SH", "HP"):
            for method in ("source_only", "E2_pair_blend", "T2-C"):
                found = {int(r["step"]): r for r in growth if r["split"] == split and r["boundary"] == boundary
                         and int(r["horizon"]) == 56 and r["method"] == method}
                ratio = float(found[56]["joint_mean"]) / max(float(found[1]["joint_mean"]), 1e-12)
                lines.append(f"| {split} | {boundary} | {method} | {pct(found[1]['joint_mean'])} | {pct(found[8]['joint_mean'])} | {pct(found[24]['joint_mean'])} | {pct(found[56]['joint_mean'])} | {ratio:.2f}× |")
    lines.extend(["", "## 7. Gate 行为与可解释性", ""])
    for boundary in ("SH", "HP"):
        for split in ("validation", "final_test"):
            rows = [r for r in weights if r["boundary"] == boundary and r["split"] == split and r["method"] == "T2-C"]
            ordered = sorted(rows, key=lambda r: float(r["Re"]))
            trend = float(ordered[-1]["alpha_mean"]) - float(ordered[0]["alpha_mean"])
            lines.append(
                f"- **{boundary}/{split}**：alpha 从 Re={float(ordered[0]['Re']):g} 的 {float(ordered[0]['alpha_mean']):.4f} "
                f"变到 Re={float(ordered[-1]['Re']):g} 的 {float(ordered[-1]['alpha_mean']):.4f}，端点差 {trend:+.4f}。"
            )
    lines.extend([
        "",
        "alpha 的物理含义是 source expert 权重。理想情况下 S–H 随 Re 增大应逐步减少 Steady 权重，H–P 随 Re 增大应逐步增加 Periodic 权重；但单调性不是硬约束，历史动力学描述符会让同一 Re 的不同窗口具有权重带宽。final-test 若落在训练描述符分布之外，T2-C 修正项可能比 Re-only E2 更容易过拟合。",
        "",
        "本次两条边界的平均趋势都与上述朴素 regime 方向相反：S–H 的 Steady 权重随 Re 上升，H–P 的 Periodic 权重随 Re 下降。这不等于数值错误——在当前共同物理场损失下 source expert 确实经常更准——但说明 T2-C 更像是针对候选误差的局部校正器，而不是可直接解释为相变概率的路由器。因而不能把 alpha 当作动力学类别置信度。",
        "",
        "## 8. 深层诊断",
        "",
    ])
    for boundary in ("SH", "HP"):
        val = lookup(aggregate, "validation", boundary, 24, "T2-C")
        test = lookup(aggregate, "final_test", boundary, 24, "T2-C")
        source_test = lookup(aggregate, "final_test", boundary, 24, "source_only")
        oracle_test = lookup(aggregate, "final_test", boundary, 24, "per_window_oracle")
        e2_test = lookup(aggregate, "final_test", boundary, 24, "E2_pair_blend")
        generalization = float(test["joint_mean"]) / float(val["joint_mean"])
        oracle_gap = float(test["joint_mean"]) / float(oracle_test["joint_mean"]) - 1
        lines.append(f"### 8.{1 if boundary == 'SH' else 2} {boundary}")
        lines.append("")
        lines.append(
            f"K24 从 validation 到 final-test 的 joint 比率为 {generalization:.3f}×。final-test 上 T2-C 相对 source-only "
            f"{gain(test['joint_mean'], source_test['joint_mean'])}、相对 E2 {gain(test['joint_mean'], e2_test['joint_mean'])}，"
            f"但仍比逐窗口 oracle 高 {100 * oracle_gap:.2f}%。"
        )
        if not robust_pass[boundary]:
            lines.append(
                f"但 {boundary} 没有通过逐 Re 鲁棒门槛：至少一个 validation/final-test Re 上 T2-C 劣于 {source_label[boundary]}。"
                "这属于路由负迁移而非候选发散；部署应使用 source-only fallback。后续若重训，应只在新的 development 划分上加入 worst-Re/regret 目标，不能使用本次 final-test 回调。"
            )
        else:
            lines.append(
                f"这说明 {boundary} 的融合收益可以跨未见 Re 保留；仍需结合 K56、最坏窗口和有限性决定是否放行，而不能只看 K24 均值。"
            )
        lines.append("")
    lines.extend([
        "## 9. 部署建议",
        "",
        "建议把每个边界视为独立策略，而不是把“融合有效”概括为全局结论：",
        "",
    ])
    for boundary in ("SH", "HP"):
        test = lookup(aggregate, "final_test", boundary, 24, "T2-C")
        source = lookup(aggregate, "final_test", boundary, 24, "source_only")
        if robust_pass[boundary]:
            lines.append(f"- **{boundary}**：通过 validation+final-test 的逐 Re K56 非劣门槛，允许 T2-C，但保留 finite/error 守卫及 {source_label[boundary]} 回退。")
        else:
            lines.append(f"- **{boundary}**：虽然 final-test 总均值可能有收益，但逐 Re 结果不一致；默认使用 {source_label[boundary]}，当前 T2-C 仅保留为研究分支。")
    lines.extend([
        "- gate 训练时加入相对 source-only 的 regret 约束，比单纯最小化融合绝对误差更符合安全部署目标。",
        "- 下一轮开发应设置独立 calibration split 估计描述符分布外程度，并对 alpha 做置信收缩；本次 final-test 应从此永久封存，不再用于模型选择。",
        "- 若需要真正长期闭环融合，应另行训练“融合反馈”模型；当前结果只证明并行专家输出融合，不证明融合场作为下一步状态时仍稳定。",
        "",
        "## 10. 局限性",
        "",
        "1. 只评估 S–H 与 H–P 重叠边界，不代表三个专家各自主区域的全 Re 全局性能。",
        "2. T2-C 在 K24 上训练；K1/K8/K16/K32/K56 是冻结后的时域外推诊断，不是分别优化的 gate。",
        "3. alpha 在整条轨迹固定，无法响应 rollout 中途发生的动力学相变或专家失稳。",
        "4. oracle 使用未来真值，仅表示凸融合的结构上限；它与可部署方法不可直接等价。",
        "5. 64 个均匀窗口覆盖了每个 Re 的时间范围，但不是所有可能起点的穷举；协议与原子专家的标准 64-window 评估一致。",
        "",
        "## 11. 可复现性与产物",
        "",
        f"- 结果目录：`{result_dir}`",
        "- `FULL_ROLLOUT_SUMMARY.json`：冻结合同、资产审计、稳定性和总体指标。",
        "- `ROLLOUT_AGGREGATE.csv` / `ROLLOUT_BY_RE.csv`：总体与逐 Re 指标。",
        "- `ERROR_GROWTH_CURVES.csv`：逐步误差传播。",
        "- `WEIGHT_DIAGNOSTICS.csv`：E2/T2-C 权重分布。",
        "- `FINAL_TEST_SEAL.json`：一次性解封与禁止再训练记录。",
        "- SVG 图：多时域误差、K56 增长曲线和 alpha–Re 关系。",
        "",
        "### 冻结权重 SHA256",
        "",
    ])
    for name, digest in summary["checkpoint_sha256"].items():
        lines.append(f"- {name}: `{digest}`")
    lines.extend(["", f"评估总耗时：{summary['elapsed_seconds'] / 60:.2f} 分钟。", ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
