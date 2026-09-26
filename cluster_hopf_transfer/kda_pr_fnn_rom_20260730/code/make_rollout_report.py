#!/usr/bin/env python3
"""Build SVG plots and the detailed validation rollout report."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rollout_evaluation_validation"
MODELS = ("b1", "b2", "b3")
LABEL = {
    "b1": "B1 Deep-FNN-H3",
    "b2": "B2 KDA radial-off",
    "b3": "B3 KDA radial-on",
}
COLOR = {"b1": "#315b8a", "b2": "#c94c5d", "b3": "#e29b35"}
HORIZONS = (1, 2, 4, 8, 16, 32, 56)


def read_csv(name):
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pct(x):
    return f"{100*x:.4f}%"


def delta(new, old):
    return 100 * (new / old - 1)


def svg_line(path, title, xlabel, ylabel, x_values, series):
    width, height = 920, 520
    left, right, top, bottom = 90, 30, 60, 70
    pw, ph = width - left - right, height - top - bottom
    all_y = [y for _, values, _ in series for y in values]
    ymin, ymax = min(all_y), max(all_y)
    pad = max((ymax - ymin) * 0.12, ymax * 0.03, 1e-9)
    ymin, ymax = max(0.0, ymin - pad), ymax + pad

    def sx(x):
        return left + (x - min(x_values)) / max(max(x_values) - min(x_values), 1) * pw

    def sy(y):
        return top + (ymax - y) / max(ymax - ymin, 1e-12) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-size="22" '
        f'font-family="Arial">{title}</text>',
    ]
    for i in range(6):
        y = ymin + (ymax - ymin) * i / 5
        yy = sy(y)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left+pw}" y2="{yy:.2f}" '
                     'stroke="#dddddd"/>')
        parts.append(f'<text x="{left-10}" y="{yy+5:.2f}" text-anchor="end" '
                     f'font-size="12" font-family="Arial">{y:.4f}</text>')
    for x in x_values:
        xx = sx(x)
        parts.append(f'<line x1="{xx:.2f}" y1="{top+ph}" x2="{xx:.2f}" '
                     f'y2="{top+ph+6}" stroke="#333"/>')
        parts.append(f'<text x="{xx:.2f}" y="{top+ph+24}" text-anchor="middle" '
                     f'font-size="12" font-family="Arial">{x}</text>')
    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#333"/>',
        f'<text x="{left+pw/2}" y="{height-18}" text-anchor="middle" '
        f'font-size="15" font-family="Arial">{xlabel}</text>',
        f'<text x="22" y="{top+ph/2}" text-anchor="middle" font-size="15" '
        f'font-family="Arial" transform="rotate(-90 22 {top+ph/2})">{ylabel}</text>',
    ])
    for index, (name, values, color) in enumerate(series):
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(x_values, values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                     'stroke-width="3"/>')
        for x, y in zip(x_values, values):
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" '
                         f'fill="{color}"/>')
        lx, ly = left + 20 + index * 245, top + 18
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+28}" y2="{ly}" '
                     f'stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+36}" y="{ly+5}" font-size="13" '
                     f'font-family="Arial">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    summary = json.loads((OUT / "ROLLOUT_SUMMARY.json").read_text(encoding="utf-8"))
    results = read_csv("ROLLOUT_RESULTS.csv")
    growth = read_csv("ERROR_GROWTH_CURVES.csv")
    memory = read_csv("MEMORY_DIAGNOSTICS.csv")
    agg = summary["aggregate"]

    svg_line(
        OUT / "error_growth_by_horizon.svg",
        "Validation rollout error growth", "Rollout horizon K",
        "Mean joint field error (%)", list(HORIZONS),
        [(LABEL[m], [100 * agg[m][str(k)]["joint_field_time_mean"]
                     for k in HORIZONS], COLOR[m]) for m in MODELS],
    )
    by_model_step = defaultdict(lambda: defaultdict(list))
    for row in growth:
        if int(row["horizon"]) == 56:
            by_model_step[row["model"]][int(row["step"])].append(
                float(row["joint_field_mean"]))
    steps = list(range(1, 57))
    svg_line(
        OUT / "k56_error_trajectory.svg",
        "K56 autonomous validation trajectory", "Autonomous step",
        "Mean joint field error (%)", steps,
        [(LABEL[m], [100 * statistics.mean(by_model_step[m][step])
                     for step in steps], COLOR[m]) for m in MODELS],
    )
    by_memory = defaultdict(lambda: defaultdict(list))
    for row in memory:
        for key in ("tau_mean", "alpha_mean", "beta_mean",
                    "memory_effective_rank_mean"):
            by_memory[(row["model"], int(row["head"]))][key].append(float(row[key]))
    heads = [0, 1, 2, 3]
    svg_line(
        OUT / "memory_time_scales.svg",
        "Learned KDA memory time scales", "Memory head", "Mean tau",
        heads,
        [(LABEL[m], [statistics.mean(by_memory[(m, h)]["tau_mean"])
                     for h in heads], COLOR[m]) for m in ("b2", "b3")],
    )
    svg_line(
        OUT / "memory_effective_rank.svg",
        "KDA matrix-memory effective rank", "Memory head", "Effective rank",
        heads,
        [(LABEL[m], [statistics.mean(
            by_memory[(m, h)]["memory_effective_rank_mean"])
                     for h in heads], COLOR[m]) for m in ("b2", "b3")],
    )

    k56 = {m: agg[m]["56"] for m in MODELS}
    per_re = {
        m: sorted(
            [r for r in results if r["model"] == m and int(r["horizon"]) == 56],
            key=lambda r: float(r["Re"]),
        )
        for m in MODELS
    }
    mem_summary = {}
    mem_keys = (
        "tau_mean", "half_life_mean", "alpha_mean", "beta_mean",
        "memory_frobenius_mean", "memory_max_singular_mean",
        "memory_effective_rank_mean", "delta_write_error_mean",
        "query_u_p_cosine_mean",
    )
    for model in ("b2", "b3"):
        selected = [r for r in memory if r["model"] == model]
        mem_summary[model] = {
            key: statistics.mean(float(r[key]) for r in selected)
            for key in mem_keys
        }

    lines = [
        "# CenteredSquare Hopf KDA-PR-FNN-ROM rollout 实验报告",
        "",
        "## 1. 实验目的与结论摘要",
        "",
        "本实验在已经冻结的 CenteredSquare Hopf r11 工程上，严格比较固定三状态深层 FNN "
        "与 KDA 有限状态记忆闭合。评估仅使用六个 validation Reynolds 数；由于 B2 "
        "未达到预先冻结的 B1 对比门禁，held-out test 始终未加载。",
        "",
        "三种模型在 K1–K56 的全部评估窗口上均保持 finite fraction=1、零发散。"
        "但是主方法 B2 在 K56 的平均联合场误差、终端误差和最坏窗口误差上均显著劣于 B1。"
        "B3 的 radial regularization 能改善 B2 的长期误差，但仍不足以超过 B1。"
        "因此当前证据不支持用 KDA 分支替换现有固定三状态 Hopf specialist，也没有证据支持"
        "继续加入 full attention。",
        "",
        "## 2. 数据与冻结合同",
        "",
        "- 数据：CenteredSquare CN09 graded mesh，Hopf specialist。",
        "- POD：速度 `ru=11`，压力 `rp=11`，均仅由 23 个 train Re 拟合。",
        "- Validation Re：94.5、95.25、95.5、97.5、99、101.5。",
        "- Held-out Re：95.1、95.3、96.5、100.5、102；本轮硬禁用。",
        "- 每个 Re 最多选取 64 个合法连续窗口；不得跨轨迹拼接。",
        "- 评估 horizon：K1、K2、K4、K8、K16、K32、K56。",
        "",
        "附件中的 560/192/176/67 维假设不适用于当前方柱 r11 工程。运行时实际合同为："
        "当前块 46 维；每个历史块 66 维；两个历史块后 B1 输入为 178 维。"
        "冻结训练器中声明的 Hopf augmentation helper 并未进入正式前向，故 B2/B3 使用"
        "真实的 46 维 current-only 输入。",
        "",
        "## 3. 方法详细描述",
        "",
        "### 3.1 共同物理与数值合同",
        "",
        "三组模型共享同一 train-only POD、Galerkin 算子、Pressure–Poisson 算子、"
        "归一化统计量、AdamW/余弦学习率、8000 optimizer-step 预算和"
        "K4→K8→K16→K32→K56 curriculum。方柱快照间隔下直接使用连续 Galerkin RK4 "
        "会失稳，因此遵循冻结工程：Galerkin RHS 作为当前物理特征，速度闭合学习有限差分"
        "导数；压力仍使用 Pressure–Poisson 基线、逐模态 sigmoid gate 和代数残差。",
        "",
        "速度采用冻结 memory 的四阶段 RK 更新：",
        "",
        "`a_(n+1) = a_n + dt/6 * (k1 + 2 k2 + 2 k3 + k4)`。",
        "",
        "压力宏步闭合为：",
        "",
        "`b_(n+1) = gate_n ⊙ P_H(a_(n+1), Re) + rho_p,n`。",
        "",
        "### 3.2 B1：Deep-FNN-H3",
        "",
        "B1 保留当前状态与两个历史状态构成的 178 维输入，删除全部 MoE router、shared/routed "
        "expert、Top-k、线性专家和低秩二次专家，改为六层宽度 512 的普通 FNN。"
        "三个输出头分别预测速度导数、压力残差和压力门控。参数量 1,745,797。",
        "",
        "### 3.3 B2：KDA-PR-FNN-ROM radial-off",
        "",
        "B2 删除显式历史拼接，只读取 46 维当前特征。Encoder 将当前特征映射到 128 维 token；"
        "四个 head 各维护一个 16×16 的 float32 动态矩阵，总动态状态为 1024。"
        "对于每个 head，遗忘、写入误差与更新为：",
        "",
        "`alpha = exp(clamp(-dt/tau, -20, 0))`",
        "",
        "`M_bar = Diag(alpha) M_prev`",
        "",
        "`e = v - M_bar^T k`",
        "",
        "`M = M_bar + beta k e^T`。",
        "",
        "其中 `tau` 为逐 key-channel 正时间尺度，`beta∈[0,0.5]` 为逐 head 写入率。"
        "速度与压力使用不同 query 读取同一 memory。每条轨迹以三个真实初始状态按时间顺序"
        "warm-up；自主预测后只用预测状态更新 memory。memory 仅在宏步边界更新一次，四个 RK "
        "stage 内完全冻结，并在 AMP 下保持 float32。B2 不使用 radial loss。"
        "参数量 1,830,621，比 B1 增加 4.86%。",
        "",
        "### 3.4 B3：KDA radial-on",
        "",
        "B3 与 B2 完全相同，仅恢复 train-only 临界平面半径的对数幅值和增长正则。"
        "它用于判断 radial loss 是否能抑制 KDA 长时漂移，而不是主候选。",
        "",
        "## 4. Rollout 指标",
        "",
        "- `velocity_field_time_mean`：面积加权速度物理场相对 L2，在窗口与时间上平均。",
        "- `pressure_field_time_mean`：去规范压力物理场相对 L2，在窗口与时间上平均。",
        "- `joint_field_time_mean`：上述速度和压力误差之和。",
        "- `joint_field_terminal`：每个窗口末端联合场误差的均值。",
        "- `joint_field_worst_window_time`：所有窗口、所有时间中的最大联合场误差。",
        "- `pressure_drift_time_mean`：压力 POD 能量相对漂移。",
        "- 有限性门：finite fraction=1 且 divergent windows=0。",
        "",
        "## 5. 聚合结果",
        "",
        "| 模型 | K56速度均值 | K56压力均值 | K56联合均值 | K56终端 | 最坏窗口 | 压力漂移 | 推理时间* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = k56[model]
        lines.append(
            f"| {LABEL[model]} | {pct(row['velocity_field_time_mean'])} | "
            f"{pct(row['pressure_field_time_mean'])} | "
            f"{pct(row['joint_field_time_mean'])} | "
            f"{pct(row['joint_field_terminal'])} | "
            f"{pct(row['joint_field_worst_window_time'])} | "
            f"{pct(row['pressure_drift_time_mean'])} | "
            f"{row['inference_seconds_total']:.3f}s |"
        )
    lines.extend([
        "",
        "\\* 推理时间是六个 validation Re 分批评估的合计 wall-clock，只用于同机相对比较。",
        "",
        f"B2 相对 B1：联合均值 {delta(k56['b2']['joint_field_time_mean'], k56['b1']['joint_field_time_mean']):+.1f}%，"
        f"终端 {delta(k56['b2']['joint_field_terminal'], k56['b1']['joint_field_terminal']):+.1f}%，"
        f"最坏窗口 {delta(k56['b2']['joint_field_worst_window_time'], k56['b1']['joint_field_worst_window_time']):+.1f}%。",
        "",
        f"B3 相对 B2：联合均值 {delta(k56['b3']['joint_field_time_mean'], k56['b2']['joint_field_time_mean']):+.1f}%，"
        f"终端 {delta(k56['b3']['joint_field_terminal'], k56['b2']['joint_field_terminal']):+.1f}%，"
        f"最坏窗口 {delta(k56['b3']['joint_field_worst_window_time'], k56['b2']['joint_field_worst_window_time']):+.1f}%。",
        "",
        "![Horizon error growth](error_growth_by_horizon.svg)",
        "",
        "![K56 trajectory](k56_error_trajectory.svg)",
        "",
        "## 6. K56 逐 Reynolds 数结果",
        "",
        "| Re | B1联合均值 | B2联合均值 | B3联合均值 | B2/B1变化 | B3/B2变化 |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for i in range(6):
        r1, r2, r3 = per_re["b1"][i], per_re["b2"][i], per_re["b3"][i]
        v1, v2, v3 = (float(r["joint_field_time_mean"]) for r in (r1, r2, r3))
        lines.append(
            f"| {float(r1['Re']):g} | {pct(v1)} | {pct(v2)} | {pct(v3)} | "
            f"{delta(v2, v1):+.1f}% | {delta(v3, v2):+.1f}% |"
        )
    lines.extend([
        "",
        "B2 在 Re=95.25 和 95.5 的中长 horizon 上表现相对接近 B1，但在 Re=97.5、99 "
        "和尤其 101.5 上明显恶化；Re=101.5 的 K56 联合时间均值达到 "
        f"{pct(float(per_re['b2'][-1]['joint_field_time_mean']))}，是同一 Re 下 B1 的 "
        f"{float(per_re['b2'][-1]['joint_field_time_mean'])/float(per_re['b1'][-1]['joint_field_time_mean']):.2f} 倍。"
        "这说明 KDA 的主要问题不是数值爆炸，而是随 Re 变化的闭合偏差。",
        "",
        "## 7. KDA memory 诊断",
        "",
        "| 指标 | B2 radial-off | B3 radial-on |",
        "|---|---:|---:|",
    ])
    names = {
        "tau_mean": "平均 tau",
        "half_life_mean": "平均 half-life",
        "alpha_mean": "平均 alpha",
        "beta_mean": "平均 beta",
        "memory_frobenius_mean": "memory Frobenius norm",
        "memory_max_singular_mean": "最大奇异值",
        "memory_effective_rank_mean": "effective rank",
        "delta_write_error_mean": "delta write error",
        "query_u_p_cosine_mean": "u/p query cosine",
    }
    for key, name in names.items():
        lines.append(
            f"| {name} | {mem_summary['b2'][key]:.6f} | "
            f"{mem_summary['b3'][key]:.6f} |"
        )
    lines.extend([
        "",
        "两组 KDA 的 alpha 均约 0.87–0.90，并非接近 0，说明 memory 没有退化为纯短记忆；"
        "beta 约 0.068–0.074，也未接近 0，因此网络没有完全绕过写入。memory norm 和最大"
        "奇异值保持有限，没有爆炸。effective rank 约 2.9，高于 1，未发生严格 rank-1 "
        "collapse，但相对于 16 维 value 空间仍然较低，表明有效记忆子空间较窄。"
        "平均物理 half-life 约 34–36 个时间单位，确实形成了超过三状态窗口的长记忆，"
        "但该记忆并未转化为更好的 validation 泛化。",
        "",
        "![Memory time scales](memory_time_scales.svg)",
        "",
        "![Memory effective rank](memory_effective_rank.svg)",
        "",
        "## 8. 计算成本",
        "",
        "- B1：1,745,797 参数；训练 66.5 分钟；K56 六 Re 推理 2.384 秒。",
        "- B2/B3：各 1,830,621 参数；训练约 94–97 分钟；K56 六 Re 推理约 3.38 秒。",
        f"- B2 相对 B1 的 K56 推理 wall-clock 增加约 "
        f"{delta(k56['b2']['inference_seconds_total'], k56['b1']['inference_seconds_total']):+.1f}%。",
        "- KDA 额外动态状态为每条轨迹 1024 个 float32 数，不属于模型参数。",
        "",
        "## 9. 预设门禁与科学判断",
        "",
        "预设门禁要求 B2 相对 B1 至少在 mean joint、terminal 或 worst-window 中一项改善 "
        "10%，且速度或压力均值不得恶化超过 5%。B2 三项均恶化，因此明确失败。"
        "按照冻结协议，不生成 TEST_RESULTS.csv，不访问 held-out test，也不允许事后调整阈值。",
        "",
        "对原始问题的逐项回答：",
        "",
        "1. 删除 MoE、使用深层 FNN：B1 在本次 validation rollout 上稳定且非常准确，"
        "说明普通 FNN 是有效的轻量替代候选。",
        "2. 相同 FNN 下 KDA 是否优于固定三状态历史：否，B2 明显更差。",
        "3. KDA 是否改善长期速度误差：总体否；K56 速度均值高于 B1。",
        "4. KDA 是否改善压力稳定性：否；压力场误差和压力漂移均高于 B1。",
        "5. 是否只改善训练而未改善 validation：结果与此一致；最终训练 loss 很低，"
        "但 validation rollout 未获益。",
        "6. 有效物理记忆时间：平均 half-life 约 34–36 个时间单位。",
        "7. memory 是否 collapse/爆炸/被绕过：没有爆炸或完全绕过；effective rank 较低，"
        "存在显著低维化，但不是严格 rank-1 collapse。",
        "8. 额外成本：参数增加 4.86%，K56 推理时间增加约 42%。",
        "9. 是否值得替换当前 Hopf specialist：当前不值得。",
        "10. 是否需要 full attention：当前没有必要。首先应解决 current-only 特征不足、"
        "memory 有效秩偏低和高 Re 闭合偏差；直接增加 attention 会混淆因果并显著增加成本。",
        "",
        "## 10. 局限与后续建议",
        "",
        "本报告是单 seed validation 结果，不是 held-out 泛化结论。由于主方法未通过预设门禁，"
        "继续多 seed 或 held-out 测试不符合当前协议。若开展下一轮，应作为新的预注册实验："
        "优先检查 KDA token/value 归一化、提高 memory 有效秩、针对高 Re 采用稳定的"
        "时间尺度条件化，并保留 B1 作为强基线；不建议直接加入 Transformer/full attention。",
        "",
        "## 11. 复现产物",
        "",
        "- `ROLLOUT_RESULTS.csv`：逐模型、逐 Re、逐 horizon 的完整指标。",
        "- `ERROR_GROWTH_CURVES.csv`：逐 rollout step 的速度/压力/联合场误差。",
        "- `MEMORY_DIAGNOSTICS.csv`：逐模型、逐 Re、逐 head 的 KDA 诊断。",
        "- `ROLLOUT_SUMMARY.json`：资产审计、检查点哈希和聚合结果。",
        "- SVG 图：horizon 增长、K56 轨迹、memory 时间尺度与有效秩。",
        "",
    ])
    report = OUT / "FINAL_ROLLOUT_COMPARATIVE_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    artifacts = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (OUT / "ROLLOUT_ARTIFACT_MANIFEST.json").write_text(
        json.dumps({"schema_version": 1, "files": artifacts}, indent=2),
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
