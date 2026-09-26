#!/usr/bin/env python3
"""Aggregate V15 Physics-Generalizable experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


METRIC_PATHS = {
    "rhs_l2": ("deep_moe", "rhs_relative_l2"),
    "one_step_velocity_l2": ("one_step_autonomous_pressure", "a_relative_l2"),
    "one_step_pressure_l2": ("one_step_autonomous_pressure", "b_relative_l2"),
    "one_step_pressure_energy_error": (
        "one_step_autonomous_pressure",
        "b_energy_relative_error",
    ),
    "rollout_velocity_l2": ("rollout_autonomous_pressure", "a_relative_l2_mean"),
    "rollout_pressure_l2": ("rollout_autonomous_pressure", "b_relative_l2_mean"),
    "rollout_pressure_energy_error": (
        "rollout_autonomous_pressure",
        "b_energy_relative_error",
    ),
    "pressure_base_l2": ("deep_moe", "pressure_surrogate_base_relative_l2"),
    "pressure_closure_l2": ("deep_moe", "pressure_head_relative_l2"),
    "pressure_residual_only_l2": ("deep_moe", "pressure_residual_only_relative_l2"),
    "pressure_residual_l2": ("deep_moe", "pressure_residual_relative_l2"),
    "pressure_effective_residual_l2": (
        "deep_moe",
        "pressure_effective_residual_relative_l2",
    ),
    "alpha_mean": ("deep_moe", "closure_alpha_mean"),
    "beta_mean": ("deep_moe", "closure_beta_mean"),
    "base_contribution_ratio_mean": (
        "deep_moe",
        "closure_base_contribution_ratio_mean",
    ),
    "residual_contribution_ratio_mean": (
        "deep_moe",
        "closure_residual_contribution_ratio_mean",
    ),
    "router_entropy": ("routing_analysis_test", "entropy_mean"),
    "router_active_experts": ("routing_analysis_test", "active_experts_mean"),
    "group_router_entropy": ("group_routing_analysis_test", "entropy_mean"),
}

CASE_ORDER = ["V15_Base", "V15_LargeROM", "V15_BalancedTraining"]
COLORS = {
    "V15_Base": "#1f77b4",
    "V15_LargeROM": "#2ca02c",
    "V15_BalancedTraining": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def nested_metric(item: Dict[str, object], path: Iterable[str]) -> float:
    cur: object = item
    for key in path:
        if not isinstance(cur, dict):
            return float("nan")
        cur = cur.get(key)
    try:
        return float(cur)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def case_name(data: Dict[str, object], path: Path) -> str:
    settings = data.get("settings", {})
    if isinstance(settings, dict) and settings.get("experiment_tag"):
        return str(settings["experiment_tag"])
    name = str(data.get("settings", {}).get("experiment_name", path.stem))
    for case in CASE_ORDER:
        if case in name:
            return case
    return path.parent.parent.name


def regime_group(regime: str) -> str:
    if regime in {"steady_wake", "pre_hopf_steady"}:
        return "Steady"
    if regime == "hopf_transition":
        return "Hopf"
    return "Periodic"


def load_rows(path: Path) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    case = case_name(data, path)
    rows: List[Dict[str, object]] = []
    for item in data.get("results", []):
        if not isinstance(item, dict):
            continue
        row: Dict[str, object] = {
            "case": case,
            "Re": float(item["test_Re"]),
            "label": str(item["test_Re_label"]),
            "regime": str(item.get("test_regime", "unknown")),
            "regime_group": regime_group(str(item.get("test_regime", "unknown"))),
            "num_test": int(item.get("num_test", 0)),
            "best_epoch": int(item.get("best_epoch", -1)),
        }
        for name, metric_path in METRIC_PATHS.items():
            row[name] = nested_metric(item, metric_path)
        rows.append(row)
    return data, sorted(rows, key=lambda row: float(row["Re"]))


def finite_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def aggregate(rows: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {"overall": {}, "by_regime_group": {}, "by_regime": {}}
    cases = sorted({str(row["case"]) for row in rows}, key=lambda x: CASE_ORDER.index(x) if x in CASE_ORDER else 99)
    for case in cases:
        sub = [row for row in rows if row["case"] == case]
        out["overall"][case] = {
            metric: finite_stats([float(row.get(metric, float("nan"))) for row in sub])
            for metric in METRIC_PATHS
        }
    for key in ["regime_group", "regime"]:
        target = out["by_regime_group" if key == "regime_group" else "by_regime"]
        groups = sorted({str(row[key]) for row in rows})
        for group in groups:
            target[group] = {}
            for case in cases:
                sub = [row for row in rows if row["case"] == case and row[key] == group]
                target[group][case] = {
                    metric: finite_stats([float(row.get(metric, float("nan"))) for row in sub])
                    for metric in METRIC_PATHS
                }
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "case",
        "Re",
        "label",
        "regime",
        "regime_group",
        "num_test",
        "best_epoch",
        *METRIC_PATHS.keys(),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def svg_line_chart(path: Path, title: str, metric: str, rows: List[Dict[str, object]]) -> None:
    width, height = 1180, 700
    left, right, top, bottom = 88, 35, 58, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    cases = [case for case in CASE_ORDER if any(row["case"] == case for row in rows)]
    points = [
        (float(row["Re"]), float(row.get(metric, float("nan"))))
        for row in rows
        if np.isfinite(float(row.get(metric, float("nan"))))
    ]
    if not points:
        path.write_text("", encoding="utf-8")
        return
    xs = np.asarray([p[0] for p in points])
    ys = np.asarray([p[1] for p in points])
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = min(0.0, float(ys.min()) * 0.95), max(0.1, float(ys.max()) * 1.08)

    def sx(x: float) -> float:
        return left + (x - x_min) / max(x_max - x_min, 1.0e-12) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - y_min) / max(y_max - y_min, 1.0e-12) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#1f2d3a}.axis{stroke:#536575;stroke-width:1.3}.grid{stroke:#d8e0e8;stroke-width:1}.line{fill:none;stroke-width:2.5}.dot{stroke:#fff;stroke-width:1.2}</style>",
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="{left}" y="34" font-size="23" font-weight="700">{title}</text>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
    ]
    for frac in np.linspace(0, 1, 6):
        y = top + plot_h - frac * plot_h
        val = y_min + frac * (y_max - y_min)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text x="18" y="{y + 4:.1f}" font-size="12">{val:.3g}</text>')
    for x in np.linspace(x_min, x_max, 6):
        parts.append(f'<text x="{sx(float(x)) - 18:.1f}" y="{top + plot_h + 26}" font-size="12">{x:.0f}</text>')
    for idx, case in enumerate(cases):
        sub = sorted(
            [(float(row["Re"]), float(row.get(metric, float("nan")))) for row in rows if row["case"] == case],
            key=lambda item: item[0],
        )
        sub = [(x, y) for x, y in sub if np.isfinite(y)]
        if not sub:
            continue
        color = COLORS.get(case, "#444")
        d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in sub)
        parts.append(f'<polyline class="line" points="{d}" stroke="{color}"/>')
        for x, y in sub:
            parts.append(f'<circle class="dot" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
        ly = top + 20 + 22 * idx
        parts.append(f'<line x1="{left + 10}" y1="{ly}" x2="{left + 34}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{left + 42}" y="{ly + 4}" font-size="13">{case}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 1.0:
        return f"{value:.4g}"
    return f"{value:.5f}"


def report_markdown(path: Path, rows: List[Dict[str, object]], stats: Dict[str, object], data: List[Dict[str, object]]) -> None:
    lines = [
        "# V15 Physics-Generalizable 综合实验报告",
        "",
        "本报告由 `aggregate_v15_physics_generalizable.py` 自动生成。三组实验使用同一套 ROM_PhysicsGeneralizable Re=20-200 数据库，保持 HPRS-MoE、Galerkin、RK4、Pressure Poisson Surrogate、pressure target=closure、loss 和优化超参数一致。",
        "",
        "## 实验组",
        "",
        "- `V15_Base`: ru=16, rp=16，V14 最优 baseline closure。",
        "- `V15_LargeROM`: 仅改 ru=32, rp=32。",
        "- `V15_BalancedTraining`: ru=16, rp=16，仅启用 regime-balanced mini-batch sampling。",
        "",
        "## POD 能量",
        "",
    ]
    for item in data:
        settings = item.get("settings", {})
        meta = item.get("data_meta", {})
        if not isinstance(settings, dict) or not isinstance(meta, dict):
            continue
        pod = meta.get("pod_energy", {})
        if not isinstance(pod, dict):
            continue
        lines.append(
            f"- {settings.get('experiment_tag')}: velocity_first_{settings.get('r_u')}="
            f"{fmt(float(pod.get('velocity_first_' + str(settings.get('r_u')), float('nan'))))}, "
            f"pressure_first_{settings.get('r_p')}="
            f"{fmt(float(pod.get('pressure_first_' + str(settings.get('r_p')), float('nan'))))}."
        )
    lines.extend(["", "## Overall Mean Metrics", ""])
    lines.append("| Case | 1-step u | 1-step p | 24-step u | 24-step p | RHS | p base | p energy | active experts |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for case in CASE_ORDER:
        if case not in stats["overall"]:
            continue
        s = stats["overall"][case]
        lines.append(
            f"| {case} | {fmt(s['one_step_velocity_l2']['mean'])} | "
            f"{fmt(s['one_step_pressure_l2']['mean'])} | "
            f"{fmt(s['rollout_velocity_l2']['mean'])} | "
            f"{fmt(s['rollout_pressure_l2']['mean'])} | "
            f"{fmt(s['rhs_l2']['mean'])} | {fmt(s['pressure_base_l2']['mean'])} | "
            f"{fmt(s['rollout_pressure_energy_error']['mean'])} | "
            f"{fmt(s['router_active_experts']['mean'])} |"
        )
    lines.extend(["", "## Regime Group Mean Metrics", ""])
    lines.append("| Regime group | Case | 1-step u | 1-step p | 24-step u | 24-step p | RHS |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for group in ["Steady", "Hopf", "Periodic"]:
        group_stats = stats["by_regime_group"].get(group, {})
        for case in CASE_ORDER:
            if case not in group_stats:
                continue
            s = group_stats[case]
            lines.append(
                f"| {group} | {case} | {fmt(s['one_step_velocity_l2']['mean'])} | "
                f"{fmt(s['one_step_pressure_l2']['mean'])} | "
                f"{fmt(s['rollout_velocity_l2']['mean'])} | "
                f"{fmt(s['rollout_pressure_l2']['mean'])} | {fmt(s['rhs_l2']['mean'])} |"
            )
    lines.extend(["", "## Held-out Reynolds Numbers", ""])
    lines.append("| Case | Re | Regime | 24-step u | 24-step p | p base | alpha | beta |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda r: (CASE_ORDER.index(str(r["case"])) if r["case"] in CASE_ORDER else 99, float(r["Re"]))):
        lines.append(
            f"| {row['case']} | {float(row['Re']):.3f} | {row['regime']} | "
            f"{fmt(float(row['rollout_velocity_l2']))} | {fmt(float(row['rollout_pressure_l2']))} | "
            f"{fmt(float(row['pressure_base_l2']))} | {fmt(float(row['alpha_mean']))} | "
            f"{fmt(float(row['beta_mean']))} |"
        )
    lines.extend(
        [
            "",
            "## 初步解释",
            "",
            "- 若 `V15_LargeROM` 在 Steady/Hopf/Periodic 三组均显著降低 one-step 和 rollout，则瓶颈偏向 ROM 空间容量。",
            "- 若 `V15_BalancedTraining` 主要改善 Steady/Hopf 但不显著改变成熟周期流，则瓶颈偏向训练数据分布。",
            "- 若 Pressure BaseOnly 指标在某些 regime 持续高于 Closure，后续应优先重诊断 Pressure Poisson Surrogate 或 residual-target conditioning。",
            "- Router/Expert 的 regime 分化需要结合 `routing_by_regime_*` JSON 字段和 SwanLab 曲线进一步解释。",
            "",
            "完整原始 metrics JSON、checkpoint 和 SwanLab 缓存保留在集群结果目录；Git 只应提交轻量 CSV、SVG、JSON 摘要和报告。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, object]] = []
    raw_data: List[Dict[str, object]] = []
    for metric_path in args.metrics:
        data, rows = load_rows(metric_path)
        raw_data.append(data)
        all_rows.extend(rows)
    stats = aggregate(all_rows)
    write_csv(args.output_dir / "v15_physics_generalizable_combined.csv", all_rows)
    (args.output_dir / "v15_physics_generalizable_summary_metrics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for metric, title in [
        ("one_step_velocity_l2", "V15 one-step velocity vs Re"),
        ("one_step_pressure_l2", "V15 one-step pressure vs Re"),
        ("rollout_velocity_l2", "V15 24-step rollout velocity vs Re"),
        ("rollout_pressure_l2", "V15 24-step rollout pressure vs Re"),
        ("pressure_base_l2", "V15 Pressure BaseOnly error vs Re"),
        ("router_active_experts", "V15 active experts vs Re"),
    ]:
        svg_line_chart(args.output_dir / f"{metric}.svg", title, metric, all_rows)
    report_markdown(
        args.output_dir / "TECHNICAL_REPORT_V15_PHYSICS_GENERALIZABLE.md",
        all_rows,
        stats,
        raw_data,
    )
    aggregate_json = {
        "metrics": [str(path) for path in args.metrics],
        "rows": all_rows,
        "stats": stats,
    }
    (args.output_dir / "v15_physics_generalizable_aggregate.json").write_text(
        json.dumps(aggregate_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "num_rows": len(all_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
