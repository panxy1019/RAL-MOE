#!/usr/bin/env python3
"""Create the immutable Hopf A/B heldout report and artifact inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


NAMES = (
    "HopfLocal32_V16Common_K1248",
    "HopfLocal32_V16ScaleAware_K1248",
)
SHORT = {NAMES[0]: "A / V16Common", NAMES[1]: "B / V16ScaleAware"}


def pct(value: float) -> str:
    if value != value or value in (float("inf"), -float("inf")):
        return "NaN"
    return f"{100 * value:.4f}%"


def number(value: float, digits: int = 4) -> str:
    if value != value or value in (float("inf"), -float("inf")):
        return "NaN"
    return f"{value:.{digits}f}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalization-dir", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    args = parser.parse_args()
    root = args.finalization_dir
    selection = json.loads((root / "checkpoint_selection_manifest.json").read_text())
    heldout = json.loads((root / "heldout_metrics.json").read_text())

    val_rows = []
    for name in NAMES:
        item = selection["experiments"][name]
        val_rows.append([SHORT[name], item["optimizer_step"], f'{item["validation_score"]:.8f}',
                         "PASS" if item["hard_gate"] else "FAIL", item["sha256"][:16] + "…"])
    a_score = selection["experiments"][NAMES[0]]["validation_score"]
    b_score = selection["experiments"][NAMES[1]]["validation_score"]
    val_improvement = (a_score - b_score) / a_score

    horizon_rows = []
    k48_rows = []
    diag_rows = []
    for name in NAMES:
        experiment = heldout["experiments"][name]
        for re_value, row in experiment["by_re"].items():
            for horizon in (1, 8, 16, 24, 48):
                metric = row["horizons"][str(horizon)]
                horizon_rows.append([
                    SHORT[name], re_value, f"K{horizon}",
                    pct(metric["velocity_area_weighted_physical_relative_l2"]),
                    pct(metric["pressure_area_weighted_physical_relative_l2"]),
                    pct(metric["finite_fraction"]), metric["divergent_windows"],
                    metric["first_divergence_step"] or "—",
                ])
            k48 = row["horizons"]["48"]
            diag = row["hopf_diagnostics_k48"]
            k48_rows.append([
                SHORT[name], re_value,
                pct(k48["velocity_area_weighted_physical_relative_l2"]),
                pct(k48["pressure_area_weighted_physical_relative_l2"]),
                pct(k48["velocity_modal_relative_l2"]), pct(k48["pressure_modal_relative_l2"]),
                pct(k48["finite_fraction"]), k48["divergent_windows"],
                "PASS" if row["hopf_attractor_preserved"] else "FAIL",
            ])
            diag_rows.append([
                SHORT[name], re_value, pct(diag["rms_amplitude_error"]),
                pct(diag["peak_to_peak_amplitude_error"]), pct(diag["growth_sign_accuracy"]),
                f'{diag["true_local_log_growth"]:.3e}', f'{diag["predicted_local_log_growth"]:.3e}',
                pct(diag["frequency_relative_error"]), number(diag["terminal_phase_drift_cycles"]),
                number(diag["normalized_orbit_distance"]), "是" if diag["false_growth"] else "否",
            ])

    b47 = heldout["experiments"][NAMES[1]]["by_re"]["47.081355"]
    a47 = heldout["experiments"][NAMES[0]]["by_re"]["47.081355"]
    preserved = {name: heldout["experiments"][name]["heldout_preserved_count"] for name in NAMES}
    lines = [
        "# HopfLocal32 V16Common vs V16ScaleAware 最终实验报告",
        "",
        "## 1. 最终结论",
        "",
        f"两组都没有达到预注册的 Hopf 成功标准（至少 2/3 heldout attractor preserved）：A 为 {preserved[NAMES[0]]}/3，B 为 {preserved[NAMES[1]]}/3。",
        "",
        f"B 的 validation 主分数相对 A 改善 {pct(val_improvement)}，并把 Re=47.081355 从 K48 首次在 step {a47['horizons']['48']['first_divergence_step']} 发散、"
        f"finite={pct(a47['horizons']['48']['finite_fraction'])} 改善到 finite={pct(b47['horizons']['48']['finite_fraction'])}、divergent windows=0。"
        "但是 B 在该 near-onset Re 的 RMS 振幅误差仍为 "
        f"{pct(b47['hopf_diagnostics_k48']['rms_amplitude_error'])}，peak-to-peak 振幅误差为 "
        f"{pct(b47['hopf_diagnostics_k48']['peak_to_peak_amplitude_error'])}，因此只能判定为“数值稳定性改善”，不能判定 Hopf 吸引子被保留。",
        "",
        "最关键的观察是：B 在三个 heldout 的 K48 速度/压力 physical-field relative L2 都低于 0.5%，但振幅、增长和相位诊断仍失败。这里的物理场分母受训练均值场主导，不能单独证明小幅 near-onset 振荡被正确复现。",
        "",
        "## 2. 冻结 checkpoint 与 validation 选择",
        "",
        "checkpoint 只依据 validation 选择；heldout 在冻结完成后才首次解封。最终冻结文件位于独立、不可覆盖的 finalization 目录，并设置为只读。",
        "",
        table(["版本", "step", "validation score", "硬门", "SHA-256"], val_rows),
        "",
        "主分数定义为两个 validation Re 中最差的 K8/K16 area-weighted physical velocity+pressure relative L2。A 选择 step 7500，B 选择 step 7200。",
        "",
        "## 3. Heldout K48 主结果",
        "",
        table(["版本", "Re", "u physical", "p physical", "u modal", "p modal", "finite", "divergent", "attractor"], k48_rows),
        "",
        "以上所有 relative error 均已乘 100，以百分数表示。A 在 Re=47.081355 的超大误差来自真实数值发散；NaN 轨道诊断不做有限值替代。",
        "",
        "## 4. Hopf 吸引子诊断",
        "",
        table(["版本", "Re", "RMS振幅误差", "P2P振幅误差", "growth-sign", "真增长", "预测增长", "频率误差", "相位漂移/周期", "orbit距离", "符号型false growth"], diag_rows),
        "",
        "`false growth` 列采用严格的增长符号判据。本次 Re=47.081355 的 heldout 真值在所评 K48 窗口内具有很小的正平均增长，因此该布尔量为“否”；但 B 的巨大振幅过冲仍表明 near-onset false oscillation/scale inflation 没有消除。",
        "",
        "B 相比 A 的实际收益主要是：",
        "",
        "- Re=47.081355：从 K48 数值发散变为全程 finite。",
        "- Re=49.022357：RMS 振幅误差明显下降，频率误差降至约 0.0275%，但振幅仍远超 10% 门槛。",
        "- Re=51.786450：RMS 振幅误差由约 31.43% 降至约 18.48%，仍未达到 10% 门槛。",
        "- 三个 Re 的 growth-sign accuracy 均提高，但 Re=51.786450 只有约 66.50%。",
        "",
        "## 5. 多步 physical-field 结果",
        "",
        table(["版本", "Re", "horizon", "u physical", "p physical", "finite", "divergent", "首次发散step"], horizon_rows),
        "",
        "## 6. 评价合同与成功判据",
        "",
        "终评使用 one-step、K2/K4/K8/K16/K24/K48 的共同 K48-capable 起点，window stride=8；递推只保留固定 Re、观测 dt 和预测状态/历史。主二维平面固定为训练数据识别的 Hopf-local POD 第 2/3 模态（1-based），center、r_floor、growth tolerance 和归一化尺度全部只由 12 个训练 Re 重建。",
        "",
        "单个 Re 的 attractor-preserved 联合门包括：全程 finite、零 divergent window、K48 u/p physical error≤5%、RMS/P2P 振幅误差≤10%、频率误差≤5%、末端相位漂移≤0.25 周期、orbit distance≤0.10、u/p 能量漂移≤10%，且无符号型 false growth。",
        "",
        "## 7. 数据隔离与模型差异",
        "",
        "POD 均值、r32 基、scaler、Galerkin/pressure tensor、Hopf 主平面、r_floor 及增长统计只使用 12 个训练 Re。validation 仅用于 step/checkpoint 选择；3 个 heldout Re 只在冻结 checkpoint 后进入本次 evaluator。A 与 B 的唯一科学变量仍是 B 的 floor-aware log-amplitude 与短窗增长约束。",
        "",
        "## 8. SwanLab 与产物",
        "",
        "- A: https://swanlab.cn/@panxy1019/V17_HopfLocal32_MoEROM/runs/58657mp1",
        "- B: https://swanlab.cn/@panxy1019/V17_HopfLocal32_MoEROM/runs/1ryusvfl",
        "- `checkpoint_selection_manifest.json`: validation-only 选择证据与冻结 SHA。",
        "- `heldout_metrics.json`: 完整逐版本、逐 Re、逐 horizon 指标。",
        "- 每个实验目录下的 `final.pt`: 只读冻结 checkpoint。",
    ]
    report = root / "FINAL_HOPF_AB_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "schema_version": 1,
        "status": "FINAL_HELDOUT_COMPLETE",
        "success_criterion": "at least 2/3 heldout attractors preserved",
        "success": False,
        "validation_relative_improvement_B_vs_A": val_improvement,
        "preserved_count": preserved,
        "near_onset_Re_47p081355": {
            "A_K48_first_divergence_step": a47["horizons"]["48"]["first_divergence_step"],
            "B_K48_finite_fraction": b47["horizons"]["48"]["finite_fraction"],
            "B_K48_divergent_windows": b47["horizons"]["48"]["divergent_windows"],
            "B_rms_amplitude_error": b47["hopf_diagnostics_k48"]["rms_amplitude_error"],
            "B_peak_to_peak_amplitude_error": b47["hopf_diagnostics_k48"]["peak_to_peak_amplitude_error"],
            "interpretation": "numerical divergence removed, attractor-scale failure remains",
        },
    }
    (root / "FINAL_HOPF_AB_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    inventory = {
        "schema_version": 1,
        "trainer": {"path": str(args.trainer), "sha256": sha256(args.trainer)},
        "evaluator": {"path": str(args.evaluator), "sha256": sha256(args.evaluator)},
        "artifacts": [],
    }
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_inventory.json":
            inventory["artifacts"].append({
                "path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    (root / "artifact_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(json.dumps({"event": "report_complete", "report": str(report),
                      "success": False, "preserved": preserved}), flush=True)


if __name__ == "__main__":
    main()
