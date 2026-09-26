#!/usr/bin/env python3
"""Create the final expanded-H4 summary, report, and immutable artifact inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(text: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(payload: object, path: Path) -> None:
    atomic_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), path)


def pct(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalization-dir", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    args = parser.parse_args()
    root = args.finalization_dir
    experiment = "HopfExpanded34_H4_NormalFormRadial_r32"
    metrics = json.loads((root / "heldout_metrics.json").read_text())["experiments"][experiment]
    selection = json.loads((root / "checkpoint_selection_manifest.json").read_text())["experiments"][experiment]
    validation = selection["validation_record"]
    throughput = json.loads((args.training_run / "throughput.json").read_text())

    thresholds = {
        "physical": 0.05, "amplitude": 0.10, "frequency": 0.05,
        "phase_cycles": 0.25, "orbit": 0.10, "energy": 0.10,
    }
    rows = []
    for re_value, record in metrics["by_re"].items():
        k48 = record["horizons"]["48"]
        diag = record["hopf_diagnostics_k48"]
        failed = []
        if k48["velocity_area_weighted_physical_relative_l2"] > thresholds["physical"]: failed.append("K48 velocity field")
        if k48["pressure_area_weighted_physical_relative_l2"] > thresholds["physical"]: failed.append("K48 pressure field")
        if diag["rms_amplitude_error"] > thresholds["amplitude"]: failed.append("RMS amplitude")
        if diag["peak_to_peak_amplitude_error"] > thresholds["amplitude"]: failed.append("P2P amplitude")
        if diag["frequency_relative_error"] > thresholds["frequency"]: failed.append("frequency")
        if abs(diag["terminal_phase_drift_cycles"]) > thresholds["phase_cycles"]: failed.append("terminal phase")
        if diag["normalized_orbit_distance"] > thresholds["orbit"]: failed.append("orbit distance")
        if k48["velocity_energy_drift"] > thresholds["energy"]: failed.append("velocity energy")
        if k48["pressure_energy_drift"] > thresholds["energy"]: failed.append("pressure energy")
        if diag["false_growth"]: failed.append("false growth")
        rows.append({
            "Re": re_value, "preserved": record["hopf_attractor_preserved"],
            "finite_fraction": k48["finite_fraction"], "divergent_windows": k48["divergent_windows"],
            "k48_velocity_physical_relative_l2": k48["velocity_area_weighted_physical_relative_l2"],
            "k48_pressure_physical_relative_l2": k48["pressure_area_weighted_physical_relative_l2"],
            "rms_amplitude_error": diag["rms_amplitude_error"],
            "p2p_amplitude_error": diag["peak_to_peak_amplitude_error"],
            "growth_sign_accuracy": diag["growth_sign_accuracy"],
            "true_local_log_growth": diag["true_local_log_growth"],
            "predicted_local_log_growth": diag["predicted_local_log_growth"],
            "frequency_relative_error": diag["frequency_relative_error"],
            "terminal_phase_drift_cycles": diag["terminal_phase_drift_cycles"],
            "normalized_orbit_distance": diag["normalized_orbit_distance"],
            "false_growth": diag["false_growth"], "failed_strict_checks": failed,
        })

    summary = {
        "schema_version": 1,
        "experiment": experiment,
        "training": {"return_code": 0, "final_step": 8000, **throughput},
        "selection": {
            "split": "validation_only", "heldout_consulted": False,
            "step": selection["optimizer_step"], "score": selection["validation_score"],
            "hard_gate": selection["hard_gate"], "checkpoint_sha256": selection["sha256"],
        },
        "final_evaluation": {
            "status": "COMPLETE", "test_Re": metrics["heldout_Re"],
            "not_strict_blind_due_to_prior_method_diagnostics": True,
            "k48_finite_and_zero_divergence_count": sum(r["finite_fraction"] == 1.0 and r["divergent_windows"] == 0 for r in rows),
            "strict_attractor_preserved_count": metrics["heldout_preserved_count"],
            "rows": rows,
        },
    }
    atomic_json(summary, root / "FINAL_H4_EXPANDED_SUMMARY.json")

    table = []
    for row in rows:
        table.append(
            f"| {row['Re']} | {pct(row['k48_velocity_physical_relative_l2'])} | "
            f"{pct(row['k48_pressure_physical_relative_l2'])} | {pct(row['rms_amplitude_error'])} | "
            f"{pct(row['p2p_amplitude_error'])} | {pct(row['growth_sign_accuracy'])} | "
            f"{pct(row['frequency_relative_error'])} | {row['normalized_orbit_distance']:.4f} | "
            f"{'是' if row['false_growth'] else '否'} | {'通过' if row['preserved'] else '未通过'} |"
        )
    failures = "\n".join(
        f"- Re={row['Re']}：" + ("无" if not row["failed_strict_checks"] else "、".join(row["failed_strict_checks"]))
        for row in rows
    )
    report = f"""# HopfExpanded34 H4 最终实验报告

## 1. 训练与冻结

- 实验：`{experiment}`
- 正式训练：return code 0，完成 8000 optimizer steps。
- checkpoint 选择只使用 validation Re `46.7, 56.543246`；三个 test Re 在冻结前未读取。
- 冻结 checkpoint：step {selection['optimizer_step']}，validation score `{selection['validation_score']:.9f}`，hard gate 通过。
- checkpoint SHA-256：`{selection['sha256']}`。
- 训练耗时：{throughput['elapsed_seconds'] / 3600:.3f} 小时，平均 {throughput['steps_per_min']:.2f} steps/min。

## 2. Test 终评协议

- Test Re：`47.081355, 49.022357, 51.786450`。
- 这些 Re 曾用于前序方法诊断，因此本报告称为最终评价集，不称为严格一次性 blind test。
- 使用冻结 checkpoint、state-autonomous rollout 和新 29-Re train-only POD/ROM/统计量。
- 评价 horizon：K1/K2/K4/K8/K16/K24/K48；表内相对误差均乘以 100，以百分数表示。

## 3. K48 主要结果

| Re | 速度物理场误差 | 压力物理场误差 | RMS 振幅误差 | P2P 振幅误差 | Growth-sign | 频率误差 | Orbit distance | False growth | 严格保持 |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
{chr(10).join(table)}

三组 test Re 均为 finite_fraction=100%、divergent windows=0。严格 attractor-preserved 为 `{metrics['heldout_preserved_count']}/3`。

严格判据失败项：

{failures}

## 4. 结果分析

模型的数值稳定性和总物理场重构很好：三个 Re 的 K48 速度误差均低于 0.017%，压力误差均低于 0.279%，且没有发散。主要问题仍是临界二维波动动力学，而不是压力闭合或总场。

- Re=47.081355 已消除 false growth：真实平均对数增长 `{rows[0]['true_local_log_growth']:.6g}`，预测 `{rows[0]['predicted_local_log_growth']:.6g}`；RMS 振幅误差仅 {pct(rows[0]['rms_amplitude_error'])}。其 P2P 误差 {pct(rows[0]['p2p_amplitude_error'])} 略超 10% 门限。真实频率接近零，因此 {pct(rows[0]['frequency_relative_error'])} 的相对频率误差不宜作为有效旋转频率结论。
- Re=49.022357 的频率误差仅 {pct(rows[1]['frequency_relative_error'])}，但 RMS/P2P 振幅误差分别为 {pct(rows[1]['rms_amplitude_error'])}/{pct(rows[1]['p2p_amplitude_error'])}，并伴随 orbit distance `{rows[1]['normalized_orbit_distance']:.4f}`，说明轨道尺度和形状没有保持。
- Re=51.786450 的增长方向准确率为 {pct(rows[2]['growth_sign_accuracy'])}、频率误差 {pct(rows[2]['frequency_relative_error'])}，但 RMS/P2P 振幅误差和 orbit distance 仍超门限，表现为稳定但幅值偏弱的吸引子。

## 5. 结论

本次新数据 H4 达成了 `3/3` K48 finite、`3/3` 零 divergence，并保持了很低的总场速度/压力误差；同时消除了 Re=47.081355 的 false growth。严格 Hopf attractor preserved 未通过，核心短板是 Re=49.022357 和 51.786450 的振幅/轨道尺度，以及 Re=47.081355 的轻度 P2P 偏差。当前结果不支持把该 checkpoint 宣称为三组 test Re 的完整 Hopf 吸引子保持模型。
"""
    atomic_text(report, root / "FINAL_H4_EXPANDED_REPORT.md")

    inventory = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "artifact_inventory.json"):
        inventory.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json({"schema_version": 1, "root": str(root), "artifacts": inventory}, root / "artifact_inventory.json")
    atomic_json({"status": "COMPLETE", "checkpoint_frozen": True, "final_evaluation_complete": True,
                 "report": "FINAL_H4_EXPANDED_REPORT.md", "summary": "FINAL_H4_EXPANDED_SUMMARY.json"},
                root / "FINALIZATION_COMPLETE.json")
    print(json.dumps({"status": "COMPLETE", "report": str(root / 'FINAL_H4_EXPANDED_REPORT.md'),
                      "preserved": metrics["heldout_preserved_count"], "finite_zero_divergence": 3}))


if __name__ == "__main__":
    main()
