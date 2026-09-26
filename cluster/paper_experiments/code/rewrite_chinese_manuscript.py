#!/usr/bin/env python3
"""Rewrite the numerical-experiment draft in Chinese with KLong-only percent tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path("/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE")
REPORTS = ROOT / "paper_experiments/reports"
TABLES = REPORTS / "tables"
MANUSCRIPT = ROOT / "paper_experiments/manuscript"

S3 = ROOT / "steady_specialist_v1/evaluation_inputs/s3_heldout_metrics.json"
VAN_S = ROOT / "paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/steady/vanilla-fnn/frozen_user_stop_step6200_20260724/evaluation/test/metrics.json"
DATA_S = ROOT / "paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/steady/data-only/evaluation/test/metrics.json"
VAN_H_VAL = ROOT / "paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/hopf_v6/hopf/vanilla-fnn/evaluation/validation/metrics.json"
DATA_H = ROOT / "paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/hopf/data-only/evaluation/test/metrics.json"
PROP_H = ROOT / "Hopf/migrated_h4_expanded/native_attractor_audit_20260723_v1/results/HOPF_NATIVE_ATTRACTOR_AUDIT.json"
VAN_P = ROOT / "paper_experiments/runs/specialist_vanilla_recovery_v1_20260724/retries/periodic_v4/periodic/vanilla-fnn/evaluation/test/native/periodic_r32_multihorizon_evaluation.json"
DATA_P = ROOT / "paper_experiments/runs/specialist_ablations_single_seed_v3_20260724/periodic/data-only/evaluation/test_k48_supplement_20260724/metrics.json"
PROP_P = ROOT / "periodic_specialist_r32/reproduction/best_epoch85_heldout_eval_v2/periodic_r32_multihorizon_evaluation.json"
E2 = ROOT / "trajectory_router_e1_e2_e3_20260722/E1_E2_E3_TOP1_REPORT.md"
T2 = ROOT / "top2_boundary_experiments_20260722/one_sided_recovery_v2/resplit_20260723_v1/final_test_evaluation/EVALUATION_REPORT.json"
T2_ORACLE = ROOT / "top2_boundary_experiments_20260722/one_sided_recovery_v2/resplit_20260723_v1/supplements/per_re_matched_oracle_v4/PER_RE_MATCHED_ORACLE_REPORT.json"
T2_ATTR = ROOT / "top2_boundary_experiments_20260722/one_sided_recovery_v2/resplit_20260723_v1/supplements/t2c_attractor_v2/T2C_ATTRACTOR_ANALYSIS.json"
GLOBAL = ROOT / "V16_1_SteadyPressureAnchor32/reproduction/heldout_eval_frozen_checkpoint_20260723/V16_1_SteadyPressureAnchor32_ru32_rp32_reproduced_metrics.json"
HP = ROOT / "top2_boundary_experiments_20260722/one_sided_recovery_hp_20260723_v1/H_P_RECOVERY_REPORT_20260723.md"


def load(path: Path):
    return json.loads(path.read_text())


def pct(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{100.0 * float(value):.{digits}f}%"


def num(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(lines)


def write_csv(path: Path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def re_from_label(label: str) -> float:
    return float(label.removeprefix("Re_").replace("p", "."))


def keyed_by_re(entries):
    return {round(re_from_label(key), 5): value for key, value in entries.items()}


def physical_row(row):
    field = row["physical_reconstruction_area_weighted"]
    return {
        "u": field["velocity_relative_l2"],
        "p": field["pressure_relative_l2"],
        "joint": 0.5 * (field["velocity_relative_l2"] + field["pressure_relative_l2"]),
        "finite": row["finite_fraction"],
        "div": row["divergent_windows"],
        "fp": row.get("fixed_point_residual"),
        "drift": row.get("pressure_drift"),
    }


def mean_rows(rows):
    return {
        key: mean(row[key] for row in rows)
        for key in ("u", "p", "joint")
    } | {
        "finite": min(row["finite"] for row in rows),
        "div": sum(row["div"] for row in rows),
        "worst": max(row["joint"] for row in rows),
    }


def near(mapping, value, tolerance=3e-5):
    key = min(mapping, key=lambda candidate: abs(candidate - value))
    if abs(key - value) > tolerance:
        raise KeyError(f"no key near {value}; closest={key}")
    return mapping[key]


def native_times():
    configs = [
        (
            "Steady", ROOT / "steady_specialist_v1/source_artifacts/steady/Global_POD_AreaWeighted_L2/pod_snapshot_index.csv",
            [24.630436, 32.740068, 39.685479, 45.142703], 56,
        ),
        (
            "Hopf", ROOT / "Hopf/artifacts/hopf/projection_snapshots_velocity_hopf.csv",
            [47.081355, 49.022357, 51.786450, 49.3, 49.6, 50.0], 56,
        ),
        (
            "Periodic", ROOT / "periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/pod_snapshot_index.csv",
            [70.314635, 100.352251, 149.059229, 189.862278], 48,
        ),
    ]
    rows = []
    for regime, path, values, horizon in configs:
        data = list(csv.DictReader(path.open()))
        for value in values:
            times = sorted(float(row["time"]) for row in data if abs(float(row["Re"]) - value) < 2e-5)
            if len(times) <= horizon:
                continue
            # The first interval can be a retained trajectory-boundary irregularity.
            # The exact horizon span is therefore reported, not horizon*global median dt.
            span = times[horizon] - times[0]
            rows.append([regime, value, f"K{horizon}", span])
    return rows


def specialist_long_table():
    s3 = load(S3)["experiments"]["S3-B"]["horizons"]["k56"]["by_re"]
    van_s = load(VAN_S)["horizons"]["k56"]["by_re"]
    data_s = load(DATA_S)["by_re"]
    common_s = [24.63044, 32.74007, 39.68548]
    s3_map, van_s_map = keyed_by_re(s3), keyed_by_re(van_s)
    data_s_map = {round(float(key), 5): value["k56"] for key, value in data_s.items()}
    steady = {
        "Vanilla-FNN-MoE": mean_rows([physical_row(near(van_s_map, x)) for x in common_s]),
        "DataOnly-MoE": mean_rows(
            [
                {
                    "u": near(data_s_map, x)["velocity_physical_relative_l2"],
                    "p": near(data_s_map, x)["pressure_physical_relative_l2"],
                    "joint": near(data_s_map, x)["joint_field_error"],
                    "finite": near(data_s_map, x)["finite_fraction"],
                    "div": near(data_s_map, x)["divergent_windows"],
                }
                for x in common_s
            ]
        ),
        "Proposed Specialist MoE": mean_rows([physical_row(near(s3_map, x)) for x in common_s]),
    }

    hopf_audit = load(PROP_H)
    prop_h_map = {
        round(row["Re"], 5): row["horizons"]["56"]
        for row in hopf_audit["per_Re"]
        if row["split"] == "heldout"
    }
    data_h = load(DATA_H)["by_re"]
    data_h_map = {round(float(key), 5): value["k56"] for key, value in data_h.items()}
    common_h = [47.08136, 49.02236, 51.78645]
    proposed_h = mean_rows(
        [
            {
                "u": near(prop_h_map, x)["velocity_area_weighted_physical_relative_l2"],
                "p": near(prop_h_map, x)["pressure_area_weighted_physical_relative_l2"],
                "joint": 0.5
                * (
                    near(prop_h_map, x)["velocity_area_weighted_physical_relative_l2"]
                    + near(prop_h_map, x)["pressure_area_weighted_physical_relative_l2"]
                ),
                "finite": near(prop_h_map, x)["finite_fraction"],
                "div": near(prop_h_map, x)["divergent_windows"],
            }
            for x in common_h
        ]
    )
    dataonly_h = mean_rows(
        [
            {
                "u": near(data_h_map, x)["velocity_physical_relative_l2"],
                "p": near(data_h_map, x)["pressure_physical_relative_l2"],
                "joint": near(data_h_map, x)["joint_field_error"],
                "finite": near(data_h_map, x)["finite_fraction"],
                "div": near(data_h_map, x)["divergent_windows"],
            }
            for x in common_h
        ]
    )

    def periodic_map(payload, method):
        if method == "data":
            return {
                round(float(key), 5): {
                    "u": value["k48"]["velocity_physical_relative_l2"],
                    "p": value["k48"]["pressure_physical_relative_l2"],
                    "joint": value["k48"]["joint_field_error"],
                    "finite": value["k48"]["finite_fraction"],
                    "div": value["k48"]["divergent_windows"],
                }
                for key, value in payload["by_re"].items()
            }
        return {
            round(float(row["Re"]), 5): {
                "u": row["horizons"]["48"]["velocity_area_weighted_l2"],
                "p": row["horizons"]["48"]["pressure_area_weighted_l2"],
                "joint": 0.5
                * (
                    row["horizons"]["48"]["velocity_area_weighted_l2"]
                    + row["horizons"]["48"]["pressure_area_weighted_l2"]
                ),
                "finite": row["horizons"]["48"]["finite_fraction"],
                "div": row["horizons"]["48"]["divergent_windows"],
            }
            for row in payload["results"]
        }

    common_p = [70.31464, 100.35225, 149.05923, 189.86227]
    maps_p = {
        "Vanilla-FNN-MoE": periodic_map(load(VAN_P), "native"),
        "DataOnly-MoE": periodic_map(load(DATA_P), "data"),
        "Proposed Specialist MoE": periodic_map(load(PROP_P), "native"),
    }

    rows = []
    re_text = {
        "Steady": "24.630436, 32.740068, 39.685479（共同可评估集）",
        "Hopf": "47.081356, 49.022357, 51.786450",
        "Periodic": "70.314635, 100.352251, 149.059229, 189.862278",
    }
    for method, result in steady.items():
        rows.append(["Steady", method, re_text["Steady"], "K56", result])
    rows.extend(
        [
            ["Hopf", "Vanilla-FNN-MoE", re_text["Hopf"], "K56", None],
            ["Hopf", "DataOnly-MoE", re_text["Hopf"], "K56", dataonly_h],
            ["Hopf", "Proposed Specialist MoE", re_text["Hopf"], "K56", proposed_h],
        ]
    )
    for method, mapping in maps_p.items():
        rows.append(["Periodic", method, re_text["Periodic"], "K48", mean_rows([near(mapping, x) for x in common_p])])
    return rows


def steady_attractor_rows():
    s3 = load(S3)["experiments"]["S3-B"]
    van = load(VAN_S)
    s3_h = keyed_by_re(s3["horizons"]["k56"]["by_re"])
    van_h = keyed_by_re(van["horizons"]["k56"]["by_re"])
    s3_a = s3["attractivity"]
    van_a = van["attractor"]
    values = [24.63044, 32.74007, 39.68548, 45.14270]

    def method_rows(method, hmap, amap):
        output = []
        for value in values:
            label = next(k for k in amap["by_re"] if abs(re_from_label(k) - value) < 2e-5)
            clean = physical_row(near(hmap, value))
            fp = amap["by_re"][label]["fixed_point_residual"]["k56"]["physical_reconstruction_area_weighted"]["pressure_relative_l2"]
            output.append(
                [
                    method, value, clean["u"], clean["p"], fp,
                    clean["drift"], clean["finite"], clean["div"],
                ]
            )
        return output

    rows = method_rows("Proposed Specialist MoE (S3-B)", s3_h, s3_a)
    rows += method_rows("Vanilla-FNN-MoE", van_h, van_a)
    # DataOnly lacks the perturbation-bank/fixed-point evaluator; preserve its field values.
    for key, value in load(DATA_S)["by_re"].items():
        k56 = value["k56"]
        rows.append(
            [
                "DataOnly-MoE", float(key),
                k56["velocity_physical_relative_l2"],
                k56["pressure_physical_relative_l2"],
                None, None, k56["finite_fraction"], k56["divergent_windows"],
            ]
        )
    return rows, {
        "proposed_gain_worst": s3_a["overall_worst"]["k56"]["paired_gain_worst_all_scales"],
        "vanilla_gain_worst": van_a["overall_worst"]["k56"]["paired_gain_worst_all_scales"],
        "proposed_perturb_div": s3_a["overall_worst"]["k56"]["divergent_windows_sum"],
        "vanilla_perturb_div": van_a["overall_worst"]["k56"]["divergent_windows_sum"],
        "data_generic": load(DATA_S)["attractor"],
    }


def hopf_strict_rows():
    wanted = [49.3, 49.6, 50.0]
    rows = []
    for item in load(PROP_H)["per_Re"]:
        if not any(abs(item["Re"] - value) < 2e-4 for value in wanted):
            continue
        field = item["horizons"]["56"]
        attr = item["attractor_K56"]
        rows.append(
            [
                item["Re"],
                field["velocity_area_weighted_physical_relative_l2"],
                field["pressure_area_weighted_physical_relative_l2"],
                attr["rms_amplitude_error"],
                attr["peak_to_peak_amplitude_error"],
                attr["frequency_relative_error"],
                attr["terminal_phase_drift_cycles"],
                attr["normalized_orbit_distance"],
                field["velocity_energy_drift"],
                field["pressure_energy_drift"],
                field["finite_fraction"],
                field["divergent_windows"],
                attr["false_growth"],
                item["attractor_preserved_K56"],
            ]
        )
    return sorted(rows)


def periodic_strict_rows():
    rows = []
    for item in load(PROP_P)["results"]:
        if not item["preserved"]:
            continue
        field = item["horizons"]["48"]
        attr = item["k48_cycle_diagnostics"]
        rows.append(
            [
                item["Re"],
                field["velocity_area_weighted_l2"],
                field["pressure_area_weighted_l2"],
                attr["rms_amplitude_error"],
                attr["peak_to_peak_amplitude_error"],
                attr["strouhal_error"],
                attr["phase_rms_rad"],
                attr["terminal_cycle_drift"],
                attr["normalized_orbit_distance"],
                field["velocity_energy_drift"],
                field["pressure_energy_drift"],
                field["finite_fraction"],
                field["divergent_windows"],
                item["preserved"],
            ]
        )
    return rows


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT.mkdir(parents=True, exist_ok=True)

    times = native_times()
    write_csv(TABLES / "TABLE_KLONG_NATIVE_TIME.csv", ["Regime", "Re", "KLong", "exact_time_span"], times)
    time_md = md_table(
        ["流态", "评价 Re", "KLong", "数据库 time 坐标跨度（沿用原数据单位）"],
        [[r, f"{re:.6f}", k, f"{span:.6f}"] for r, re, k, span in times],
    )

    long_rows = specialist_long_table()
    write_csv(
        TABLES / "TABLE_A1_KLONG_PERCENT.csv",
        ["Regime", "Method", "Test_Re", "KLong", "velocity_percent", "pressure_percent", "joint_percent", "worst_joint_percent", "finite_fraction", "divergent_windows"],
        [
            [
                regime, method, re_values, horizon,
                *(None if result is None else 100 * result[key] for key in ("u", "p", "joint", "worst")),
                None if result is None else result["finite"],
                None if result is None else result["div"],
            ]
            for regime, method, re_values, horizon, result in long_rows
        ],
    )
    long_md = md_table(
        ["流态", "方法", "测试 Re", "KLong", "速度误差", "压力误差", "联合误差", "worst", "finite", "发散窗"],
        [
            [
                regime, method, re_values, horizon,
                "验证失败，未开放 test" if result is None else pct(result["u"]),
                "—" if result is None else pct(result["p"]),
                "—" if result is None else pct(result["joint"]),
                "—" if result is None else pct(result["worst"]),
                "—" if result is None else num(result["finite"], 3),
                "—" if result is None else str(result["div"]),
            ]
            for regime, method, re_values, horizon, result in long_rows
        ],
    )

    steady_rows, steady_summary = steady_attractor_rows()
    steady_md = md_table(
        ["方法", "评价 Re", "K56速度", "K56压力", "压力固定点误差", "压力漂移", "finite", "clean发散窗"],
        [
            [
                method, f"{re:.6f}", pct(u), pct(p), pct(fp), pct(drift),
                num(finite, 3), str(div),
            ]
            for method, re, u, p, fp, drift, finite, div in steady_rows
        ],
    )
    write_csv(
        TABLES / "TABLE_B1A_STEADY_ATTRACTOR_PERCENT.csv",
        ["Method", "Re", "u_K56_percent", "p_K56_percent", "pressure_fixed_point_percent", "pressure_drift_percent", "finite_fraction", "clean_divergent_windows"],
        [[m, re, 100*u, 100*p, None if fp is None else 100*fp, None if drift is None else 100*drift, fin, div] for m,re,u,p,fp,drift,fin,div in steady_rows],
    )

    hopf_rows = hopf_strict_rows()
    hopf_md = md_table(
        ["指定 Re", "K56速度", "K56压力", "RMS振幅误差", "峰峰值误差", "频率误差", "末端相位漂移(周期)", "轨道距离", "速度能量漂移", "压力能量漂移", "finite", "发散窗", "false growth", "Strict"],
        [
            [
                f"{re:.6f}", pct(u), pct(p), pct(rms), pct(p2p), pct(freq),
                num(phase, 4), pct(orbit), pct(u_energy), pct(p_energy),
                num(finite, 3), str(divergent), "是" if false_growth else "否",
                "PASS" if strict else "FAIL",
            ]
            for re,u,p,rms,p2p,freq,phase,orbit,u_energy,p_energy,finite,divergent,false_growth,strict in hopf_rows
        ],
    )
    write_csv(
        TABLES / "TABLE_B1B_HOPF_STRICT_PERCENT.csv",
        ["Re", "u_K56_percent", "p_K56_percent", "rms_amplitude_percent", "p2p_amplitude_percent", "frequency_percent", "terminal_phase_cycles", "orbit_percent", "u_energy_drift_percent", "p_energy_drift_percent", "finite_fraction", "divergent_windows", "false_growth", "strict"],
        [[re,100*u,100*p,100*rms,100*p2p,100*freq,phase,100*orbit,100*u_energy,100*p_energy,finite,divergent,false_growth,strict] for re,u,p,rms,p2p,freq,phase,orbit,u_energy,p_energy,finite,divergent,false_growth,strict in hopf_rows],
    )

    periodic_rows = periodic_strict_rows()
    periodic_md = md_table(
        ["测试 Re", "K48速度", "K48压力", "RMS振幅误差", "峰峰值误差", "频率误差", "相位RMS(rad)", "末周期漂移", "轨道距离", "速度能量漂移", "压力能量漂移", "finite", "发散窗", "Strict"],
        [
            [
                f"{re:.6f}", pct(u), pct(p), pct(rms), pct(p2p), pct(freq),
                num(phase, 4), num(drift, 4), pct(orbit), pct(u_energy),
                pct(p_energy), num(finite, 3), str(divergent),
                "PASS" if strict else "FAIL",
            ]
            for re,u,p,rms,p2p,freq,phase,drift,orbit,u_energy,p_energy,finite,divergent,strict in periodic_rows
        ],
    )
    write_csv(
        TABLES / "TABLE_B1C_PERIODIC_STRICT_PERCENT.csv",
        ["Re", "u_K48_percent", "p_K48_percent", "rms_amplitude_percent", "p2p_amplitude_percent", "frequency_percent", "phase_rms_rad", "terminal_cycle_drift", "orbit_percent", "u_energy_drift_percent", "p_energy_drift_percent", "finite_fraction", "divergent_windows", "strict"],
        [[re,100*u,100*p,100*rms,100*p2p,100*freq,phase,drift,100*orbit,100*u_energy,100*p_energy,finite,divergent,strict] for re,u,p,rms,p2p,freq,phase,drift,orbit,u_energy,p_energy,finite,divergent,strict in periodic_rows],
    )

    t2 = load(T2)
    t2o = load(T2_ORACLE)["per_Re"]
    t2a = load(T2_ATTR)["per_Re"]
    sh_rows = []
    for key in ("42.359071", "43.5", "43.9"):
        report = t2o[key]
        for method in ("E2_Top1", "T2_C"):
            a = t2a[key][method]
            weights = report["T2_C_weights"] if method == "T2_C" else None
            sh_rows.append(
                [
                    float(key), "E2 Top-1" if method == "E2_Top1" else "T2-C",
                    report[method]["K56_mean"],
                    a["tail_center_field_error"]["mean"],
                    a["tail_amplitude_absolute_error"]["mean"],
                    a["tail_orbit_geometry_error"]["mean"],
                    None if weights is None else weights["alpha_S_mean"],
                    None if weights is None else weights["alpha_H_mean"],
                ]
            )
    sh_md = md_table(
        ["测试 Re", "方法", "K56联合误差", "尾段中心误差", "振幅绝对误差", "轨道几何误差", "αS", "αH"],
        [
            [
                f"{re:.6f}", method, pct(joint), pct(center), pct(amp), pct(orbit),
                num(a_s, 4), num(a_h, 4),
            ]
            for re,method,joint,center,amp,orbit,a_s,a_h in sh_rows
        ],
    )

    prop_gain = steady_summary["proposed_gain_worst"]
    van_gain = steady_summary["vanilla_gain_worst"]
    fp_prop = max(row[4] for row in steady_rows if row[0].startswith("Proposed"))
    fp_van = max(row[4] for row in steady_rows if row[0].startswith("Vanilla"))
    gain_reduction = 1.0 - prop_gain / van_gain
    fp_reduction = 1.0 - fp_prop / fp_van

    chinese_report = f"""# 最终消融实验中文报告

## 1. 报告口径

所有相对误差均转换为百分数，即原始 relative error × 100。主表只保留长时自主 rollout：Steady 与 Hopf 使用 K56，Periodic 使用三种方法共同具备的 K48。每个表均显式列出评价 Reynolds 数。KLong 表示数据库中的原生预测步数，而不是三个流态共享的统一物理时间。

### KLong 对应的原生数据库时间

{time_md}

同一流态不同 Re 的时间跨度也不完全相同，而且当前资产未给出可跨数据库直接比较的统一量纲单位。因此正文不把“K56”简单换写为一个全局物理时间；采用“K56 个原生数据库步，并逐 Re 报告原始 `time` 坐标跨度”的表述更准确。

## 2. Specialist 长时物理场误差

{long_md}

Steady 表采用三种方法共同具备 K56 结果的 Re=24.630436、32.740068、39.685479，避免 DataOnly 缺少 Re=45.142703 时产生不公平平均。权威 Proposed Steady checkpoint 是 S3-B step 600，而不是后来跨硬件严格门失败的 S4 候选。

Hopf 中 Proposed 在相同 held-out Re 上保持有限且无发散，DataOnly 虽没有 NaN/Inf，但存在 39 个 K56 norm-divergent windows；Vanilla 在 validation 长时 rollout 已非有限，因此按合同不开放 test。

Periodic 三种方法使用相同四个测试 Re，并通过本轮补充的冻结 DataOnly K48 evaluator 实现 horizon 对齐。Proposed 在 K48 的联合误差和 worst-case 均优于 Vanilla；与 DataOnly 相比，Proposed 的主要优势进一步体现在经过认证的频率、相位与轨道保持，而不是只看短时场误差。

## 3. Specialist attractor preservation

### 3.1 Steady：固定点与扰动吸引性

{steady_md}

Steady 当前不存在一个三种方法均完整执行、且预注册为同一阈值的 binary Strict-attractor test，因此不能把缺失字段强行写成 PASS。可比较证据仍显示出 Proposed 的吸引子优势：S3-B 的 K56 压力固定点 worst 为 {pct(fp_prop)}，Vanilla 为 {pct(fp_van)}，相对降低 {pct(fp_reduction)}；K56 全扰动尺度 paired-gain worst 从 Vanilla 的 {van_gain:.3f} 降至 Proposed 的 {prop_gain:.3f}，相对降低 {pct(gain_reduction)}。这说明 proposed physics–data expert 与收缩训练显著削弱了最坏扰动放大。

但两者最坏 paired gain 仍大于 1，且扰动库存在 norm-divergent cases，因此不能宣称 Steady 已通过“所有扰动均收缩”的最严格二值判据。更准确的结论是：Proposed 显著改善固定点与扰动鲁棒性，但尚未实现全方向严格收缩。DataOnly 没有运行同一 perturbation-bank evaluator，不能认证 Strict attractor；其整体 tail-center modal error 为 {pct(steady_summary['data_generic']['tail_center_modal_relative_l2'])}，tail-amplitude relative error 为 {pct(steady_summary['data_generic']['tail_amplitude_relative_error'])}，terminal increment RMS 为 {steady_summary['data_generic']['terminal_increment_rms']:.6e}。

### 3.2 Hopf：指定三个严格 K56 保持案例

{hopf_md}

三个指定案例均同时满足预注册的有限性、零发散、速度/压力场误差、振幅、频率、相位、轨道距离、能量漂移和无 false-growth 联合门，因此 Strict 结果均为 PASS。它们清楚展示了 Proposed Hopf specialist 在适配参数点上保持极限环局部几何和时序特征的能力。为保证数据说明完整，49.3、49.6、50.0 在原 split 中属于训练 Re；正文将其称为“指定严格保持案例”，而不称为独立 held-out 泛化结果。

### 3.3 Periodic：Proposed 的三个严格保持测试 Re

{periodic_md}

这里只列出 Proposed Specialist MoE 在 held-out 中通过完整 K48 周期吸引子联合门的三个 Re。三者的振幅、频率、相位漂移和归一化轨道距离均处于门限内，说明 proposed continuous-time physics–data coupling 不仅降低物理场误差，而且能够保持周期吸引子的闭合几何与相位推进。完整 held-out 为 4 个 Re；未列入本 strict-pass 子表的 Re=70.314635 因压力场门未通过而判为非严格保持，不能删除这一负结果，已在限制部分保留。

## 4. S–H 边界与路由

{sh_md}

T2-C 在 Re=42.359071 基本退化为 S-only；在 Re=43.50 和 43.90 增大 Hopf 权重。Re=43.50 的全 K56 场误差比 E2 更差，但尾段中心、振幅与轨道几何更优，说明瞬态误差与吸引子保持必须分别报告。Re=43.90 则同时改善长时场误差与吸引子指标。

## 5. 结论边界

支持的结论包括：Physics–Data 结构显著改善 Hopf 相对 DataOnly 的长时压力稳定性；Periodic Proposed 相对 Vanilla 提升场误差和严格周期保持；Steady Proposed 显著降低固定点误差及最坏扰动增益；T2-C 在 S–H 平均 K56 和 2/3 测试 Re 上有收益。

不支持的结论包括：Proposed 在所有流态、所有 Re、所有指标全面占优；Steady 所有扰动方向严格收缩；Hopf 三个指定案例是 held-out 泛化；H–P Top-2 已经验证；Global MoE 与多坐标系统已经在同一物理场 evaluator 上完成全域数值排名。
"""
    (REPORTS / "FINAL_EXPERIMENT_REPORT_ZH.md").write_text(chinese_report)
    (REPORTS / "SPECIALIST_ABLATION_REPORT.md").write_text(chinese_report)
    (REPORTS / "ATTRACTOR_ANALYSIS.md").write_text(
        "# Specialist 吸引子评价（中文）\n\n"
        + chinese_report.split("## 3. Specialist attractor preservation", 1)[1].split("## 4.", 1)[0]
    )

    draft = f"""# 数值实验部分草稿（中文修订版）

## 4.1 流动构型、数据划分与多坐标图 POD 空间

本文研究跨越 Steady、Hopf 与 Periodic 三类尾流动力学的参数化降阶预测问题。三个 specialist 分别保留其原生速度 POD、压力 POD、均值场、scaler、Galerkin 算子、pressure closure、历史特征构造器以及原生时间推进合同。不同局部坐标图中的模态系数不直接混合；只有各 specialist 独立推进并重构到同一物理网格后，才允许在经过认证的相邻共同可行域中实施输出级凸融合。

训练、验证和测试均以完整 Reynolds 数和完整轨迹隔离。新训练的 Vanilla-FNN-MoE 与 DataOnly-MoE 每个流态只使用一个固定随机种子，因此下文是确定性单次实验比较，不报告不存在的跨 seed 均值与标准差。Steady 的权威 Proposed 模型采用跨硬件审计后保留的 S3-B contraction checkpoint；后续 S4 fixed-point-anchor 候选因 4090 上的 P95 contraction 保护门失败，不作为最终 Proposed specialist。

本文所有物理场相对误差均以百分数表示。主结果只展示长时自主预测 KLong，而不再同时罗列 K1 和 K16。Steady、Hopf 使用 K56；Periodic 为保证三种消融具备严格一致的 horizon，统一使用 K48。

{time_md}

需要强调的是，KLong 是“原生数据库步数”，而不是统一物理时间。由于三个数据库的采样合同不同，甚至同一流态不同 Re 的原生时间间隔也不同，表中逐 Re 给出从首个合法窗口计算的原始 `time` 坐标跨度；其单位沿用各数据库原合同，不预先解释为可跨数据库比较的统一物理单位。后续若使用无量纲时间比较，应单独定义统一的无量纲化，而不能把不同 specialist 的 K56 直接解释为相同时间。

## 4.2 长时物理场预测

### 4.2.1 流态专用 specialist 消融

{long_md}

在 Steady 流态中，为保证三种方法使用同一评价人口，平均值只基于共同具备 K56 窗口的 Re=24.630436、32.740068、39.685479。Vanilla 的提前冻结 checkpoint 在部分 clean-field 指标上具有较小误差，因此结果不支持“特殊 expert 结构在 Steady 所有场误差上全面优于普通 FNN”。然而 DataOnly 的压力误差达到远高于可接受范围的水平，并出现 norm-divergent windows，说明仅学习离散数据映射不能稳定复现 pressure closure。Physics–Data specialist 对 DataOnly 的主要优势来自显式连续时间动力学、Galerkin 物理项、pressure coupling 与 rollout 约束共同提供的稳定性，而不能简单归因于网络宽度。

在 Hopf 流态中，Proposed 在 held-out Re=47.081356、49.022357、51.786450 上完成 K56 自主 rollout，finite fraction 为 1 且无发散。DataOnly 虽保持数值有限，但具有 39 个 K56 divergent windows，压力误差也显著高于 Proposed。Vanilla 在 validation K16/K56 已出现非有限轨迹，按照 validation-first 合同禁止进入 test。该结果说明 Hopf specialist 中的 normal-form/physics–data expert、连续时间推进与压力 closure 并非可被普通 FNN 或纯离散映射无损替换。

在 Periodic 流态中，三种方法均在 Re=70.314635、100.352251、149.059229、189.862278 上采用 K48。Proposed 相对 Vanilla 明显降低速度、压力、联合误差及 worst-case；相对 DataOnly 的优势不仅体现在物理场误差，还体现在下节经过严格认证的振幅、频率、相位和轨道闭合保持。这表明周期流态中的 physics–data coupling 对长期相位推进和极限环几何尤其重要。

### 4.2.2 完整参数域系统比较

Global MoE 已有冻结 checkpoint 和 held-out 结果，但其归档 evaluator 主要输出模态一步及 rollout 误差；E2/T2-C 的 S–H cache 使用面积加权物理场误差。两组数值合同并不相同，因此本文不把它们强行合并为一个全域排名表。当前能够严格比较的完整系统结果集中在 S–H 边界：E2 Top-1 的平均 K56 联合误差为 {pct(t2['baselines']['E2_pair_Top1']['horizons']['K56']['joint_mean'])}，T2-C 为 {pct(t2['routes']['T2-C_LearnedConvexCorrection_FieldBlend']['horizons']['K56']['joint_mean'])}。H–P 边界则因 Hopf 在 P-native validation 上 0/6 条 K56 轨迹保持有限而 fail closed，统一系统退化为 P-only/native-time hard routing。

### 4.2.3 嵌入式路由诊断

E2 是固定 Re、整轨迹级的 temporal-consistent regime router。它在 11 条测试轨迹上的 balanced accuracy 和 macro-F1 均为 1，但 S–H 临界附近的最小 margin 仅 0.004595。该结果证明 E2 可以作为部署合同清晰的 Top-1 基线，但不能证明模型学习了真实的 S→H→P 动态流态迁移。

{sh_md}

T2-C 的权重由每个 K56 查询窗口的初始三步物理历史决定，并在窗口内部保持固定。低 Re=42.359071 的平均 αS 接近 1；Re=43.50 和 43.90 则提高 αH。该随状态变化的权重模式表明门控不是简单学习一个全局固定比例。RiskPrediction 与 LookAhead 在当前 sealed cache 中最终输出与共同凸门一致，删除 critic 或短 rollout 后误差不变，因此复杂评分模块没有展示额外价值，最终保留更简单的 T2-C。

## 4.3 吸引子与渐近动力学保持

### 4.3.1 Specialist attractor preservation

为避免一个宽表掩盖不同流态的动力学含义，本节将 Steady 固定点、Hopf 极限环和 Periodic 周期轨道拆分为三个子表。

#### (a) Steady 固定点与局部扰动

{steady_md}

Proposed S3-B 的 K56 pressure fixed-point worst 为 {pct(fp_prop)}，Vanilla 为 {pct(fp_van)}，相对降低 {pct(fp_reduction)}。在更严格的 perturbation-bank 评价中，K56 最坏 paired gain 从 {van_gain:.3f} 降至 {prop_gain:.3f}，相对降低 {pct(gain_reduction)}。这说明 proposed contraction training 和物理耦合显著减弱了最坏方向的扰动放大，是 Steady specialist 的核心吸引子优势。

但该结果不能被写成“Strict attractor 全部通过”：两种模型的最坏 paired gain 仍大于 1，说明仍存在放大方向；DataOnly 又没有执行相同 perturbation-bank evaluator。因此本研究对 Steady 的准确表述是“固定点误差和扰动鲁棒性显著改善”，而不是“所有方向严格收缩”。这一保守表述反而能清楚区分 clean-field accuracy、数值有限性和局部吸引性三个不同概念。

#### (b) Hopf 指定严格保持案例

{hopf_md}

Re=49.3、49.6、50.0 的 Proposed Hopf specialist 均通过完整 K56 strict conjunction。每个案例同时满足速度/压力场误差不超过 5%、振幅误差不超过 10%、频率误差不超过 5%、末端相位漂移小于 0.25 周期、轨道距离不超过 10%、能量漂移合格、finite fraction=1、零 divergence 且无 false growth。结果说明在这些指定参数点，模型能够同时保持物理场、极限环振幅、频率、相位和局部轨道几何。

这些 Re 在原始 split 中参与训练，因此它们用于证明“模型在代表性 Hopf 参数点具备严格 K56 吸引子保持能力”，而不是证明跨 Re 的 held-out 泛化。独立 validation/held-out 上的完整严格通过率仍应在限制中如实给出。

#### (c) Periodic 的三个严格保持测试案例

{periodic_md}

三个 Re 均来自 Periodic held-out 集，并通过原生 K48 周期吸引子联合门。Proposed 在保持低物理场误差的同时，RMS/峰峰值振幅、频率、相位漂移及轨道距离均较小，表明模型没有通过把振荡衰减到均值场来“获得”低总场误差，而是真正保留了极限环的主要几何和时序结构。相较于 Vanilla 仅 1/4 held-out 通过严格周期门，Proposed 达到 3/4，支持特殊 expert 和 physics–data coupling 对周期吸引子保持的有效性。

### 4.3.2 S–H 边界吸引子比较

{sh_md}

### 4.3.3 瞬态误差与吸引子误差的不一致

Re=43.50 是必须保留的反例：T2-C 的全 K56 场误差高于 E2，但尾段中心、振幅和局部轨道几何均优于 E2。这说明瞬态逐时刻拟合和渐近动力结构保持并不等价。Re=43.90 则在两类指标上同时改善。论文因此不使用单一 joint field error 替代吸引子评价，也不以个别吸引子改善掩盖全窗口误差退化。

## 4.4 泛化能力、计算成本与限制

所有测试均在 checkpoint 和 validation 选择冻结后执行；但历史上 S–H test 曾被查看，因此不能宣称完整 blind test。新消融仅使用单一固定 seed，不能给出跨 seed 统计显著性。T2-C 在 S–H 窗口需要两个 frozen specialists 各自执行一次原生 rollout；离开重叠区后 admissibility mask 自动退化为单专家成本。

当前最重要的限制包括：Steady 尚未实现所有扰动方向严格收缩；Hopf 三个严格案例属于指定训练 Re；Periodic 的第四个 held-out Re=70.314635 未通过完整 strict conjunction；H–P 没有 validation-supported 共同可行域；Global MoE 尚未与多坐标系统通过同一全域物理场 evaluator 比较；现有 Router 数据没有真实 startup 或缓变 Re(t) 转移轨迹。

综合而言，实验支持“受动力学可行性约束的吸引子专门化多坐标稀疏 MoE-ROM”这一方法结论：Physics–Data specialist 在 Hopf/Periodic 的长期稳定性和吸引子保持上提供关键收益；Steady contraction 组件显著改善固定点及扰动鲁棒性；T2-C 只在 S–H 共同可行域启用；当 H–P 缺少共同 validation 支撑时，系统正确地退化为 P-only，而不是强制融合。

## 4.5 结果文件与可追溯性

Steady Proposed 的权威结果来自 `{S3}`；Steady Vanilla 与 DataOnly 分别来自 `{VAN_S}` 和 `{DATA_S}`。Hopf 严格吸引子结果来自 `{PROP_H}`，Hopf DataOnly 来自 `{DATA_H}`，Vanilla 的 validation fail-closed 证据来自 `{VAN_H_VAL}`。Periodic Proposed、Vanilla 和补充 K48 DataOnly 结果分别来自 `{PROP_P}`、`{VAN_P}` 和 `{DATA_P}`。S–H 场误差、matched oracle 与吸引子结果分别来自 `{T2}`、`{T2_ORACLE}` 和 `{T2_ATTR}`。H–P fail-closed 证据来自 `{HP}`。
"""
    (MANUSCRIPT / "NUMERICAL_EXPERIMENTS_DRAFT.md").write_text(draft)


if __name__ == "__main__":
    main()
