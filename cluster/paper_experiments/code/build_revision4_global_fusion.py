#!/usr/bin/env python3
"""Build revision 4 without modifying the frozen revision-3 manuscript."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import rewrite_chinese_manuscript as base


ROOT = base.ROOT
OUT = ROOT / "paper_experiments/revisions/revision4_global_fusion_20260724"
MANUSCRIPT = OUT / "manuscript"
REPORTS = OUT / "reports"
TABLES = REPORTS / "tables"


def load(path: Path):
    return json.loads(path.read_text())


def write_csv(name: str, headers, rows):
    path = TABLES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def closest(items, re_value):
    return min(items, key=lambda row: abs(float(row["test_Re"]) - re_value))


def global_rows():
    payload = load(base.GLOBAL)
    values = {
        "Steady": [24.630436, 32.740067, 39.685478],
        "Hopf": [47.081356, 49.022358, 51.786449],
        "Periodic": [70.314636, 100.352249, 149.059235, 189.862274],
    }
    output = {}
    for regime, re_values in values.items():
        rows = [closest(payload["results"], value) for value in re_values]
        u = mean(row["rollout_autonomous_pressure"]["a_relative_l2_mean"] for row in rows)
        p = mean(row["rollout_autonomous_pressure"]["b_relative_l2_mean"] for row in rows)
        output[regime] = {
            "re": re_values,
            "u": u,
            "p": p,
            "joint": 0.5 * (u + p),
            "rows": rows,
        }
    return output


def long_horizon_table():
    rows = []
    for regime, method, re_values, horizon, result in base.specialist_long_table():
        rows.append(
            [
                regime,
                method,
                re_values,
                horizon,
                "面积加权物理场",
                "验证失败，未开放 test" if result is None else base.pct(result["u"]),
                "—" if result is None else base.pct(result["p"]),
                "—" if result is None else base.pct(result["joint"]),
                "—" if result is None else base.pct(result["worst"]),
                "—" if result is None else base.num(result["finite"], 3),
                "—" if result is None else str(result["div"]),
            ]
        )
    for regime, result in global_rows().items():
        rows.append(
            [
                regime,
                "Global MoE†",
                ", ".join(f"{value:.6f}" for value in result["re"]),
                "K24",
                "POD 系数域†",
                base.pct(result["u"]),
                base.pct(result["p"]),
                base.pct(result["joint"]),
                "—",
                "—",
                "—",
            ]
        )
    order = {"Steady": 0, "Hopf": 1, "Periodic": 2}
    rows.sort(key=lambda row: (order[row[0]], row[1] == "Global MoE†", row[1]))
    headers = [
        "流态",
        "方法",
        "测试 Re",
        "长时域",
        "误差空间",
        "速度误差",
        "压力误差",
        "联合误差",
        "worst",
        "finite",
        "发散窗",
    ]
    write_csv("TABLE_4_2_1_GLOBAL_INCLUDED.csv", headers, rows)
    return base.md_table(headers, rows)


def steady_attractor_table():
    rows, _ = base.steady_attractor_rows()
    output = []
    for method, re_value, u, p, fp, drift, finite, divergent in rows:
        output.append(
            [
                method,
                f"{re_value:.6f}",
                "K56 / 物理场",
                base.pct(u),
                base.pct(p),
                base.pct(fp),
                base.pct(drift),
                base.num(finite, 3),
                str(divergent),
                "未建立三方法同口径二值门",
            ]
        )
    for row in global_rows()["Steady"]["rows"]:
        metric = row["rollout_autonomous_pressure"]
        amp = metric["hopf_amplitude"]
        output.append(
            [
                "Global MoE†",
                f"{row['test_Re']:.6f}",
                "K24 / POD 系数域†",
                base.pct(metric["a_relative_l2_mean"]),
                base.pct(metric["b_relative_l2_mean"]),
                "—",
                f"压力能量 {base.pct(metric['b_energy_relative_error'])}; "
                f"伪振荡峰值比 {amp['overshoot_ratio_max']:.2f}×",
                "—",
                "—",
                "未运行 Steady 同口径 strict evaluator",
            ]
        )
    headers = [
        "方法",
        "评价 Re",
        "时域/误差空间",
        "速度误差",
        "压力误差",
        "压力固定点误差",
        "漂移/诊断",
        "finite",
        "发散窗",
        "Strict 结论",
    ]
    write_csv("TABLE_4_3_1A_STEADY_GLOBAL_INCLUDED.csv", headers, output)
    return base.md_table(headers, output)


def hopf_attractor_table():
    proposed = {round(row[0], 3): row for row in base.hopf_strict_rows()}
    output = []
    unavailable = {
        "Vanilla-FNN-MoE": "validation K16/K56 非有限，按 fail-closed 合同未开放指定案例评估",
        "DataOnly-MoE": "冻结归档未在 49.3/49.6/50.0 运行同一 strict evaluator",
        "Global MoE†": "Global held-out 资产不含这三个精确 Re，且仅有 K24 系数域 evaluator",
    }
    for re_value in (49.3, 49.6, 50.0):
        row = proposed[round(re_value, 3)]
        (
            _re,
            u,
            p,
            rms,
            p2p,
            freq,
            phase,
            orbit,
            u_energy,
            p_energy,
            finite,
            divergent,
            false_growth,
            strict,
        ) = row
        output.append(
            [
                f"{re_value:.1f}",
                "Proposed Specialist MoE",
                "K56 / 物理场",
                base.pct(u),
                base.pct(p),
                base.pct(rms),
                base.pct(p2p),
                base.pct(freq),
                base.num(phase, 4),
                base.pct(orbit),
                f"u {base.pct(u_energy)}, p {base.pct(p_energy)}",
                f"{base.num(finite, 3)} / {divergent}",
                "否" if not false_growth else "是",
                "PASS" if strict else "FAIL",
            ]
        )
        for method, reason in unavailable.items():
            output.append(
                [
                    f"{re_value:.1f}",
                    method,
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    reason,
                    "—",
                    "—",
                    "N/A",
                ]
            )
    headers = [
        "指定 Re",
        "方法",
        "时域/误差空间",
        "速度误差",
        "压力误差",
        "RMS 振幅",
        "峰峰值",
        "频率误差",
        "末端相位漂移(周期)",
        "轨道距离",
        "能量漂移/不可用原因",
        "finite/发散窗",
        "false growth",
        "Strict",
    ]
    write_csv("TABLE_4_3_1B_HOPF_ALL_METHODS.csv", headers, output)
    return base.md_table(headers, output)


def native_periodic_map(path: Path):
    return {round(float(row["Re"]), 3): row for row in load(path)["results"]}


def periodic_attractor_table():
    wanted = (100.352249, 149.059235, 189.862274)
    proposed = native_periodic_map(base.PROP_P)
    vanilla = native_periodic_map(base.VAN_P)
    dataonly = {
        round(float(key), 3): value["k48"]
        for key, value in load(base.DATA_P)["by_re"].items()
    }
    global_map = {
        round(float(row["test_Re"]), 3): row
        for row in load(base.GLOBAL)["results"]
    }
    output = []
    for re_value in wanted:
        key = round(re_value, 3)
        for method, mapping in (
            ("Proposed Specialist MoE", proposed),
            ("Vanilla-FNN-MoE", vanilla),
        ):
            row = mapping[key]
            field = row["horizons"]["48"]
            attr = row["k48_cycle_diagnostics"]
            output.append(
                [
                    f"{re_value:.6f}",
                    method,
                    "K48 / 物理场",
                    base.pct(field["velocity_area_weighted_l2"]),
                    base.pct(field["pressure_area_weighted_l2"]),
                    base.pct(attr["rms_amplitude_error"]),
                    base.pct(attr["peak_to_peak_amplitude_error"]),
                    base.pct(attr["strouhal_error"]),
                    base.num(attr["phase_rms_rad"], 4),
                    base.num(attr["terminal_cycle_drift"], 4),
                    base.pct(attr["normalized_orbit_distance"]),
                    f"{base.num(field['finite_fraction'], 3)} / {field['divergent_windows']}",
                    "PASS" if row["preserved"] else "FAIL",
                ]
            )
        data = dataonly[key]
        output.append(
            [
                f"{re_value:.6f}",
                "DataOnly-MoE",
                "K48 / 物理场",
                base.pct(data["velocity_physical_relative_l2"]),
                base.pct(data["pressure_physical_relative_l2"]),
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                f"{base.num(data['finite_fraction'], 3)} / {data['divergent_windows']}",
                "N/A：未运行周期 strict evaluator",
            ]
        )
        global_row = global_map[key]["rollout_autonomous_pressure"]
        attr = global_row["hopf_amplitude"]
        output.append(
            [
                f"{re_value:.6f}",
                "Global MoE†",
                "K24 / POD 系数域†",
                base.pct(global_row["a_relative_l2_mean"]),
                base.pct(global_row["b_relative_l2_mean"]),
                base.pct(attr["amplitude_relative_l2"]),
                "—",
                base.pct(attr["frequency_increment_mae"]),
                base.num(attr["phase_abs_error_mean"], 4),
                "—",
                "—",
                "—",
                "N/A：时域与 strict 定义不同",
            ]
        )
    headers = [
        "测试 Re",
        "方法",
        "时域/误差空间",
        "速度误差",
        "压力误差",
        "RMS 振幅",
        "峰峰值",
        "频率误差",
        "相位误差",
        "末周期漂移",
        "轨道距离",
        "finite/发散窗",
        "Strict",
    ]
    write_csv("TABLE_4_3_1C_PERIODIC_ALL_METHODS.csv", headers, output)
    return base.md_table(headers, output)


def sh_per_re_table():
    oracle = load(base.T2_ORACLE)["per_Re"]
    attractor = load(base.T2_ATTR)["per_Re"]
    rows = []
    for key in ("42.359071", "43.5", "43.9"):
        for source, label in (("E2_Top1", "E2 Top-1"), ("T2_C", "T2-C")):
            error = oracle[key][source]["K56_mean"]
            attr = attractor[key][source]
            weights = oracle[key].get("T2_C_weights") if source == "T2_C" else None
            rows.append(
                [
                    f"{float(key):.6f}",
                    label,
                    base.pct(error),
                    base.pct(attr["tail_center_field_error"]["mean"]),
                    base.pct(attr["tail_amplitude_absolute_error"]["mean"]),
                    base.pct(attr["tail_orbit_geometry_error"]["mean"]),
                    "—" if weights is None else base.num(weights["alpha_S_mean"], 4),
                    "—" if weights is None else base.num(weights["alpha_H_mean"], 4),
                ]
            )
    headers = [
        "测试 Re",
        "方法",
        "K56 联合误差",
        "尾段中心误差",
        "振幅绝对误差",
        "轨道几何误差",
        "αS",
        "αH",
    ]
    write_csv("TABLE_4_4_2_SH_PER_RE.csv", headers, rows)
    return base.md_table(headers, rows)


def fusion_summary_table():
    t2 = load(base.T2)
    t2_attr = load(base.T2_ATTR)["aggregate"]
    e2 = t2["baselines"]["E2_pair_Top1"]["horizons"]["K56"]
    t2c = t2["routes"]["T2-C_LearnedConvexCorrection_FieldBlend"]["horizons"]["K56"]
    periodic = native_periodic_map(base.PROP_P)
    p_rows = [periodic[round(value, 3)] for value in (70.314635, 100.352249, 149.059235, 189.862274)]
    p_u = mean(row["horizons"]["48"]["velocity_area_weighted_l2"] for row in p_rows)
    p_p = mean(row["horizons"]["48"]["pressure_area_weighted_l2"] for row in p_rows)
    p_joint = 0.5 * (p_u + p_p)
    global_data = global_rows()
    rows = [
        [
            "S–H",
            "E2 Top-1",
            "42.359071, 43.500000, 43.900000",
            "K56 / 物理场",
            base.pct(e2["velocity_mean"]),
            base.pct(e2["pressure_mean"]),
            base.pct(e2["joint_mean"]),
            f"尾段中心 {base.pct(t2_attr['E2_Top1']['tail_center_field_error']['mean'])}; "
            f"轨道 {base.pct(t2_attr['E2_Top1']['tail_orbit_geometry_error']['mean'])}",
            "Top-1",
        ],
        [
            "S–H",
            "T2-C",
            "42.359071, 43.500000, 43.900000",
            "K56 / 物理场",
            base.pct(t2c["velocity_mean"]),
            base.pct(t2c["pressure_mean"]),
            base.pct(t2c["joint_mean"]),
            f"尾段中心 {base.pct(t2_attr['T2_C']['tail_center_field_error']['mean'])}; "
            f"轨道 {base.pct(t2_attr['T2_C']['tail_orbit_geometry_error']['mean'])}",
            "共同可行域内窗口级凸融合",
        ],
        [
            "S–H",
            "Global MoE†",
            "无同一 sealed-cache 结果",
            "K24 / POD 系数域†",
            "—",
            "—",
            "—",
            "—",
            "不能用其他 Re 替代边界 test",
        ],
        [
            "H–P",
            "E2 Top-1",
            "70.314635, 100.352249, 149.059235, 189.862274",
            "K48 / 物理场",
            base.pct(p_u),
            base.pct(p_p),
            base.pct(p_joint),
            "Periodic strict 3/4 PASS",
            "P-only/native-time hard routing",
        ],
        [
            "H–P",
            "T2-C（admissibility 退化）",
            "70.314635, 100.352249, 149.059235, 189.862274",
            "K48 / 物理场",
            base.pct(p_u),
            base.pct(p_p),
            base.pct(p_joint),
            "Periodic strict 3/4 PASS",
            "mH=0, mP=1；未训练 H–P 凸门",
        ],
        [
            "H–P",
            "Global MoE†",
            "70.314636, 100.352249, 149.059235, 189.862274",
            "K24 / POD 系数域†",
            base.pct(global_data["Periodic"]["u"]),
            base.pct(global_data["Periodic"]["p"]),
            base.pct(global_data["Periodic"]["joint"]),
            "仅 Global 原生振幅/相位诊断；无同口径 strict 判定",
            "全局单模型；不可与 K48 物理场直接排名",
        ],
    ]
    headers = [
        "边界",
        "方法",
        "评价 Re",
        "时域/误差空间",
        "速度误差",
        "压力误差",
        "联合误差",
        "吸引子/渐近指标",
        "实际机制与结论",
    ]
    write_csv("TABLE_4_4_1_FUSION_LONG_AND_ATTRACTOR.csv", headers, rows)
    return base.md_table(headers, rows)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    long_md = long_horizon_table()
    steady_md = steady_attractor_table()
    hopf_md = hopf_attractor_table()
    periodic_md = periodic_attractor_table()
    fusion_md = fusion_summary_table()
    sh_md = sh_per_re_table()

    draft = f"""# 数值实验部分草稿（中文 revision 4：Global 与边界融合重组版）

> 本文件是新版本，不覆盖 revision 3。所有相对误差均为原始 relative error × 100%。本文只展示 KLong。带 † 的 Global MoE 数值来自其冻结归档的 K24 POD 系数域 evaluator；其余 specialist 与融合结果来自 K48/K56 面积加权物理场 evaluator。两类数值用于并列呈现模型行为，但不能直接作等口径大小排序。

## 4.1 实验协议、数据隔离与评价口径

Steady、Hopf、Periodic specialist 分别保留自己的 POD 基、均值场、scaler、Galerkin 算子、pressure closure、history builder 和原生积分合同。Vanilla-FNN-MoE 仅替换 expert 网络；DataOnly-MoE 移除连续时间物理推进并学习离散映射；Global MoE 使用一套全局 POD 空间和统一全域网络。所有可用于 checkpoint 选择的决策只使用 train/validation，测试集不用于调参。

KLong 在 Steady/Hopf specialist 中为 K56，在 Periodic specialist 中为 K48，在冻结 Global MoE 归档中为 K24。它们是各自原生数据库步数，不代表统一物理时间。由于原始采样合同和 evaluator 不同，本节保留“时域/误差空间”列，避免把 K24 系数误差伪装为 K56 物理场误差。

## 4.2 长时物理场预测

### 4.2.1 流态专用 specialist、消融与 Global MoE

{long_md}

Global MoE 已按要求纳入本表，并明确给出速度与压力误差。其 K24 结果在三个流态上分别反映全局模型的速度/压力 POD 系数 rollout 误差；由于没有经过同一 K48/K56 物理场重构 evaluator，本稿不据此宣称 Global 与 specialist 的严格数值排名。这个限制是测量口径限制，而不是删除 Global 结果。

Steady 的 Proposed 权威模型是 S3-B contraction checkpoint。在共同可评价的三个 Re 上，Vanilla 的 clean-field 平均误差较低，因此不能声称 Proposed 在 Steady 所有场误差上全面占优；Proposed 的主要优势应结合固定点和扰动吸引性判断。Hopf 中，Proposed 在 held-out K56 保持 finite=1 且零发散，而 DataOnly 出现 39 个 norm-divergent windows，Vanilla 在 validation 已非有限。Periodic 中，Proposed 在相同四个 held-out Re 的 K48 联合误差和 worst-case 均优于 Vanilla，并在吸引子联合门上取得更高通过率。

### 4.2.2 长时场误差的解释边界

物理场误差衡量整段轨迹的逐时刻预测质量，但不能单独判断渐近吸引子的中心、振幅、频率、相位和轨道几何是否保持。尤其在 S–H 边界 Re=43.50，T2-C 的全窗口误差略有退化，但尾段中心和轨道几何明显改善。因此下一节将 attractor preservation 独立列出，而不把它折叠进单一 joint error。

## 4.3 吸引子与渐近动力学保持

### 4.3.1 Specialist attractor preservation

#### (a) Steady 固定点与局部扰动

{steady_md}

Proposed S3-B 的 K56 pressure fixed-point worst 为 27.1317%，Vanilla 为 49.4685%；更严格的 perturbation-bank K56 最坏 paired gain 从 107702.891 降至 585.262。该结果支持“固定点与扰动鲁棒性显著改善”，但由于最坏 gain 仍大于 1，不支持“所有扰动方向严格收缩”。Global 的低 Re 归档出现较大的伪振荡 overshoot，因此其存在于表中并不等于通过 Steady strict-attractor 判据。

#### (b) Hopf 指定严格 K56 保持案例

{hopf_md}

Proposed 在 Re=49.3、49.6、50.0 上均通过完整 strict conjunction，包括物理场、振幅、频率、相位、轨道、能量、finite、divergence 和 false-growth 条件。这三个 Re 参与原始训练，故其科学含义是证明代表性参数点的严格保持能力，而不是 held-out 泛化。Vanilla、DataOnly 和 Global 已在同一子表中出现；然而现有冻结产物没有三者在这三个精确 Re 上的同口径 strict 结果，必须标为 N/A，不能用邻近 Re 或不同 horizon 冒充。因而目前可宣称 Proposed 的三个严格案例，但不能据该表宣称它在这三个案例上定量优于全部消融。

#### (c) Periodic 三个严格保持测试案例

{periodic_md}

Periodic 的三条 Re 均为 held-out。Proposed 在三者上均 PASS；Vanilla 只在 Re=100.352249 上 PASS，在更高 Re 的相位/轨道或压力相关条件下失败。DataOnly 在相同 Re 有 K48 物理场误差，但没有运行周期 strict evaluator；Global 在相同 Re 有 K24 系数域振幅、频率和相位诊断，但 horizon 与定义不同。因此表中同时展示四种方法，并把“缺少同口径认证”和“实际 FAIL”严格区分。

### 4.3.2 吸引子优势的综合解释

Hopf 的核心证据是 Proposed 在指定案例中同时保持振幅、频率、相位和轨道，而不是只获得很小的瞬态场误差；Periodic 的核心证据是 Proposed 在三个 held-out Re 通过联合门，并将严格通过率从 Vanilla 的 1/4 提升到 3/4。Steady 的证据性质不同：它表现为固定点误差与扰动放大显著下降，但尚未达到全方向严格收缩。三类流态因此不能共用一个未经定义的“Strict”标签。

## 4.4 Admissibility-constrained 边界融合算法

### 4.4.1 统一边界机制

边界层先执行动力学 admissibility gate，再决定 Top-1 或 Top-2。S–H 的 S-native development 域同时支持冻结 S/H K56 rollout，因此允许 T2-C 在公共物理场做窗口条件凸融合。H–P 的 P-native validation 中，Hopf 为 0/6 finite K56，而 Periodic 合法，因此 mask 固定为 mH=0、mP=1；所谓 T2-C 系统在此边界机械退化为 P-only，而不是训练一个不合法的 H–P 融合门。

### 4.4.2 长时物理场与吸引子联合对比

{fusion_md}

这张表同时覆盖 S–H 和 H–P。S–H 中 T2-C 将平均 K56 联合物理场误差从 4.6710% 降至 3.0450%，并改善平均尾段中心与轨道几何；Global 尚未在同一 S–H sealed cache 上评估，所以不能填入伪造的边界数值。H–P 中 E2 与 admissibility-constrained T2-C 产生相同的 P-only 输出：这不是缺失实验，而是 mH=0 后的预注册稀疏退化。Global 在 Periodic held-out Re 有结果，但仍是 K24 系数域口径。

### 4.4.3 S–H 逐 Re 行为与权重

{sh_md}

T2-C 并非在每个 Re 上都改善全窗口误差。Re=42.359071 基本退化为 S-only；Re=43.50 增大 H 权重后吸引子指标改善但 K56 总误差上升；Re=43.90 则同时改善总误差与渐近指标。这一逐 Re 结果支持“能力感知、状态条件融合”，同时否定“融合必然逐点支配”的过强表述。

### 4.4.4 H–P fail-closed 结果

H–P 当前仅允许 P-only/native-time hard routing。Hopf 在 P-native validation 没有共同稳定支持域，因此 RiskPrediction、LookAhead 和 T2-C 均不应启动。论文应把该结果表述为统一 admissibility 原则的有效输出：当候选 specialist 不具备有限、稳定、validation-supported rollout 时，其融合权重归零。

## 4.5 限制与可支持的论文结论

本轮使用单一固定 seed，不能声称跨 seed 显著性；历史 S–H test 曾被查看，不能声称全程 blind test；Hopf 三个指定严格案例属于训练 Re；Global 尚未通过与 specialist 完全相同的 K48/K56 物理场 evaluator；H–P 没有共同可行域；现有 Router 数据不包含经认证的真实跨流态连续迁移轨迹。

实验支持的结论是：physics–data specialist 在 Hopf/Periodic 的长期稳定性和吸引子保持上提供关键收益；Steady contraction 机制明显改善固定点与扰动鲁棒性；T2-C 只在 S–H 共同可行域启用；H–P 按 admissibility mask 退化为 P-only。实验不支持 Proposed 在所有指标全面支配、Steady 全方向严格收缩、Global 与多坐标系统已完成同口径全域排名，或 Router 已学习真实 S→H→P 时间迁移。

## 4.6 可追溯性

Revision 3 已在独立目录冻结并生成 SHA256；本 revision 4 只写入 `{OUT}`。Global 原始指标来自 `{base.GLOBAL}`；specialist、T2-C、attractor 与 H–P fail-closed 的来源路径均记录在同目录 `REVISION4_AUDIT.json`。
"""
    manuscript_path = MANUSCRIPT / "NUMERICAL_EXPERIMENTS_DRAFT_REVISION4.md"
    manuscript_path.write_text(draft)
    (REPORTS / "REVISION4_EXPERIMENT_REPORT_ZH.md").write_text(draft)

    audit = {
        "revision": "revision4_global_fusion_20260724",
        "does_not_overwrite": str(ROOT / "paper_experiments/manuscript/NUMERICAL_EXPERIMENTS_DRAFT.md"),
        "metric_contract": {
            "specialists_and_T2C": "area-weighted reconstructed physical-field relative error, K48/K56",
            "global_moe": "POD coefficient relative error, K24",
            "direct_numeric_ranking_allowed": False,
        },
        "sources": {
            "global": str(base.GLOBAL),
            "steady_proposed": str(base.S3),
            "steady_vanilla": str(base.VAN_S),
            "steady_dataonly": str(base.DATA_S),
            "hopf_proposed": str(base.PROP_H),
            "hopf_vanilla_validation": str(base.VAN_H_VAL),
            "hopf_dataonly": str(base.DATA_H),
            "periodic_proposed": str(base.PROP_P),
            "periodic_vanilla": str(base.VAN_P),
            "periodic_dataonly": str(base.DATA_P),
            "sh_test": str(base.T2),
            "sh_attractor": str(base.T2_ATTR),
            "hp_fail_closed": str(base.HP),
        },
        "known_missing_same_contract_results": {
            "hopf_49p3_49p6_50_vanilla_dataonly_global": True,
            "global_on_sh_sealed_cache": True,
            "global_k48_k56_physical_field": True,
        },
    }
    (OUT / "REVISION4_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    lines = [f"{sha256(path)}  {path.relative_to(OUT).as_posix()}" for path in files]
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    print(manuscript_path)
    print((OUT / "SHA256SUMS.txt").read_text())


if __name__ == "__main__":
    main()
