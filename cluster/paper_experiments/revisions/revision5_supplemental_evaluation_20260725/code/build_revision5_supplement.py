from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
DOWNLOADS = Path(__file__).resolve().parent / "revision5_downloads"
OUT = (
    WORKSPACE
    / "final_paper_reports"
    / "revisions"
    / "revision5_supplemental_evaluation_20260725"
)
REV4 = (
    WORKSPACE
    / "final_paper_reports"
    / "revisions"
    / "revision4_global_fusion_20260724"
    / "manuscript"
    / "NUMERICAL_EXPERIMENTS_DRAFT_REVISION4.md"
)


def load(name: str) -> dict:
    return json.loads((DOWNLOADS / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{100.0 * float(value):.{digits}f}%"


def num(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def re_key(value: float) -> str:
    return f"{float(value):.6f}"


def nearest(mapping: dict[str, dict], value: float, tolerance: float = 5.0e-4) -> dict:
    key = min(mapping, key=lambda candidate: abs(float(candidate) - float(value)))
    if abs(float(key) - float(value)) > tolerance:
        raise KeyError(f"No Re within {tolerance} of {value}; closest={key}")
    return mapping[key]


def global_status(row: dict) -> str:
    metric = row["K56"]
    attempted = int(metric.get("attempted_windows", 1))
    completed = int(metric.get("num_windows", 0))
    divergent = int(metric.get("divergent_windows", attempted - completed))
    if completed == 0:
        return f"FAIL-CLOSED（{completed}/{attempted} finite，发散 {divergent}）"
    return f"{completed}/{attempted} finite，发散 {divergent}"


def build() -> None:
    global_data = load("GLOBAL_K56_PHYSICAL_COMPACT.json")
    dataonly = load("DATAONLY_MISSING_CONTRACTS.json")
    steady = load("STEADY_PROPOSED_STRICT_PRESSURE_SCAN.json")
    selection = load("FIXED_POINT_QUALIFIED_SELECTION.json")

    global_by_re = {re_key(row["Re"]): row for row in global_data["results"]}
    proposed_by_re: dict[str, dict] = {}
    for split, split_payload in steady["splits"].items():
        for row in split_payload["rows"]:
            proposed_by_re[re_key(row["Re"])] = {**row, "split": split}
    dataonly_steady = dataonly["results"]["steady"]["by_re"]

    global_rows = []
    for row in global_data["results"]:
        metric = row["K56"]
        global_rows.append(
            "| "
            + " | ".join(
                [
                    re_key(row["Re"]),
                    str(row["regime"]),
                    pct(metric["velocity_area_weighted_physical_relative_l2"]),
                    pct(metric["pressure_area_weighted_physical_relative_l2"]),
                    pct(metric["velocity_coefficient_relative_l2"]),
                    pct(metric["pressure_coefficient_relative_l2"]),
                    global_status(row),
                ]
            )
            + " |"
        )

    regime_sets = {
        "Steady": [24.630436, 32.740067, 39.685478, 45.142704],
        "Hopf": [47.081356, 49.022358, 51.786449],
        "Periodic": [70.314636, 100.352249, 149.059235, 189.862274],
    }
    global_aggregate_rows = []
    for regime, re_values in regime_sets.items():
        rows = [nearest(global_by_re, value) for value in re_values]
        metrics = [row["K56"] for row in rows]
        global_aggregate_rows.append(
            "| "
            + " | ".join(
                [
                    regime,
                    ", ".join(f"{value:.6f}" for value in re_values),
                    pct(
                        sum(
                            row["velocity_area_weighted_physical_relative_l2"]
                            for row in metrics
                        )
                        / len(metrics)
                    ),
                    pct(
                        sum(
                            row["pressure_area_weighted_physical_relative_l2"]
                            for row in metrics
                        )
                        / len(metrics)
                    ),
                    pct(
                        sum(row["velocity_coefficient_relative_l2"] for row in metrics)
                        / len(metrics)
                    ),
                    pct(
                        sum(row["pressure_coefficient_relative_l2"] for row in metrics)
                        / len(metrics)
                    ),
                    str(sum(int(row["num_windows"]) for row in metrics)),
                ]
            )
            + " |"
        )

    steady_scan_rows = []
    for split in ("train", "validation", "heldout"):
        for row in steady["splits"][split]["rows"]:
            steady_scan_rows.append(
                "| "
                + " | ".join(
                    [
                        split,
                        re_key(row["Re"]),
                        pct(row["K56_velocity_physical_relative_l2"]),
                        pct(row["K56_pressure_physical_relative_l2"]),
                        pct(row["pressure_fixed_point_physical_relative_l2"]),
                        pct(row["pressure_drift_coefficient_relative_l2"]),
                        str(int(row["divergent_windows"])),
                        "PASS" if row["strict_pressure_pass"] else "FAIL",
                    ]
                )
                + " |"
            )

    fixed_rows = []
    fixed_re = (
        selection["fixed_point_qualified_only"]["train_Re"]
        + selection["fixed_point_qualified_only"]["test_Re"]
    )
    for re_value in fixed_re:
        key = re_key(re_value)
        proposed = nearest(proposed_by_re, re_value)
        data = nearest(dataonly_steady, re_value)
        data_k56 = data["K56"]
        data_diag = data["terminal_diagnostic"]
        glob = nearest(global_by_re, re_value)
        glob_k56 = glob["K56"]
        fixed_rows.extend(
            [
                "| "
                + " | ".join(
                    [
                        key,
                        proposed["split"],
                        "Proposed Specialist MoE",
                        pct(proposed["K56_velocity_physical_relative_l2"]),
                        pct(proposed["K56_pressure_physical_relative_l2"]),
                        pct(proposed["pressure_fixed_point_physical_relative_l2"]),
                        pct(proposed["pressure_drift_coefficient_relative_l2"]),
                        f"{proposed['finite_fraction']:.3f}/{int(proposed['divergent_windows'])}",
                        "仅固定点合格",
                    ]
                )
                + " |",
                "| "
                + " | ".join(
                    [
                        key,
                        proposed["split"],
                        "DataOnly-MoE",
                        pct(data_k56["velocity_physical_relative_l2"]),
                        pct(data_k56["pressure_physical_relative_l2"]),
                        pct(data_diag["pressure_fixed_point_physical_relative_l2"]),
                        pct(data_diag["pressure_drift_coefficient_relative_l2"]),
                        f"{data_k56['finite_fraction']:.3f}/{int(data_k56['divergent_windows'])}",
                        "FAIL",
                    ]
                )
                + " |",
                "| "
                + " | ".join(
                    [
                        key,
                        proposed["split"],
                        "Global MoE†",
                        pct(glob_k56["velocity_area_weighted_physical_relative_l2"]),
                        pct(glob_k56["pressure_area_weighted_physical_relative_l2"]),
                        "—",
                        "—",
                        global_status(glob),
                        "K56有限；strict诊断不支持",
                    ]
                )
                + " |",
            ]
        )

    hopf_rows = []
    for re_value, row in dataonly["results"]["hopf"]["by_re"].items():
        metric = row["K56"]
        attr = row["attractor_K56"]
        hopf_rows.append(
            "| "
            + " | ".join(
                [
                    re_value,
                    row["split"],
                    pct(metric["velocity_area_weighted_physical_relative_l2"]),
                    pct(metric["pressure_area_weighted_physical_relative_l2"]),
                    pct(attr["rms_amplitude_error"]),
                    pct(attr["peak_to_peak_amplitude_error"]),
                    pct(attr["frequency_relative_error"]),
                    num(attr["terminal_phase_drift_cycles"]),
                    pct(attr["normalized_orbit_distance"]),
                    f"{metric['finite_fraction']:.3f}/{int(metric['divergent_windows'])}",
                    "是" if attr["false_growth"] else "否",
                    "PASS" if row["strict_preserved"] else "FAIL",
                ]
            )
            + " |"
        )

    periodic_rows = []
    for re_value, row in dataonly["results"]["periodic"]["by_re"].items():
        metric = row["K48"]
        attr = row["cycle_K48"]
        periodic_rows.append(
            "| "
            + " | ".join(
                [
                    re_value,
                    row["split"],
                    pct(metric["velocity_area_weighted_physical_relative_l2"]),
                    pct(metric["pressure_area_weighted_physical_relative_l2"]),
                    pct(attr["rms_amplitude_error"]),
                    pct(attr["peak_to_peak_amplitude_error"]),
                    pct(attr["strouhal_error"]),
                    num(attr["phase_rms_rad"]),
                    num(attr["terminal_cycle_drift"]),
                    pct(attr["normalized_orbit_distance"]),
                    f"{metric['finite_fraction']:.3f}/{int(metric['divergent_windows'])}",
                    "PASS" if row["strict_preserved"] else "FAIL",
                ]
            )
            + " |"
        )

    report = f"""# Revision 5：缺失评价补全与 Steady 压力固定点审计

> 本报告是独立补充版本，不覆盖 revision 3 或 revision 4。除相位（rad/周期）和计数外，所有相对误差均按 `relative error × 100%` 报告。Global MoE† 使用冻结的 all-Re checkpoint；DataOnly-MoE 使用各流态 validation-selected 冻结 checkpoint。本轮没有训练或修改任何 checkpoint。

## 1. 结论摘要

1. Global MoE† 已在其数据库中可精确匹配的 16 个 Re 上执行 K56，其中包含原 11 个 held-out Re 和 5 个额外 Steady 固定点候选。16 个 Re 均完成有限 K56；Steady 短轨迹各提供 1 个合法窗口，Hopf/Periodic 长轨迹各提供 2 个合法窗口。
2. Global 数据库不含精确的 Re=49.3、49.6、50.0。本报告不以近邻节点替代，所以这三个指定 Hopf 严格案例仍没有 Global 同 Re 结果。
3. DataOnly-MoE 已补做 Hopf 指定 Re=49.3、49.6、50.0 的 K56 strict evaluator：三点全部 FAIL，均出现 false growth、明显频率/相位/轨道误差以及 5 个 norm-divergent windows。
4. DataOnly-MoE 已补做 Periodic Re=100.352249、149.059235、189.862274 的 K48 strict cycle evaluator：三点全部 PASS。该结果必须如实保留；它说明 DataOnly 在这三个冻结测试案例上也能保持周期吸引子，不能再写成 N/A。
5. Proposed Steady 在 train/validation/heldout 全集上没有任何 Re 同时满足预注册的“压力固定点误差 ≤5% 且压力漂移 ≤5%”联合 strict 门。因此论文不能宣称 Steady 存在通过该联合门的严格压力固定点案例。
6. 为完成横向诊断，保留 6 个“仅固定点误差 <5%”的次级子集（train 五点、heldout 一点）。这 6 点不是 strict joint pass。DataOnly 在六点均严重失稳；Global 在六点完成有限 K56 且全场误差较低，但其数据库索引式终点续推合同不能生成与局部 Steady evaluator 相同的固定点/漂移诊断，故不授予 strict 标签。

## 2. Global MoE† K56 补充 rollout

### 2.1 与三个 specialist 测试集一一对应的聚合结果

| Regime | 测试 Re | 平均速度物理场误差 | 平均压力物理场误差 | 平均速度系数误差 | 平均压力系数误差 | 完整窗口数 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(global_aggregate_rows)}

### 2.2 逐 Re 结果

| Re | 流态标签 | 速度物理场误差 | 压力物理场误差 | 速度系数误差 | 压力系数误差 | K56 状态 |
|---:|---|---:|---:|---:|---:|---|
{chr(10).join(global_rows)}

Global 的物理场误差和 POD 系数误差同时列出，是因为二者回答不同问题：物理场相对误差包含 Re-specific mean field 的能量，而系数误差只衡量脉动/模态坐标。较小的全场误差不能被解释为同样小的模态动力学误差。初次恢复 evaluator 时还发现了一个窗口索引问题：原循环先按 K56 跳步再检查三步 history，会跳过 Steady 短轨迹唯一合法起点；最终结果使用“先筛合法 history 起点、再按 K56 stride 取窗”的修正版。

## 3. Proposed Steady 全 split 压力固定点与漂移扫描

联合 strict 门冻结为：finite fraction=1、发散窗口=0、压力固定点物理相对误差≤5%、压力漂移系数相对误差≤5%。

| Split | Re | 速度场误差 | 压力场误差 | 压力固定点误差 | 压力漂移 | 发散窗 | 联合 Strict |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(steady_scan_rows)}

联合门通过数为 **0**。最接近固定点条件的若干 Re 虽然终态压力固定点误差小于 5%，但漂移诊断仍显著高于 5%。这说明“终点靠近固定点”和“整段压力动力学无漂移”不能互换。

## 4. 固定点合格次级子集上的横向诊断

该表只使用 Proposed 的 `pressure fixed-point error <5%` 选择六个 Re；选择不读取其他方法结果。`finite/发散` 中前者为 finite fraction，后者为 divergence count。

| Re | Proposed split | 方法 | 速度场误差 | 压力场误差 | 压力固定点误差 | 压力漂移 | finite/发散 | 结论 |
|---:|---|---|---:|---:|---:|---:|---|---|
{chr(10).join(fixed_rows)}

Proposed 在六点均满足“固定点误差 <5%”，但不满足联合 strict 漂移门。DataOnly 的压力场、终态固定点和漂移误差均大幅恶化，并在每个 Re 产生一个 norm-divergent K56 窗口。Global 在这些 Re 上完成有限 K56，且含均值场的速度/压力全场误差较低；但冻结 Global wrapper 的 phase/time 与 history 依赖数据库索引，轨迹终点之后没有合法的下一索引，不能在不改变原合同的情况下从终态再续推 56 步。因此 Global 的固定点和漂移列严格保留为“—”，不能把普通轨迹 K56 场误差冒充 fixed-point strict 结果。

## 5. DataOnly-MoE 缺失吸引子评价

### 5.1 Hopf 指定案例（K56）

| Re | Split | 速度场误差 | 压力场误差 | RMS 振幅误差 | 峰峰值误差 | 频率误差 | 末端相位漂移/周期 | 轨道距离 | finite/发散 | false growth | Strict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
{chr(10).join(hopf_rows)}

DataOnly 在三个指定 Hopf 训练 Re 上全部 strict FAIL。虽然系数仍为有限数值，5 个窗口均越过预注册发散范数阈值；频率误差约 94%–97%，且出现 false growth。因此 Proposed 在这三个案例上的 strict K56 保持优势仍成立，并且现在已有同 Re、同 horizon 的 DataOnly 反事实对照。

### 5.2 Periodic 指定 held-out 案例（K48）

| Re | Split | 速度场误差 | 压力场误差 | RMS 振幅误差 | 峰峰值误差 | Strouhal 误差 | phase RMS/rad | 末周期漂移 | 轨道距离 | finite/发散 | Strict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
{chr(10).join(periodic_rows)}

DataOnly 在这三个 Periodic held-out Re 全部 strict PASS。Proposed 的优势因此应表述为多数误差指标更低或跨更广测试集合的鲁棒性，而不能声称只有 Proposed 能保持这三个周期吸引子。

## 6. 可用于论文的客观表述

- 可以写：Global MoE† 在全部受测数据库节点上能够完成有限 K56；其含均值全场误差较低，但脉动系数误差明显更大，且当前冻结接口不能执行与 Steady specialist 相同的终点 fixed-point continuation。
- 可以写：Proposed Hopf 在 Re=49.3、49.6、50.0 通过 strict K56；同 Re 的 DataOnly 全部失败，支持连续时间物理结构对 Hopf 渐近动力学保持的重要性。
- 可以写：DataOnly 在三个指定 Periodic held-out Re 也通过 strict K48，因此 Periodic 证据体现为误差大小与整体鲁棒性差异，而不是二元“能/不能保持”的绝对差异。
- 不可以写：Proposed Steady 已通过“固定点+漂移”联合严格门。本次完整扫描中通过数为 0。
- 不可以写：Global 在 Re=49.3、49.6、50.0 已被同 Re 测试；其数据库没有这些精确节点。

## 7. 可追溯性与限制

- Global K56 原始结果 SHA256：`{global_data['source_sha256']}`。
- Global compact SHA256：`{sha256(DOWNLOADS / 'GLOBAL_K56_PHYSICAL_COMPACT.json')}`。
- DataOnly 补充结果 SHA256：`{sha256(DOWNLOADS / 'DATAONLY_MISSING_CONTRACTS.json')}`。
- Proposed Steady 扫描 SHA256：`{sha256(DOWNLOADS / 'STEADY_PROPOSED_STRICT_PRESSURE_SCAN.json')}`。
- 固定点子集选择文件 SHA256：`{sha256(DOWNLOADS / 'FIXED_POINT_QUALIFIED_SELECTION.json')}`。
- 本轮仅单固定 seed；没有开展显著性统计。
- Hopf 指定 49.3/49.6/50.0 属于训练 Re，证明代表点保持能力，不证明 held-out 泛化。
- Global 的 POD、mean 和压力 closure 合同与局部 specialist 不同；本轮统一了 K56 和面积加权物理场误差定义，但并未把它改造成局部坐标图 specialist。
"""

    manuscript_supplement = f"""# 数值实验部分补充稿（revision 5）

> 本文件在 revision 4 基础上追加新的同 horizon 评价，不覆盖旧版本。全文相对误差统一为 `relative error × 100%`。

## 4.2.3 Global MoE† 的 K56 同时域复核

此前 Global MoE† 仅有 K24 系数域结果，因此只能作为不同口径的旁证。本轮保持 checkpoint、全局 POD、scaler、Galerkin 与 pressure closure 不变，将 rollout horizon 延长到 K56，并增加面积加权物理场误差。结果显示，Global 在四个 Steady held-out Re、三个 Hopf held-out Re、四个 Periodic held-out Re和五个额外 Steady 候选 Re 上均形成完整有限窗口。故原表中的 Global K56 “—”可以由新的逐 Re 数值替换。

需要强调的是，Global 全场误差以含均值场的物理范数归一化，而系数误差聚焦脉动模态；在 Hopf 区域，二者可相差数个数量级。因此应同时报告，不能只选较小的一列。Global 的 K56 成功也不等价于通过局部 specialist 的 strict attractor evaluator，因为后者还要求终点续推、漂移、相位和轨道等合同。

## 4.3.3 Steady 压力固定点联合门

我们对 Proposed Steady 的全部 train、validation 与 heldout Re 执行同一 K56 扫描。预注册联合门要求输出有限、零发散、压力固定点物理相对误差不超过 5%，且压力漂移系数相对误差不超过 5%。扫描没有产生任何联合 PASS。六个 Re 的终态固定点误差低于 5%，但漂移仍超阈值，因此只能称为“fixed-point-qualified”，不能称为严格压力固定点保持。

在这六个次级诊断点上，DataOnly 的压力预测和漂移显著恶化，并在每个 K56 窗口触发 norm-divergence；Global 完成有限 K56 且全场误差较小，但由于终点之后缺少合法数据库索引，不能执行同合同 fixed-point continuation。该对照说明 Proposed 的固定点锚定相对 DataOnly 确有实质作用，但现有证据支持的是相对鲁棒性改善，而不是严格渐近收敛证明，也不足以对 Global 授予或否决同一 strict 标签。

## 4.3.4 DataOnly 的 Hopf 与 Periodic 补充吸引子结果

在 Hopf 指定 Re=49.3、49.6、50.0 上，DataOnly 的 K56 strict 结果均为 FAIL：频率相对误差约为 94%–97%，末端相位漂移约为 -6.2 至 -6.7 个周期，且三个 Re 均出现 false growth。与 Proposed 在相同三点的 strict PASS 相比，该消融清楚表明，连续时间局部向量场、RK4、Galerkin 与压力物理结构对于 Hopf 吸引子保持不可由离散数据映射自动替代。

另一方面，在 Periodic held-out Re=100.352249、149.059235、189.862274 上，DataOnly 均通过 K48 strict cycle 门。这一负向/竞争性结果必须保留。它表明成熟周期区间的数据驱动离散映射在当前 horizon 上足以维持周期几何；Proposed 的论证重点应放在更低误差、跨流态一致性和边界稳定性，而不是声称 DataOnly 无法保持周期吸引子。

## 4.5.1 修订后的结论边界

综合补充实验，可支持的表述为：Proposed 的物理结构对 Hopf 严格保持和 Steady 固定点鲁棒性具有关键贡献；Global 单模型在 Steady K56 失稳；DataOnly 在 Hopf 指定案例失败，但在三个 Periodic held-out 案例成功。因而组件有效性具有流态依赖性，不能用单一“全面优于”概括。完整逐 Re 数值见 `REVISION5_SUPPLEMENTAL_EVALUATION_ZH.md`。
"""

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reports").mkdir(exist_ok=True)
    (OUT / "manuscript").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "final_inputs").mkdir(exist_ok=True)
    for name in (
        "GLOBAL_K56_PHYSICAL_COMPACT.json",
        "DATAONLY_MISSING_CONTRACTS.json",
        "STEADY_PROPOSED_STRICT_PRESSURE_SCAN.json",
        "FIXED_POINT_QUALIFIED_SELECTION.json",
    ):
        (OUT / "final_inputs" / name).write_bytes((DOWNLOADS / name).read_bytes())
    report_path = OUT / "reports" / "REVISION5_SUPPLEMENTAL_EVALUATION_ZH.md"
    supplement_path = (
        OUT / "manuscript" / "NUMERICAL_EXPERIMENTS_DRAFT_REVISION5_SUPPLEMENT.md"
    )
    report_path.write_text(report, encoding="utf-8")
    supplement_path.write_text(manuscript_supplement, encoding="utf-8")
    if REV4.exists():
        combined = REV4.read_text(encoding="utf-8") + "\n\n---\n\n" + manuscript_supplement
        (
            OUT / "manuscript" / "NUMERICAL_EXPERIMENTS_DRAFT_REVISION5_FULL.md"
        ).write_text(combined, encoding="utf-8")

    audit = {
        "schema": "revision5_supplement_audit/v1",
        "revision4_preserved": True,
        "inputs": {
            name: {
                "path": str(DOWNLOADS / name),
                "sha256": sha256(DOWNLOADS / name),
            }
            for name in (
                "GLOBAL_K56_PHYSICAL_COMPACT.json",
                "DATAONLY_MISSING_CONTRACTS.json",
                "STEADY_PROPOSED_STRICT_PRESSURE_SCAN.json",
                "FIXED_POINT_QUALIFIED_SELECTION.json",
            )
        },
        "outputs": {
            "report": str(report_path),
            "manuscript_supplement": str(supplement_path),
        },
        "scientific_decisions": {
            "steady_joint_strict_pass_count": 0,
            "steady_fixed_point_qualified_secondary_count": len(fixed_re),
            "no_nearest_Re_substitution": True,
            "global_exact_hopf_49p3_49p6_50_available": False,
            "global_window_selection": (
                "filter native-history-legal start positions first, then apply "
                "K56 stride"
            ),
        },
        "code": {
            "global_evaluator_sha256": sha256(
                Path(__file__).resolve().parent / "global_eval56_physical.py"
            ),
            "report_builder_sha256": sha256(Path(__file__).resolve()),
        },
        "preserved_versions": {
            "revision3_manuscript_sha256": (
                "cac3130685d08a543887b8088dec48309e4c4d18e9b9b8630e72b640ac23dff8"
            ),
            "revision4_manuscript_sha256": (
                "3845f3033d5c6312b9dcba9c1209537d81c80bf091a07b575e388e77eda7e493"
            ),
        },
    }
    audit_path = OUT / "REVISION5_AUDIT.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    checksums = []
    for path in sorted(
        p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
    ):
        checksums.append(f"{sha256(path)}  {path.relative_to(OUT).as_posix()}")
    (OUT / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
