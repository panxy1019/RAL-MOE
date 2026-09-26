from __future__ import annotations

import hashlib
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
REV4 = (
    WORKSPACE
    / "final_paper_reports"
    / "revisions"
    / "revision4_global_fusion_20260724"
    / "manuscript"
    / "NUMERICAL_EXPERIMENTS_DRAFT_REVISION4.md"
)
REV5_ROOT = (
    WORKSPACE
    / "final_paper_reports"
    / "revisions"
    / "revision5_supplemental_evaluation_20260725"
)
REV5_REPORT = REV5_ROOT / "reports" / "REVISION5_SUPPLEMENTAL_EVALUATION_ZH.md"
OUT = (
    WORKSPACE
    / "final_paper_reports"
    / "revisions"
    / "revision6_complete_20260725"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def section(text: str, start: str, end: str | None) -> str:
    start_index = text.index(start) + len(start)
    end_index = len(text) if end is None else text.index(end, start_index)
    return text[start_index:end_index].strip()


def replace_required(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Required source fragment not found: {old[:120]!r}")
    return text.replace(old, new, 1)


def remap_headings(text: str, mapping: dict[str, str]) -> str:
    for old, new in mapping.items():
        text = text.replace(old, new)
    return text


def build() -> None:
    manuscript = REV4.read_text(encoding="utf-8")
    supplement = REV5_REPORT.read_text(encoding="utf-8")

    manuscript = manuscript.replace(
        "# 数值实验部分草稿（中文 revision 4：Global 与边界融合重组版）",
        "# 数值实验部分草稿（中文 revision 6：完整合并与同 KLong 复核版）",
        1,
    )
    manuscript = replace_required(
        manuscript,
        "> 本文件是新版本，不覆盖 revision 3。所有相对误差均为原始 relative error × 100%。本文只展示 KLong。带 † 的 Global MoE 数值来自其冻结归档的 K24 POD 系数域 evaluator；其余 specialist 与融合结果来自 K48/K56 面积加权物理场 evaluator。两类数值用于并列呈现模型行为，但不能直接作等口径大小排序。",
        "> 本文件是独立完整版本，不覆盖 revision 3、4 或 5。所有相对误差均为原始 relative error × 100%。本文只展示 KLong：Steady/Hopf/Global 为 K56，Periodic specialist 为 K48。Global MoE† 同时报告面积加权物理场误差和 POD 系数误差；前者包含 Re-specific mean field，后者刻画脉动模态动力学，二者不可相互替代。",
    )
    manuscript = replace_required(
        manuscript,
        "KLong 在 Steady/Hopf specialist 中为 K56，在 Periodic specialist 中为 K48，在冻结 Global MoE 归档中为 K24。它们是各自原生数据库步数，不代表统一物理时间。由于原始采样合同和 evaluator 不同，本节保留“时域/误差空间”列，避免把 K24 系数误差伪装为 K56 物理场误差。",
        "KLong 在 Steady/Hopf specialist 与 Global MoE 中为 K56，在 Periodic specialist 中为 K48。它们是各自原生数据库步数，不代表统一物理时间。Global K56 使用合法三步 history 起点，并在同一次 rollout 中同时计算面积加权物理场误差和 POD 系数误差；所有比较仍保留测试 Re、时域和误差空间，避免把含均值全场误差与脉动系数误差混为一谈。",
    )

    global_main_rows = {
        "| Steady | Global MoE† | 24.630436, 32.740067, 39.685478 | K24 | POD 系数域† | 22.4582% | 46.6276% | 34.5429% | — | — | — |":
        "| Steady | Global MoE† | 24.630436, 32.740067, 39.685478, 45.142704 | K56 | 面积加权物理场 | 0.1809% | 0.6639% | 0.4224% | 0.8327% | 1.000 | 0 |",
        "| Hopf | Global MoE† | 47.081356, 49.022358, 51.786449 | K24 | POD 系数域† | 49.5338% | 46.2694% | 47.9016% | — | — | — |":
        "| Hopf | Global MoE† | 47.081356, 49.022358, 51.786449 | K56 | 面积加权物理场 | 0.0187% | 0.1880% | 0.1033% | 0.1532% | 1.000 | 0 |",
        "| Periodic | Global MoE† | 70.314636, 100.352249, 149.059235, 189.862274 | K24 | POD 系数域† | 4.9334% | 6.1781% | 5.5557% | — | — | — |":
        "| Periodic | Global MoE† | 70.314636, 100.352249, 149.059235, 189.862274 | K56 | 面积加权物理场 | 1.2147% | 4.9544% | 3.0846% | 4.8076% | 1.000 | 0 |",
    }
    for old, new in global_main_rows.items():
        manuscript = replace_required(manuscript, old, new)
    manuscript = replace_required(
        manuscript,
        "Global MoE 已按要求纳入本表，并明确给出速度与压力误差。其 K24 结果在三个流态上分别反映全局模型的速度/压力 POD 系数 rollout 误差；由于没有经过同一 K48/K56 物理场重构 evaluator，本稿不据此宣称 Global 与 specialist 的严格数值排名。这个限制是测量口径限制，而不是删除 Global 结果。",
        "Global MoE† 已按三个 specialist 的测试集重新执行 K56，并在主表中采用面积加权物理场误差。对应的平均速度/压力 POD 系数误差分别为：Steady 30.4250%/60.5898%，Hopf 55.1998%/46.9329%，Periodic 13.5459%/16.1058%。Global 的全场误差明显小于系数误差，说明均值场能量会降低全场相对误差；因此论文必须同时给出两类指标，而不能只选择数值更小的物理场列。",
    )

    steady_global_rows = {
        "| Global MoE† | 24.630436 | K24 / POD 系数域† | 30.6107% | 62.7433% | — | 压力能量 127.0489%; 伪振荡峰值比 207.75× | — | — | 未运行 Steady 同口径 strict evaluator |":
        "| Global MoE† | 24.630436 | K56 / 面积加权物理场 | 0.3404% | 1.3249% | — | 系数误差 u/p=44.2564%/101.0296%；终点续推不受原合同支持 | 1.000 | 0 | K56 有限；strict N/A |",
        "| Global MoE† | 32.740067 | K24 / POD 系数域† | 20.6771% | 46.7996% | — | 压力能量 66.6479%; 伪振荡峰值比 98.11× | — | — | 未运行 Steady 同口径 strict evaluator |":
        "| Global MoE† | 32.740067 | K56 / 面积加权物理场 | 0.1800% | 0.6831% | — | 系数误差 u/p=31.2366%/74.5202%；终点续推不受原合同支持 | 1.000 | 0 | K56 有限；strict N/A |",
        "| Global MoE† | 39.685478 | K24 / POD 系数域† | 16.0867% | 30.3400% | — | 压力能量 30.6771%; 伪振荡峰值比 40.88× | — | — | 未运行 Steady 同口径 strict evaluator |":
        "| Global MoE† | 39.685478 | K56 / 面积加权物理场 | 0.1186% | 0.3976% | — | 系数误差 u/p=25.4158%/41.5445%；终点续推不受原合同支持 | 1.000 | 0 | K56 有限；strict N/A |",
    }
    for old, new in steady_global_rows.items():
        manuscript = replace_required(manuscript, old, new)
    manuscript = manuscript.replace(
        "Proposed S3-B 的 K56 pressure fixed-point worst 为 27.1317%，Vanilla 为 49.4685%；更严格的 perturbation-bank K56 最坏 paired gain 从 107702.891 降至 585.262。该结果支持“固定点与扰动鲁棒性显著改善”，但由于最坏 gain 仍大于 1，不支持“所有扰动方向严格收缩”。Global 的低 Re 归档出现较大的伪振荡 overshoot，因此其存在于表中并不等于通过 Steady strict-attractor 判据。",
        "Proposed S3-B 的 K56 pressure fixed-point worst 为 27.1317%，Vanilla 为 49.4685%；更严格的 perturbation-bank K56 最坏 paired gain 从 107702.891 降至 585.262。该结果支持“固定点与扰动鲁棒性显著改善”，但由于最坏 gain 仍大于 1，不支持“所有扰动方向严格收缩”。Global 在三个相同 heldout Re 上完成有限 K56，物理场误差较小，但冻结 Global 接口不支持数据库终点之后的同合同 fixed-point continuation，因此不能据普通轨迹误差授予 Steady strict-attractor 标签。",
    )

    manuscript = manuscript.replace(
        "Global held-out 资产不含这三个精确 Re，且仅有 K24 系数域 evaluator",
        "Global 数据库不含该精确 Re；K56 evaluator 禁止使用近邻节点替代",
    )

    periodic_global_rows = {
        "| 100.352249 | Global MoE† | K24 / POD 系数域† | 2.8343% | 3.6636% | 0.9267% | — | 0.7457% | 0.0159 | — | — | — | N/A：时域与 strict 定义不同 |":
        "| 100.352249 | Global MoE† | K56 / 物理场 | 1.1484% | 4.3387% | — | — | — | — | — | — | 1.000 / 0 | N/A：未运行周期 strict evaluator |",
        "| 149.059235 | Global MoE† | K24 / POD 系数域† | 2.5471% | 3.2347% | 0.7537% | — | 0.5134% | 0.0093 | — | — | — | N/A：时域与 strict 定义不同 |":
        "| 149.059235 | Global MoE† | K56 / 物理场 | 0.7461% | 2.7039% | — | — | — | — | — | — | 1.000 / 0 | N/A：未运行周期 strict evaluator |",
        "| 189.862274 | Global MoE† | K24 / POD 系数域† | 4.1608% | 5.0987% | 1.2367% | — | 0.7290% | 0.0132 | — | — | — | N/A：时域与 strict 定义不同 |":
        "| 189.862274 | Global MoE† | K56 / 物理场 | 1.4043% | 4.7200% | — | — | — | — | — | — | 1.000 / 0 | N/A：未运行周期 strict evaluator |",
    }
    for old, new in periodic_global_rows.items():
        manuscript = replace_required(manuscript, old, new)
    manuscript = manuscript.replace(
        "DataOnly-MoE | — | — | — | — | — | — | — | — | 冻结归档未在 49.3/49.6/50.0 运行同一 strict evaluator | — | — | N/A",
        "DataOnly-MoE | 见 4.3.5.1 | 见 4.3.5.1 | 见 4.3.5.1 | 见 4.3.5.1 | 见 4.3.5.1 | 见 4.3.5.1 | 见 4.3.5.1 | 见 4.3.5.1 | 同 Re、同 K56 strict evaluator 已补齐 | 1.000 / 5 | 是 | FAIL",
    )
    manuscript = manuscript.replace(
        "DataOnly-MoE | K48 / 物理场 | 0.7633% | 2.7966% | — | — | — | — | — | — | 1.000 / 0 | N/A：未运行周期 strict evaluator",
        "DataOnly-MoE | K48 / 物理场 | 0.8891% | 3.1271% | 0.7753% | 2.0613% | 0.0852% | 0.0376 | -0.0097 | 4.7728% | 1.000 / 0 | PASS",
    )
    manuscript = manuscript.replace(
        "DataOnly-MoE | K48 / 物理场 | 0.8276% | 2.2308% | — | — | — | — | — | — | 1.000 / 0 | N/A：未运行周期 strict evaluator",
        "DataOnly-MoE | K48 / 物理场 | 0.7854% | 2.3207% | 0.3444% | 0.5408% | 0.0905% | 0.0326 | -0.0097 | 4.3304% | 1.000 / 0 | PASS",
    )
    manuscript = manuscript.replace(
        "DataOnly-MoE | K48 / 物理场 | 0.6416% | 1.3018% | — | — | — | — | — | — | 1.000 / 0 | N/A：未运行周期 strict evaluator",
        "DataOnly-MoE | K48 / 物理场 | 0.8509% | 2.2511% | 0.5757% | 0.8865% | 0.0507% | 0.0314 | 0.0009 | 4.6841% | 1.000 / 0 | PASS",
    )
    manuscript = manuscript.replace(
        "DataOnly 在相同 Re 有 K48 物理场误差，但没有运行周期 strict evaluator；Global 在相同 Re 有 K24 系数域振幅、频率和相位诊断，但 horizon 与定义不同。",
        "DataOnly 已在相同 Re 运行 K48 周期 strict evaluator并取得 3/3 PASS；Global 已补做 K56 物理场 rollout，但没有运行同一周期 strict evaluator。",
    )
    manuscript = manuscript.replace(
        "Vanilla、DataOnly 和 Global 已在同一子表中出现；然而现有冻结产物没有三者在这三个精确 Re 上的同口径 strict 结果，必须标为 N/A，不能用邻近 Re 或不同 horizon 冒充。因而目前可宣称 Proposed 的三个严格案例，但不能据该表宣称它在这三个案例上定量优于全部消融。",
        "Vanilla、DataOnly 和 Global 均列在同一子表中。DataOnly 已补齐三个精确 Re 的同 K56 strict 结果并全部 FAIL，因此可以在该对照下支持 Proposed 相对 DataOnly 的严格保持优势；Vanilla 因 validation 长时非有限而 fail-closed，Global 数据库不含这三个精确 Re，二者仍不得用邻近 Re 或不同 horizon 冒充定量结果。",
    )

    manuscript = replace_required(
        manuscript,
        "| H–P | Global MoE† | 70.314636, 100.352249, 149.059235, 189.862274 | K24 / POD 系数域† | 4.9334% | 6.1781% | 5.5557% | 仅 Global 原生振幅/相位诊断；无同口径 strict 判定 | 全局单模型；不可与 K48 物理场直接排名 |",
        "| H–P | Global MoE† | 70.314636, 100.352249, 149.059235, 189.862274 | K56 / 面积加权物理场 | 1.2147% | 4.9544% | 3.0846% | K56 finite=1；无同口径周期 strict 判定 | 全局单模型；同时报告系数误差 13.5459%/16.1058% |",
    )
    manuscript = replace_required(
        manuscript,
        "| S–H | Global MoE† | 无同一 sealed-cache 结果 | K24 / POD 系数域† | — | — | — | — | 不能用其他 Re 替代边界 test |",
        "| S–H | Global MoE† | 无同一 sealed-cache 结果 | — | — | — | — | — | Global 数据库缺少完整精确边界 Re，不能用其他 Re 替代 sealed-cache test |",
    )
    manuscript = manuscript.replace(
        "Global 在 Periodic held-out Re 有结果，但仍是 K24 系数域口径。",
        "Global 在 Periodic held-out Re 已有 K56 面积加权物理场和系数域双口径结果，但仍无同一周期 strict 判定。",
    )
    manuscript = manuscript.replace(
        "Global 尚未通过与 specialist 完全相同的 K48/K56 物理场 evaluator",
        "Global 已完成 K56 面积加权物理场评价，但仍未通过各局部 specialist 的终点续推与吸引子 strict evaluator",
    )

    frozen_summary = section(
        supplement,
        "## 1. 结论摘要",
        "## 2. Global MoE† K56 补充 rollout",
    )
    manuscript = manuscript.replace(
        "\n## 4.2 长时物理场预测",
        "\n### 4.1.1 Revision 5 补充评价冻结结论\n\n"
        + frozen_summary
        + "\n\n## 4.2 长时物理场预测",
        1,
    )

    global_detail = remap_headings(
        section(
            supplement,
            "## 2. Global MoE† K56 补充 rollout",
            "## 3. Proposed Steady 全 split压力固定点与漂移扫描"
            if "## 3. Proposed Steady 全 split压力固定点与漂移扫描" in supplement
            else "## 3. Proposed Steady 全 split 压力固定点与漂移扫描",
        ),
        {
            "### 2.1 ": "#### 4.2.3.1 ",
            "### 2.2 ": "#### 4.2.3.2 ",
        },
    )
    global_block = (
        "\n\n### 4.2.3 Global MoE† K56 物理场与系数误差完整结果\n\n"
        + global_detail
        + "\n"
    )
    manuscript = manuscript.replace("\n## 4.3 吸引子与渐近动力学保持", global_block + "\n## 4.3 吸引子与渐近动力学保持", 1)

    steady_scan = section(
        supplement,
        "## 3. Proposed Steady 全 split 压力固定点与漂移扫描",
        "## 4. 固定点合格次级子集上的横向诊断",
    )
    fixed_subset = section(
        supplement,
        "## 4. 固定点合格次级子集上的横向诊断",
        "## 5. DataOnly-MoE 缺失吸引子评价",
    )
    dataonly_detail = remap_headings(
        section(
            supplement,
            "## 5. DataOnly-MoE 缺失吸引子评价",
            "## 6. 可用于论文的客观表述",
        ),
        {
            "### 5.1 ": "#### 4.3.5.1 ",
            "### 5.2 ": "#### 4.3.5.2 ",
        },
    )
    attractor_block = f"""

### 4.3.3 Proposed Steady 全 split 压力固定点与漂移扫描

{steady_scan}

### 4.3.4 固定点合格次级子集横向诊断

{fixed_subset}

### 4.3.5 DataOnly-MoE 的 Hopf 与 Periodic strict 补充结果

{dataonly_detail}
"""
    manuscript = manuscript.replace(
        "\n## 4.4 Admissibility-constrained 边界融合算法",
        attractor_block + "\n\n## 4.4 Admissibility-constrained 边界融合算法",
        1,
    )

    objective = section(
        supplement,
        "## 6. 可用于论文的客观表述",
        "## 7. 可追溯性与限制",
    )
    traceability = section(
        supplement,
        "## 7. 可追溯性与限制",
        None,
    )
    manuscript = manuscript.replace(
        "\n## 4.6 可追溯性",
        "\n### 4.5.1 Revision 5 补充评价后的客观表述边界\n\n"
        + objective
        + "\n\n## 4.6 可追溯性",
        1,
    )
    manuscript += "\n\n### 4.6.1 Revision 5 补充资产与限制\n\n" + traceability + "\n"

    outdated_claims = [
        "冻结 Global MoE 归档中为 K24",
        "Global 尚未通过与 specialist 完全相同的 K48/K56 物理场 evaluator",
        "DataOnly 在相同 Re 有 K48 物理场误差，但没有运行周期 strict evaluator",
    ]
    found = [claim for claim in outdated_claims if claim in manuscript]
    if found:
        raise RuntimeError(f"Outdated claims remain: {found}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manuscript").mkdir(exist_ok=True)
    (OUT / "inputs").mkdir(exist_ok=True)
    output_path = (
        OUT / "manuscript" / "NUMERICAL_EXPERIMENTS_DRAFT_REVISION6_COMPLETE.md"
    )
    output_path.write_text(manuscript, encoding="utf-8")
    for source in (
        REV5_ROOT / "final_inputs" / "GLOBAL_K56_PHYSICAL_COMPACT.json",
        REV5_ROOT / "final_inputs" / "DATAONLY_MISSING_CONTRACTS.json",
        REV5_ROOT / "final_inputs" / "STEADY_PROPOSED_STRICT_PRESSURE_SCAN.json",
        REV5_ROOT / "final_inputs" / "FIXED_POINT_QUALIFIED_SELECTION.json",
    ):
        (OUT / "inputs" / source.name).write_bytes(source.read_bytes())

    audit = {
        "schema": "revision6_complete_manuscript_audit/v1",
        "output": str(output_path),
        "inputs": {
            "revision4_manuscript": {
                "path": str(REV4),
                "sha256": sha256(REV4),
            },
            "revision5_supplement_report": {
                "path": str(REV5_REPORT),
                "sha256": sha256(REV5_REPORT),
            },
        },
        "coverage": {
            "global_k56_aggregate_physical_and_coefficient_errors": True,
            "global_k56_per_Re": True,
            "steady_all_split_pressure_scan": True,
            "steady_fixed_point_qualified_cross_method_table": True,
            "dataonly_hopf_strict_table": True,
            "dataonly_periodic_strict_table": True,
            "objective_claim_boundary": True,
            "traceability_and_limits": True,
        },
        "preserved": {
            "revision3_sha256": "cac3130685d08a543887b8088dec48309e4c4d18e9b9b8630e72b640ac23dff8",
            "revision4_sha256": "3845f3033d5c6312b9dcba9c1209537d81c80bf091a07b575e388e77eda7e493",
            "revision5_full_sha256": sha256(
                REV5_ROOT
                / "manuscript"
                / "NUMERICAL_EXPERIMENTS_DRAFT_REVISION5_FULL.md"
            ),
        },
        "builder_sha256": sha256(Path(__file__).resolve()),
    }
    audit_path = OUT / "REVISION6_AUDIT.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    checksum_lines = []
    for path in sorted(
        p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
    ):
        checksum_lines.append(f"{sha256(path)}  {path.relative_to(OUT).as_posix()}")
    (OUT / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    build()
