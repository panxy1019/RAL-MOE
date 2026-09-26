# 圆柱扰流案例：流场图位置、内容与选帧说明

本文档汇总圆柱扰流（cylinder-wake）数值实验中用于论文展示的流场图、图像含义、冻结选帧规则及可用于正文的解释边界。它整合自 `CANDIDATE_FIELD_FIGURE_SELECTION.md` 与 `batlowW_style_haline_final_v7` 的四份最终 manifest。

> 所有图均为由冻结预测逐帧机械筛出的 best-case qualitative snapshot，仅用于定性结构对照。它们不代表某个 Re 的全时段表现、全部 Re 的平均表现或所有指标上的全面优势；论文的总体定量结论仍必须以冻结的统一 evaluator 结果表为准。

## 1. 最终图像位置

最终论文配色版本为 `batlowW_style_haline_final_v7`，共 8 张 400 dpi PNG，不生成 PDF：

```text
/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/
paper_experiments/visualization/results/batlowW_style_haline_final_v7/
├── steady/
│   ├── Steady_specialist_best_case_physical_fields.png
│   └── Steady_specialist_best_case_pointwise_errors.png
├── hopf/
│   ├── Hopf_specialist_best_case_physical_fields.png
│   └── Hopf_specialist_best_case_pointwise_errors.png
├── periodic/
│   ├── Periodic_specialist_best_case_physical_fields.png
│   └── Periodic_specialist_best_case_pointwise_errors.png
└── sh_fusion/
    ├── SH_boundary_best_case_physical_fields.png
    └── SH_boundary_best_case_pointwise_errors.png
```

每个子目录还保存 `FIELD_FIGURE_MANIFEST.json` 和 `candidate_scan.json`。前者记录图像 SHA256、输入资产、选中窗口及绘图合同；后者记录全部候选帧及其资格判定。

## 2. 全部图的共同阅读方式

物理场图由两列组成：左列为速度模值 $|\mathbf{u}|$，右列为压力 $p$。每一行对应 CFD Truth 或一种模型预测；同一物理变量在整张图内使用由该帧 Truth 决定的共同色标。因此，不同方法之间的颜色可以直接比较，未为任何失稳或误差较大的方法单独缩放色标。

误差图同样由两列组成：左列为点态速度误差 $|\Delta\mathbf{u}|$，右列为点态压力误差 $|\Delta p|$。各行标题中的 $\epsilon_u$ 与 $\epsilon_p$ 是对应预测在该冻结帧上的面积加权相对 $L_2$ 误差；图内色块显示的是空间分布的绝对点态误差，两者不可混同。

所有压力场和压力误差都执行了与统一 evaluator 相同的面积加权零均值 pressure gauge。绘图窗口固定为 $x\in[-1.5,10]$、$y\in[-3,3]$。最终版本采用：速度 `cividis`、压力零中心 `RdBu_r`、误差 `haline_r`（batlowW 风格的感知均匀配色）；色条使用 LaTeX 科学计数法。

## 3. 图像清单与比较对象

| 图组 | 选中样本 | 输出图 | 逐行比较对象 | 该图回答的问题 |
|---|---|---|---|---|
| Steady specialist | held-out Re=39.685478，K56，物理时刻 1478.290527 | `steady/Steady_specialist_best_case_physical_fields.png`；`steady/Steady_specialist_best_case_pointwise_errors.png` | CFD Truth、FNN-MoE、DataOnly-MoE、Proposed Specialist MoE | 在稳态吸引子区域，物理结构与压力恢复是否优于两类机制消融。 |
| Hopf specialist | train Re=50.000000，K56，物理时刻 2427.914795 | `hopf/Hopf_specialist_best_case_physical_fields.png`；`hopf/Hopf_specialist_best_case_pointwise_errors.png` | CFD Truth、FNN-MoE、DataOnly-MoE、Proposed Specialist MoE | 在指定 Hopf 严格保持案例中，振荡尾迹和压力结构是否被正确保持。 |
| Periodic specialist | held-out Re=70.314636，K48，物理时刻 2981.031738 | `periodic/Periodic_specialist_best_case_physical_fields.png`；`periodic/Periodic_specialist_best_case_pointwise_errors.png` | CFD Truth、FNN-MoE、DataOnly-MoE、Proposed Specialist MoE | 在周期极限环区域，速度尾迹与压力周期结构的长期保持能力。 |
| S–H boundary fusion | held-out Re=42.359071，K56，物理时刻 1456.932007 | `sh_fusion/SH_boundary_best_case_physical_fields.png`；`sh_fusion/SH_boundary_best_case_pointwise_errors.png` | CFD Truth、Steady Specialist、Hopf Specialist、Proposed RA-LMoE-ROM | 在经认证的 S–H 共同可行域内，冻结的窗口级凸物理场融合是否较两个单专家更接近 Truth。 |

注：K56/K48 是以 1 为起点的 rollout 长度；manifest 内的 `time_index=55/47` 分别是零起点数组索引，并不构成矛盾。

## 4. 每组图的定量锚点

以下数值来自图中所选帧，均为相对误差乘以 $100\%$。`joint` 为速度与压力相对误差的算术平均，只用于候选帧资格排序与辅助解释。

### 4.1 Steady specialist：held-out Re=39.685478，K56

| 方法 | $\epsilon_u$ | $\epsilon_p$ | joint |
|---|---:|---:|---:|
| FNN-MoE | 0.0356% | 3.2755% | 1.6555% |
| DataOnly-MoE | 0.2893% | 7752.9498% | 3876.6196% |
| Proposed Specialist MoE | 0.0488% | 1.4920% | 0.7704% |

该帧属于预注册的 joint-priority tier 2：Proposed Specialist MoE 在压力和联合误差上优于两项消融，但速度误差略高于 FNN-MoE。因此正文应表述为“该选帧的联合场和压力恢复更优”，而不能将此单帧扩大为速度误差也严格优于 FNN-MoE 的结论。

### 4.2 Hopf specialist：train Re=50.000000，K56

| 方法 | $\epsilon_u$ | $\epsilon_p$ | joint |
|---|---:|---:|---:|
| FNN-MoE | 0.0127% | 0.1422% | 0.0775% |
| DataOnly-MoE | 0.1060% | 2888.1087% | 1444.1074% |
| Proposed Specialist MoE | 0.0027% | 0.0150% | 0.0089% |

该帧为 tier 0：Proposed Specialist MoE 的速度、压力和联合误差均小于两个消融。其 split 为 train，原因是 FNN-MoE 的 Hopf held-out 长时有限性合同 fail-closed；该图只能作为指定严格保持案例的结构性定性对照，不能替代 held-out 消融结论。

### 4.3 Periodic specialist：held-out Re=70.314636，K48

| 方法 | $\epsilon_u$ | $\epsilon_p$ | joint |
|---|---:|---:|---:|
| FNN-MoE | 5.7889% | 21.6859% | 13.7374% |
| DataOnly-MoE | 2.3553% | 16.5425% | 9.4489% |
| Proposed Specialist MoE | 0.4462% | 1.8140% | 1.1301% |

该帧为 tier 0，Proposed Specialist MoE 在三项帧级指标上均优于两个消融。物理场图用于显示周期尾迹的结构恢复，误差图用于定位两个消融在空间上的偏差区域。

### 4.4 S–H boundary fusion：held-out Re=42.359071，K56

| 方法 | $\epsilon_u$ | $\epsilon_p$ | joint |
|---|---:|---:|---:|
| Steady Specialist | 0.0573% | 3.1862% | 1.6217% |
| Hopf Specialist | 0.2968% | 9.9141% | 5.1054% |
| Proposed RA-LMoE-ROM | 0.0564% | 3.1604% | 1.6084% |

该帧为 tier 0：Proposed RA-LMoE-ROM 在速度、压力及联合误差上均小于 Steady Specialist 和 Hopf Specialist。该融合是冻结的 trajectory-level T2-C 窗口级凸物理场融合；两套专家各自在原生坐标图独立 rollout，融合只发生在严格对齐的输出时间点，不向下一时步反馈融合状态。

## 5. 候选筛选覆盖范围

下表保留原候选筛选报告中的完整 Re 扫描范围。最终图只从“合格”候选帧中机械选取；最后一列中的“是”表示该 Re 存在合格候选帧，而非该 Re 的所有时间点均合格。

| 图组 | Re | split | 扫描帧记录数 | 合格帧数 | 最终采用该 Re |
|---|---:|---|---:|---:|---|
| Steady | 24.630436 | heldout | 56 | 5 | 否 |
| Steady | 32.740067 | heldout | 56 | 17 | 否 |
| Steady | 39.685478 | heldout | 1 | 1 | 是 |
| Steady | 45.142704 | heldout | 56 | 28 | 否 |
| Hopf | 49.300000 | train | 13 | 13 | 否 |
| Hopf | 49.600000 | train | 13 | 13 | 否 |
| Hopf | 50.000000 | train | 13 | 13 | 是 |
| Periodic | 70.314636 | heldout | 60 | 30 | 是 |
| Periodic | 100.352249 | heldout | 248 | 125 | 否 |
| Periodic | 149.059235 | heldout | 60 | 41 | 否 |
| Periodic | 189.862274 | heldout | 624 | 188 | 否 |
| S–H fusion | 42.359071 | heldout | 170 | 136 | 是 |
| S–H fusion | 43.500000 | heldout | 115 | 32 | 否 |
| S–H fusion | 43.900000 | heldout | 60 | 45 | 否 |

## 6. 可直接用于论文的图注草案

**Steady 图注。** Re=39.685478 的 K56 held-out 窗口流场对比。左列为速度模值 $|\mathbf{u}|$，右列为经统一 pressure gauge 校正的压力 $p$；下图给出对应的点态绝对误差。所有方法共享以 CFD Truth 确定的色标。该选帧中，Proposed Specialist MoE 的压力相对误差为 1.4920%，低于 FNN-MoE 的 3.2755%，并获得最低联合场误差；速度相对误差并未严格低于 FNN-MoE。

**Hopf 图注。** Re=50.000000 的 K56 指定 Hopf 严格保持案例。Proposed Specialist MoE 同时降低速度和压力误差，分别为 0.0027% 与 0.0150%。该图来自 train split，仅作为流动结构和局部吸引子保持的定性对照，不替代 held-out 消融结论。

**Periodic 图注。** Re=70.314636 的 K48 held-out 周期流场对比。Proposed Specialist MoE 的速度和压力相对误差分别为 0.4462% 和 1.8140%，均低于 FNN-MoE 与 DataOnly-MoE，显示出对周期尾迹和压力分布的更稳定恢复。

**S–H 融合图注。** Re=42.359071 的 K56 held-out S–H 共同可行重叠域对比。Steady Specialist 与 Hopf Specialist 先在各自原生坐标图独立推进，Proposed RA-LMoE-ROM 仅在当前输出时间点进行冻结窗口级凸物理场融合。该选帧的融合结果在速度、压力和联合相对误差上均优于两个单专家；该结果不代表在未通过 admissibility gate 的 H–P 区域也启用软融合。

## 7. 使用限制与可追溯性

1. 本文档不将 best-case 快照作为平均性能或 worst-case 性能的替代证据。
2. DataOnly-MoE 的大误差会造成色彩饱和；这是真实的共享色标结果，而不是对 DataOnly-MoE 单独改变颜色范围。
3. S–H 的正融合结果只适用于已经通过共同 rollout 可行性认证的 S-native development 域；H–P 当前采用 P-only/native-time hard routing，不能借用这张 S–H 图推出 H–P 融合同样有效。
4. 原始候选筛选文字与扫描统计见 [CANDIDATE_FIELD_FIGURE_SELECTION.md](C:/Users/panxy1019/Desktop/STABLEMOE/V16STABLE/paper_experiments/visualization/CANDIDATE_FIELD_FIGURE_SELECTION.md)；最终图像版本、输入资产哈希和输出哈希以各 `final_v7/*/FIELD_FIGURE_MANIFEST.json` 为准。
