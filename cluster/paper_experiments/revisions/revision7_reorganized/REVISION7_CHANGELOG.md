# Revision 7 变更日志

## 1. 版本关系

- 冻结输入：`revision6_complete_20260725/manuscript/NUMERICAL_EXPERIMENTS_DRAFT_REVISION6_COMPLETE.md`
- Revision 6 SHA256：`f4a5fd04b981a90f665f9e8abb462ff508d9011bd13075f986aeb7c1ededb35c`
- 新输出：`revision7_reorganized/NUMERICAL_EXPERIMENTS_REVISION7_CN.md`
- Revision 6 未被修改或覆盖。
- 本轮未训练模型、未运行 evaluator、未修改 checkpoint、未生成新科学指标。

## 2. Revision 6 → Revision 7 章节映射

| Revision 6 内容 | Revision 7 位置 | 处理 |
|---|---|---|
| 4.1 实验协议、数据隔离与评价口径 | 4.1 Flow configuration, datasets, and evaluation protocol | 重写；补入冻结 CFD case 中的几何、域、边界、时间步与网格合同 |
| 4.1.1 Revision 5 补充评价冻结结论 | 4.1、4.5、附录 A | 拆分到协议、限制和相应附录，不再作为时间顺序日志 |
| 4.2.1 specialist、消融与 Global 混合主表 | 4.2 表 1、4.3 表 3、4.4 表 5 | 按“局部组件—完整系统—局部消融”拆成三张表 |
| 4.2.2 KLong 物理时间跨度 | 4.1.1、4.1.3 | 压缩为原生步数和非统一物理时间说明 |
| 4.2.3 Global 全部结果 | 4.3 表 3、附录 A.1 | 聚合进入完整系统表；逐 Re 移入附录 |
| 4.3.1 Specialist attractor preservation | 4.2 表 2、4.4 表 6 | Proposed 能力与内部消融分离 |
| 4.3.2 吸引子综合解释 | 4.2.2、4.4.3 | 合并到各科学问题之后，删除重复结论 |
| 4.3.3 Steady 全 split 扫描 | 附录 A.2；正文 4.2.2 保留 0/20 主结论 | 细表移入附录 |
| 4.3.4 固定点六 Re 子集 | 附录 A.2 末段 | 压缩为选择列表；完整源仍由 source map 指向 revision 6 |
| 4.3.5 DataOnly Hopf/Periodic | 附录 A.3/A.4；正文表 6 摘要 | 逐 Re 细表移入附录 |
| 4.4 边界融合算法 | 4.3 Full-system comparison | 提升到完整系统主结果；T2-C 明确为局部机制 |
| 4.4.2 边界综合表 | 4.3 表 3 | 重排为 Steady core、S–H、Hopf core、H–P、Periodic core |
| 4.4.3 S–H 逐 Re | 4.3 表 4 | 增加已有 routing regret、matched oracle gap、tail/terminal 字段 |
| 4.4.4 H–P fail closed | 4.3.3 | 扩写 formal gate 的 53/6 development 计数 |
| 4.5 限制与可支持结论 | 4.5 | 统一整理为“可以支持/不可以支持” |
| 4.6 可追溯性 | 4.6 + `REVISION7_SOURCE_MAP.json` | 从路径说明升级为逐表、逐结论结构化映射 |

## 3. 正文六张主表

1. Proposed 三个 specialist 的原生长时场结果。
2. Proposed 三个 specialist 的吸引子证据。
3. Global MoE† / E2 / Proposed 的分区域完整系统比较。
4. S–H sealed cache 的逐 Re 场误差、吸引子、权重与 oracle 诊断。
5. Vanilla / DataOnly / Proposed 的 specialist 长时消融。
6. Vanilla / DataOnly / Proposed 的吸引子消融。

没有任何主表同时混合 specialist 内部消融与完整系统方法。

## 4. 删除、合并和移入附录

### 删除的重复内容

- 多处重复出现的 KLong 定义、Global representation 警告和 Hopf train-Re 限制。
- 同一数值在“主表—补充表—综合解释”中的三次重复。
- 按 revision 时间顺序排列的“Revision 5 补充结论”日志式叙述。

删除仅指从 revision 7 正文移除；revision 6 原文件仍完整保留。

### 合并的内容

- Global 的物理场误差与 POD 系数误差合并到完整系统解释中。
- Proposed specialist 的长时场能力和吸引子能力合并为 4.2。
- S–H field、attractor、gate weights、routing regret 和 oracle gap 合并为表 4。
- Vanilla/DataOnly 的长时与吸引子结果分别集中到 4.4 的两张消融表。

### 移入附录的内容

- Global 全部逐 Re K56 物理场/系数结果。
- Proposed Steady 全 train/validation/heldout fixed-point 扫描。
- “仅 pressure fixed-point error <5%”的六 Re 次级子集说明。
- DataOnly Hopf 三个指定训练 Re 的 strict 结果。
- DataOnly Periodic 三个指定 held-out Re 的 strict 结果。
- 所有主要 N/A、失败与 fail-closed 原因。
- 完整 SHA256 与源键存入 `REVISION7_SOURCE_MAP.json`。

## 5. 关键措辞修正

- 将 `Global MoE` 统一改为 `Global MoE†`，并在首次出现时说明其 transductive all-Re representation。
- 不再把完整 Proposed 系统简称为 T2-C；T2-C 仅指 S–H overlap 的窗口条件物理场凸融合。
- Hopf 49.3、49.6、50.0 统一标为 train/in-sample mechanism cases。
- Steady strict 结论改为 pressure fixed-point + drift 联合门 0/20。
- H–P 改写为 admissibility fail-closed，明确没有训练软融合门。
- 把 finite、低全场误差和 strict attractor PASS 明确区分。

## 6. 结构终检

```text
局部 specialist 可用             PASS
→ 完整系统比较                   PASS
→ S–H/H–P 边界机制              PASS
→ specialist 内部消融           PASS
→ 限制与结论边界                PASS
```
