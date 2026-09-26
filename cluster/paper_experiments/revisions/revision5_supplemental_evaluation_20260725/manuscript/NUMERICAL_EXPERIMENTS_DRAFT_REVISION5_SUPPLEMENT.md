# 数值实验部分补充稿（revision 5）

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
