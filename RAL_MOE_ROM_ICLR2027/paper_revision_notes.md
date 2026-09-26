# Required manuscript updates

本文件仅列出当前稿件需要修改/确认的事项。本轮未修改论文；证据见 `network_architecture_audit.md` 与 `resolved_architecture.json`。不能以重新设计网络替代说明真实实验。

## P0 — Must fix before submission

1. **PAPER-CODE MISMATCH：T2-C descriptor 6→8、输入7→9。** 修改 `appendix/routing_details.tex:66–78` 中六维向量、标准化后的维数及MLP输入；完整枚举实际8通道，不能只加两个名称而保留现有不一致公式。实际还有前一速度增量、相邻增长差、log/log1p/tanh变换；curvature也不等同当前二阶导数公式。保留实际 `9→64→64→1`、4,865参数，避免改实验网络来迎合旧文。
2. **PAPER-CODE MISMATCH：pressure closure 不是现写的凸混合。** `sections/method.tex:61–64`、`appendix/specialist_details.tex:34–41` 写 `(1−gamma)` 缩放 data candidate；Circular 已查 adaptive_gate 实际为 `gamma*PP + residual`，没有该缩放。应按实际 residual 定义改写，并单列Square Steady的baseline模式及Square Hopf的direct-output例外；不能在未经定义转换说明时称两者一致。
3. **PAPER-CODE MISMATCH：Square Hopf 最终候选并非加性 Galerkin RHS。** 最终 `build_boundary_cache.py` 调用 H4 `make_ops`，其 RHS 直接learned，pressure直接learned，不使用PP/gamma组装；Galerkin只保留作输入。调整统一方法适用范围/benchmark实现说明，并核对该变体是否符合论文希望声明的方法。不要仅因base trainer另有`g+residual`函数就忽略最终调用链。
4. **PAPER-CODE MISMATCH：Pinball Periodic 表中方法身份。** Table2 K32/K56误差与B1 Deep-FNN-H3的final-test记录数值及checkpoint路径相符，网络为6×512而非MoE。需要作者确认方法命名、实验角色和适用结论；恢复历史checkpoint hash链后再作正式勘误。不得直接填成MoE结构，也不得把尚未关联的Pinball Hopf validation结果充作test证据。
5. **PAPER-CODE MISMATCH：统一训练课程声明。** `appendix/specialist_details.tex:192` 附近的`{4,8,12,16}`不适用于H4的`{1,2,4,8}`或Square S的`{1,4,8,16}`。按benchmark/chart列出预算课程与selected step/epoch；预算不是选中权重实际经历的全部训练阶段。
6. **PAPER-CODE MISMATCH：bilinear 参数化。** 当前`U,V,W_o`共享rank描述和实际`quad_left/right[out,4,state]`每输出独立低秩参数化不同。按代码给出逐输出公式；没有独立W_o。不影响已核验q=4、kappa=.05，但影响参数形状和复现。

## P1 — Strongly recommended

1. 插入本轮独立LaTeX结构表，但先处理上述实现身份差异。区分Circular H/P与Square S/H/P，不将未确认Circular S/各baseline配置混入。
2. 补充encoder三层、两个refinement块、expert的4×1024或3×768残差块、GELU/SiLU/LN/dropout和bias。区分`num_blocks=3`与实际refinement数2。
3. 明确每组每输出channel为6 routed+1 shared。u/p router和expert参数独立；不要写成两个通道共用一个shared expert。
4. 说明Hopf group router单组且frozen；Square P `fixed_regime_group=1`，不是所有chart都学动态group selection。Circular P的18 routed ID与Square P固定组的现象不可互相推广。
5. 说明压力gate结构：adaptive模式`h→64→rp`，componentwise sigmoid；Square S保存的`224→56→2`head冻结/跳过，Square H gate不参与最终压力组装。
6. 分别报告optimizer、lr、weight decay、batch、gradient accumulation、scheduler、validation/selector、seed、AMP及curriculum；区分预算和选中step（如Circular P预算240epoch，checkpoint85；Square P预算720，checkpoint490）。
7. 补充实际loss项和selector，不把所有实现概括为单一rollout MSE。E2选择分数含`.001×training batch loss`，不应误写为只由validation CE决定；T2-C训练平方误差与选择用的开方误差须区分。
8. Table13的Circular H/P数值无需更改；明确trainable和state_dict总元素不同，Hopf frozen router/训练专属normal-form不在部署trainable总数中。Square S也有约22.91M冻结参数，不可全算trainable。
9. Baseline必须先恢复最终checkpoint身份。现有Vanilla源码宽度近似参数匹配但不证明完整capacity matching；DataOnly候选源码连expert、u/p输出组织、RK4都不同，不能在没核实最终run前声称“只去掉物理项”。Global MoE字段保持UNRESOLVED。

## P2 — Optional reproducibility improvements

1. 随源码提供去除私人路径的resolved配置、所有state key/shape、checkpoint hash和模型repr；不发布集群凭据。当前审计JSON为内部证据，应整理路径再匿名公开。
2. 发布明确的“single prediction decision”参数计数脚本；Square完整物理路径active count尚未核验，不能外推Circular比例。参数量不等于FLOPs或runtime。
3. 为Square Periodic补充固定组、bf16、batched-expert执行及8个RK子步设置；将架构层级图与实际benchmark例外分开说明。
4. 增补每个benchmark独立E2/gate的run-to-paper对应表。当前Square已确认，Pinball检查到独立正式outer权重但Appendix H各行完整关联尚未闭环，Circular E2本轮UNRESOLVED。
5. 硬件型号、完整训练停止时刻、各早停触发记录没有证据时不填默认值；用具体日志补齐。
