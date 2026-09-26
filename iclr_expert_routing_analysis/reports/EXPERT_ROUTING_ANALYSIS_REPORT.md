# Circular-cylinder 三套冻结模型的内部专家路由分析

完成日期：2026-09-05。分析对象：用户指定的 Steady S4、Hopf H4 与 Periodic epoch 85 checkpoints。用途：为后续论文整合提供可追溯的 post-hoc 统计，不直接修改论文。

## 1. 结论先行

本轮完成了三套原始模型恢复、原生 held-out rollout、旁路路由记录、独立统计重算、三张 PDF 图和完整 CSV 附表。没有训练、重新选择 checkpoint、调整路由参数，也没有修改原模型源文件和 checkpoint。

最重要的结论不是“三个模型均表现出丰富的专家切换”，而是不同模型呈现明显不同的路由行为：

1. **Periodic**：在宏步第一阶段样本中，速度使用全部 18 个 routed experts、33 种 group/pair 组合；压力使用 14 个 routed experts、16 种组合。没有出现总体单一 group 或单一专家对占比超过 95% 的现象。
2. **Hopf**：只有一个 group，这是结构设置，不是 group 路由塌缩；但速度固定选择 `(g0,e1,e3)`，压力固定选择 `(g0,e2,e4)`，所选专家对在全部记录的四个 RK4 阶段都不变。速度 `g0:e3` 承担约 **99.9344% 的 routed mass**，存在显著权重集中。压力虽然专家对固定，权重分配仍随 Re 和状态改变。
3. **Steady S4**：总体使用多个 group/专家组合，但 Re≈45.142703 的 group 1 使用率达到 **95.8333%**，不能用总体平均掩盖局部集中。更重要的是，S4 不能被标为当前论文 Steady headline 结果的对应模型。
4. **shared mass 是固定配置**：S4 为 0.5405405，Hopf/Periodic 为 0.5714286。它们不是学出来的全局共享门，也不是 shared expert 对物理预测贡献的测量。
5. Hopf、Periodic 的逐 Re 场误差与相同 checkpoint 的历史评估数值完全一致；Steady 指定 S4 的结果与任务文件中的范围不一致，已定位到结果归属问题，详见第 4 节。

因此，建议论文采用“structured sparse expert parameterization with chart-dependent routing profiles”的表述，不建议声称所有 charts 均表现出强专家专门化或均未出现固定配置。

## 2. 交付范围与证据位置

本地分析根目录：

```text
C:\Users\panxy1019\Documents\CHANNEL\iclr_expert_routing_analysis
```

远端分析根目录：

```text
/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/iclr_expert_routing_analysis
```

后文所有相对文件名均以分析根目录为基准。完整原始代码/资产绝对路径及 SHA256 见 `reports/model_asset_inventory.md` 和 `logs/*_routing_verification.json`。

| 交付内容 | 文件 |
|---|---|
| 本报告 | `reports/EXPERT_ROUTING_ANALYSIS_REPORT.md` |
| checkpoint、代码、POD、scaler、operator 清单 | `reports/model_asset_inventory.md` |
| 可选用的英文正文与图注 | `reports/PAPER_INTEGRATION_SNIPPETS.md` |
| 完整逐专家附表，含零使用专家 | `reports/APPENDIX_ROUTING_TABLE.md` |
| 三模型主图，明确 S4 为独立诊断 | `figures/circular_internal_expert_routing.pdf` |
| Group 使用与固定 shared 比例 | `figures/group_usage_and_fixed_shared_mass.pdf` |
| Periodic 按 rollout step 的统计 | `figures/periodic_rollout_step_routing.pdf` |
| 完整旁路路由日志 | `raw/{steady,hopf,periodic}_routing.jsonl.gz` |
| 全部窗口预测与目标系数 | `raw/{steady,hopf,periodic}_predictions.npz` |
| 逐 Re 误差核验 | `stats/error_reproduction.csv` |
| 统计主表与独立检查 | `stats/routing_summary.csv`、`stats/validation_checks.json` |

这是一份“指定 checkpoints 的诊断包”，不是已经可以无条件替换现有论文 S/H/P 主图的包。Steady 的模型归属需要在论文整合前明确处理。

## 3. 模型身份与运行环境

### 3.1 精确 checkpoint

以下路径相对于远端项目根目录 `particalMOE`；未采用任何替代文件。

| 模型 | 相对路径 | 冻结节点 | 文件字节数 |
|---|---|---:|---:|
| Steady S4 | `steady_specialist_v1/checkpoint/frozen_s4_validation_step_1200.pt` | step 1200 | 397948073 |
| Hopf H4 | `Hopf/migrated_h4_expanded/final_evaluation/20260722_h4_expanded_final/HopfExpanded34_H4_NormalFormRadial_r32/final.pt` | step 7200 | 383345403 |
| Periodic | `periodic_specialist_r32/checkpoint/FINAL_PERIODIC_SPECIALIST.pt` | epoch 85 | 780313433 |

```text
Steady:   bcab5661af39e3262103ed887ce9846415467452b453f3dcc14e133f0f8bd80d
Hopf:     02148741ed8bc9fbec88709f69b10492e263514b2edcee76652485b399962235
Periodic: b2052a746fdaab60f83e65d1b91d038a9dded8255ce74f41e3f6fca4c9e717c5
```

三者均严格加载，missing/unexpected keys 为空。Steady 加载 `model`；Hopf 和 Periodic 加载 `model_state`。没有用 Periodic 的其他 state 字段替换指定部署状态，也没有将 Hopf 的辅助 normal-form 参数作为新的推理分支。

### 3.2 计算环境

远端 Python 3.11.15、PyTorch 2.11.0+cu126、NumPy 2.4.4、CUDA runtime 12.6，GPU 为 NVIDIA GeForce RTX 4090。

| 模型 | 原生推理精度 | matmul TF32 | cuDNN TF32 |
|---|---|---|---|
| Steady S4 | BF16 autocast | 开启 | 开启 |
| Hopf H4 | FP32，无 autocast | 关闭 | 开启 |
| Periodic | FP32，无 autocast | 关闭 | 开启 |

每个 chart 在独立 Python 进程执行，避免 Steady 设置的 TF32 状态传播至其他模型。上述设置依据实际运行器，而不是根据训练 args 中的 `amp` 或 `allow_tf32` 字段推断。绘图在本地已有 Anaconda 环境完成，matplotlib 3.8.4、NumPy 1.26.4；没有安装或升级远端推理环境。

## 4. 原结果核验与 Steady 归属冲突

### 4.1 本轮实际场误差

均为原生面积加权物理场 relative L2 ×100%，不是 POD coefficient error，不是逐窗百分误差的简单平均。

| 模型 | Re（原始标记） | K | 窗口数 | Eu (%) | Ep (%) | 与同 checkpoint 历史数值 |
|---|---:|---:|---:|---:|---:|---|
| S4 独立诊断 | 24.630436 | 56 | 6 | 0.309191 | 18.655330 | 不对应任务给出的论文范围 |
| S4 独立诊断 | 32.740068 | 56 | 6 | 0.088507 | 5.452247 | 同上 |
| S4 独立诊断 | 39.685479 | 56 | 6 | 0.038528 | 1.593066 | 同上 |
| S4 独立诊断 | 45.142703 | 56 | 6 | 0.061621 | 2.444002 | 同上 |
| Hopf H4 | 47.081355 | 56 | 5 | 0.003208 | 0.035519 | 精确一致 |
| Hopf H4 | 49.022357 | 56 | 5 | 0.028788 | 0.432134 | 精确一致 |
| Hopf H4 | 51.786450 | 56 | 5 | 0.006096 | 0.041412 | 精确一致 |
| Periodic | 70.314635 | 48 | 13 | 1.347254 | 5.960086 | 精确一致 |
| Periodic | 100.352251 | 48 | 13 | 0.397196 | 1.708284 | 精确一致 |
| Periodic | 149.059229 | 48 | 13 | 0.380851 | 1.682540 | 精确一致 |
| Periodic | 189.862278 | 48 | 13 | 1.127427 | 4.427241 | 精确一致 |

CSV 保留实际 FP32 Re 数值，例如 49.02235412597656；表格保留原始数据的 Re 标记。二者的微小差异来自浮点存储，不是额外测试 case。

三套原生 evaluator 的 finite fraction 均为 1，按各自原始 norm-divergence 定义发散窗均为 0。这仅描述这些有限时域窗口，不构成固定点收缩、极限环保持或任意长时间稳定性的证明。

### 4.2 为什么 Steady 不能标为论文复现

任务文档列出的 Steady 目标范围为 Eu=0.0330–0.1723%、Ep=0.2500–1.3249%。但远端原始逐 Re 报告显示：

| 被列作 Steady 目标的数值 | 在原始详细报告中的实际归属 |
|---|---|
| Eu=0.0330–0.1723% | Vanilla-FNN-MoE 的四个 Steady Re 速度误差 |
| Ep=0.2500–1.3249% | Global MoE 的四个 Steady Re 压力误差 |
| Proposed Steady 身份 | 原始 specialist 消融报告明确指定 S3-B step 600，而非 S4 |

可核对证据：本包 `source/NUMERICAL_EXPERIMENTS_REVISION8_DETAILED_CN.md` 的 Steady 全部方法逐 Re 表，以及 `source/SPECIALIST_ABLATION_REPORT.md` 的“Specialist 长时物理场误差”解释段。两份文件按原样保存，没有改写其历史结果。

因此，这不是单纯的四舍五入误差或路径迁移误差。目标范围混合了不同方法的分量，且指定 checkpoint 与原报告 Proposed 身份不同。原 S4 handoff 文件还记录了跨硬件严格验证方面的限制，见 `source/FINAL_REPRODUCIBLE_STEADY_SPECIALIST.md`。本轮没有重新执行其扰动收缩认证，也不将历史限制宣称为已修复。

本轮处理：在披露冲突并收到继续执行指令后，只对用户指定 S4 做独立诊断。没有擅自切换 S3-B，没有改写论文表格，也没有为了满足预期误差修改精度、积分器或数据资产。

整合建议：若论文仍保留 S3-B 为 Proposed，S4 路由列只能标为独立候选诊断或放入附录；不能作为 S3-B 的内部路由证据。若将来决定论文改用 S4，需要另外统一 checkpoint、误差表、方法描述和严格验证结论，不能只替换一张路由图。

## 5. 实际 rollout 与取样合同

### 5.1 原生推进，不强行统一

| 项目 | Steady S4 | Hopf H4 | Periodic |
|---|---|---|---|
| 速度推进 | Euler | RK4 | RK4 |
| 压力更新 | 原生 additive algebraic closure | 原生 adaptive closure | 原生 adaptive closure |
| 内部 forward 次数/宏步 | 1 | 4 | 4 |
| 压力 closure 使用的 forward | 唯一一次 | RK4 stage 1 | RK4 stage 1 |
| 历史长度 | 3 | 3 | 3 |
| K | 56 | 56 | 48 |
| 窗口选择 | 全部合法 held-out K56 起点 | 原 native audit 的 5 个等间隔合法起点/Re | 原 portable evaluator 的 stride-8 K48 起点 |

Steady 的 `Experiment.rollout` 明确执行 `a_next = a + dt * rhs`。本轮没有将其改成 RK4。“三个 specialist 都使用 RK4”不适用于这套 S4 运行器。

Hopf 每个 held-out Re 有 102 个合法 K56 起点，本轮遵循历史 native audit 使用其中固定等间隔的 5 个，而非临时选择看起来更好的窗口。Steady 每个 Re 恰有 6 个合法起点。Periodic 每个 Re 恰有 13 个 stride-8 起点。具体 `start_index`、起始时间、窗口 ID 均写入原始路由日志，起点列表写入 verification JSON。

### 5.2 自回归状态与外部元数据

模型从合法初始状态及其历史开始，后续 a/b/history 使用预测值推进；未来 a/b 真值仅用于误差计算，不回灌 router。Hopf 运行器保留已知 Re、native dt，不使用 phase features。

Steady 和 Periodic 保留原运行器的 `arrays['phase'][cur]` 与 Re/time 合同。该 phase 是数据索引提供或由其 period/time 元数据构造的输入，而不是本轮重新估计的涡脱落相位。本轮不重新审计这些历史 phase/period 元数据的构建来源，也不把结果宣称为仅依赖状态、完全不含时钟信息的自治系统。模型状态的自回归与外部 phase/time 条件输入应在论文中分别说明。

### 5.3 两套统计总体

主图/正文统计使用 `population=macro_stage1`：每个窗口每个宏步取一次路由。这使速度与压力在同一宏步状态上比较，且压力统计对应真正被使用的 closure 输出。

另存 `population=used_forward_calls`：速度计入全部有效 RK4 stages 1–4；压力仍只计 stage 1。它是运行器实际使用的 forward 决策的等权使用统计，不是 RK4 的 1:2:2:1 加权输出贡献，也不是误差归因。

| 模型 | 窗口总数 | 主统计决策数/通道 | 有效速度 forward 数 | 有效压力 forward 数 | 原始双通道路由行数 |
|---|---:|---:|---:|---:|---:|
| S4 | 24 | 1344 | 1344 | 1344 | 2688 |
| Hopf | 15 | 840 | 3360 | 840 | 6720 |
| Periodic | 52 | 2496 | 9984 | 2496 | 19968 |
| 合计 | 91 | 4680 | 14688 | 4680 | 29376 |

原始文件也保留 Hopf/Periodic stage 2–4 的压力路由，但 `used_in_update=false`，不计入有效压力统计。窗口可能重叠；91 个窗口不是 91 条独立物理轨迹，实际独立 Re case 数为 4+3+4=11。不同 Re 的 K 步不等于相同物理时间。

同一 chart 内每个 Re 的窗口数和 horizon 相同，因此本轮 pooled 平均等价于对各 Re 平均统计再等权平均；并未跨三个 chart 汇总一个总体“专家熵”。不为大量相关的窗口/步数套用独立样本置信区间。

## 6. Router 定义与统计公式

三个运行器均使用其原始 `OperatorSpaceMoEROM`。外层 `group_router` 在 u/p 两通道间共享；组内 `velocity_group_routers[g]` 和 `pressure_group_routers[g]` 分别产生 Top-2 选择。专家实现为原始 `PhysicsAwareExpert`，没有重新搭建近似替代模型。

| 配置 | S4 | Hopf | Periodic |
|---|---:|---:|---:|
| group 数 G | 3 | 1 | 3 |
| routed experts/组/通道 | 6 | 6 | 6 |
| shared experts/组/通道 | 1 | 1 | 1 |
| group Top-k | 1 | 1 | 1 |
| local Top-k | 2 | 2 | 2 |
| local temperature | 0.95 | 0.8 | 0.8 |
| group temperature | 0.9 | 1.0 | 0.9 |
| shared_scale / routed_scale | 1 / 0.85 | 1 / 0.75 | 1 / 0.75 |
| local/group gate floor | 0 / 0 | 0 / 0 | 0 / 0 |

索引全部从 0 开始。一个 routed expert 的完整身份是 `(chart, channel, group_id, expert_id)`；绝不能把不同 group 下的同号专家合并。Top-2 pair 统计忽略二者顺序，但包含 group。

令当前选择的组为 g*，组内稀疏归一化权重为 r_e。模型实际返回的 combined gate 对应：

\[
\omega_{g,e}=\mathbf 1(g=g^*)\rho r_e,
\qquad \omega_{g,0}^{\mathrm{shared}}=\mathbf 1(g=g^*)s,
\qquad s+\rho=1.
\]

其中 S4 的 s=1/1.85，Hopf/Periodic 的 s=1/1.75。所谓 shared “always active”，指每次所选 group 的 shared expert 激活；不是所有 group 的 shared experts 同时参与预测。

对同一统计总体中的 N 个决策：

\[
f_g=N^{-1}\sum_n\mathbf1(g_n^*=g),\qquad
f_{g,e}=\frac{\sum_n I_{n,g,e}}{2N},\qquad
m_{g,e}=N^{-1}\sum_n\omega_{n,g,e}I_{n,g,e}.
\]

这里 normalized activation frequency 的总和为 1。另存 `selection_probability=count/N`，其跨专家总和为 2；两者不能混用。固定专家对的两个专家都有 f=0.5，但这不意味着各自权重相等。

\[
H=-\frac{\sum_{g,e:f>0}f_{g,e}\log f_{g,e}}{\log(6G)}.
\]

H 使用该 chart 全部 routed expert 的数量，不包含 shared。固定 Top-2 pair 在 G=1 时 H=log(2)/log(6)=0.3868528，并不是 0；这正是同时报告 pair 集中度的原因。跨 G=1 与 G=3 的 H 比较还受支持集大小影响，不能把 H 当作性能分数。

为识别权重集中，额外报告 q_e=m_e/Σm 与其最大值，以及有效专家数 1/Σq_e²。后者仅是 routing-mass 分布的逆 Simpson 指数，不是可压缩模型参数量或必要专家数。

记录以模型真实返回的 `combined_gates` 为统计依据。保存的 dense probabilities 根据捕获的 logits 与 temperature 在 CPU float64 重算，并在字段名中明确标记；它们仅用于诊断，不回写模型，也不假装是原精度下逐位相同的 softmax 中间值。Top-2 在权重数值相等时保存顺序只用于展示，不据此解释 tie-breaking。

## 7. 主统计结果

### 7.1 Group 使用

外层 group 相同，因此只需一份 u/p 共用解释。

| 模型 | Re | g0 (%) | g1 (%) | g2 (%) |
|---|---:|---:|---:|---:|
| S4 | pooled | 22.7679 | 68.4524 | 8.7798 |
| S4 | 24.630436 | 39.8810 | 45.5357 | 14.5833 |
| S4 | 32.740068 | 48.2143 | 38.6905 | 13.0952 |
| S4 | 39.685479 | 0.5952 | 93.7500 | 5.6548 |
| S4 | 45.142703 | 2.3810 | 95.8333 | 1.7857 |
| Hopf | 各 Re 及 pooled | 100 | 不存在 | 不存在 |
| Periodic | pooled | 30.4087 | 33.0529 | 36.5385 |
| Periodic | 70.314635 | 57.3718 | 20.8333 | 21.7949 |
| Periodic | 100.352251 | 30.6090 | 35.7372 | 33.6538 |
| Periodic | 149.059229 | 14.4231 | 27.2436 | 58.3333 |
| Periodic | 189.862278 | 19.2308 | 48.3974 | 32.3718 |

S4 高 Re 对 g1 更集中，但不能从这四个 Re 的使用比例推断 group 1 的物理语义。Periodic pooled 比例接近不等于对每个 Re 均匀；例如 Re≈70.31 的 g0 为 57.37%，Re≈149.06 的 g2 为 58.33%。

### 7.2 专家对、熵与集中度

| 模型 | 通道 | 使用专家/可用专家 | 不同 pair 数 | 最常用 pair | 该 pair 比例 (%) | H |
|---|---|---:|---:|---|---:|---:|
| S4 | u | 18/18 | 33 | g1:{e3,e4} | 34.0774 | 0.773943 |
| S4 | p | 16/18 | 28 | g1:{e2,e5} | 19.7173 | 0.783105 |
| Hopf | u | 2/6 | 1 | g0:{e1,e3} | 100 | 0.386853 |
| Hopf | p | 2/6 | 1 | g0:{e2,e4} | 100 | 0.386853 |
| Periodic | u | 18/18 | 33 | g0:{e0,e5} | 16.2660 | 0.929230 |
| Periodic | p | 14/18 | 16 | g2:{e0,e2} | 31.2500 | 0.783768 |

在本次主样本中零激活的 routed experts：S4 pressure 的 g0:e3、g2:e1；Periodic pressure 的 g0:e5、g1:e1、g2:e3、g2:e4；Hopf 每个通道除固定 pair 外的四个专家。它们仍保留在 CSV/主图横坐标，不因零值删除。有限 held-out 窗口内零使用不等于训练期间从未使用，也不等于可以直接安全删除。

三模型 u/p 选中 pair 不同的比例分别为 94.4196%、100%、95.3926%。这支持“通道有不同路由配置”，但 Hopf 的 100% 只是两条通道始终使用不同固定 pair，不是丰富的状态条件切换证据。

### 7.3 Hopf：固定 pair 与可变 mass 必须分开

| Re | u: g0:e1 mass | u: g0:e3 mass | p: g0:e2 mass | p: g0:e4 mass |
|---:|---:|---:|---:|---:|
| 47.081355 | 0.00000651 | 0.42856492 | 0.42842261 | 0.00014883 |
| 49.022357 | 0.00083059 | 0.42774085 | 0.41247720 | 0.01609424 |
| 51.786450 | 0.00000581 | 0.42856562 | 0.24119582 | 0.18737561 |
| pooled | 0.00028097 | 0.42829046 | 0.36069854 | 0.06787289 |

虽然所有激活频率都是固定 pair 内各 0.5，实际权重并不均衡。速度 g0:e3 占全部 routed mass 的 99.9344%，mass 有效专家数约 1.0013。不能把它写成“两个同等贡献的速度专家”。但也不能称整个模型只有一个有效专家，因为 shared 分支、编码器、物理项等仍存在，而且 gate mass 并非输出范数贡献。

压力 g0:e2 占 pooled routed mass 的 84.1630%，但在最高测试 Re 时 e4 的 mass 达到 0.18738，表明固定支持集内的软权重分配仍改变。准确描述是“fixed selected pairs with nonuniform, state/parameter-dependent mixing weights”，不是“完全恒定的路由权重”。

### 7.4 Steady S4 与 Periodic 的细节

S4 pooled 最大速度质量专家是 g1:e4，m=0.138777，占 routed mass 的 30.2043%；最大压力质量专家是 g1:e2，m=0.104801，占 22.8097%。速度/压力 mass 有效专家数分别约 6.52/7.34。总体并非单 pair 固定，但在 Re≈45.14 存在 group 层面的高度集中。

Periodic pooled 最大速度质量专家是 g2:e5，m=0.054248，占 routed mass 的 12.6579%；最大压力质量专家是 g2:e0，m=0.122033，占 28.4743%。对应有效专家数约 12.21/6.55。压力支持集比速度更集中；这描述统计配置，不说明压力物理机制更简单。

在单个窗口相邻宏步之间，S4 速度/压力 pair 切换比例为 55.4545%/60.3030%；Periodic 为 84.3290%/77.6596%；Hopf 均为 0。切换统计不跨窗口边界计数。较高切换率可能与轨迹状态、时钟输入、Top-k 决策边界或数值阶段有关，不能直接作为有益的物理专门化证据。

## 8. RK4 子阶段敏感性

主统计不是把所有 forward 混在一起。现有日志支持检查这一选择是否影响结论：

| 模型/速度通道 | 统计总体 | 决策数 | 不同 pair 数 | 最大 pair 比例 (%) | H |
|---|---|---:|---:|---:|---:|
| Hopf u | stage 1 | 840 | 1 | 100 | 0.386853 |
| Hopf u | stages 1–4 | 3360 | 1 | 100 | 0.386853 |
| Periodic u | stage 1 | 2496 | 33 | 16.2660 | 0.929230 |
| Periodic u | stages 1–4 | 9984 | 36 | 26.1118 | 0.823540 |

Hopf 固定 pair 的结论对全部 RK4 阶段成立。Periodic stage 2–4 相对同一宏步 stage 1：group 改变比例 46.1405%，速度 pair 改变比例 72.0753%。因此不能把 stage-1 主图不加说明地称为“所有速度 RHS evaluations 的统计”。

Periodic 中间压力 pair 也有 49.5994% 的变化，但这部分压力输出被原积分器丢弃；不能纳入实际压力更新的使用频率。完整对比见 `stats/rk4_stage_sensitivity.csv`。

总体结论依然成立：Periodic 使用多个配置，Hopf 固定支持集；但定量熵、最大 pair 比例依赖明确的统计总体。论文若选择全部速度 forward 作为主口径，应同步更改主图、正文表格和图注，不能混用两套数值。

## 9. 图表阅读与可用范围

主图采用 3 列×2 行 grouped bars：列为 S4/Hopf/Periodic，行为 u/p，横轴为 group:expert，颜色为各 held-out Re，纵轴为 mean routed mass。全部零使用专家保留。顶部提供 pooled group 比例，角落标注固定 shared mass 和不同 pair 数。

本地预览：

![Internal routing](C:/Users/panxy1019/Documents/CHANNEL/iclr_expert_routing_analysis/figures/circular_internal_expert_routing.png)

第二张图展示逐 Re group 使用以及固定 shared/routed 比例。第三张图仅作 Periodic 附录诊断：横轴是 k/K，不是 vortex-shedding physical phase；对不同起点的窗口作平均也不是相位对齐的流场周期统计。

三张 PDF 已通过 Poppler 渲染检查；修正了主图压力面板注释可能遮挡柱顶的位置。图中不将专家编号命名为增长、耗散、涡脱落等物理机制。

建议将当前三列图作为带 S4 警示的完整诊断图；若主文需要严格对应已确认的论文 checkpoints，优先仅讨论 Hopf/Periodic，或在明确处理 Steady 身份后再决定排版。不要删除 S4 独立诊断标签后直接使用。

## 10. 质量检查与不变性验证

每个原生窗口批次都执行了无 hooks 与有 hooks 两次 rollout，并比较完整 `pred_a`、`pred_b` 数组。结果为所有 91 个窗口逐元素完全一致，而不仅是最终平均误差接近。原始 baseline 预测没有重复保存一份，比较由运行时断言完成，验证结果和有 hooks 的完整预测已保存。

| 检查项 | 结果 |
|---|---|
| 三个精确 checkpoint 存在、可读、哈希可追溯 | 通过 |
| 严格 state_dict 加载 | 三者均无缺失或多余参数 |
| Hopf 重算 train normalization 与 checkpoint 十个 norm 字段比较 | 最大绝对差均为 0 |
| 无 hooks / 有 hooks 全窗口预测逐位相等 | 91/91 |
| 模型 state digest 推理前后不变 | 三者通过 |
| checkpoint 文件 SHA256 推理前后不变 | 三者通过 |
| 路由记录唯一性与理论行数 | 29376/29376，通过 |
| 每条记录精确一个活动 group、两个 routed experts | 通过 |
| u/p 外层 group 一致 | 通过 |
| 保存 Top-2 ID 与实际 combined gates 非零位置一致 | 通过 |
| 全部 combined gate 非负且有限 | 通过 |
| 权重和与 1 的最大绝对差 | 小于 9×10^-8 |
| normalized activation frequency 求和为 1 | 全部 chart/通道/Re/总体通过 |
| Hopf/Periodic 历史逐 Re Eu/Ep 复现 | 全部精确一致 |

Steady 的数值正确恢复不等于论文归属正确：本报告刻意将这两项验收分开。

代码改动限于新增的分析工具；修复了迁移后的绝对路径、零维张量哈希序列化、NumPy 布尔 JSON 序列化和图形环境选择问题。没有通过改变模型来“修复”S4 结果不匹配。运行耗时约为 S4 17.2 秒、Hopf 19.4 秒、Periodic 238.8 秒；这些时间包含恢复、两次 rollout、日志与哈希等，不是模型推理性能基准。

## 11. 论文能够说什么、不能说什么

### 可以由本轮数据支持

- 指定 Periodic 模型在固定 held-out 协议下使用多组、多专家对，u/p 路由统计不同。
- 指定 Hopf 模型使用固定的通道特定专家对，其中速度 routed mass 高度集中；压力固定 pair 内的质量分配随 Re 明显改变。
- 指定 S4 独立诊断存在总体多配置使用和部分 Re 的 group 集中，但不代表 S3-B 的行为。
- shared/routed balance 由固定配置控制。

### 本轮不能支持

- “所有三个 specialists 均未塌缩到固定专家配置”。Hopf 不支持。
- “每个专家对应一个明确物理机制”。未做因果或机制实验。
- “路由熵高导致预测更准确”。没有干预实验，也不能跨 chart 简单比较。
- “shared mass 大说明共享专家贡献更大”。权重质量不等于输出贡献。
- “没被选的专家可以删掉且不影响性能”。未做剪枝/替换/扰动测试。
- “本轮解决了多 seed、参数容量匹配或数据构建泄漏审计”。这些不在本轮范围。
- “所有原论文 headline 结果均成功复现”。Steady 存在明确的模型/数值归属冲突。

完整英文正文和 caption 草案单列在 `reports/PAPER_INTEGRATION_SNIPPETS.md`，包含谨慎的 Hopf/Periodic 主文版本及明确标注 S4 的全诊断版本。

## 12. 可复现命令

### 12.1 远端完整重跑

以下重跑会更新分析目录中的本轮输出，不会更改原 checkpoint 或原实验目录。应先保留当前交付包；远端需继续保有原模型和数据资产。不会在本地打包约 1.56 GB 的三个 checkpoints。

```bash
ssh -p 20381 root@10.10.164.243
export WORK_ROOT=/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE
export ANALYSIS_DIR="$WORK_ROOT/iclr_expert_routing_analysis"
export PYTHONDONTWRITEBYTECODE=1
cd "$WORK_ROOT"
set -eo pipefail

"$WORK_ROOT/.runtime/pt_env/bin/python" "$ANALYSIS_DIR/scripts/run_routing.py" --root "$WORK_ROOT" --chart steady
"$WORK_ROOT/.runtime/pt_env/bin/python" "$ANALYSIS_DIR/scripts/run_routing.py" --root "$WORK_ROOT" --chart hopf
"$WORK_ROOT/.runtime/pt_env/bin/python" "$ANALYSIS_DIR/scripts/run_routing.py" --root "$WORK_ROOT" --chart periodic
"$WORK_ROOT/.runtime/pt_env/bin/python" "$ANALYSIS_DIR/scripts/summarize_routing.py" --analysis-dir "$ANALYSIS_DIR"
"$WORK_ROOT/.runtime/pt_env/bin/python" "$ANALYSIS_DIR/scripts/supplemental_diagnostics.py" --analysis-dir "$ANALYSIS_DIR"
```

SSH 使用现有密钥或交互式认证。密码未写入代码、报告、日志或 shell 命令。

### 12.2 本地仅重算统计/图形

不需要 checkpoint 或 GPU，只读取交付的原始路由日志：

```powershell
$analysisRoot = 'C:\Users\panxy1019\Documents\CHANNEL\iclr_expert_routing_analysis'
& 'C:\ProgramData\anaconda3\python.exe' "$analysisRoot\scripts\summarize_routing.py" --analysis-dir $analysisRoot
& 'C:\ProgramData\anaconda3\python.exe' "$analysisRoot\scripts\supplemental_diagnostics.py" --analysis-dir $analysisRoot
& 'C:\ProgramData\anaconda3\python.exe' "$analysisRoot\scripts\plot_routing.py" --analysis-dir $analysisRoot
```

原始日志使用 gzip 压缩 JSONL，无需 parquet 依赖；CSV 可直接用于自定义图表。静态报告是本次经过复核的叙述版本，不会在重跑模型后自动改写，因此如更换统计协议，必须重新核对报告中的所有数值与文字。

## 13. 剩余问题与最终建议

本次要求的统计和详细报告已完成，没有正在等待的训练或后台 evaluator。

仍需由论文整合决策解决的不是程序错误，而是科学归属与表述边界：明确 Steady 使用 S3-B 还是 S4；纠正任务目标中将 Vanilla 速度范围与 Global 压力范围混在一起的归属；披露 Hopf 固定专家对/速度质量集中；注明 S4 原生 Euler、H/P 原生 RK4 和不同统计总体；保留原 phase/time 输入合同的限定。

建议先处理这几项，再把本报告的结论和图表整合进论文。路由分析可补充“模型实际怎样使用专家”的证据，但不能替代公平消融、稳定性认证、容量匹配或多随机种子实验。
