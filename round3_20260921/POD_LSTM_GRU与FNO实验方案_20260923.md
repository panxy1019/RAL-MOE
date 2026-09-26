# POD--LSTM/GRU 与 FNO 对比实验方案

日期：2026-09-23  
适用项目：RAL-MoE-ROM / ICLR 实验补充  
性质：预注册式实验方案；本文档不代表实验已经完成

## 1. 实验目的与方法定位

本轮补充两类外部学习基线，但二者回答的问题不同，不应混为同一类消融：

1. **POD--LSTM/GRU**：在与 RAL-MoE-ROM 相同的局部 POD 坐标、训练/验证/测试划分和预测窗口下，检验不使用 Galerkin 骨架、稀疏专家或内部路由的黑箱循环模型能达到何种精度与稳定性。
2. **FNO**：在物理场网格上学习参数化时间推进算子，检验全场神经算子在相同 CFD 数据与预测时长下的表现。FNO 不属于局部 ROM，因此其参数量、显存、参考场和插值误差必须单独披露。

本轮优先在 **Circular Hopf 与 Circular Periodic** 上完成，因为现有受控比较已经固定了这两个流态的窗口、预测长度、三种子和误差聚合规则。若两类基线在 Circular 上闭环，再决定是否扩展到 Square 或 Pinball。第一阶段不评价跨流态 E2/T2-C 融合，避免把“局部预测能力”和“跨流态组装能力”混在一个实验中。

## 2. 共用数据与评价协议

### 2.1 冻结的数据划分

必须复用现有 Circular local-control study 的轨迹级训练、验证和测试划分，禁止重新随机划分快照：

- Hopf 测试集：42 个窗口，来自 `Re=47.081355, 49.022357, 51.786450`；
- Periodic 测试集：32 个窗口，来自 `Re=70.314635, 100.352251, 149.059229, 189.862278`；
- 每个测试窗口含 48 个预测步；`K=24` 使用同一窗口的前 24 步；
- 超参数、early stopping 和 checkpoint 只能使用训练集和验证集；测试集在配置冻结后一次性解封；
- 重叠窗口不能作为相互独立的统计样本，三种子统计的独立重复单位是训练种子，而不是窗口。

训练前应生成并保存一份 `split_manifest.json`，至少记录轨迹身份、Re、起始时间/索引、训练/验证/测试标签和原始文件哈希。两个基线必须读取同一份清单。

### 2.2 统一预测任务

两类模型均执行 autonomous rollout：模型只接收窗口起点可用的参数和历史信息，后续预测递归使用自身输出，不得读取未来真实场或真实 POD 系数。

- 输入历史长度首先固定为现有 specialist 使用的历史长度；若当前 Circular 协议为 3 帧，则两个基线都使用 3 帧；
- 输出间隔与现有 Circular specialist 完全一致；
- 报告 `K=24` 和 `K=48`；
- 不以 one-step teacher-forced error 替代 autonomous rollout error；one-step 指标只能作为训练诊断放在附录。

### 2.3 统一误差定义

沿用现有物理场相对误差及聚合顺序：

1. 对每个预测步计算速度和压力物理场误差 `E_u`、`E_p`；
2. 先在预测步上平均；
3. 再在同一 Re 的测试窗口上平均；
4. 最后对测试 Re 等权平均；
5. 神经模型报告三个预先指定训练种子 `1248, 1600, 2026` 的均值和样本标准差；
6. 同时报告 joint error `E_joint = E_u + E_p`、完成率和非有限/发散窗口数。

需要区分两种参考：

- POD--LSTM/GRU 的主比较使用与现有局部模型相同的 **POD-reconstructed truth**，从而排除原始 CFD 投影余量；
- FNO 的自然输出是全场，主结果应对 **原始 CFD 场或固定规则网格上的 CFD 场** 计算。若还要加入 ROM 主表，必须额外把 FNO 输出通过固定映射送回现有有限体积积分点，并报告插值误差下限。不能把两种参考下的百分数不加说明地并列。

## 3. POD--LSTM/GRU 实验

### 3.1 模型输入与输出

每个流态独立训练，不共享 Hopf 与 Periodic 参数。输入为过去 `L` 帧标准化局部 POD 系数、标准化参数和必要的时间步信息：

```text
x_n = [a_{n-L+1:n}, b_{n-L+1:n}, standardized(1/Re)]
```

主实现直接预测下一步系数增量：

```text
Delta[a,b]_{n+1} = RNN(x_n)
[a,b]_{n+1} = [a,b]_n + Delta[a,b]_{n+1}
```

采用增量输出可以保留恒等映射附近的归纳偏置，但仍属于纯数据驱动基线。速度和压力均由 RNN 预测，不能调用 proposed model 的 Galerkin RHS、PPE 或 pressure-correction head，否则不再是独立黑箱基线。预测系数通过相应流态被冻结的 POD decoder 重构为物理场。

建议同时实现 LSTM 与 GRU，但把它们视为两个预注册基线：二者使用相同输入、损失、搜索空间和种子，不能测试后只保留更好的一个而不披露另一个。

### 3.2 训练目标

训练目标由多步闭环系数误差和物理场误差组成：

```text
L = lambda_a L_a + lambda_b L_b + lambda_u L_u + lambda_p L_p
```

推荐先令各分量通过训练集尺度标准化，再固定 `lambda_a=lambda_b=lambda_u=lambda_p=1`。如果物理场重构使显存或 I/O 成本过高，可先使用标准化系数损失训练，再用验证集确认其与物理场指标的一致性；任何权重调整都必须在测试前完成并记录。

采用与现有 local specialist 相近的 rollout curriculum，例如 `1 -> 4 -> 8 -> 16`；最终验证必须直接使用完整 `K=48` autonomous rollout。teacher forcing 只允许出现在 curriculum 早期，并记录比例；checkpoint 选择不能依据 one-step loss。

### 3.3 受控超参数搜索

在每个流态的验证集上搜索，不使用测试误差：

| 项目 | 预注册候选 |
|---|---|
| cell | LSTM；GRU |
| hidden width | 32，64，128 |
| recurrent layers | 1，2 |
| history length | 主实验固定为与 proposed 相同；可将其他长度作为附录敏感性分析 |
| dropout | 0，0.1 |
| optimizer | AdamW |
| initial learning rate | `1e-3`，`3e-4` |
| weight decay | `0`，`1e-5`，`1e-4` |
| gradient clipping | global norm 1.0 |

为避免组合爆炸，先用种子 1248 完成结构筛选；随后冻结每个流态的 cell、宽度、层数、学习率和 checkpoint 规则，再运行三个正式种子。LSTM 与 GRU 均使用相同的搜索预算。选择准则建议复用现有规则：先最小化最差 Re 的验证集 full-window joint error，再以 pooled joint error 作为平局判据。

### 3.4 公平性与必须报告的量

- 训练轨迹、POD basis、均值场、标准化统计和测试窗口与 proposed 模型一致；
- 报告总参数量、训练时激活参数量、选中 checkpoint、训练时间和 batch-one `K=48` 延迟；
- 统计每个种子的验证接受状态，失败种子不得事后替换；
- 报告 `E_u`、`E_p`、`E_joint`，以及三种子中成功完成全部测试窗口的数量；
- 参数量可以另给 active-matched 版本，但不能为了匹配参数量而给 LSTM/GRU 更大的超参数搜索预算。

### 3.5 POD--LSTM/GRU 的论文结论边界

该实验只能回答“在相同局部 POD 表示和数据下，结构化混合 Galerkin 模型相对于通用循环动力学模型表现如何”。它不能单独证明跨流态路由、T2-C 融合或任意时间稳定性。

## 4. FNO 实验

### 4.1 为什么需要单独设计

标准 2D FNO 假设规则网格，而当前 Circular 数据来自有限体积网格并含固体圆柱。直接插值会引入额外误差，因此 FNO 首先需要通过“网格可行性门槛”，之后才能进入正式训练。FNO 是全场 surrogate，不应被描述为参数匹配的 ROM 消融。

### 4.2 网格与场表示

为 Circular 建立一个固定笛卡尔网格，覆盖所有训练、验证和测试场：

- 通道：`u_x, u_y, p`；
- 静态输入：`x, y` 坐标、流体/固体 mask；
- 条件输入：广播后的标准化 `1/Re`；
- 历史输入：过去 `L` 帧的三个物理通道；
- 输出：下一时间步的 `u_x, u_y, p` 或相应增量。

所有有限体积场到规则网格、规则网格回有限体积点的算子必须在训练前冻结，并保存权重/邻接与哈希。圆柱内部值固定为零或明确的掩码值，损失和误差只在流体区域计算。

### 4.3 插值可行性门槛

正式训练前，对真实 CFD 场执行闭环映射：

```text
FV field -> Cartesian grid -> FV field
```

分别计算速度和压力的插值往返误差，并按 Re 汇总。建议至少比较两个候选网格分辨率，在以下条件下选择较小者：

1. 流场关键尾迹结构在可视化中没有明显损失；
2. 往返误差显著低于准备比较的模型误差，例如不超过现有最优模型误差的 10%--20%；
3. 单次训练可在现有 4090 显存内完成。

如果任何可承受分辨率都无法满足门槛，应停止标准 FNO 实验，并将原因记录为“几何/离散不匹配”，而不是报告一个被插值误差主导的结果。此时可转向 Geo-FNO、图神经 PDE solver，或仅保留 POD--LSTM/GRU。

### 4.4 FNO 架构与训练

使用 autoregressive 2D FNO，而不是一次性输出整段未来，以便与现有闭环预测任务一致：

```text
[past fields, coordinates, mask, 1/Re]
    -> lifting layer
    -> four Fourier layers
    -> projection head
    -> next field / field increment
```

建议验证搜索空间：

| 项目 | 预注册候选 |
|---|---|
| Fourier layers | 固定 4 |
| retained modes per direction | 12，16，24；不得超过网格 Nyquist 限制 |
| channel width | 32，48，64 |
| history length | 与 proposed 相同 |
| output form | direct field；field increment，二选一由验证集决定 |
| optimizer | AdamW |
| initial learning rate | `1e-3`，`3e-4` |
| scheduler | cosine decay 或 plateau，预先固定其一 |
| weight decay | `1e-5`，`1e-4` |

训练使用流体 mask 下的归一化 `u_x/u_y/p` 损失，并逐步增加闭环 rollout 长度。由于长 rollout 反向传播开销较高，可以采用截断反向传播，但最终验证和测试必须完整递归到 `K=48`。不得在测试 rollout 中重新注入真实帧。

### 4.5 FNO 的两套评价

FNO 应报告两套相互区分的结果：

1. **Native-grid evaluation**：在固定笛卡尔网格上相对于插值后的 CFD 真值计算误差，反映 FNO 自身的学习误差；
2. **Common physical-field evaluation**：将 FNO 预测固定映射回有限体积点，使用与 ROM 相同的质量/体积权重计算 `E_u`、`E_p` 和 `E_joint`，用于谨慎的横向比较。

此外单独报告 `FV -> grid -> FV` 的 interpolation floor。若 FNO 的 common-space error 接近该下限，应明确说明结果受到插值瓶颈限制。

### 4.6 FNO 的计算成本报告

- 总参数量和实际激活参数量；
- 训练 GPU、精度、batch size、epoch/step 数和选中 checkpoint；
- 三个训练种子的总训练时间；
- batch-one `K=24/48` 预测时间，包含和不包含网格映射各报告一次；
- 峰值 GPU 显存；
- 不把不同硬件或不同 I/O 范围下的时间直接解释为端到端加速比。

## 5. 实验执行顺序与停止条件

### 阶段 A：数据与协议冻结

- [ ] 生成唯一 `split_manifest.json`；
- [ ] 核对 Circular Hopf/Periodic POD basis、均值场和标准化统计；
- [ ] 用现有 evaluator 重算一个已知 checkpoint，确认误差口径能够复现论文数值；
- [ ] 固定三种子、checkpoint 选择规则和失败判据。

验收证据：manifest、数据哈希、复现误差与现有表格在舍入容差内一致。

### 阶段 B：POD--LSTM/GRU

- [ ] 单种子 smoke test：one-step 与 4-step rollout 均有限；
- [ ] 验证集结构筛选；
- [ ] 冻结配置；
- [ ] Hopf/Periodic × LSTM/GRU × 三种子正式训练；
- [ ] 统一 evaluator 生成 `K=24/48` 结果与延迟统计。

停止条件：如果某配置在预先规定的三次种子中出现训练失败，保留失败率，不追加替代种子；如果两种 RNN 都无法完成测试 rollout，先审计标准化和闭环训练，而不是缩短测试窗口。

### 阶段 C：FNO 可行性与正式训练

- [ ] 构建并冻结 FV/规则网格双向映射；
- [ ] 完成插值误差与网格分辨率审计；
- [ ] 通过可行性门槛后进行单种子 smoke test；
- [ ] 验证集架构选择；
- [ ] Hopf/Periodic 三种子训练；
- [ ] native-grid 与 common-space 双重评价。

停止条件：插值往返误差主导目标误差、显存无法支持最低可接受网格，或模型无法在不注入真值的情况下完成 `K=48`。停止时保留可行性报告，不把失败隐藏为“未报告”。

## 6. 结果文件与可复核性

每个正式 run 至少保存：

```text
config.yaml
split_manifest.json
data_audit.json
best_checkpoint.pt
checkpoint_selection.json
train_history.csv
test_predictions_or_hashes.json
metrics_by_window.csv
metrics_by_re.csv
summary.json
latency.json
environment.txt
```

`summary.json` 应包含模型身份、代码提交/源码哈希、种子、参数量、选中 step、验证目标、测试完成率、`K=24/48` 的 `E_u/E_p/E_joint` 和错误状态。聚合脚本只读取这些冻结结果，不从日志手工抄数。

## 7. 论文中的建议呈现

### 7.1 主文

主文局部预测表优先放：

```text
POD--Galerkin
OpInf
POD--GRU（或同时给 POD--LSTM）
ql-ROM（完成后）
RAL-MoE-ROM / relevant frozen configuration
```

Dense 与 Structured 仍属于内部结构消融，建议与外部方法表分开。若 LSTM 和 GRU 结果接近，可在主文给验证预先指定的主要 RNN，并在附录完整给出二者；但正文必须说明另一模型的位置，不能根据测试集选择后静默删除。

FNO 建议单独作为“full-field neural-operator comparison”表或附录小节，列出其参考场、网格分辨率和插值下限。只有 common-space evaluator 完全闭环后，才能把 FNO 数值放入 ROM 主表，并应通过分组或脚注明确其全场模型身份。

### 7.2 建议表格字段

```latex
\begin{tabular}{llcccccc}
\toprule
Regime & Model & $K$ & $E_u$ & $E_p$ & $E_{\rm joint}$
& Accepted & Params \\
\midrule
% Hopf and Periodic results
\bottomrule
\end{tabular}
```

FNO 另表增加 `Grid`、`Interpolation floor` 和 `Peak GPU memory`。速度比较单独成表，避免把预测精度与硬件不匹配的时间混为一个结论。

## 8. 可支持与不可支持的结论

完成 POD--LSTM/GRU 后，可以讨论：在相同 POD 表示、数据划分和闭环窗口下，结构化物理--数据混合模型相对通用循环模型的精度、可靠性与成本。

完成 FNO 后，可以讨论：在固定 Circular 几何和相同轨迹划分下，降阶混合模型与全场神经算子的精度--成本权衡。

这些实验仍不能单独支持：

- 对所有神经算子、PINN 或所有局部 ROM 的普遍优越性；
- 任意长时间稳定性；
- 跨几何或跨离散分辨率泛化；
- 若只测试 Circular，则不能声称三个 benchmark 上均优于这些外部方法。

## 9. 方法来源

- FNO：Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021.  
  <https://openreview.net/pdf/53c47f849d1cd4d21b865caf7d774e07a5c42aa4.pdf>
- 时间依赖 PDE 中显式记忆建模的近期 ICLR 参考：Buitrago Ruiz et al., *On the Benefits of Memory for Modeling Time-Dependent PDEs*, ICLR 2025.  
  <https://proceedings.iclr.cc/paper_files/paper/2025/hash/89379d5fc6eb34ff98488202fb52b9d0-Abstract-Conference.html>

