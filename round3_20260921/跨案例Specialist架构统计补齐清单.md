# 跨案例 Specialist 架构统计补齐清单

## 1. 目的

现有附录段落 `Audited circular-cylinder architectures` 主要描述参与内部路由分析的 Circular H/P，不能作为 Circular、Square、Pinball 全部局部模型的架构说明。

本任务应补充各案例实际使用模型的结构、输入、输出、推进路径及训练/冻结状态，并建立它们与论文结果的对应关系。完成后可形成跨案例架构表，而不是把 Circular 的配置复制到其他案例。

**本文件是待执行的统计清单，不是新代码审计结果。** 已有线索来自用户提供的 2026-09-21 专家路由核实报告；需明确区分已有证据、仍需核实和不适用项。优先只读整理，不启动重训，不改写现有检查点或结果。

## 2. 与 E2/T2-C 统计的边界

本清单统计的是产生局部预测的 **specialist**，包括 POD 后的特征编码器、内部组路由、通道专家、压力模块以及实际更新规则。

以下另由《E2与T2C跨案例统计补齐清单》处理，不应混入 specialist 参数量：

- 外层 E2：参数到 S/H/P 偏好的网络。
- T2-C gate：物理历史描述量到融合修正的网络。
- 跨 chart 候选集合与外层结果映射。

但两份统计必须通过相同的候选检查点和评价入口相互关联。

## 3. 已有证据与关键缺口

| 实例 | 用户核实报告已有结论 | 本轮补齐重点 |
|---|---|---|
| Circular S | 论文结果身份未闭环；已有 S4 仅诊断模型 | 找到论文 Steady 行实际 checkpoint/评价入口；不能用诊断架构替代 |
| Circular H/P | sparse-MoE 身份、shape、严格加载和真实路由前向证据较完整 | 统一架构表；区分原版和二次初始化修正版、不同训练种子 |
| Square S | 三组 sparse-MoE；baseline 压力；velocity experts frozen | 精确冻结范围、实际输出缩放、参数计数及与缓存候选绑定 |
| Square H | 单组 MoE 参数存在；最终 overlap evaluator 为 data-only 输出路径 | 同时报告存储架构和实际执行路径，不能把存储 PPE/gate 当成部署使用 |
| Square P | 三组结构，部署固定 group index 1；adaptive PPE gate；内部子步 | 固定组索引约定、实际活跃模块、子步规则及计数 |
| Pinball S/H/P | 已定位 B1 候选为 Deep-FNN，不含 MoE 路由和结构化专家；论文映射程度不同 | 先绑定表格结果，再核实普通网络的层数、输入/输出、历史和更新规则；MoE 项填不适用 |

特别注意：报告指出 Pinball P 的数值和路径关联 Deep-FNN-H3；Pinball H 已有核查记录 `heldout_loaded=false`。这些事实不能升级为三个 Pinball 论文结果均已最终闭环。

## 4. 第一步：模型身份与运行路径

对所有 S/H/P 实例和论文使用的 baseline，记录：

1. 论文表格/行、方法名称和对应评价 split。
2. 正式评价 JSON、评价入口、导入的模型类与 forward。
3. 检查点绝对路径、SHA-256、保存配置及源码版本/哈希。
4. 独立 specialist 评价与 overlap 候选评价是否使用相同模型和包装器。
5. 原版、初始化修正版、稳定化修正版及不同种子是否分别存档。
6. 严格加载是否成功；有无 missing/unexpected keys、非严格加载或额外覆盖参数。
7. 结构是否由 checkpoint 内配置恢复，而非当前脚本默认值。

### 身份表

| 案例/流态 | 原版/修正版/seed | 论文结果位置 | 独立或 overlap | 模型类 | Checkpoint/hash | 评价入口 | 映射状态 |
|---|---|---|---|---|---|---|---|
| 待填写 | | | | | | | |

没有完成身份绑定时，可以描述为“已定位候选”，不能写成“论文使用的模型”。

## 5. 第二步：所有模型共有的输入与输出统计

### 5.1 局部坐标与特征

- 速度/压力 reduced dimension：d_u、d_p。
- POD mean/basis、压力 gauge 和训练标准化统计的来源。
- 历史长度：当前状态是否计入；时间顺序与滞后间隔。
- 输入是否含 a、b、Galerkin RHS、参数、phase harmonics、差分、norm/energy 派生量。
- 各特征块的精确排列、维数及总 D。
- 特征与 target 是否分别标准化，scaler 的拟合 split 和反标准化位置。
- 输入中的 Galerkin RHS 是否真正作为输出 backbone 相加：二者必须分别记录。

### 5.2 输出和推进

- 网络输出是速度 RHS、系数增量、下一步系数还是物理修正项。
- 压力输出是 residual、next pressure 还是其他 target。
- 反标准化、时间步乘法、输出缩放和 correction bias 的顺序。
- 使用 additive Galerkin+correction，还是 data-only update。
- 实际 Step：RK、其他积分器或直接离散更新；子步数、最大内部步长。
- 压力何时重构，历史何时移位，阶段内重算哪些特征和路由。

### 通用结构表

| 实例 | 模型家族 | d_u/d_p | 历史长度 | 输入 D/特征块 | 隐藏表示 | 输出含义 | 推进规则 | 压力路径 |
|---|---|---|---|---|---|---|---|---|
| 待填写 | | | | | | | | |

## 6. 第三步：sparse-MoE 特有配置

仅对确认使用该结构的实例填写；普通 Deep-FNN 填“不适用”，而不是“0 个专家”。

### 6.1 特征编码器

- 主干各层宽度、LayerNorm/其他归一化、激活、dropout、bias。
- refinement residual block 的数量和展开结构。
- 输入 D、输出 h；velocity/pressure 是否共享 encoder。
- 配置中的 `num_blocks` 与实际主干/refinement 数量分别记录，避免把 `num_blocks=3` 解释成三个 refinement blocks。

### 6.2 内部组与通道路由

- 组数 G、组选择策略：学习 hard Top-1、单组、固定组。
- 固定组实际索引及 0-based/1-based 约定。
- 组/通道路由器输入、层宽、激活、dropout、温度。
- 是否使用 straight-through、gate floor 或其他训练/推理差异。
- 每组每通道 routed/shared 数量、Top-k、共享专家是否排除竞争。
- shared/routed 配置比例及归一化方式；不要统一套用 4/7。

### 6.3 单个专家

- 非线性分支的拼接输入、projection width、残差块数 B、展开宽度、激活、LayerNorm、dropout、输出 head。
- 残差块可学习缩放的形式和初值。
- 线性分支是否无 bias，输入状态切片和输出维数。
- 二次分支 U/V 张量形状、rank、kappa、收缩方式，有无额外投影。
- shared/routed、不同组及不同通道是否同构/共享参数。
- 分支相加、专家混合和输出反标准化的先后顺序。

### 6.4 压力模块

- confidence head 结构、输入和输出维数，是否实际执行。
- adaptive gate、baseline PPE+residual 或 data-only 的实际选择。
- 是否为 gamma*PPE+residual，不得误写为凸组合。
- 是否启用 pressure state override，其与原 context 编码的关系。

### MoE 配置表

| 实例 | D/h | Encoder/refinement | G/组策略 | Routers/温度 | Routed/shared/Top-k | Expert B/展开宽度 | Rank/kappa | Shared:routed | 压力模式 |
|---|---|---|---|---|---|---|---|---|---|
| 待填写 | | | | | | | | | |

### 已有数值线索（不能代替逐实例证据）

| 实例 | d_u,d_p | D | h | G/策略 | Expert B/展开宽度 | Shared:routed |
|---|---|---|---|---|---|---|
| Circular H | 32,32 | 493 | 256 | 1/单组 | 4/1024 | 1:0.75 |
| Circular P | 32,32 | 501 | 224 | 3/学习 | 3/768 | 1:0.75 |
| Square S | 5,4 | 91 | 224 | 3/学习 | 3/768 | 1:0.85 |
| Square H | 11,11 | 178 | 256 | 1/单组 | 4/1024 | 1:0.75 |
| Square P | 28,26 | 431 | 224 | 3/部署固定索引1 | 3/768 | 1:0.75 |

报告还给出五实例每组每通道 6 routed+1 shared、Top-2、rank=4、kappa=0.05；这些描述的是确认的模型结构，实际 evaluator 是否绕过部分模块需另列。

## 7. 第四步：Pinball Deep-FNN 专项

不能将 Pinball 的统计强行放进 MoE 模板。对每个绑定的实际模型确认：

- 主干深度与宽度，明确“6×512”等简称究竟计哪些层。
- 输入 projection、输出 head 是否计入层数；各层 bias。
- 激活、归一化、dropout、残差连接和参数共享。
- H3 是否确实对应三状态历史，历史中是否包含压力/RHS/差分。
- velocity/pressure 是否共享主干，是否有两个独立 head。
- 网络输出 target、尺度转换及 Galerkin/PPE 的实际作用。
- 训练过程中哪些参数可训练，最终 evaluator 实际调用哪些模块。
- S/H/P 是否真的是同一结构族，不能仅凭 B1 文件名推断。

### Deep-FNN 配置表

| 实例/身份状态 | 输入与历史 | 主干层数/宽度 | 激活/归一化 | 残差/dropout | 输出 head | 参数共享 | Target/推进 | 压力路径 |
|---|---|---|---|---|---|---|---|---|
| Pinball S | | | | | | | | |
| Pinball H | | | | | | | | |
| Pinball P | | | | | | | | |

## 8. 第五步：参数量、冻结与分支激活

分别统计，不用单个“参数量”混淆多个口径：

1. 完整神经网络参数数目，按唯一参数张量计数，避免共享参数重复。
2. 训练时可更新参数数目：需训练脚本/优化器参数组证据，不能只看加载后默认 `requires_grad`。
3. 冻结模块及原因，例如 Square S 的 velocity experts；核实是否还训练 encoder/router。
4. 存储但部署未调用的模块及参数量。
5. 单次局部前向实际执行参数的口径：激活组、shared+Top-2、路由器、encoder、pressure head 是否计入。
6. 若完整 forward 计算但结果随后丢弃，也应区分“实际执行”与“对最终输出有贡献”，特别是 RK 中间压力输出。
7. POD bases、scalers、Galerkin/PPE tensors 与神经参数分开计数。
8. 不把参数量直接作为 FLOPs、实测延迟或加速比。

### 参数与部署表

| 实例/版本 | 总神经参数 | 训练可更新参数 | 冻结模块 | 未调用存储模块 | 活跃参数定义/值 | POD/物理算子是否排除 | 证据 |
|---|---|---|---|---|---|---|---|
| 待填写 | | | | | | | |

### 二次分支专门记录

- 构造器初始化与训练脚本后续重置分别记录。
- 原版实际 checkpoint 中 U/V 是否为零，是否仅能产生零二次输出。
- 修正版只置零一个因子，是否有实际非零分支输出证据。
- 原版/修正版、种子和接受/失败状态分开统计。
- “具有该参数化”“分支激活”“精度收益”是不同结论，不能互相替代。

## 9. 最小只读验证与证据等级

优先复用已有审计结果，再补缺项：

- 检查配置与 checkpoint shape 是否一致，严格加载检查。
- 用实际输入检查 encoder、router、expert 和输出形状。
- 少量前向 hook 验证实际调用模块、状态切片和输出尺度；不改缓存和权重。
- 参数计数用明确口径的脚本输出，报告是否含冻结/未调用模块。
- 没有对应 held-out 数据时，可以报告 shape/source 级确认，不能升级为真实窗口 forward 验证。

报告已有：Circular H/P 有真实窗口路由诊断；Square 主要是源码/hash/shape 级确认。无需为了补架构表默认重跑全部内部路由实验。

| 核对项 | 实例 | 结论/数值 | 配置键或源码函数/行号 | Checkpoint 身份 | 证据等级 | 状态 |
|---|---|---|---|---|---|---|
| 待填写 | | | | | | |

状态使用：已确认一致、已确认需更正、配置相关、证据不足、不适用。

## 10. 论文建议呈现

将单独的 `Audited circular-cylinder architectures` 扩展为：

### Specialist architectures and deployment configurations

- 一张跨案例总览表：模型家族、局部维数、输入/历史、主干、更新与压力路径。
- 一张 sparse-MoE 配置表：仅 Circular H/P、Square S/H/P 等已确认实例。
- 一张 Deep-FNN 配置表：仅有证据绑定的 Pinball 模型。
- 一段部署例外：Square H data-only、Square S 冻结/pressure baseline、Square P 固定组。
- 一段身份和初始化边界：Circular S 未闭环、Pinball 候选与正式结果区别、原版与修正版二次分支。

若表格过宽，可将参数计数和逐层配置放在独立表中。正文 3.2 描述确认的 sparse-MoE 机制，但不得称全部案例均采用这一结构。

## 11. 最终交付和完成标准

交付：

1. 实例身份表与缺失证据清单。
2. 跨案例通用架构表。
3. MoE/Deep-FNN 分族配置表。
4. 参数量、冻结和部署路径表。
5. 附录替换建议及必要的正文适用范围更正。

先身份，再结构，再部署，再参数量。证据缺失时明确留缺，不为九个 S/H/P 格子形式整齐而虚构统一架构。若最终确认某论文行对应 Deep-FNN，应建议更正模型归类，而不是给它补造 MoE 参数。
