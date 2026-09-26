# Centered-square T2-C 描述量消融与三种子实验报告

日期：2026-09-11。状态：**两组 overlap、两种输入设置、各三个种子，共 12 次 gate 训练及测试全部完成。**

本报告接续 `ICLR_EXPERIMENT_P1_P2_P3_REPORT.md` 的前置核对。用户已明确授权暂缓 Table 4 和论文描述量维数问题，先执行消融与三种子实验。本轮只使用原运行实现开展新实验，不修改 Table 4、论文描述量定义、原始结果或 Overleaf。

## 1. 核心结果

下表报告 K24 **末步**联合物理场误差，单位为 %；“±”是三个训练种子间的 sample SD（ddof=1），不是跨 Re 的 SD，也不是置信区间。

| Overlap | T2-C μ-only | T2-C full | full 相对 μ-only 的均值误差降低 |
|---|---:|---:|---:|
| S–H | 2.185039 ± 0.000382 | **1.195722 ± 0.000103** | 45.28% |
| H–P | 1.151164 ± 0.000702 | **0.850040 ± 0.003475** | 26.16% |

两个 overlap 内，三个配对种子均是 full 优于 μ-only。结果支持：**在当前固定候选轨迹、E2 和训练选择规则下，历史描述量对门控修正有明确的增益**。它不是“整个 RAL-MoE-ROM 已完成多种子验证”，也不是跨数据划分或跨 benchmark 的不确定性结论。

一个需要如实保留的细节：S–H 的 μ-only 三个种子都选择第 1 步 checkpoint，性能接近 E2 probability blend；H–P 的 μ-only 都选择第 400 步，能明显改善 E2 probability blend，但仍不及 full。因此不能把两个 overlap 都写成“参数修正先改善、历史修正再改善”的同一严格递进故事。

## 2. 实验范围和对照定义

本轮完成：

- S–H：full / μ-only，seed 42001、42002、42003。
- H–P：full / μ-only，seed 42001、42002、42003。
- 原始 full checkpoint 前向复现。
- 每个新 run 的验证选择、held-out K24 分量误差、逐 Re 和逐窗口权重记录。
- 训练后重新加载 checkpoint 复核，以及 μ-only 历史扰动不变性检查。

未执行：Circular Periodic specialist 的可选三种子实验、Table 4 统计转换、论文重编译或上传。这里的“12 次”全部是轻量 gate 训练，不是 12 次 specialist 训练。

### 2.1 两种输入

沿用原始运行的实际输入和网络，不调整论文定义：

$$
x_{full}=[\widetilde\mu;\widetilde d_1;\ldots;\widetilde d_8],\qquad
x_{\mu}=[\widetilde\mu;0;\ldots;0].
$$

所有特征首先使用同一原训练集 mean/std 标准化，**然后**将 μ-only 的八个历史通道置零。不能在原始未标准化特征上置零后再标准化，因为那会留下非零常量输入。

两组均为 9→64→64→1 的 SiLU MLP，共 **4865 个可训练参数**，输出层零初始化。μ-only 保留全部参数和矩阵形状，但历史输入为零，因此其实际信息能力不同；“参数量匹配”不等于“有效输入信息相同”。

两组使用完全相同的 E2 pair logit：

$$
\alpha_w=\sigma\!\left(\operatorname{logit}(\bar\pi_r(\mu_w))+c_\psi(x_w)\right).
$$

每窗口权重由初始信息计算一次，在 K24 内固定。候选轨迹独立推进；融合不反馈给 specialist。

## 3. 固定项、随机项和实际训练设置

| 项目 | 实际设置 |
|---|---|
| 原始运行 | E2_T2C_K24_20260730_STRICT_V3 |
| Seed | 42001、42002、42003，训练前固定，不按结果筛选 |
| 候选 specialist | 原缓存中的固定候选 rollout，不重新训练或选择 |
| E2 | 原 `e2_router/best.pt`，不重新训练 |
| Horizon | 24 |
| 网络 | 9→64→64→1，SiLU，输出层零初始化 |
| 优化器 | AdamW，lr=3e-4，weight decay=1e-4 |
| 梯度裁剪 | global norm 1.0 |
| 训练步数 | 每次 8000 |
| Batch | 有放回采样 min(256, n_train)，S–H 为 56，H–P 为 32 |
| 验证频率 | 第 1 步、随后每 100 步，共 81 次 |
| 选择规则 | 验证集 `(worst_Re_mean, joint_mean)` 字典序最小；同分保留先出现者 |
| 训练损失 | 全窗口、全预测步的速度/压力平方归一化场误差之和 |
| 测试报告 | K24 末步 Eu、Ep、Eu+Ep；另外保留其他 horizon 和 all-step 指标 |
| 归一化 | 同一原训练缓存的 mean/std；小于 1e-8 的 std 替换为 1 |
| 随机性 | Python / NumPy / PyTorch seed；网络初始化和随机 minibatch 顺序 |
| GPU / 软件 | RTX 4090；PyTorch 2.11.0+cu126；NumPy 2.4.4 |
| 外部日志服务 | disabled；未向 Swanlab 在线上传 |

没有为了 μ-only 改学习率、训练步数、模型大小、选择目标或测试集。每个种子训练均完成 8000 步，“best step=1”不是只训练了一步。

12 次训练脚本记录的耗时合计约 **217.56 秒**，单次 16.88–19.41 秒，不含前向复现、进程启动、验证及文件传输。代价较低是因为重用已保存的候选 rollout 二次型缓存，未重新运行流场求解器或 specialist。

## 4. 数据划分与候选顺序

| Overlap | Train Re | Validation Re | Held-out Re |
|---|---|---|---|
| S–H | 94, 95, 95.05, 95.15, 95.2, 95.35, 95.4 | 94.5, 95.25 | 95.1, 95.3 |
| H–P | 98.5, 99.5, 100, 101 | 99, 101.5 | 100.5, 102 |

每个 Re 有 8 个窗口。因此 S–H 开发缓存 72 个窗口（56 train + 16 validation），H–P 开发缓存 48 个窗口（32 + 16）。两组测试均为 16 个窗口。

S–H 的 candidate 1/2 = Steady/Hopf；H–P 的 candidate 1/2 = Periodic/Hopf。H–P 的 α 是 **Periodic 权重**，不是 Hopf 权重。

已检查 gate 缓存内三种 split 的 Re 两两不相交，candidate 顺序与 checkpoint 一致。这里验证的是当前 gate 实验的划分；不能据此声称已经重新审计全部 specialist、E2 或基底构建的历史数据流程。

## 5. 误差、聚合和 oracle 口径

缓存给出物理场误差二次型。设 $A=\|q_1-q_*\|_M^2$、$B=\|q_2-q_*\|_M^2$、$C=\langle q_1-q_*,q_2-q_*\rangle_M$、$T=\|q_*\|_M^2$，则：

$$
E_q(w,k)=100\sqrt{\max\left(\frac{\alpha_w^2 A_{wk}+(1-\alpha_w)^2 B_{wk}+2\alpha_w(1-\alpha_w)C_{wk}}{\max(T_{wk},10^{-12})},0\right)}.
$$

速度按单元面积对两个分量加权；压力的参考和候选均先去除面积加权均值。参考物理场沿用原缓存的重建参考，没有在本轮换成另一种 truth 定义。

$$
E_{joint}=E_u+E_p,\qquad
\overline E_q^{K24}=\frac1{16}\sum_w E_q(w,24).
$$

不是 24 步平均，也不是所有字段合并后的单一归一化误差。本次每个 Re 窗口数相同，所以 pooled mean 与先求逐 Re 均值再平均恰好相等。

原 oracle 在 0 到 1、间隔 1e-4 的 10001 点网格上，为每个窗口最小化 **24 步平均 Eu+Ep**，随后报告该权重的 K24 末步误差。它与训练的平方误差目标不同，且不是 K24 指标的严格最优下界。本轮保留其原实现作为诊断参照。

## 6. 原 fixed run 复现

本轮实际加载原 gate 和 router，重新前向计算 held-out 缓存，结果为：

| Overlap | 原论文 E_joint (%) | 本轮原 checkpoint 重算 (%) |
|---|---:|---:|
| S–H | 1.1958 | 1.1958408357404098 |
| H–P | 0.8465 | 0.8465289562715951 |

除了这两个四位小数，还检查原比较表所有方法的 K24 分量及 worst 指标与原报告在比例值 1e-7 内一致。新训练的 full seed 42001 也复现上述两个联合误差值。

原模型、开发缓存、held-out 缓存的身份经过哈希核对。原始结果没有被本轮 seed 平均值替换。

## 7. 全部逐种子结果

单位为 %，best step 只依据验证集选择。

| Overlap | 模式 | Seed | Best step | Eu | Ep | E_joint |
|---|---|---:|---:|---:|---:|---:|
| S–H | full | 42001 | 5500 | 0.461510 | 0.734331 | 1.195841 |
| S–H | full | 42002 | 7100 | 0.461586 | 0.734089 | 1.195675 |
| S–H | full | 42003 | 5900 | 0.461579 | 0.734072 | 1.195651 |
| S–H | μ-only | 42001 | 1 | 0.789761 | 1.394837 | 2.184598 |
| S–H | μ-only | 42002 | 1 | 0.790201 | 1.395043 | 2.185244 |
| S–H | μ-only | 42003 | 1 | 0.790221 | 1.395053 | 2.185274 |
| H–P | full | 42001 | 3500 | 0.450912 | 0.395617 | 0.846529 |
| H–P | full | 42002 | 3500 | 0.451791 | 0.398323 | 0.850113 |
| H–P | full | 42003 | 3200 | 0.453617 | 0.399860 | 0.853477 |
| H–P | μ-only | 42001 | 400 | 0.587480 | 0.562888 | 1.150368 |
| H–P | μ-only | 42002 | 400 | 0.588571 | 0.563126 | 1.151696 |
| H–P | μ-only | 42003 | 400 | 0.595955 | 0.555471 | 1.151426 |

### 7.1 分量的三种子均值与 SD_seed

| Overlap | 模式 | Eu (%) | Ep (%) | E_joint (%) |
|---|---|---:|---:|---:|
| S–H | full | 0.461558 ± 0.000042 | 0.734164 ± 0.000145 | 1.195722 ± 0.000103 |
| S–H | μ-only | 0.790061 ± 0.000260 | 1.394977 ± 0.000122 | 2.185039 ± 0.000382 |
| H–P | full | 0.452107 ± 0.001380 | 0.397933 ± 0.002148 | 0.850040 ± 0.003475 |
| H–P | μ-only | 0.590669 ± 0.004611 | 0.560495 ± 0.004353 | 1.151164 ± 0.000702 |

联合误差的 SD 从每个种子的 Eu+Ep 重新计算，**不能把 Eu 的 SD 和 Ep 的 SD 相加**。所有 CSV 保留未四舍五入数值。

### 7.2 配对差值

以同一个 seed 的 `μ-only − full` 定义差值，单位为百分点：

| Overlap | 平均差值 ± SD_seed | full 更优的配对数 |
|---|---:|---:|
| S–H | 0.989316 ± 0.000485 | 3/3 |
| H–P | 0.301124 ± 0.002972 | 3/3 |

本报告不做小样本显著性宣称，不把三个种子的低 SD 解释成整个方法的统计误差已经充分刻画。

## 8. 保留原 fixed run 的基线对照表

以下 full 与基线来自原 checkpoint 重算；μ-only 使用预先固定的 seed 42001，不是挑三个种子中测试最好的那个。

| Overlap | 方法 | Eu (%) | Ep (%) | E_joint (%) |
|---|---|---:|---:|---:|
| S–H | Steady only | 2.597588 | 2.245047 | 4.842635 |
| S–H | Hopf only / E2 Top-1 | 0.494206 | 1.235593 | 1.729799 |
| S–H | Equal | 1.364401 | 1.667656 | 3.032057 |
| S–H | E2 probability | 0.789843 | 1.394875 | 2.184717 |
| S–H | T2-C μ-only | 0.789761 | 1.394837 | 2.184598 |
| S–H | T2-C full | 0.461510 | 0.734331 | 1.195841 |
| S–H | Oracle | 0.465865 | 0.718349 | 1.184215 |
| H–P | Periodic only | 0.642001 | 0.565943 | 1.207944 |
| H–P | Hopf only | 1.003397 | 1.868420 | 2.871817 |
| H–P | E2 Top-1 | 0.742165 | 1.105184 | 1.847349 |
| H–P | Equal | 0.607745 | 1.006882 | 1.614627 |
| H–P | E2 probability | 0.615396 | 1.046445 | 1.661841 |
| H–P | T2-C μ-only | 0.587480 | 0.562888 | 1.150368 |
| H–P | T2-C full | 0.450912 | 0.395617 | 0.846529 |
| H–P | Oracle | 0.447877 | 0.390517 | 0.838394 |

适合后续 Table 3 的 μ-only 新列为 **S–H 2.1846、H–P 1.1504**。原 full 列仍为 1.1958、0.8465；三种子均值另列，不覆盖原数值。

## 9. 结果分析与可支持的机制结论

### 9.1 S–H：增益主要来自历史条件化，而非纯参数修正

μ-only 的三个最佳 checkpoint 均为 step 1，联合误差与 E2 probability 的 2.184717% 几乎相同，而且仍差于 E2 Top-1 的 1.729799%。full 则稳定达到约 1.1957%。

这支持“在当前设置中，仅有参数输入的修正没有得到有效验证改进，加入历史后取得明显增益”，不支持“学习任何 correction 都能改善 S–H”的笼统说法。

为什么选 step 1？不是训练未执行，也不是权重全被冻结：代码训练了 8000 步。以 μ-only seed 42001 为例，验证 worst-Re 全步平均联合误差从 step 1 的 **1.822292%** 变为 step 8000 的 **2.069412%**，因此原选择规则保留最早 checkpoint。这一现象与“参数信息不足以按窗口区分候选质量”相容，但也可能受固定损失和验证规则影响；本轮未做额外超参数搜索，不能证明所有 parameter-only 设计均无效。

full seed 42001 在 Re 95.1、95.3 的平均 Steady 权重分别约 0.0967、0.0971；μ-only 对应约 0.2460、0.2288。full 还可以在同一 Re 的不同窗口产生不同权重，μ-only 则不能。

### 9.2 H–P：参数修正有贡献，历史仍提供额外增益

E2 probability 为 1.661841%，μ-only seed 42001 降到 1.150368%，full 降到 0.846529%。三种子均值的 full 相对 μ-only 进一步降低 26.16%。这里可以使用“参数修正改善，再由历史条件化进一步改善”的叙述。

μ-only 的 Periodic 权重偏高；seed 42001 在两个 held-out Re 的权重约为 0.8975 和 0.9357。full 的相应窗口平均权重约 0.5870 和 0.7094，并保留窗口级变化。这与两类专家的状态依赖互补性相容，但权重本身不是独立的物理因果证据。

### 9.3 逐 Re 检查：固定种子的改进不是仅由一个 Re 拉动

| Overlap | Re | μ-only E_joint (%) | full E_joint (%) |
|---|---:|---:|---:|
| S–H | 95.1 | 2.194146 | 1.182653 |
| S–H | 95.3 | 2.175051 | 1.209029 |
| H–P | 100.5 | 1.287181 | 0.857059 |
| H–P | 102.0 | 1.013556 | 0.835999 |

这是 seed 42001 的逐 Re 对照；全部三个种子的逐 Re 数据另存 CSV。两组各只有两个测试 Re，应避免外推为整个参数域都已充分验证。

### 9.4 小 seed SD 的解释边界

这里只有轻量门控被重新训练；候选轨迹、基底、E2、数据划分和 scaler 均固定。低 seed SD 表明这个条件下的门控优化较稳定，**不包括 specialist 训练不确定性、数据抽样不确定性或数值求解误差**。μ-only 的 SD 很小也部分来自其选择点很早，不能单凭 SD 小就说模型更好。

## 10. 适合后续论文整合的英文表述

以下是基于本轮结果的备选文字，尚未写入论文；后续方法与实验定义仍需保持一致。

### Descriptor ablation

> With the candidate specialists and E2 router fixed, we compare the full correction gate with an architecture-matched parameter-only gate obtained by zeroing the standardized history channels. Over three paired training seeds, the mean K=24 joint error decreases from 2.18504% to 1.19572% on S–H and from 1.15116% to 0.85004% on H–P. The parameter-only correction improves upon probability blending on H–P but remains close to it on S–H, whereas history conditioning improves both overlaps under the same optimization and validation-selection protocol.

### Seed variability

> Across three training seeds, the full T2-C gate achieves joint errors of 1.19572 ± 0.00010% on S–H and 0.85004 ± 0.00347% on H–P. The reported standard deviations quantify gate-training seed variability conditional on fixed specialists, routing, and data partitions.

### Limitations

> Repeated-seed variability is evaluated for the T2-C correction gate. A full multi-seed study of the local specialists and the complete hierarchical pipeline remains future work.

不建议使用 “statistically significant”“all settings”“the descriptor is universally necessary” 等超出本轮证据范围的表述。

## 11. 验证与文件完整性

训练前核对包括：原始 asset 哈希、原 evaluator 回放、原训练 history 的 8000 步/81 次验证日程、scaler 与原 checkpoint 完全一致、两个 overlap 的候选顺序、split 不相交、缓存候选 finite 且未超限。

训练后 **118 项检查全部通过**，包括：

- 12 个 run 全部存在，训练 history 完整。
- 按保存的验证 history 重新求 argmin，等于实际 best step。
- 日志中的 loss、梯度范数均有限。
- 两组模型均为 4865 参数，原 checkpoint 和训练代码哈希未改变。
- 重新加载每个 checkpoint，逐窗口 α 和 K24 指标与保存结果一致。
- μ-only 将历史特征任意增加 1000 后，预测 α 完全不变。
- μ-only 同一 Re 的所有窗口 α 完全相同。

统计端额外验证全部 12 个 `(overlap, mode, seed)` 组合齐全，`E_joint=E_u+E_p`，sample SD 使用 ddof=1。对原论文 manifest 内 **31 个源文件/素材**重新计算 SHA-256，全部未改变。

原始训练代码在 `centeredsquare_fusion_v1` 下保持不变；本轮改动只在独立 study 副本加入输入 masking、配置记录和模式元数据。项目执行管理与代码修改技能用于约束这一最小改动和验证流程。

## 12. 交付文件与复现入口

论文项目根目录：`C:\Users\panxy1019\Documents\CHANNEL\RAL_MOE_ROM_ICLR2027`。

### 汇总结果

- `artifacts/t2c_descriptor_ablation.csv`：原 checkpoint 各基线分量 + 新 μ-only 固定 seed 42001 分量。
- `artifacts/t2c_seed_results.csv`：12 个 run 的逐种子分量和 best step。
- `artifacts/t2c_seed_summary.csv`：Eu/Ep/E_joint 的三种子均值与 sample SD。
- `artifacts/t2c_seed_per_re.csv`：全部种子的逐 Re 分量、窗口数及权重。
- `artifacts/t2c_paired_seed_differences.csv`：配对差值与相对均值改善。
- `artifacts/t2c_experiment_summary.json`：本轮完成状态与完整摘要。

### 原始运行和代码

`experiments/iclr_seed_study_20260911/results_v1/` 保存预注册配置、原始前向复现、12 个 best.pt、全部 history、config、test_metrics、验证结果和运行状态。

对应代码在 `experiments/iclr_seed_study_20260911/code/`，复现说明见该 study 的 `README.md`。

集群结果目录：

```text
/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/experiments/iclr_seed_study_20260911/results_v1
```

本地重新统计命令：

```powershell
python C:\Users\panxy1019\Documents\CHANNEL\RAL_MOE_ROM_ICLR2027\scripts\verify_iclr_experiment_tables.py --t2c-study-only
```

不加 `--t2c-study-only` 仍运行上一轮 Table 4 前置核对，按历史设计返回未通过；它不是本轮 gate 实验失败。本轮单独状态为 `T2C_ABLATION_AND_THREE_SEEDS_COMPLETE`，不把被暂缓的问题伪装为已解决。

合并原始新实验结果 `results.json` 的 SHA-256：

`2c9301fda6684332fdb7d59714b4cf8a40577e27becb6cd35fd434ba2fe1935c`

## 13. 交付边界

本轮消融与门控三种子实验已完成，无仍在运行的本轮训练任务。没有自动扩大到 specialist 重训，没有更改 Table 4，没有修订论文中的描述量定义，没有更新原论文数值、PDF 或 Overleaf。后续整合时可使用本报告的固定种子列与独立 seed 表，但不得混淆两种统计口径。
