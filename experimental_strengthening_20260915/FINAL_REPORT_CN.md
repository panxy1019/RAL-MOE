# RAL-MoE-ROM 实验强化最终报告

日期：2026-09-15。当前批次状态：**训练、对齐评估、计时及汇总均完成**。

本报告对应 `ICLR_Experimental_Strengthening_and_Ablation_Plan.md`，补齐本轮已确定范围的任务。范围为：Centered-square S–H/H–P 固定权重消融；Circular Periodic 单种子 Dense 对照、四方法 K48 同窗口评估及原生运行成本。**不是三套算例、全部 S/H/P 的新训练矩阵，也不是整个 RAL 的三种子或 Top-2 端到端性能认证。**

原始模型、数据、论文和 Overleaf 未覆盖。本轮新文件位于独立目录。可供论文整合的文字和表格见 `paper_ready_inserts.tex`。

## 1. 清单完成情况

| 计划项目 | 本轮交付 |
|---|---|
| Constant-alpha | 两个 overlap 的训练拟合常数、验证选择敏感性对照、测试分量与逐 Re 统计 |
| Dense correction | Circular Periodic，总参数匹配，原生训练/早停、选定模型最终评估 |
| Architecture ablation 解释 | 本报告第 7 节与可粘贴 LaTeX，明确各对照能支持和不能支持的归因 |
| 误差随时间曲线 | 四方法共同 12 窗口 K1–K48；另有按 Re 的原生时间轴曲线 |
| Cross-regime 范围 | 提供限制性英文表述，不声称 unseen physics 或任意切换 |
| Router/fusion 图 | S–H/H–P 实测 Re 点上的 E2、T2-C、固定常数，含 split 标识 |
| Runtime/memory | 同一主机、无并发 GPU 负载，原生模态 rollout 耗时和峰值 GPU 分配量 |
| Oracle 说明 | 明确未来信息诊断、不可部署、不是每一步的严格下界 |

其中语言修改以独立插入文件交付，没有擅自选取或修改一个不确定的“最新版论文目录”。

## 2. 固定权重融合：T2-C 的收益不只是一个固定混合比例

### 2.1 协议

沿用 centered-square `E2_T2C_K24_20260730_STRICT_V3` 的冻结候选轨迹、原 E2 和 T2-C checkpoint，以及既有开发/测试划分。每个 overlap 拟合一个标量，无参数或历史输入。

训练拟合目标是训练窗口全 24 步的归一化速度/压力平方误差之和，用解析凸二次最优解并限制 alpha 在 [0,1]。额外提供仅使用验证集选择的固定权重敏感性对照：网格间隔 0.0001，按 worst-Re 优先、pooled mean 次级选择。测试集不参与权重选择。

| Overlap | alpha 对应候选 | 训练拟合 alpha | 验证敏感性 alpha |
|---|---|---:|---:|
| S–H | Steady 权重 | 0.35785055 | 0.0349 |
| H–P | Periodic 权重 | 0.92868350 | 0.89 |

H–P 的 alpha **不是 Hopf 权重**。确定性的单变量拟合不需要随机种子重复。

### 2.2 测试结果

以下为 K24 **末步**窗口平均联合误差，Ejoint=Eu+Ep，单位 %，不是全 24 步累计均值。

| 方法 | S–H | H–P |
|---|---:|---:|
| Equal blend | 3.0321 | 1.6146 |
| E2 probability blend | 2.1847 | 1.6618 |
| Constant-alpha：训练拟合 | 2.5590 | 1.1549 |
| Constant-alpha：验证敏感性 | 1.7340 | 1.1607 |
| 原 T2-C | 1.1958 | 0.8465 |
| Convex oracle：诊断 | 1.1842 | 0.8384 |

两个 overlap 中，T2-C 均优于两种固定权重对照。结果支持“本实验中的单一固定混合比例不足以解释 T2-C 收益”。不能将它提升为所有分布、所有优化目标或每个时间步都必然占优的定理。

同一 Re 下 T2-C 权重随初始历史窗口变化；这与单纯参数路由和固定常数有区别。但对历史输入的单独归因应结合已有 mu-only 消融，不能只靠这张权重图。

Oracle 利用未来参考轨迹按全 24 步平均 Eu+Ep 选择每窗口 alpha，随后报告 K24。因此它不可部署，也不是 K24 或每一个时间步指标的严格最优下界。

## 3. Dense correction：训练和 checkpoint 已核实

### 3.1 模型对照

移除所有 group/channel sparse routers、shared experts 和 routed experts，用一个三隐藏层 GELU 全连接网络联合输出速度和压力修正。保留原编码器、输入、物理 Galerkin 骨架、压力基线、闭合置信度头、优化器、数据划分、rollout 训练和验证选择规则。闭合置信度头不是稀疏路由。仅移除没有专家/路由后不适用的相关正则项。

| 项目 | Full specialist | Dense correction |
|---|---:|---:|
| 总可训练参数 | 48,617,547 | 48,609,532 |
| 单次推理激活参数估计 | 8,308,959 | 48,609,532 |
| Dense 隐藏层宽度 | — | 4,820 |
| 验证选中轮数 | 85 | 20 |

总参数差约 -0.01649%。Full 激活量按一个选中 group、每个速度/压力分支一个 shared 加两个 routed experts 统计，计入公共模块及路由参数，不包含训练时多样性分析的额外 expert stacks。它是单次推理路径参数量，不是每条轨迹的参数并集，也不是 FLOP。

### 3.2 训练实际完成情况

- Seed：1600；AdamW，lr=5.5e-4，weight decay=1.5e-4。
- 配置最大 240 epochs；原生 min_epochs=180、patience=70、early_stop_min_delta=0.001。
- **在第 180 轮正常早停**，总训练及原生收尾耗时 2293.65 秒，约 38.23 分钟。
- 配置 curriculum=4,8,12,16；本次早停前实际推进到 12 步，不能声称完整训练了 16 步阶段。
- 最终评估使用 `best_validation.pt`，其模型来自第 20 轮，验证分数约 0.2836883。后续有小于该分数的记录，但未达到 min_delta=0.001 的更新幅度，因此不能用“所有记录中最小数字对应的轮数”替换原生选择规则。
- 不按测试表现挑 checkpoint。权重全部有限，最终评估 12/12 个窗口完成全部 48 步。

Dense checkpoint SHA256：`59304c96e99ef8a3e349330aa388f4c9431db7994dcb43d9ce8663e1e703b4e1`。

原 Full checkpoint SHA256：`b2052a746fdaab60f83e65d1b91d038a9dded8255ce74f41e3f6fca4c9e717c5`。

最终选中的 Dense 尚处于短 curriculum 阶段，因此结论是**在这套固定原生训练与选择协议下**的对照，不是穷尽 Dense 架构、超参数和长时选择指标后的最强 Dense 上限。

## 4. 共同窗口 K48：保留有利与不利结果

### 4.1 统一评估

Circular Periodic held-out Re 约为 70.314635、100.352251、149.059229、189.862278；每个三个窗口，共 12 个。旧 Global/local 缓存起始时间不同，本轮按 Global 缓存的起始时刻重跑 local/Dense/POD-Galerkin，验证三元键 `(Re,start,step)` 完全相同。

不同 POD 坐标不直接相减。Global 预测通过跨基 Gram 矩阵计算共同物理误差，包含基和均值场差异；网格、面积权重已核对。参考是 local Periodic **POD 重构真值，不是完整 FOM 快照**。所有方法的压力使用相同面积加权去均值规范。

压力规范核对很重要：Global 原压力基不保证零面积均值，直接使用旧压力误差会把常数偏置计入误差。本报告不使用去均值前的 Global 比较结果。

### 4.2 K48 末步统计

| 方法 | Eu (%) | Ep (%) | Eu+Ep (%) |
|---|---:|---:|---:|
| POD-Galerkin | 9.1217 | 36.2473 | 45.3691 |
| Dense correction | 5.4590 | 20.1153 | 25.5743 |
| Global MoE | 1.6072 | 5.8646 | 7.4719 |
| RAL Periodic specialist | 1.9352 | 7.4275 | 9.3627 |

Full local 相比总参数匹配 Dense 的联合误差更低，支持本协议下 structured sparse specialist 的价值。但 Global 在这一组 K48 窗口的速度、压力和联合误差均更低。**不能写成 local 在所有流态、所有时长和所有指标上优于 Global。**

本组是 Periodic 单 specialist 测试，不是 overlap T2-C 双模型融合测试。不能用它替代第 2 节的 overlap 结论。

逐 Re 的 K48 联合误差进一步说明差异不是统一方向：

| Re | Dense (%) | Global (%) | Full local (%) |
|---|---:|---:|---:|
| 70.3146 | 39.9827 | 11.9628 | 19.6748 |
| 100.3522 | 24.9758 | 6.5862 | 2.9414 |
| 149.0592 | 16.3223 | 4.0289 | 8.0188 |
| 189.8623 | 21.0164 | 7.3095 | 6.8157 |

Full 在四个 Re 均优于这个 Dense 对照；相对 Global 则是 2/4 更优。Global 的总体均值优势不能改写为每个 Re 都更优，反之也不能只展示 local 更好的两个 Re。

提供逐步对数纵轴图和逐 Re 原生 elapsed-time 图。各 Re 的数据库时间步不同，混合平均图用 rollout step，不能当成统一物理秒。所有窗口都被保留，没有以发散或误差大小筛除窗口。

### 4.3 原生辅助输入边界

为保持原训练合同，本轮保留 phase_harmonics=4。local 索引没有 phase/period 列，原 loader 从保留时间段的起止时间归一化构造 phase；Global 索引有 estimated_period 列，原 loader 由周期信息构造 phase。Global 还使用已有 per-Re mean fields。

因此，这是**既有辅助时间/周期和均值信息下的原生协议对照**，不能直接升级为“只给初始物理历史、完全不依赖外部时间/周期元数据”的部署认证。Global/local 同时涉及这些合同差异，不能声称严格只改了 locality 一个因素。本轮没有为了结果改动这些输入或重新训练原模型。

## 5. 同机原生运行成本

主机 CPU：Intel Xeon Platinum 8358P @ 2.60 GHz；GPU：RTX 4090（24564 MiB），驱动 595.71.05。各模型独立进程、同一第一个对齐窗口、batch=1、K48，3 次 warm-up、10 次重复；GPU 同步后计时。OMP/OpenBLAS 线程数为 2。

集群运行环境：Python 3.11.15、PyTorch 2.11.0+cu126、NumPy 2.4.4。

| 方法 | K48 时间中位数 (s) | GPU peak allocated (MiB) |
|---|---:|---:|
| POD-Galerkin，原生 CPU 路径 | 0.3745 | 0.00 |
| Dense correction | 0.5761 | 194.80 |
| RAL Periodic specialist | 2.1264 | 194.78 |
| Global MoE | 4.3295 | 194.78 |

计时包含：原生模态推进、特征计算、CPU/GPU 传输、模态压力重构。排除：模型/数据加载、物理场解码和误差统计。逐次耗时及 peak reserved 也已保存。

这是同机**原生实现**成本，不是所有物理算子都迁移 GPU 后的严格设备对等比较。POD 的 GPU 分配为零不代表 CPU 内存为零；本表不是进程全部内存。单窗口重复结果也不能代表所有 Re、batch size 或吞吐量。

Dense 虽激活更多参数，却比 sparse specialist 更快；不能从激活参数少推断加速。Full 与 Global 的当前延迟差异也不能推断为普遍算法复杂度优势。全部权重仍驻留设备，因此 active parameter 数较小没有自动变成相同比例的显存降低。

本次测量不包含 E2 选择、两 specialist 并行/串行推进、物理空间解码后 T2-C 融合的完整链路，**不得据此报告整个 RAL Top-2 pipeline 的端到端 speedup**。

## 6. 修复与验证证据

训练结束后进程被 PID 1 保留为 zombie，原调度脚本仅用 `kill(pid,0)` 判断进程存在，因此无限等待。已改为优先查看完成标记，并检查 `/proc/<pid>/stat` 的 Z/X 状态。训练未重跑，原输出保留。

四项调度测试通过：运行进程、休眠进程、消失进程、名称含空格的 zombie。恢复后的队列已依次完成 dense/full/global/galerkin，`post_training_status.json` 明确为 COMPLETED。

其他检查：

- Constant-alpha 保留训练/验证/测试划分与哈希，原 T2-C 重放通过。
- Dense 去路由断言、参数匹配断言、模型有限权重检查通过。
- Full/Dense selected checkpoint 的身份与选择轮数已存档。
- Global 原生重新运行与冻结缓存全部 12×48 位置的原生 Eu/Ep 最大差异约 8.88e-16 个百分点；正式跨 chart 表另行统一压力规范。
- 同窗口键一致，最终每条轨迹均完成 K48；未做 survivor-only 平均。
- 原生训练输出自带的通用 MoE 模板文字不适用于 Dense 架构；本报告和 architecture.json 才是本轮 Dense 的准确说明。

## 7. 论文可用解释

1. **DataOnly vs Full**：讨论物理骨架与原生推进合同的整体作用；若同时改变离散/连续推进，不能视为严格单因素因果实验。
2. **Vanilla-FNN-MoE vs Full**：路由保持时，分析 structured expert 的贡献，不是证明 MoE 本身带来全部收益。
3. **Global vs local**：分析表示、模型分解及既有训练/均值/辅助输入合同的组合效果。当前 periodic 长时负结果需要保留。
4. **Dense vs Full**：总参数匹配下 Full 精度更好，但 Dense 原生延迟更低。精度与效率必须分开讨论。
5. **Constant-alpha vs T2-C**：支持已评估 overlap 中自适应融合的价值；历史的独立贡献需结合 mu-only。

推荐英文：

> The tested structured local specialist improves K48 accuracy over a total-parameter-matched dense correction under the same native training and selection protocol. However, Global MoE yields lower errors on these particular periodic windows, and the dense control has lower latency in the current implementation. These results indicate regime- and protocol-dependent accuracy–cost trade-offs rather than uniform superiority.

> Our cross-regime claims concern evaluated regime transitions within parameterized incompressible flows, not unseen physics or arbitrary regime switching. The convex oracle uses future reference trajectories and serves only as a non-deployable diagnostic.

## 8. 文件与复现

- `constant_alpha_v1/`：固定权重结果、划分协议、逐窗口权重、逐步误差及 PDF/PNG。
- `final_periodic_{dense,full,global,galerkin}_v1/`：独立评估、完整计时样本、checkpoint 哈希、完成标记。
- `global_common_truth_v1/`：统一参考与压力规范后的 Global 曲线。**跨模型精度表必须用这里，不能用 final_periodic_global_v1 的原生未对齐 pressure gauge 指标。**
- `aligned_rollout_figures/`：最终四方法汇总曲线、逐 Re 时间曲线及聚合数字。
- `FINAL_CHECKPOINT_AUDIT.json`、`GLOBAL_REPLAY_VERIFIED.json`、`post_training_status.json`：验证证据。
- `paper_ready_inserts.tex`：可直接整合的表格、图和限制性文字；需 booktabs、graphicx。
- Python 脚本与原训练配置、完整日志用于复现。训练/评估依赖既有集群数据，不把结果包误认为独立完整数据集。

Dense 权重保留在原集群新实验目录的 `dense_periodic_v1/best_validation.pt`，其绝对路径和 SHA256 见审计 JSON。结果包不包含大体积原始数据与 checkpoint；不会因用户下载报告而丢失这些集群资产。

原 `EXPERIMENT_REPORT.md` 为分阶段记录，遇到状态冲突以本最终报告及完成 JSON 为准。没有尚在等待的本批次计算任务。若后续扩展到其他算例/流态、多随机种子、完整 FOM 真值、无辅助周期信息部署或完整 Top-2 端到端基准，应作为新的明确实验矩阵，不能把本批次结果直接泛化过去。
