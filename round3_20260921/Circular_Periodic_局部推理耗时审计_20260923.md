# Circular Periodic 局部推理耗时审计（2026-09-23）

## 结论先行

在不改动原始源码、检查点或原始计时文件的条件下，复现中位数为 Proposed **2.1587 s**、Dense **0.6486 s**、Structured **0.8714 s**；相对存档值分别偏离 −1.20%、−1.41%、+0.72%，排序和量级稳定。这里比较的是 **Circular Periodic 单个局部模型的一条 48 步轨迹**，不是完整 E2/T2-C 系统的端到端耗时。

差异由模型结构与当前实现的调度开销共同造成。三模型使用相同的 RK4、CPU NumPy Galerkin、代数压力、物理场解码和显式 CPU↔GPU 数据传输。Proposed **没有执行所有专家的前向**：每次右端项只执行速度/压力各 1 个共享专家和各 2 个被选路由专家。然而它每次仍运行 1 个组路由器及全部 6 个通道路由器，并在 Python 循环内逐项检查组/专家是否激活。单条轨迹产生 1,344 次 Top-k 函数调用和 3,456 次 GPU 标量判定；Dense/Structured 无此路径。因而不能把额外耗时单独归咎于三者共有的 CPU Galerkin 或传输。当前证据**不能**把架构固有成本和可消除的实现开销精确拆成两个不重叠百分比，也不能把 active-matched 参数量解释成 FLOPs 或速度匹配。

## 1. 比较身份与公平性

存档基准为远端 `experiments/round2_20260917/code/benchmark_periodic_idle.py`（SHA256 `e687516988f8d207a96dc456886307944f726dfbda8a2462ca7c16bb31ed8c2e`）；其调用 `experiments/strengthening_20260915/code/evaluate_periodic_dense.py`（`9a57c91c8751a52926d79dcd23d72503a9edf483351e295f39779f0a4838764f`）并加载 `periodic_controls.py`（`4dee4775ef63a149a5cfc94e3aa24da32a3fcab56c8e269b06092cc430cbe152`）。实际原生专家源码 `iclr_expert_routing_analysis/source/periodic_specialist_r32/code/train_periodic_moe.py` 的 SHA256 为 `500c2179734e6c3a6db6469629d4727637ae4552843b4d3154ca7b281a102314`。该目录无可用 Git 提交号，故以文件哈希标识代码版本。

共同协议：seed 1248、同一 held-out 窗口起点 976、batch 1、历史长度 3、降阶速度/压力维数均 32、K=48、RK4 每步 4 次右端项，即每条轨迹 192 次模型前向和 CPU Galerkin、48 次压力代理算子。三者都以 FP32 的模型输入/输出在 RTX 4090 上推理；NumPy Galerkin 和物理场解码留在 CPU。均在 `torch.inference_mode()` 下，且模型内部前向还包有 `torch.no_grad()`。每次基准重复前后各执行 `torch.cuda.synchronize()`；预热 3 次、重复 10 次。`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=2`，PyTorch intra-op 线程数实测 2（interop 默认 64）。硬件/运行时：Intel Xeon Platinum 8358P、RTX 4090、NVIDIA 驱动 595.71.05、Python 3.11.15、PyTorch 2.11.0+cu126、NumPy 2.4.4。

| 模型 | 存档检查点（下列路径均相对远端 `experiments/round2_20260917/`） | 检查点 SHA256 | 初始化身份与修正 |
|---|---|---|---|
| Proposed | `P_proposed/1248/Circular_P_proposed_round2_Re_70p314635_checkpoint.pt` | `b48eaacc1deb923721f9fa0244169861a6b8a4111cab7a5255a61110679fa625` | 原始初始化；不是后续低秩二次分支修正版 |
| Dense | `P_dense/1248/Circular_P_dense_round2_Re_70p314635_checkpoint.pt` | `ff2b6456b49ce1834a97b573445f48558815c6dc5ebce11f8ae8a6554405f48b` | 原始控制：非路由联合 Dense 修正；不是修正版 |
| Structured | `P_structured/1248/Circular_P_structured_round2_Re_70p314635_checkpoint.pt` | `aea99d3eea9ceaf55241be5d7a4561e8af4d28f9dbf3a9582337b92d9ef1887f` | 原始控制：每通道一个非路由结构专家；不是修正版 |

训练配置来自各检查点旁的 `protocol.json`；三者共享 `r_u=r_p=32`、`integrator=rk4`、`history_len=3`、`pressure_input_mode=pressure_only` 等设置。原生初始化函数会将低秩二次分支的两个因子都置零；Structured 的架构记录也明示这一身份。本审计仅比较上述**同一轮原始配置**，不把后续修正版结果掺入。

速度路径是四阶段 RK4：每阶段计算 CPU Galerkin 基项、构造带历史的特征、一次联合模型前向，速度修正与投影项合成阶段右端项。压力路径在每阶段模型前向也产生压力修正及闭合参数，但 k2/k3/k4 的压力输出不用于本宏步末的压力更新；步末仅一次代数压力基线，再以 k1 的闭合输出作修正。这个“后 3 阶段仍计算压力头”的工作发生在**三种**模型中，因压力分支结构不同，其成本并不相等；属于潜在的诊断线索，而非已经验证的等价优化。

Dense 的 “active-matched” 是对**原始已核验 Circular Periodic 模型单样本激活路径参数数** 8,308,959 的匹配：Dense 总可训练参数 8,312,230（差约 0.039%），单个三隐藏层联合 GELU 修正宽度 1898；Structured 总参数 8,308,782、每通道一个宽度 2860 的结构专家。两控制删除了组/通道路由器和共享/路由专家库，保留编码器、物理投影、压力基线和置信门控。这不匹配 FLOPs、GPU kernel 次数、Python 调度次数或墙钟时间，也不意味着三者执行相同数量的模块。

三模型计时范围相同：原生模态推进、特征与历史更新、主机/设备传输、每步代数压力重构、48 步速度与压力的物理场解码及压力零均值规范修正。时间内没有误差计算、日志或文件写入；加载数据/检查点、物理历史编码、E2、T2-C 及最终误差统计在时间外。三模型都走相同的解码与压力规范修正代码；Proposed 无额外的仅它执行的误差计算。这个局部计时不能用于声称完整推理链路加速。

## 2. A：基准复现表

复现脚本仅将存档基准的输出根从 `P_isolated_runtime` 改为 `P_latency_audit_20260923`，其余执行逻辑未变。每个模型独立进程、顺序运行。原始文件仍在 `P_isolated_runtime/{proposed,dense,structured}/runtime.json`；新原始日志与清单在远端 `P_latency_audit_20260923/`，本地副本见同目录 `latency_audit_20260923/runtime_*.json`、`original_runtime_*.json`、`reproduction_manifest.json`。单位：秒；标准差为 10 次重复的**样本**标准差。

| 模型 | 10 次新重复耗时 | 新中位数 | 新均值 | 样本标准差 | 存档中位数 | 中位数偏差 |
|---|---|---:|---:|---:|---:|---:|
| Proposed | 2.1922, 2.1595, 2.1594, 2.1574, 2.1589, 2.1538, 2.1678, 2.1585, 2.1514, 2.1534 | 2.1587 | 2.1612 | 0.0118 | 2.1848 | −1.20% |
| Dense | 0.6501, 0.6481, 0.6579, 0.6471, 0.6473, 0.6492, 0.6489, 0.6484, 0.6536, 0.6427 | 0.6486 | 0.6493 | 0.0040 | 0.6579 | −1.41% |
| Structured | 0.8735, 0.8815, 0.8713, 0.8701, 0.8695, 0.8773, 0.8759, 0.8715, 0.8700, 0.8708 | 0.8714 | 0.8731 | 0.0039 | 0.8652 | +0.72% |

复现中 Proposed/Dense = **3.33 倍**、Proposed/Structured = **2.48 倍**。这是单窗口本机局部耗时比，未附置信区间、跨负载/设备泛化或 FOM 加速解释。

## 3. B：模块耗时与调用次数

为避免改变正式计时，另以独立诊断目录 `P_latency_decode_20260923` 对相同检查点、窗口、48 步轨迹执行一次 PyTorch profiler 和前向钩子。三模型被钩子标记前后的模态输出最大绝对差均为 **0**。下表耗时为该**带 profiler 的一次运行**之 CPU 区间包含时间（ms），不是原始基准耗时；行间存在父子嵌套，不可相加。完整事件与调用清单见本地 `latency_audit_20260923/decode_*.json`，脚本为 `latency_profile_20260923.py`。Profiler 自身使 Proposed 的总墙钟升至约 3.60 s、Dense 0.81 s、Structured 1.17 s，故绝不以这些数替代上节正式基准。

| 模块 / 诊断区间 | Proposed 调用 / ms | Dense 调用 / ms | Structured 调用 / ms | 解释 |
|---|---:|---:|---:|---|
| 全部原生模态推进 | 1 / 3524 | 1 / 733 | 1 / 1089 | 父区间，包含下列大部分项目 |
| 积分宏步 | 48 | 48 | 48 | 每步 RK4 4 个阶段 |
| 右端项/模型输出 | 192 / 3413 | 192 / 631 | 192 / 986 | 含 CPU 物理算子与模型前向 |
| Galerkin NumPy | 192 / 334 | 192 / 293 | 192 / 304 | 三者共有；实测差别远小于总体差别 |
| 特征构造 | 192 / 24 | 192 / 22 | 192 / 22 | 历史拼接在 `model_outputs` 的未单独标注部分 |
| 编码器 | 192 / 71 | 192 / 70 | 192 / 70 | 每个右端项一次，无速度/压力双重编码 |
| 组路由 | 192 / 34 | 0 | 0 | Proposed 专有 |
| 速度+压力通道路由 | 1152 / 214 | 0 | 0 | Proposed 每次右端项运行 3+3 个组内路由器 |
| Top-k 函数 | 1344 / 278 | 0 | 0 | 192×(1 组+6 通道) |
| 共享专家 | 384 / 384 | 0 | 0 | 每次右端项速度/压力各 1 个；两通道总数 |
| 被选 Top-2 专家 | 768 / 782 | 0 | 0 | 每次右端项速度/压力各 2 个；**未选专家未前向** |
| Dense 联合修正 | 0 | 192 / 49 | 0 | 一个联合输出网络 |
| Structured 非路由专家 | 0 | 0 | 384 / 392 | 每次右端项两个通道各一次 |
| 代数压力代理 | 48 / 102 | 48 / 94 | 48 / 95 | 每宏步一次；压力修正另在模型前向中 |
| 全步物理场解码与压力 gauge | 1 批（48 帧）/ 73 | 1 批（48 帧）/ 72 | 1 批（48 帧）/ 77 | 单一向量化批区间；不是 48 次函数调用 |

细项：Proposed 的 `aten::item` / `_local_scalar_dense` 为 **3,456** 次，`aten::nonzero` 为 **6,912** 次；代码中的 `bool(torch.any(active))` 对每个右端项执行 18 次（6 组检查 + 12 专家检查），构成上述标量同步的直接来源。Top-k 算子调用 1,344 次。被选专家前向为 768 次、共享专家前向 384 次；因此“先算全部专家再选 Top-2”与轨迹不符，但“所有组内路由器都运行”成立。训练用全专家 stack 在该推理调用中被 `return_expert_stack=False` 禁用，未见推理阶段执行训练辅助损失。模块权重由检查点一次加载到设备，时间循环中未见重复重建或搬运；固定 POD 基及统计在时间循环外准备，但每阶段仍新建 FP32 输入张量。

显式主机↔设备传输（通过诊断钩子计数，三模型完全相同）：每条轨迹 **H→D 192 次 / 384,768 字节**（每右端项一个 501 维 FP32 输入）；**D→H 768 次 / 98,304 字节**（每右端项 4 个张量）。两方向合计 960 次 / 483,072 字节。这只统计源码所用 `torch.tensor(np.ndarray, device='cuda')` 和 `Tensor.cpu()`，不声称覆盖 CUDA 内部拷贝/所有隐式同步；Profiler 的 `aten::_to_copy` 亦为三者各 960 次。反而 Proposed 特有的 GPU 标量读取次数显著更多。CPU 和 CUDA 事件可能重叠，不能把表中耗时相加；未标注的张量创建、索引、拼接、Python 调度、历史更新及 profiler 开销保留在父区间内，无法从本次轨迹无偏地逐项分摊。

## 4. C：原因证据与确认程度

| 候选原因 | 代码位置 | 运行证据 / 受控核对 | 判定 |
|---|---|---|---|
| 先执行全部专家再选 Top-2 | `train_periodic_moe.py` `_group_mix` / `_sparse_mix`（约 1800–1940 行） | 192 次右端项仅有 768 次路由专家前向（=2 通道×2 专家×192），不是所有组×所有专家；未选组/专家前向数为 0 | **否定** |
| 所有组通道路由仍执行 | 同上 `_group_mix` | 速度和压力各 576 次通道路由（=3 组×192）；只有选中组执行专家 | **确认** |
| Python 稀疏循环和 GPU 标量同步 | 同上 `bool(torch.any(active))` | Proposed 3,456 次 `aten::item`、6,912 次 `aten::nonzero`；控制组无相应路由路径。Profiler 中 Top-k 约 278 ms、通道路由约 214 ms、`nonzero` 约 394 ms，但区间与算子存在嵌套，不能相加为“可优化收益” | **确认存在且与额外耗时相关**；可消除比例未测 |
| 更多实际专家计算与小算子调度 | 模型前向、`PhysicsAwareExpert.forward` | Proposed 每条轨迹 1,152 次专家前向（384 共享+768 路由），Structured 384 次，Dense 192 次联合修正；单独 profiled 专家区间约 1166/392/49 ms（不同模型结构，非同 FLOPs） | **确认结构工作量不同**；精确 FLOPs 未测 |
| CPU/GPU 混合路径 | `model_outputs_from_states_np`、`integrate_autonomous_step_np`（约 4001–4160 行） | 三者 CPU Galerkin 均 192 次、压力代理 48 次；显式传输次数及字节数相同。Proposed 另有路由标量同步 | **共有路径确认，但不足以单独解释差值** |
| 时间范围、积分/重构工作量不同 | `benchmark_periodic_idle.py`、`evaluate_periodic_dense.py` | 同一窗口、RK4、K=48、FP32、推理模式、全部物理场解码与压力 gauge；复现差异 <1.5% | **未见不公平范围** |
| 仅 Proposed 的误差/文件写入/训练损失 | 计时块与模型调用 | 误差及文件写入在计时外；`return_expert_stack=False`；三模型均未见额外训练辅助损失 | **未发现** |

本轮没有固定/回放路由输出、关闭模块或改变积分器去制造“优化版”正式速度。现有代码轨迹和单次 profiler 足以确认上述执行事实；若后续要量化**路由决策本身**对正式墙钟的独立贡献，应另建诊断脚本，逐次缓存并回放同一轨迹的路由权重、验证模态输出逐步完全一致、单独重复计时，且仍不得把这种回放速度列为真实推理速度。

## 5. 论文表述建议与边界

可写为：“在公平的 Circular Periodic 单专家局部 K=48 推理协议下，Proposed 的中位墙钟耗时高于 active-parameter-matched Dense 和非路由 Structured 对照。逐模块审计表明，Top-2 选择确实跳过未选专家前向；差异主要与多共享/路由专家调用、所有组通道路由计算以及逐专家稀疏判定的同步/小算子调度相关。三模型共有的 CPU Galerkin 与物理场解码不是单独解释。”

不应写成“Proposed 推理更快”“active-matched 等于算力公平”“CPU Galerkin 是唯一瓶颈”，也不应把带 profiler 的区间时间加总成正式延迟。若要报告完整方法相对 CFD/FOM 的加速比，必须另做相同物理时长、几何、I/O 边界的配对端到端计时；本审计不提供该结论。

## 6. 可追溯材料

- 只改输出目录的复现脚本：`round3_20260921/latency_reproduce_20260923.py`；新基准 JSON：`round3_20260921/latency_audit_20260923/runtime_*.json`。
- 单窗口 profiler/调用与显式传输钩子：`round3_20260921/latency_profile_20260923.py`；原始事件 JSON：`round3_20260921/latency_audit_20260923/decode_*.json`。
- 原训练配置、检查点身份与控制架构：同一证据目录下的 `train_protocol_*.json`、`protocol_*.json`、`architecture_dense.json`、`architecture_structured.json`。
- 原始存档与复现日志均保留；仅新建 `P_latency_audit_20260923`、`P_latency_profile_20260923`、`P_latency_transfer_20260923`、`P_latency_decode_20260923` 等互不覆盖的远端诊断目录。
