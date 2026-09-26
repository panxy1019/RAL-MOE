# Unified Local-Chart MoE：资产审计与实现映射

审计日期：2026-07-22  
审计目标：在不训练、不修改 specialist 的前提下，核验 Steady/Hopf/Periodic 三个局部坐标图是否满足统一动态路由框架的启动门槛。  
结论：**BLOCKED；未实现、未构造适配器、未启动 A/B/C 训练。**

## 1. 硬停止结论

材料和用户指令明确要求三个 specialist 的时间步长完全一致，并规定发现时间步长不一致时必须停止、不得猜测继续。实际资产不满足该条件：

| Chart | 审计数据 | 相邻样本实际 `dt` 范围 | 典型的逐 Re 中位 `dt` 范围 |
|---|---|---:|---:|
| Steady | train+validation modal bundle | 0.619895–9.523810 | 4.101548–9.523810 |
| Hopf | expanded H4 train+validation | 0.433494–23.633552 | 4.078719–23.608121 |
| Periodic | native `pod_snapshot_index.csv` | 0.271268–17.635631 | 3.501459–17.635631 |

这不是无关的数据库元数据。现有实现直接使用轨迹时间差：

- Steady/Periodic trainer 从 `time[next] - time[current]` 或 `dt_next` 取得步长，并在 RK4/rollout 中使用；
- Hopf expanded trainer 同样从相邻 `time` 计算 `dt` 并用于自主更新；
- 因而不能在上层 Router 中假定统一的一步具有相同物理时间语义。

在没有重新生成/重采样并重新认证三个 frozen specialist 的共同时间网格前，直接训练会把不同物理时间跨度的增量当成可比较、可融合的 `Delta F_r`，违反方法定义。

第二个尚未解析的接口风险是 Periodic（以及 Steady 公共 V16 接口）的 phase/time 特征。Periodic checkpoint 使用 4 组 phase harmonics，并只在“预测状态递推 + 已知数据库时钟/phase”合同下完成认证；Hopf H4 明确 phase-free。H→P 切换时如何仅由自主状态初始化 Periodic phase，以及如何把三步 history 重编码到目标 chart，当前资产没有经验证的合同。不能用真实未来 phase，也不能自行发明未验证的 phase estimator。

## 2. 冻结 checkpoint 映射

| Chart | 冻结 checkpoint | SHA-256 | 审计状态 |
|---|---|---|---|
| Steady | `/root/panxy/particalMOE/steady_specialist_v1/checkpoint/frozen_s4_validation_step_1200.pt` | `bcab5661af39e3262103ed887ce9846415467452b453f3dcc14e133f0f8bd80d` | 文件反序列化通过；1414 tensors，48,613,781 elements |
| Hopf | `/root/panxy/particalMOE/Hopf/migrated_h4_expanded/final_evaluation/20260722_h4_expanded_final/HopfExpanded34_H4_NormalFormRadial_r32/final.pt` | `02148741ed8bc9fbec88709f69b10492e263514b2edcee76652485b399962235` | 文件反序列化通过；592 tensors，32,004,147 elements；best step 7200 |
| Periodic | `/root/panxy/particalMOE/periodic_specialist_r32/checkpoint/FINAL_PERIODIC_SPECIALIST.pt` | `b2052a746fdaab60f83e65d1b91d038a9dded8255ce74f41e3f6fca4c9e717c5` | 文件反序列化通过；1414 tensors，48,617,547 elements；best epoch 85 |

Steady 的交接材料同时保留 S3-B 跨硬件保守 fallback，但本轮要求“已有最佳 validation checkpoint”，所以实现映射指向 S4 validation step 1200；没有把 fallback 静默替换为主 checkpoint。

由于时间步长硬门禁已失败，本轮没有继续执行模型实例化后的 strict `load_state_dict` 与单 batch forward；已有交接评估/复现产物表明这些 checkpoint 曾在新 4090 上成功加载和评估，但正式统一框架仍需在门禁解除后重跑 strict restore preflight。

## 3. 网络与 specialist API 映射

| 项目 | Steady | Hopf | Periodic |
|---|---:|---:|---:|
| `r_u / r_p` | 32 / 32 | 32 / 32 | 32 / 32 |
| history length | 3 | 3 | 3 |
| checkpoint input width | 501 | 493 | 501 |
| shared width | 224 | 256 | 224 |
| 模型主类 | `OperatorSpaceMoEROM` | expanded `build_model(...)` 返回的 MoE-ROM | `OperatorSpaceMoEROM` |
| phase features | V16 phase contract | 无 | 4 harmonics |
| 状态更新 | Galerkin + MoE residual + RK4 + pressure closure | Galerkin + MoE residual + RK4 + pressure closure | Galerkin + MoE residual + RK4 + pressure closure |

统一实现不能把三者包装成只接收同一个 64 维 normalized tensor 的同构网络。需要三个只读 specialist wrapper，各自保留：本地 scaler、Galerkin/pressure tensor、feature builder、history builder、Re/time 语义和 pressure closure。上层只能在反归一化 physical modal coefficient 上执行 chart adapter。

## 4. POD、均值、gauge、网格与 scaler

三套原生 chart 均为 `r_u=r_p=32`，速度自由度为 `[u(all points), v(all points)]`，压力为 point scalar；网格点数 97,368，速度自由度 194,736。

| 检查 | Steady | Hopf | Periodic | 结论 |
|---|---:|---:|---:|---|
| pressure gauge | `subtract_area_mean_per_snapshot` | 同左 | 同左 | PASS |
| POD 内积 | area-weighted | area-weighted | area-weighted | 必须使用原始 `point_areas`，不能取 `M=I` |
| velocity weighted Gram max error | 4.64e-8 | 5.39e-8 | 5.44e-8 | PASS |
| pressure weighted Gram max error | 2.77e-8 | 5.44e-8 | 5.41e-8 | PASS |
| scaler finite/std positive | PASS | PASS | PASS | PASS |
| scaler roundtrip max abs | 4.44e-16 | 2.22e-16 | 2.22e-16 | PASS |

三套 `points` 与 `point_areas` 逐元素完全相同，确认了网格自由度对应关系和有限体积/面积权重合同。相邻跨基矩阵的初步谱审计（尚未构造正式 adapter artifact）为：

| 方向对 | 速度 `sigma_min / cond2` | 压力 `sigma_min / cond2` |
|---|---:|---:|
| S–H | 1.1907e-5 / 8.3981e4 | 5.7133e-5 / 1.7503e4 |
| H–P | 3.7537e-5 / 2.6640e4 | 3.6972e-4 / 2.7045e3 |

相邻 32 维子空间的变换显著病态，后续必须在真实 overlap snapshots 上报告 projection floor ratio、field/fluctuation/energy/roundtrip error 后才能放行；不能让 residual adapter 掩盖这些谱风险。

注意：`unified_rollout_eval_heldout_v1/periodic/pod_runtime_r32.npz` 不是 Periodic checkpoint 的原生 POD。Periodic 原生 chart 必须使用：

- `/root/panxy/particalMOE/periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz`
- `/root/panxy/particalMOE/periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/global_pressure_pod_area_weighted_l2.npz`
- `/root/panxy/particalMOE/periodic_specialist_r32/assets/normalization_periodic.npz`

统一 heldout bundle 只可作为物理真值来源，不能把其中的 Periodic modal coefficient 直接输入 frozen checkpoint。

## 5. 数据与 split 审计

| Chart | Train | Validation | Held-out | 轨迹合同 |
|---|---:|---:|---:|---|
| Steady | 14 Re / 894 snapshots | 2 Re / 127 | 4 Re / 256 | 按完整 Re 隔离 |
| Hopf H4 | 29 Re / 4092 snapshots | 2 Re / 225 | 3 Re / 483 | 按完整 Re 隔离 |
| Periodic | 53 Re / 8380 valid samples | 6 Re / 949 | 4 Re / 632 valid samples | 按完整 Re 隔离 |

交接合同、NPZ split 字段和 checkpoint split statistics 均声明 train/validation/held-out Re 零交集。A/B/C 若继续，必须共同复用这一合并后的 trajectory-level split，不能再随机拆窗口。

当前没有发现一条经认证、标签随时间真实执行 S→H→P 或相邻 chart 切换的连续轨迹。旧集群只找到各 specialist 的独立数据与训练 `startup_manifest.json`，未找到可证明为真实跨流态演化的统一 transition/startup dataset。现有每条轨迹固定 Re、固定 chart 标签，不允许跨 Re 拼接。

因此，即使时间门禁后来被解决，第一轮 Router 也必须在报告中称为 **temporal-consistent regime router**；除非新增并隔离真实启动/跨流态轨迹，否则不能声称已经验证真实跨流态时间迁移。

## 6. 已有评估入口

- Steady：`steady_specialist_v1/code/evaluate_s3_heldout.py`、`finalize_s4_one_time.py`、paired-gain audit。
- Hopf：`Hopf/migrated_h4_expanded/code/evaluate_h4_expanded.py`，已有 K1/K2/K4/K8/K16/K24/K48 指标。
- Periodic：`periodic_specialist_r32/code/evaluate_periodic_r32_portable.py`，已有 K1/K4/K8/K16/K24/K48 指标。
- 统一 heldout 物理真值：`/root/panxy/particalMOE/unified_rollout_eval_heldout_v1`。

统一框架最终 evaluator 需要复用三者的原生 decode/weight/gauge 逻辑，并新增 oracle/autonomous routing、switch/jitter/dwell/illegal-transition、overlap consistency 和 oracle gap 汇总；不能直接平均不同 chart 的 normalized coefficient L2。

## 7. 解除阻断后的实现映射

只有先提供并验证共同时间语义后才执行以下工作：

1. `ChartAssetRegistry`：绑定每个 chart 的原生 mean/basis/scaler/weights/physics tensors/checkpoint SHA。
2. `FrozenSpecialistWrapper`：分别适配 S/H/P 现有 feature/history/closure API；参数全部 `requires_grad=False`，但保留对输入状态的 autograd graph。
3. `AnalyticPairAdapter`：仅构造 S→H、H→S、H→P、P→H；physical coefficients 中执行 affine state mapping，increment mapping 不含 mean offset。
4. `DynamicRouter`：三个独立 local encoder + GRU + topology mask + probability smoothing/hysteresis/confirmation/min-dwell。
5. A/B/C 三个完全独立 candidate state：共享只读 specialists/adapters，但各自维护 Router、optimizer、scheduler、RNG、hidden/history、best/last checkpoint。
6. B/C 的 Top-2 只融合映射到当前锚点的 dynamics increments；C 的 residual adapter 为零初始化小幅修正并受 cycle/physical/size regularization。
7. 单进程 candidate-level round-robin；K>=4 后不共享 candidate rollout trajectory。

## 8. 恢复所需的明确决策/资产

要继续而不改变方法语义，至少需要满足其一并形成书面合同：

1. 提供三个流态在同一个物理/无量纲固定 `dt` 上生成的完整轨迹、POD coefficients 和相同 trajectory-level split，并证明 frozen specialists 对该 `dt` 的 step API 已重新认证；或
2. 明确授权建立一个共同无量纲时间坐标与严格重采样方案，并对三个 frozen specialists 在重采样网格上重新执行 K1/K4/K8/K16/K56 认证。若结果不等价，则不能继续把现有 checkpoint 当作已认证的固定一步算子。

还需定义 H→P/P→H 切换后的 phase/time 和 history 重编码规则。它必须只使用当前/历史自主预测状态和确定性时钟，不得读取未来真值。

## 9. 运行环境风险

新集群 `/` 仅余约 1.1 GB（97% used）。`/root/panxy/particalMOE/.runtime` 约 7.8 GB，且 Hopf runs/final evaluation 中有多份 checkpoint 副本。正式训练前应在明确确认可删除/可归档对象后释放空间；本次只读审计没有删除任何文件。

## 10. 本轮实际执行边界

- 已完成：完整方法材料阅读、三 checkpoint 定位/SHA/反序列化、POD/scaler/gauge/grid/split/dt 只读审计、旧集群 transition 资产搜索、实现映射。
- 未执行：解析 adapter artifact、projection-floor/overlap 误差正式报告、统一框架代码、编译与 forward/backward、吞吐测试、SwanLab run、A/B/C smoke/formal training、任何 specialist head 微调。
- 停止原因：时间步长硬门禁失败，符合用户指令中的 mandatory stop 条件。
