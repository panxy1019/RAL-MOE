# Final Periodic specialist：可复现交接与多坐标图接口说明

## 1. 最终结论

最终 Periodic specialist 固定为 `periodic_v16_public_r32` 的 epoch-85 模型，使用独立 Periodic local POD，`r_u=32`、`r_p=32`。该模型不再继续 K16 训练，也不采用 r80 备选路线。

该 specialist 已在新集群 `root@10.10.164.243:20073` 完成两项迁移验收：

1. 使用原科学配置、seed=1600 和随机初始化从零训练 1 epoch；
2. 使用冻结的 epoch-85 最佳 checkpoint 完成 K1/K4/K8/K16/K24/K48 heldout rollout。

最终 heldout 判定为 3/4 preserved。所有 Re、所有 horizon 的 finite fraction 均为 1，发散窗口为 0。Re=70.314635 仅因 K48 pressure area-weighted L2 为 5.960086%，超过 5% 判据而未 preserved；它没有数值发散或周期吸引子丢失。

## 2. 唯一有效模型身份

| 字段 | 值 |
|---|---|
| Experiment | `periodic_v16_public_r32` |
| Final checkpoint epoch | 85 |
| Best validation score | 0.2852946321 |
| Velocity/pressure rank | `r_u=32`, `r_p=32` |
| Checkpoint | `/root/panxy/particalMOE/periodic_specialist_r32/checkpoint/FINAL_PERIODIC_SPECIALIST.pt` |
| Checkpoint SHA256 | `b2052a746fdaab60f83e65d1b91d038a9dded8255ce74f41e3f6fca4c9e717c5` |
| Pressure gauge | `subtract_area_mean_per_snapshot` |
| Original SwanLab run | `https://swanlab.cn/@panxy1019/V17indepentMOEV2/runs/k6syaunj` |
| Status | `FINALIZED` |

checkpoint 内含 `model_state`、最佳模型状态、optimizer/scheduler 状态、scalers、RNG、训练历史、split statistics 和完整 args。模型状态包含 48,617,547 个标量，FP32 模型状态约 194.5 MB；完整 checkpoint 因同时保存训练恢复状态约为 780 MB。

## 3. 坐标图与物理合同

该模型只定义在 Periodic local POD 坐标图中：

\[
u(x,t) \approx \bar u_P(x)+\Phi^u_P a_P(t),\qquad
p(x,t) \approx \bar p_P(x)+\Phi^p_P b_P(t).
\]

- `a_P∈R^32`：Periodic velocity modal coefficients；
- `b_P∈R^32`：Periodic pressure modal coefficients；
- velocity 和 pressure 的 POD/均值/scaler 均仅由 Periodic train split 构建；
- pressure 在投影前逐快照减去 area-weighted mean；
- 动力学基线为 Periodic Galerkin ROM，并以 RK4 积分；
- pressure 基线来自静态 Pressure-Poisson surrogate；
- MoE 学习 velocity physics residual 与 pressure closure；
- autonomous rollout 中只保留已知 Re 与数据时钟 phase/time，`a/b/history` 全部由预测状态递推。

`unified_rollout_eval_heldout_v1/periodic` 中的另一套 POD 基与本模型原生 local POD 基不相同，其 `a_raw/b_raw` 不能直接喂给本 checkpoint。正式复现必须使用本目录 `assets/Global_POD_AreaWeighted_L2` 中的原生 POD 和 `pod_snapshot_index.csv`。

## 4. HPRS-MoE 架构参数

| 模块 | 配置 |
|---|---|
| Model class | `OperatorSpaceMoEROM` |
| Effective input dimension | 501 |
| Output dimensions | velocity 32，pressure 32 |
| History length | 3 |
| Physical context encoder | hidden dimension 224 |
| Encoder/refinement blocks | 3 total，2 residual refinement blocks |
| Regime groups | 3 |
| Experts per group | 6 routed + 1 shared，velocity/pressure 各自独立 |
| Sparse routing | group top-1，group 内 expert top-2 |
| Expert hidden dimension | 768 |
| Expert residual blocks | 3 |
| Quadratic correction | rank 4，scale 0.05 |
| Shared/routed scale | 1.0 / 0.75，归一化混合 |
| Router temperature | local 0.8，group 0.9 |
| Phase encoding | 4 Fourier harmonics |
| Dropout | 0.04 |
| Pressure closure | `adaptive_gate`，逐 pressure mode sigmoid gate |
| Pressure base | `static` Pressure-Poisson surrogate |
| Pressure input/target | `pressure_only` / `closure` |
| Velocity RHS target | Galerkin residual |
| Integrator | RK4 |
| Attractor-conditioned adapter | disabled |

模型输入由当前 `a_t/b_t`、Re、周期 phase Fourier 特征、Galerkin RHS、物理描述量和 history-3 状态构成。velocity 与 pressure 拥有独立的 group router、local router、shared experts 和 routed experts；公共 encoder 与 group router 提供共享物理上下文。

## 5. 训练配置与损失

| 参数 | 值 |
|---|---:|
| Seed | 1600 |
| Batch size | 256 |
| Learning rate | 5.5e-4 |
| Weight decay | 1.5e-4 |
| Planned epochs | 240 |
| Validation interval | 5 epochs |
| Selected checkpoint | epoch 85 |
| Rollout curriculum | K4 → K8 → K12 → K16 |
| Final train rollout | K16 |
| Evaluation rollout | 原训练 K24；最终报告扩展至 K48 |
| Scheduled sampling | 1.0 throughout |
| TF32 | enabled |

公共损失权重：

| Loss | Weight |
|---|---:|
| Dynamics residual | 0.90 |
| Coefficient prediction | 0.75 |
| Pressure closure | 0.95 |
| Pressure relative loss | 0.70 |
| Velocity rollout | 0.45 |
| Pressure rollout | 0.45 |
| Trajectory consistency | 0.18 |
| One-step consistency | 0.15 |
| Physical reconstruction | 0.08 |
| RHS relative loss | 0.06 |
| Energy consistency | 0.05 |
| Coefficient relative loss | 0.04 |
| Router balance/smoothness | 0.06 / 0.04 |
| Group balance | 0.04 |
| Expert diversity | 0.006 |

所有 Steady/Hopf 专属 loss、三流态监督、attractor CE 和 Periodic radius/energy 专属补丁均为 0。该结果来自公共 HPRS-MoE、物理 ROM、pressure closure 和 autonomous rollout 约束，而不是额外的 Periodic 标签监督。

## 6. 数据合同

| Split | Re 数量 | 有效样本数 |
|---|---:|---:|
| Train | 53 | 8,380 |
| Validation | 6 | 949 |
| Heldout | 4 | 632 |
| Total snapshots | 63 Re | 10,150 |
| History-3 valid samples | — | 9,961 |

Validation Re：

`66.970112, 91.792204, 121.050171, 139.642302, 169.244893, 196.160723`

Heldout Re：

`70.314635, 100.352251, 149.059229, 189.862278`

Train、validation、heldout 的 Re 和样本索引交集均为 0。53 个训练 Re 的完整标签和样本映射保存在 checkpoint 的 `split_stats`、训练 metrics JSON 及 `pod_snapshot_index.csv` 中。

## 7. 最终训练效果

原始 epoch-85 记录：

| Metric | Value |
|---|---:|
| Train loss | 0.709074 |
| Validation score | 0.285295 |
| Validation RHS relative L2 | 19.190% |
| Validation coefficient relative L2 | 9.187% |
| Validation pressure relative L2 | 0.434% |
| Curriculum horizon | K8 |

新集群从零一轮训练 smoke：

| Metric | Value |
|---|---:|
| Runtime | 137.48 s |
| Train loss | 3.981163 |
| Validation score | 0.395752 |
| Validation RHS relative L2 | 24.177% |
| Validation coefficient relative L2 | 12.958% |
| Validation pressure relative L2 | 6.970% |
| OOM/NaN | none |

原集群 epoch-1 的 train loss/validation score 为 3.963328/0.391122。新旧结果量级一致，但因 GPU、PyTorch/CUDA 和 TF32 内核不同，不承诺位级相等。

## 8. Heldout 多 horizon 物理场误差

以下均为 area-weighted relative L2，已经乘以 100，以百分数表示。

| Re | Horizon | Velocity | Pressure |
|---:|---:|---:|---:|
| 70.314635 | K1 | 0.241513% | 1.762504% |
| 70.314635 | K4 | 0.341128% | 2.246375% |
| 70.314635 | K8 | 0.401306% | 2.430177% |
| 70.314635 | K16 | 0.533032% | 3.049510% |
| 70.314635 | K24 | 0.671254% | 3.685548% |
| 70.314635 | K48 | 1.347254% | 5.960086% |
| 100.352251 | K1 | 0.153698% | 0.830304% |
| 100.352251 | K4 | 0.172310% | 0.745688% |
| 100.352251 | K8 | 0.187083% | 0.814373% |
| 100.352251 | K16 | 0.223491% | 0.958995% |
| 100.352251 | K24 | 0.263104% | 1.116317% |
| 100.352251 | K48 | 0.397196% | 1.708284% |
| 149.059229 | K1 | 0.121058% | 0.581592% |
| 149.059229 | K4 | 0.143026% | 0.704370% |
| 149.059229 | K8 | 0.160800% | 0.811019% |
| 149.059229 | K16 | 0.185642% | 0.951666% |
| 149.059229 | K24 | 0.215089% | 1.097728% |
| 149.059229 | K48 | 0.380851% | 1.682540% |
| 189.862278 | K1 | 0.227903% | 0.976393% |
| 189.862278 | K4 | 0.299345% | 1.487351% |
| 189.862278 | K8 | 0.316692% | 1.424775% |
| 189.862278 | K16 | 0.378996% | 1.602653% |
| 189.862278 | K24 | 0.475313% | 2.000258% |
| 189.862278 | K48 | 1.127427% | 4.427241% |

## 9. K48 模态、周期吸引子与稳定性诊断

| Re | Vel coeff L2 | P coeff L2 | RHS error | P closure error | RMS amp error | P2P amp error | St error | Phase RMS | Terminal drift | Orbit distance |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70.314635 | 1.928% | 2.398% | 18.552% | 0.470% | 1.396% | 2.297% | 0.553% | 0.1079 rad | 0.0237 cycle | 0.0782 |
| 100.352251 | 1.869% | 2.335% | 22.144% | 0.514% | 0.433% | 0.158% | 0.137% | 0.0223 rad | -0.0088 cycle | 0.0241 |
| 149.059229 | 1.684% | 3.858% | 21.011% | 0.505% | 0.581% | 0.530% | 0.132% | 0.0229 rad | 0.0029 cycle | 0.0276 |
| 189.862278 | 2.915% | 6.769% | 18.543% | 0.640% | 0.879% | 1.504% | 0.382% | 0.0702 rad | -0.0241 cycle | 0.0443 |

K48 velocity/pressure energy drift 分别为：

- Re=70.314635：0.1652% / 1.1547%；
- Re=100.352251：0.0712% / 0.7429%；
- Re=149.059229：0.0030% / 0.8959%；
- Re=189.862278：0.0311% / 0.8300%。

所有 K1–K48 窗口均 finite，`divergent_windows=0`，`first_divergence_step=null`。

## 10. 新集群文件路径

根目录：

`/root/panxy/particalMOE/periodic_specialist_r32`

| 用途 | 路径 |
|---|---|
| Final checkpoint | `checkpoint/FINAL_PERIODIC_SPECIALIST.pt` |
| Final metadata | `metadata/FINAL_PERIODIC_SPECIALIST.json` |
| 原始训练代码 | `code/train_periodic_moe.py` |
| 原始评估代码 | `code/evaluate_periodic_r32.py` |
| 可迁移评估入口 | `code/evaluate_periodic_r32_portable.py` |
| 从零一轮训练入口 | `code/run_periodic_migration_smoke.py` |
| Velocity POD | `assets/Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz` |
| Pressure POD | `assets/Global_POD_AreaWeighted_L2/global_pressure_pod_area_weighted_l2.npz` |
| Snapshot/time/phase/split index | `assets/Global_POD_AreaWeighted_L2/pod_snapshot_index.csv` |
| Mesh area weights | `assets/Global_POD_AreaWeighted_L2/mesh_l2_point_area_weights.npz` |
| Normalization | `assets/normalization_periodic.npz` |
| Galerkin velocity ROM | `assets/velocity_rom_periodic.npz` |
| Pressure-Poisson surrogate | `assets/pressure_poisson_surrogate_periodic.npz` |
| 原训练完整 metrics | `evaluation/periodic_v16_public_r32_metrics.json` |
| 原最终评估 JSON | `evaluation/periodic_r32_multihorizon_evaluation.json` |
| 新集群复现评估 JSON | `reproduction/best_epoch85_heldout_eval_v2/periodic_r32_multihorizon_evaluation.json` |
| 一轮训练产物 | `reproduction/one_epoch_from_scratch_seed1600/` |
| 运行关键哈希 | `MIGRATION_RUNTIME.sha256` |
| 迁移复现报告 | `PERIODIC_R32_MIGRATION_REPRODUCTION.md` |

运行环境：

- Python：`/root/miniconda3/envs/pt_env/bin/python`
- PyTorch：`2.13.0+cu126`
- GPU：RTX 4090 24 GB

## 11. 可复现命令

先验证运行关键文件：

```bash
cd /root/panxy/particalMOE/periodic_specialist_r32
sha256sum -c MIGRATION_RUNTIME.sha256
```

从相同科学配置和随机种子重新训练 1 epoch：

```bash
/root/miniconda3/envs/pt_env/bin/python code/run_periodic_migration_smoke.py \
  --trainer code/train_periodic_moe.py \
  --source-checkpoint checkpoint/FINAL_PERIODIC_SPECIALIST.pt \
  --data-root assets/Global_POD_AreaWeighted_L2 \
  --tensor-path assets/velocity_rom_periodic.npz \
  --pressure-surrogate-path assets/pressure_poisson_surrogate_periodic.npz \
  --output-dir reproduction/one_epoch_recheck
```

使用冻结最佳 checkpoint 执行最终 heldout rollout：

```bash
/root/miniconda3/envs/pt_env/bin/python code/evaluate_periodic_r32_portable.py \
  --trainer code/train_periodic_moe.py \
  --checkpoint checkpoint/FINAL_PERIODIC_SPECIALIST.pt \
  --existing-metrics evaluation/periodic_v16_public_r32_metrics.json \
  --output-dir reproduction/best_epoch85_recheck \
  --data-root assets/Global_POD_AreaWeighted_L2 \
  --tensor-path assets/velocity_rom_periodic.npz \
  --pressure-surrogate-path assets/pressure_poisson_surrogate_periodic.npz
```

## 12. 多坐标图统一学习框架的接口约束

后续将 Steady、Hopf、Periodic specialist 组合成多坐标图动力系统时，应遵守以下边界：

1. 每个 specialist 保留自己的 POD basis、mean、scaler、pressure gauge 和物理算子，不能把不同 chart 的 modal coefficient 当成同一个向量空间直接拼接或比较。
2. 全局路由器可以使用 Re、物理统计量、稳定性诊断或重建场特征选择 chart，但 specialist 内部 rollout 必须始终留在自己的原生坐标图中。
3. chart 间切换应走“局部系数 → 物理场重建 → 目标 chart 投影”，或者使用经过验证的显式 transition map；不能直接复制 `a/b`。
4. Periodic chart 的最小运行状态为 `a_P(32), b_P(32), history-3, Re, phase/time`；固定依赖为 velocity Galerkin ROM、Pressure-Poisson surrogate、scalers 和 RK4。
5. 统一框架应为每个 chart 分别记录 finite fraction、首个发散 step、pressure drift、能量漂移和吸引子保持情况，再由全局层做 preserved 判定。
6. 本 checkpoint 的 `scalers` 和模型输入约定应作为 Periodic adapter 的权威来源；外部统一数据包只能作为物理真值来源，进入模型前必须投影到本 chart。

推荐的统一接口为：

```text
encode_periodic(physical_state, Re, time) -> local_state_P
step_periodic(local_state_P, history_P, Re, phase, dt) -> next_local_state_P
decode_periodic(local_state_P) -> reconstructed_physical_state
diagnose_periodic(trajectory_P) -> stability/phase/orbit/energy metrics
```

## 13. 复现精度声明

新集群最佳 checkpoint 评估与原集群结果对照了 318 个共同数值字段：最大绝对差为 `3.8185e-6`，最大相对差为 `0.0447%`，preserved 判定完全一致。因此可以认定科学结果严格复现；由于 GPU 与数值内核不同，不宣称 bitwise reproducibility。

唯一正式有效的新集群评估目录是：

`reproduction/best_epoch85_heldout_eval_v2`

`reproduction/best_epoch85_heldout_eval` 是补齐原训练 metrics 前的失败尝试，未进入 rollout，不得作为实验结果。
