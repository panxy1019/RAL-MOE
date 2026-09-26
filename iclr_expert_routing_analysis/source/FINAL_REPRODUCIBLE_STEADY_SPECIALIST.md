# V17 最终可复现 Steady Specialist 交接说明

更新时间：2026-07-22  
目标目录：`/root/panxy/particalMOE/steady_specialist_v1`  
用途：后续多坐标图动力系统/统一 MoE 框架中的 Steady 局部坐标图与 Steady specialist。

## 1. 结论与冻结角色

本目录已具备独立完成 Steady specialist 模型加载、继续训练、validation、heldout rollout、POD 物理场重构和严格 paired-gain 计算所需的代码、数据、物理算子与 checkpoint。迁移审计状态为 `PASS`；训练/validation/heldout 雷诺数零交集，heldout 未用于训练或 checkpoint 选择。

必须区分以下两个冻结角色：

| 角色 | 文件 | 用途 |
|---|---|---|
| 原实验冻结的 S4 候选 | `checkpoint/frozen_s4_validation_step_1200.pt` | 原 RTX 3090 预注册评估判为 `S4_SUCCESS_CANDIDATE`；用于复现原实验和研究 S4 fixed-point anchor |
| 跨硬件保守默认 | `checkpoint/frozen_s3b_contraction.pt` | 新 RTX 4090 严格复算中 S4 的 P95 contraction 保护门未通过时，按预注册失败策略使用的最终 Steady sub-MoE |

因此，统一框架不得用含糊的 `best.pt`。配置中应显式写出：

```yaml
steady_specialist:
  variant: s4_original_frozen_candidate   # 或 s3b_cross_hardware_conservative
  checkpoint: /root/panxy/particalMOE/steady_specialist_v1/checkpoint/frozen_s4_validation_step_1200.pt
  ru: 32
  rp: 32
  history_len: 3
  pressure_gauge: subtract_area_mean_per_snapshot
```

原 canonical best 没有被覆盖或重新选择。本交接也未修改任何模型权重。

## 2. 模型输入输出合同

- 局部状态：`z_s = (a, b)`，其中速度系数 `a ∈ R^32`、压力系数 `b ∈ R^32`。
- 坐标系：Steady 独立 POD 基，`ru=32`、`rp=32`；不是全局 POD 坐标。
- 归一化：只使用 Steady train split 拟合的 coefficient mean/std；物理指标必须 inverse-transform 后计算。
- 历史：`history_len=3`。模型编码输入维数为 501；由 Re/相位特征、当前 `(a,b)`、Galerkin RHS 以及三步历史特征按 `make_features_np` 和 `make_history_features_np` 构建。
- 条件变量：Reynolds number；相位展开 `phase_harmonics=4`。
- 压力规范：`subtract_area_mean_per_snapshot`。压力物理场重构或跨 specialist 比较前必须执行同一 gauge。
- rollout：必须走训练器中的 physics-aware rollout 包装逻辑，不能把模型裸 `forward()` 的 operator 输出误当作下一时刻 POD 状态。
- 模型输出：velocity operator 与 pressure operator 两路、分组 router/gates 和可选 closure 参数。
- 物理重构：速度使用 `velocity_basis` 与 `velocity_mean`；压力使用 `pressure_basis`、`pressure_mean` 并保持上述 gauge；物理误差采用面积加权 L2。

统一多坐标图框架建议把以下内容作为 Steady chart adapter 的不可分割接口：

1. `Re + 三步历史 raw POD state -> Steady train-only normalization -> 501 维模型输入`；
2. specialist/physics wrapper 产生下一步 `(a,b)`；
3. inverse normalization 后通过本地 POD 基重构；
4. chart 之间切换时必须先回到 raw POD 或物理场，再投影到目标 chart，禁止直接混用不同 chart 的 normalized coefficients。

## 3. 网络结构与规模

模型类：`OperatorSpaceMoEROM`。

| 参数 | 值 |
|---|---:|
| encoder input | 501 |
| velocity output rank | 32 |
| pressure output rank | 32 |
| hidden_dim | 224 |
| refine blocks | 3 |
| regime groups | 3 |
| routed experts/group | 6 |
| shared experts/group | 1 |
| routed top-k | 2 |
| group top-k | 1 |
| expert hidden | 768 |
| expert blocks | 3 |
| quadratic rank / scale | 4 / 0.05 |
| dropout | 0.04 |
| router temperature | 0.95 |
| group temperature | 0.9 |
| shared / routed scale | 1.0 / 0.85 |
| checkpoint model tensors | 1414 |
| checkpoint state elements | 48,613,781 |
| FP32 model-state bytes | 194,455,124 |

三个冻结 checkpoint 的模型结构完全一致，可用 `strict=True` 加载。checkpoint 文件约 398 MB，因为还包含恢复训练所需的 optimizer/scheduler/RNG 等状态。

## 4. 训练合同与 S4 受控变量

基础 S2-B/S3-B 合同：

- seed：`202607251`；effective batch=64，micro batch=16，grad accumulation=4。
- learning rate：S3-B/S4 为 `1.5516372391099407e-05`；weight decay=`1e-4`；grad clip=`1.0`。
- velocity output head 冻结；pressure head 保持正常学习率；shared trunk/router 使用 pressure head 约 0.1 倍学习率。
- pressure rollout 权重=2，velocity rollout 权重=1。
- S3-B contraction loss 权重 `lambda_pair=2.9341519870712936e-09`。
- S3-B curriculum：K4 600 steps、K8 1000 steps、K16 2000 steps。

S4 的唯一新增变量是 raw-POD/面积加权物理等价 fixed-point anchor：

```text
L_fp = mean(||F_a(z*) - a*||² / ru) + mean(||F_b(z*) - b*||² / rp)
```

- 只从 S3-B step 600 模型参数初始化；optimizer/scheduler/AMP/early-stop 状态重置。
- 标定目标梯度比例=7.5%，实现值=7.5%。
- `lambda_fixed_point=193366295.035771`，前 300 optimizer steps 线性 ramp。
- 最大预算=3600；validation/checkpoint 每 200 steps。
- 用户要求在记录 step 3480 时停止；最后完整 validation=3400。
- validation 保护门通过的候选为 step 0 与 1200；按“先通过保护门，再最小化 validation K16 pressure fixed-point worst”的预注册规则冻结 step 1200。

## 5. 数据划分

| split | Re | 快照数 |
|---|---|---:|
| train | 20.000000, 22.535676, 26.667332, 30.720428, 34.737570, 36.657767, 38.357249, 40.711525, 41.576575, 42.359071, 43.093925, 44.478353, 45.795194, 46.440072 | 894 |
| validation | 28.695138, 43.797402 | 127 |
| heldout/test | 24.630436, 32.740068, 39.685479, 45.142703 | 256 |

train、validation、heldout 的 Re 集合零交集。`steady_trainval_modal_r32.npz` 只含 1021 个 train+validation 样本；heldout perturbation bank 单独存放，未进入训练文件。

## 6. 原 RTX 3090 冻结 S4 的测试结果

以下全部 relative L2 已乘 100，以 `%` 表示。所有窗口 `finite=100%`、`divergent_windows=0`。K128 因连续窗口合同不满足而不报告。

### 6.1 全部 heldout clean autonomous rollout

| Heldout Re | K | 物理速度误差 | 物理压力误差 | Raw POD 速度误差 | Raw POD 压力误差 |
|---:|---:|---:|---:|---:|---:|
| 24.630436 | 1 | 0.0514% | 2.9973% | 0.0606% | 2.4091% |
| 24.630436 | 4 | 0.0771% | 9.3747% | 0.1039% | 7.6061% |
| 24.630436 | 8 | 0.1016% | 8.6490% | 0.1345% | 7.0232% |
| 24.630436 | 16 | 0.1303% | 6.8131% | 0.1719% | 5.5395% |
| 24.630436 | 56 | 0.3092% | 18.3157% | 0.4521% | 14.9068% |
| 32.740068 | 1 | 0.0134% | 1.7287% | 0.0549% | 5.6073% |
| 32.740068 | 4 | 0.0206% | 1.6709% | 0.1039% | 5.4141% |
| 32.740068 | 8 | 0.0253% | 1.3142% | 0.1412% | 4.2557% |
| 32.740068 | 16 | 0.0313% | 1.3667% | 0.1855% | 4.4278% |
| 32.740068 | 56 | 0.0886% | 5.4462% | 0.5852% | 17.6518% |
| 39.685479 | 1 | 0.0066% | 0.2823% | 0.0420% | 2.8621% |
| 39.685479 | 4 | 0.0093% | 0.4128% | 0.0808% | 4.1994% |
| 39.685479 | 8 | 0.0118% | 0.4840% | 0.1158% | 4.9457% |
| 39.685479 | 16 | 0.0179% | 0.8811% | 0.1662% | 9.0185% |
| 39.685479 | 56 | 0.0385% | 1.6975% | 0.4088% | 17.3796% |
| 45.142703 | 1 | 0.0163% | 0.6963% | 0.0344% | 2.3362% |
| 45.142703 | 4 | 0.0238% | 1.4659% | 0.0694% | 4.9067% |
| 45.142703 | 8 | 0.0307% | 1.2927% | 0.0999% | 4.3229% |
| 45.142703 | 16 | 0.0407% | 1.1934% | 0.1383% | 3.9833% |
| 45.142703 | 56 | 0.0615% | 2.4706% | 0.2576% | 8.2754% |

### 6.2 全部 heldout fixed-point 与 physical pressure paired gain

| Heldout Re | K16 pressure FP | K56 pressure FP | K16 gain median / P95 / worst | K56 gain median / P95 / worst |
|---:|---:|---:|---:|---:|
| 24.630436 | 28.1554% | 27.8651% | 0.537 / 3.102 / 7.246 | 0.490 / 3.502 / 6.699 |
| 32.740068 | 3.8234% | 3.8112% | 0.184 / 3.972 / 8.992 | 0.181 / 3.983 / 8.966 |
| 39.685479 | 2.8486% | 2.8630% | 0.191 / 3.768 / 43.260 | 0.222 / 3.732 / 42.945 |
| 45.142703 | 6.8101% | 6.4633% | 0.513 / 11.328 / 127.722 | 0.403 / 11.368 / 145.778 |

物理压力 gain 的全体汇总：K16 median=0.312、P95=4.393；K56 median=0.313、P95=4.444。median<1 只说明多数预注册扰动收缩，不代表 P95/worst 收缩。

### 6.3 与 S3-B 的主要效果对照

| 指标 | S3-B | S4（原 3090） | 结果 |
|---|---:|---:|---|
| heldout K16 pressure FP worst | 34.9616% | 28.1554% | 改善 19.47% |
| heldout K16 clean velocity worst | 0.1303% | 0.1303% | 持平 |
| heldout K16 clean pressure worst | 7.0056% | 6.8131% | 改善 |
| heldout K56 clean velocity worst | 0.2995% | 0.3092% | +3.24%，在 5% 门限内 |
| heldout K56 clean pressure worst | 18.5603% | 18.3157% | 改善 |
| K16 pressure gain median / P95 | 0.198 / 4.263 | 0.312 / 4.393 | median<1，P95 在 +5% 门限内 |
| K56 pressure gain median / P95 | 0.243 / 4.628 | 0.313 / 4.444 | median<1，P95 未退化 |

原 3090 预注册结论：`S4_SUCCESS_CANDIDATE`。4 个 heldout Re 中有 2 个 K16 pressure FP <5%；这是非强制目标，不改变成功判定。

## 7. 新 RTX 4090 可复现性验证

### 7.1 一 epoch-equivalent 训练

- 定义：`ceil(894 / 64)=14` optimizer steps。
- 初始化：严格从 S3-B step 600 模型参数开始，未恢复旧 optimizer/scheduler。
- 参考环境：Python 3.11、PyTorch `2.11.0+cu126`、NumPy `2.4.4`。
- 状态：`PASS`；heldout 未用于训练。
- 耗时：246.113 s。
- 输出 checkpoint SHA256：`2264447bd3154ef0887b8650f1c53126e69c1557c1ecb3878fcead610def245f`。
- 同一 4090 上 PyTorch 2.11 与 2.13 生成的 14-step checkpoint SHA256 完全相同。
- 新 4090 step-1 loss=`159.0705108643`；原 3090 日志约为 `159.0680885315`，差约 0.0015%，说明跨 GPU 不是逐 bit 数值复现。

### 7.2 冻结 checkpoint 评估

同一 checkpoint、同一数据和同一 heldout perturbation bank 在新 4090 上完整执行成功；PyTorch 2.11 与 2.13 的 `raw_metrics.json` 字节一致。新 4090 得到：

- clean rollout 保护：PASS；finite=100%，divergence=0。
- heldout K16 pressure FP worst=`27.8762%`，相对 S3-B 改善 `20.27%`：PASS。
- K16 pressure gain P95=`4.788`，超过 `1.05×4.263`；K56 P95=`4.916`，超过 `1.05×4.628`：contraction 保护 FAIL。
- 因此新 4090 的硬阈值判定为 `S4_FAILED`，失败策略指向 S3-B。

这不是文件、划分或 checkpoint 不一致：相关 SHA256 全部一致；更符合 GPU 数值内核差异导致 tail-percentile 在硬阈值附近跨门的现象。该判断属于基于现有审计证据的推断。统一框架若要求跨硬件稳定晋级，应保留原始连续数值和平台信息，不能只存 PASS/FAIL。

## 8. 集群路径清单

根目录：

```text
/root/panxy/particalMOE/steady_specialist_v1
```

### 8.1 部署必需

| 内容 | 路径 |
|---|---|
| S4 frozen candidate | `checkpoint/frozen_s4_validation_step_1200.pt` |
| S3-B conservative fallback | `checkpoint/frozen_s3b_contraction.pt` |
| S2-B baseline | `checkpoint/frozen_s2b_validation.pt` |
| r32 POD/mean/scaler/mesh/area | `data/steady_pod_runtime_r32.npz` |
| train+validation modal/history/Re index | `data/steady_trainval_modal_r32.npz` |
| train+validation perturbation bank | `data/perturbation_bank_train_validation.npz` |
| velocity Galerkin operators | `physics/velocity_galerkin_trainval_r32.npz` |
| pressure Poisson/closure operators | `physics/pressure_poisson_trainval_r32.npz` |
| heldout bank | `/root/panxy/particalMOE/unified_rollout_eval_heldout_v1/steady/heldout_perturbation_bank.npz` |

### 8.2 代码入口

| 功能 | 路径 |
|---|---|
| vendor model/feature/physics implementation | `code/train_v16_4_v2_r32_compat.py` |
| S2-B trainer/runtime builder | `code/train_s2b_3090.py` |
| S3 contraction training | `code/train_s3.py` |
| S4 fixed-point anchor training | `code/train_s4_steady.py` |
| heldout rollout | `code/evaluate_s3_heldout.py` |
| four-space paired-gain audit | `code/audit_paired_gain_metrics.py` |
| frozen S4 finalizer | `code/finalize_s4_one_time.py` |
| one-epoch portability smoke | `code/reproduce_s4_one_epoch.py` |
| portable config | `code/training_s2b_portable.json` |

### 8.3 原始 POD/ROM 资产与合同

| 内容 | 路径 |
|---|---|
| full Steady velocity POD | `source_artifacts/steady/velocity_pod_steady.npz` |
| full Steady pressure POD | `source_artifacts/steady/pressure_pod_steady.npz` |
| normalization | `source_artifacts/steady/normalization_steady.npz` |
| velocity ROM | `source_artifacts/steady/velocity_rom_steady.npz` |
| pressure surrogate | `source_artifacts/steady/pressure_poisson_surrogate_steady.npz` |
| resolved split contract | `source_artifacts/split_contract_resolved.json` |
| multi-specialist handoff contract | `source_artifacts/MOE_HANDOFF_CONTRACT.json` |

### 8.4 结果与审计

| 内容 | 路径 |
|---|---|
| 原实验客观报告 | `provenance/experiment_report.md` |
| S4 preregistration | `provenance/PREREGISTRATION.json` |
| checkpoint selection | `provenance/checkpoint_selection_manifest.json` |
| 迁移审计 | `provenance/PORTABILITY_AUDIT.json` |
| 4090/PyTorch 2.11 评估报告 | `portability_runs/best_checkpoint_evaluation_torch211/experiment_report.md` |
| 4090 完整 raw metrics | `portability_runs/best_checkpoint_evaluation_torch211/raw_metrics.json` |
| 4090 four-space gains | `portability_runs/best_checkpoint_evaluation_torch211/paired_gain_four_space.json` |
| 14-step 训练证明 | `portability_runs/one_epoch_reproduction_torch211/ONE_EPOCH_REPRODUCTION.json` |

## 9. 核心 SHA256

| 文件 | SHA256 |
|---|---|
| `frozen_s2b_validation.pt` | `3ec8bdedbe2f953a181d71ae819980cbbdfd65c8fee33aac9237a16d780a682d` |
| `frozen_s3b_contraction.pt` | `b85f01bfcd8c209ee4ff6fe6f2a61dfa12acbcecd83da09b153fc0f15b19c23b` |
| `frozen_s4_validation_step_1200.pt` | `bcab5661af39e3262103ed887ce9846415467452b453f3dcc14e133f0f8bd80d` |
| `steady_pod_runtime_r32.npz` | `94e6baf066f315ee81b898269019a6cac62d14e5fd88707745f079b0b88a609f` |
| `steady_trainval_modal_r32.npz` | `044cbbe615c40fcef5d9c1f1b60865184fc40aeb716c0d226c7b0b5e12ecc70b` |
| `perturbation_bank_train_validation.npz` | `69fa522aa4c06f26cad26718dd99b5dcce0b7dd60c30b3efff8a31a154d891c8` |
| `velocity_galerkin_trainval_r32.npz` | `2a20e657a407b7562a8f58384561831089b3d21a20c9a376dbade62f3d87de40` |
| `pressure_poisson_trainval_r32.npz` | `5a473a31f49f0df684c2ef9f07620b06d938762da202a2ad96b7b8433b1c56c3` |
| heldout perturbation bank | `12b0fc179323ad4eb39b7d75f39ccf0b5fdd05bd29fa291205a78709ae224526` |

加载前建议逐项校验；任何 checkpoint、POD/scaler、pressure gauge 或 split 合同不匹配都应立即中止，而不是自动兼容。

## 10. 运行环境

严格迁移复现环境：

```text
/root/panxy/particalMOE/.runtime/pt_env
Python 3.11
PyTorch 2.11.0+cu126
CUDA runtime 12.6
NumPy 2.4.4
Pandas 3.0.2
SciPy 1.17.1
```

服务器原有 `/root/miniconda3/envs/pt_env`（PyTorch 2.13.0+cu126）也已完成同硬件核验，但为了追踪原实验软件版本，默认使用 `.runtime/pt_env`。

只读快速审计：

```bash
cd /root/panxy/particalMOE/steady_specialist_v1
/root/panxy/particalMOE/.runtime/pt_env/bin/python code/verify_steady_portability.py \
  --root /root/panxy/particalMOE/steady_specialist_v1
```

不要把 `portability_runs/*/one_epoch_step_0014.pt` 当成最终部署模型；它们只是迁移后训练可执行性的证明。不要自动重新执行 heldout 并据此重选 checkpoint。

## 11. 接入统一多坐标图框架的最低要求

1. 显式记录 specialist variant 与 checkpoint SHA256。
2. router 只能选择 chart；chart 切换必须经 raw POD/物理场投影，不直接拼接不同 chart 的 normalized POD 状态。
3. Steady normalization、POD basis、mean、area weights、pressure gauge 与 checkpoint 一起版本化。
4. 保留三步 history；切换 chart 时同步转换整个 history，而不只转换当前状态。
5. 统一 rollout 同时输出 clean physical velocity/pressure relative L2、finite/divergence、pressure drift、fixed-point residual 与 paired gain；这些概念不可互相替代。
6. 所有 relative errors 对外统一乘 100，以 `%` 表示；paired gain 为无量纲比值，不乘 100。
7. 新统一模型的 train/validation 选择不得读取四个 Steady heldout Re；heldout 只用于最终冻结后的评估。

至此，Steady specialist 的模型、坐标、数据、物理算子、评估口径、冻结角色与复现证据均已在本目录中闭合。
