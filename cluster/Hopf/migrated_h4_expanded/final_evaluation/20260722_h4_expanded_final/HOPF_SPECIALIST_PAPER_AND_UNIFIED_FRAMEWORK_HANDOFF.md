# Hopf Specialist 论文展示与统一多坐标动力系统框架交接文档

## 1. 文档目的与展示样本

本文档冻结当前 `HopfExpanded34_H4_NormalFormRadial_r32` specialist 的模型、训练合同、代表性结果和集群产物位置，供论文展示以及后续构建“多坐标图动力系统统一学习框架”使用。

论文拟展示两个 Reynolds 数：

| Reynolds 数 | 本次实验中的真实身份 | 是否参与 checkpoint 选择 | 推荐论文表述 |
|---:|---|:---:|---|
| 47.081355 | heldout/final-evaluation Re | 否 | 独立最终评价样本 |
| 56.543246 | validation Re | 是 | validation demonstration / 高 Re 展示样本 |

> 方法学注意：`Re=56.543246` 参与了早停门禁和最佳 checkpoint 选择，不能严谨地称为独立 test。若论文必须将其定义为 test，需要把它移出 validation 后重新拟合所有 train-only 统计并重训。本文后续仍按用户指定将两者作为“论文展示工况”，但保留其真实 split 身份。

## 2. 数据集与坐标系

### 2.1 数据划分

- 训练集：29 个 Re，共 4092 个 snapshots。
- Validation：`46.7, 56.543246`，共 225 个 snapshots。
- Heldout/final evaluation：`47.081355, 49.022357, 51.786450`，共 483 个 snapshots。
- POD 均值、POD 基、normalization、Galerkin/Pressure Poisson tensors 和 Hopf 二维振荡平面均只使用 29 个训练 Re 拟合。
- Pressure 在 POD 前逐 snapshot 去除 area-weighted mean gauge。

29 个训练 Re：

```text
45.5, 45.8, 46.1, 46.3, 46.5, 46.9, 47.2, 47.4, 47.6,
47.722947, 47.8, 48.0, 48.3, 48.368688, 48.7, 49.3, 49.6,
49.687640, 50.0, 50.368054, 51.066785, 52.528767, 53.294175,
54.081508, 54.887950, 55.709610, 57.389970, 58.262636, 59.201432
```

### 2.2 Hopf-local POD/ROM

- 原始新数据资产保留 `r80`，本 H4 采用报告规定的 fair-control `r_u=r_p=32`。
- 速度坐标：`a_u ∈ R^32`；压力坐标：`a_p ∈ R^32`。
- 物理场解码：

```text
u(x,t) = mean_u(x) + Phi_u(x) a_u(t)
p(x,t) = mean_p(x) + Phi_p(x) a_p(t)
```

- r32 训练集投影误差均值：速度 `0.0003716%`，gauge-fixed pressure `0.0011363%`。
- r32 能量捕获率：速度 `99.99999921%`，pressure `99.99999779%`。
- H4 径向合同的二维平面由 29 个 train Re 逐 Re 去均值后的波动协方差前两个特征向量识别；不使用 validation/heldout。

核心源资产 SHA-256：

| 资产 | SHA-256 |
|---|---|
| velocity POD | `2f951192f6eeaa69e7d911aa21552454d1a99a49ab796166b4a13df16016a192` |
| pressure POD | `cc128a4361ab2a239165ac22a48407290de7d085b857bbff6726988457cfddf3` |
| velocity ROM | `7d16ca813efa0f92109689a686de0c3de02cdd269a4b31029c53c3ed1309d3e9` |
| pressure surrogate | `a6646aa3eb3814d8648293ee3cd798a6abe268bf46a79ec0c0b8b17b30aaf06d` |

## 3. Hopf specialist 模型

### 3.1 状态更新

模型采用如下闭环：

```text
Hopf-local state/history/Re
        ↓
Galerkin RHS + history encoder
        ↓
velocity MoE residual RHS
        ↓
RK4 autonomous velocity update
        ↓
Pressure Poisson surrogate + learned pressure residual + AdaptiveGate
        ↓
next Hopf-local velocity/pressure state
```

速度方程可概括为：

```text
da_u/dt = F_Galerkin(a_u, a_p, Re) + F_MoE_residual(history, a_u, a_p, Re)
```

压力由更新后的 velocity state 经过 Pressure Poisson surrogate、独立 pressure expert bank 和 AdaptiveGate closure 得到。Rollout 只接收固定 Re、当前自主状态、历史状态和观测 dt，不读取未来真值状态或 phase。

### 3.2 网络结构

| 配置 | 数值 |
|---|---:|
| 模型输入维度 | 493 |
| velocity POD rank | 32 |
| pressure POD rank | 32 |
| history length | 3 |
| shared encoder width | 256 |
| encoder/refine blocks | 3 |
| 每个 expert hidden width | 1024 |
| 每个 expert blocks | 4 |
| routed experts | velocity 6 + pressure 6 |
| shared experts | velocity 1 + pressure 1 |
| local routing | top-2 |
| velocity/pressure router | 相互独立 |
| 激活函数 | SiLU（主干/router） |
| dropout | 0.04 |
| router temperature | 0.8 |
| quadratic rank | 4 |
| pressure closure | Pressure Poisson surrogate + learned residual + AdaptiveGate |
| 外层 group router | 单 Hopf group，数学上 inert 且冻结 |

参数量：

| 部分 | 总参数/状态元素 | 可训练参数 |
|---|---:|---:|
| 部署 MoE-ROM 主体 | 32,004,147 | 31,810,392 |
| H4 normal-form auxiliary head | 12（含 2 buffers） | 10 |
| 合计可训练参数 | — | 31,810,402 |

H4 normal-form head 参数化平滑的 `mu(Re)`、正 `beta(Re)` 和有界修正，用 derivative/step/growth-sign/saturation/consistency loss 约束临界二维径向动力学。该 head 是训练辅助约束，不直接参与冻结模型的 autonomous state transition；部署推理由 Galerkin＋MoE residual＋RK4＋pressure closure 完成。

### 3.3 初始化与训练损失

- 使用新 Hopf-local POD 坐标重新初始化，未加载旧坐标系 checkpoint。
- 初始模型为 physical ROM＋零 learned residual；normal-form head 为严格中性/零输出初始化。
- 公共目标：coefficient/state、Galerkin RHS/dynamics、pressure closure、velocity/pressure rollout、trajectory consistency、energy consistency、router balance/diversity。
- 波动目标：train-only conditional fluctuation modal/field loss、径向 `log(r+r_floor)`、多步 growth consistency。
- H4 auxiliary 目标：normal-form derivative、short-step radial trajectory、growth sign、saturation 和 full-state/normal-form consistency。
- 波动目标校准梯度比例：目标 `18%`；normal-form 目标：`8%`；阶段切换后 200 steps ramp。

## 4. 训练合同与冻结结果

| 项目 | 配置 |
|---|---|
| 随机种子 | 1248 |
| optimizer | fused AdamW |
| 初始学习率 | 1e-3 |
| weight decay | 1e-4 |
| scheduler | CosineAnnealingLR，T_max=8000 |
| gradient clip | 1.0 |
| micro/effective batch | 232 / 232 |
| 数值配置 | FP16 autocast + GradScaler + TF32 |
| curriculum | K1: 0–1200；K2: 1200–2800；K4: 2800–4800；K8: 4800–8000 |
| validation | 每 200 steps；step 4800 后每 400 steps 增加 K24/K48 |
| 正式训练 | 8000 optimizer steps，return code 0 |
| 训练耗时 | 1.979 小时 |
| 平均吞吐 | 67.36 optimizer steps/min |

Checkpoint 只按 validation 选择：

- 最佳 step：`7200`。
- Validation score：`0.878767498`。
- Numeric gate：通过。
- Hopf gate：通过。
- 冻结 checkpoint SHA-256：`02148741ed8bc9fbec88709f69b10492e263514b2edcee76652485b399962235`。

## 5. 两个论文展示工况的效果

以下相对误差均已乘以 100，以百分数表示。`Re=47.081355` 来自冻结后的 final evaluator；`Re=56.543246` 来自 step 7200 的 checkpoint-selection validation record，因此两者的窗口采样协议不同，不应直接作为同一 test 集的无偏统计平均。

### 5.1 多步物理场误差

| Re | 身份 | Horizon | 速度物理场误差 | 压力物理场误差 | Finite | Divergent windows |
|---:|---|---:|---:|---:|---:|---:|
| 47.081355 | heldout | K1 | 0.000621% | 0.073791% | 100% | 0 |
| 47.081355 | heldout | K2 | 0.000782% | 0.061592% | 100% | 0 |
| 47.081355 | heldout | K4 | 0.000951% | 0.056041% | 100% | 0 |
| 47.081355 | heldout | K8 | 0.001063% | 0.047197% | 100% | 0 |
| 47.081355 | heldout | K16 | 0.001353% | 0.041030% | 100% | 0 |
| 47.081355 | heldout | K24 | 0.001650% | 0.036561% | 100% | 0 |
| 47.081355 | heldout | K48 | 0.002374% | 0.028363% | 100% | 0 |
| 56.543246 | validation | K1 | 0.003940% | 0.096225% | 100% | 0 |
| 56.543246 | validation | K2 | 0.007901% | 0.108882% | 100% | 0 |
| 56.543246 | validation | K4 | 0.015203% | 0.122117% | 100% | 0 |
| 56.543246 | validation | K8 | 0.029180% | 0.152786% | 100% | 0 |
| 56.543246 | validation | K16 | 0.054645% | 0.235357% | 100% | 0 |
| 56.543246 | validation | K24 | 0.077259% | 0.316975% | 100% | 0 |
| 56.543246 | validation | K48 | 0.135545% | 0.538410% | 100% | 0 |

### 5.2 Hopf 动力学指标

#### Re=47.081355：near-onset heldout

| 指标 | K48 结果 |
|---|---:|
| RMS 振幅误差 | 0.5769% |
| P2P 振幅误差 | 13.0542% |
| Growth-sign accuracy | 76.8730% |
| 真实平均局部 log-growth | 3.00864e-05 |
| 预测平均局部 log-growth | -2.06555e-03 |
| False growth | 否 |
| Frequency relative error | 37.2568%（真实频率近零，不作为有效旋转频率结论） |
| Terminal phase drift | 0.00623 cycles |
| Normalized orbit distance | 0.05868 |
| 严格 attractor-preserved | 未通过：P2P 略超 10% 门限；频率指标在近零振幅下无效 |

该工况最重要的结果是消除了此前关注的 false growth，同时保持极低的 K48 总场误差。预测平均增长略为负，说明模型在临界点下方采取了轻微衰减而非虚假振荡增长。

#### Re=56.543246：高 Re validation demonstration

| 指标 | K48 结果 |
|---|---:|
| Fluctuation velocity error | 60.4359% |
| Fluctuation pressure error | 60.0331% |
| Radial RMS diagnostic | 70.8906% |
| Scale inflation | 0.26756（明显幅值衰减） |
| Growth-sign accuracy | 62.2155% |
| Finite fraction | 100% |
| Divergent windows | 0 |

该工况的总场误差和数值稳定性很好，但 K48 波动幅值约收缩到参考尺度的 26.8%。因此适合展示“高 Re 长程稳定与频率/增长方向学习”，不应表述为高 Re 吸引子幅值已完整保持。

## 6. 论文中建议的结论措辞

可支持的结论：

1. Hopf specialist 在 near-onset 与较高 Re 工况上均实现 K48 全程 finite、零 divergence。
2. 在 Hopf-local r32 坐标下，总速度场与压力场误差保持在很低水平。
3. Re=47.081355 的 false growth 被消除，near-onset 径向行为由虚假增长转为轻微衰减。
4. Re=56.543246 展示了高 Re 长程数值稳定性，但仍存在显著 fluctuation amplitude attenuation。

不应支持的结论：

1. 不应把 Re=56.543246 称为独立 test，因为它参与了 validation checkpoint 选择。
2. 不应宣称两个展示工况均完整保持 Hopf attractor。
3. 不应仅凭很低的 total-field L2 断言波动动力学准确；均值场会掩盖振幅/轨道误差。

## 7. 新集群上的数据、代码与产物路径

### 7.1 登录与环境

```bash
ssh -p 20073 root@10.10.164.243
source /root/miniconda3/bin/activate pt_env
```

### 7.2 数据与构造说明

```text
/root/panxy/particalMOE/Hopf/
├── V17_HOPF_EXPANDED_DATASET_AND_POD_ROM_REPORT.md
├── artifacts/
│   ├── HOPF_EXPANDED_POD_ACCEPTANCE.json
│   ├── split_contract_resolved.json
│   └── hopf/
│       ├── velocity_pod_hopf.npz
│       ├── pressure_pod_hopf.npz
│       ├── normalization_hopf.npz
│       ├── velocity_rom_hopf.npz
│       ├── pressure_poisson_surrogate_hopf.npz
│       ├── pod_artifact_manifest_hopf.json
│       ├── pod_projection_diagnostics_hopf.json
│       └── projection_snapshots_velocity_hopf.csv
```

### 7.3 H4 代码与封装资产

```text
/root/panxy/particalMOE/Hopf/migrated_h4_expanded/
├── code/
│   ├── train_hopf_moe_expanded.py          # 公共 Hopf MoE/物理训练逻辑
│   ├── train_h4_expanded.py                # H4 trainer/canonical rollout
│   ├── build_expanded_h4_assets.py         # 新 r80 → sealed r32 训练视图
│   ├── build_trainonly_h4_expanded_contract.py
│   ├── evaluate_h4_expanded.py             # 冻结后 K1–K48 evaluator
│   ├── freeze_h4_expanded.py               # validation-only 冻结
│   └── make_final_report.py
├── periodic_moe_3090/
│   └── train_periodic_moe.py               # V16 公共 HPRS-MoE 实现
├── assets_r32/
│   ├── expanded_h4_trainval_r32.npz
│   ├── expanded_h4_trainonly_galerkin_r32.npz
│   ├── expanded_h4_trainonly_pressure_r32.npz
│   └── TRAINING_ASSET_MANIFEST.json
├── trainonly_contract/
│   ├── trainonly_fluctuation_contract.npz
│   └── TRAINONLY_H3H4_CONTRACT.json
└── runs/HopfExpanded34_H4_NormalFormRadial_r32/
    ├── config.json
    ├── best_validation.pt
    ├── final_training.pt
    ├── latest.pt
    ├── validation_history.json
    ├── gradient_audits.json
    ├── normal_form_train_nodes.json
    └── throughput.json
```

### 7.4 冻结 checkpoint 与最终评价

```text
/root/panxy/particalMOE/Hopf/migrated_h4_expanded/final_evaluation/20260722_h4_expanded_final/
├── checkpoint_selection_manifest.json
├── heldout_metrics.json
├── FINAL_H4_EXPANDED_REPORT.md
├── FINAL_H4_EXPANDED_SUMMARY.json
├── artifact_inventory.json
└── HopfExpanded34_H4_NormalFormRadial_r32/
    ├── final.pt                             # 只读冻结 checkpoint
    ├── config.json
    ├── validation_history.json
    ├── heldout_metrics.json
    └── trainonly_manifest.json
```

冻结 checkpoint：

```text
/root/panxy/particalMOE/Hopf/migrated_h4_expanded/final_evaluation/20260722_h4_expanded_final/HopfExpanded34_H4_NormalFormRadial_r32/final.pt
```

## 8. 面向统一多坐标图动力系统框架的接口建议

### 8.1 必须整体绑定的 Hopf chart 资产

Hopf specialist 不是单独一个神经网络 checkpoint。以下对象共同定义一个可重现的坐标图：

```text
HopfChart = {
  mean_u, Phi_u_r32,
  mean_p, Phi_p_r32,
  coefficient/scaler semantics,
  Galerkin c/A/H/P tensors,
  Pressure Poisson c_tilde/A_tilde/H_tilde tensors,
  Re interpolation nodes,
  train-only fluctuation means/scales,
  frozen critical plane and radial floor,
  history_len=3 and dt semantics,
  H4 checkpoint and code SHA
}
```

缺失或替换其中任意一项都会改变坐标语义。禁止把该 checkpoint 与旧 global POD、其他 regime scaler 或不同 pressure gauge 的资产混用。

### 8.2 建议的统一 specialist API

```text
encode_Hopf(u, p) -> (a_u, a_p)
step_Hopf(a_u, a_p, history, Re, dt) -> (a_u_next, a_p_next, diagnostics)
decode_Hopf(a_u, a_p) -> (u, p)
diagnostics -> router weights, residual RHS, pressure closure gate,
               modal norms, radial coordinate, finite/divergence flags
```

统一框架应在 chart 外部管理：

1. 当前使用哪个局部坐标图；
2. 不同 chart 之间的状态投影/转换；
3. chart 切换时 history 的重新编码；
4. 各 chart 自己的均值、POD、scaler、ROM、pressure gauge 和 dt；
5. 统一物理场空间中的比较、守恒诊断和切换门禁。

### 8.3 推荐的统一表示边界

- specialist 内部继续在各自 POD 坐标中学习局部动力学，避免强迫 Steady/Hopf/Periodic 共用不合适的 modal coordinates。
- specialist 输出除了下一状态，还应暴露 uncertainty/safety diagnostics，供上层 chart selector 使用。
- chart 间一致性应在物理场或公共观测空间中定义，而不是直接比较不同 POD 系数。
- 训练上层统一框架时，冻结本 specialist 的 POD/ROM/scaler/checkpoint SHA；任何解冻实验应另建版本并重新验证。
- Re=47.081355 可作为 near-onset chart 切换与 false-growth 安全性的重点展示；Re=56.543246 可用于展示高 Re Hopf chart 的稳定 rollout，但应同时展示 amplitude attenuation，避免只展示 total-field L2。

## 9. 复现实验所需的关键 SHA

| 对象 | SHA-256 |
|---|---|
| 冻结 checkpoint | `02148741ed8bc9fbec88709f69b10492e263514b2edcee76652485b399962235` |
| H4 trainer | `bf25d3fee9ed1b51a02437eaba64ad6d9d618955eed2b54f4a218e14d25c4ee4` |
| 公共 expanded trainer | `f7facb3b6070290e9d6725f09fdd0810f0cf7e1775d35c20563a1b691d2c6975` |
| train-only fluctuation contract | `2e3aa04095ab0ccceed2a98e3582a7096d1a80993ab486a7896f2dff7afc1e19` |
| sealed coefficient view | `53d05065d518de82432e55b191dcee6c3501e09822dfedbf6162e061151f4d45` |
| train-only Galerkin r32 | `ebfb05f23c7fe909cb9b92fdea9098d849927bc732652a3881a4f32bbc6e6e26` |
| train-only pressure r32 | `9df6f17413ff884a04b5bbba9e2b791563bde412b5d17649f9badda4e151e44c` |

## 10. 最终定位

该 H4 checkpoint 可以作为统一多坐标图框架中的稳定 Hopf specialist 基线：它具备可靠的物理场重构、Pressure Poisson closure、K48 数值稳定性和 near-onset false-growth 抑制能力。当前尚未解决的是跨 Re 的波动幅值保持，尤其是高 Re 的 amplitude attenuation。因此上层统一框架应把“物理场稳定”和“局部吸引子尺度保持”作为两个独立门禁，而不能只依赖 total-field relative L2。
