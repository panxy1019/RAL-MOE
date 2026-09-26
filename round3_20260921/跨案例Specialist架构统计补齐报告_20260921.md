# 跨案例 Specialist 架构统计补齐报告

日期：2026-09-21  
性质：只读架构、检查点和部署路径统计；未重训、未修改现有结果或论文。

## 1. 结论摘要

1. 已闭环的 sparse-MoE specialist 是 Circular H/P 与 Square S/H/P，共五个实例；它们共享专家参数化，但维数、组策略、冻结状态、压力路径和推进方式并不统一。
2. Circular S 的论文结果身份仍未闭环。已有 S4 checkpoint 仅可作诊断，不能用于补写论文 Steady 架构。
3. Pinball S/H/P 的 formal outer-rollout 候选已由 final-test seal 绑定为三个 **Deep-FNN-H3 B1** 检查点；它们没有 group/channel router、shared expert 或低秩二次专家。论文不能把九个 specialist 都描述为 sparse MoE。
4. Square H 的 overlap evaluator 是 data-only velocity RHS + learned algebraic pressure；Square S 的 velocity experts 冻结且 pressure 为 baseline PPE+residual；Square P 固定使用 0-based group index 1 并含 8 个内部子步。这些部署例外必须独立披露。
5. 原版 Circular H/P 虽存有 rank-4 quadratic tensors，但 U/V 同时为零使分支始终无输出、无梯度。修正版只零一个因子后分支可训练；现有结果不足以声称稳定精度增益，H 仅 `2/3` seed accepted。

## 2. 身份与论文映射

| 实例 | 结果范围 | 模型类/家族 | Checkpoint SHA-256 | 映射状态 |
|---|---|---|---|---|
| Circular S | 论文 Steady 行 | 未确认；S4 诊断模型不得替代 | 未闭环 | **证据不足** |
| Circular H | Appendix internal-routing / 已核对局部模型 | structured sparse MoE | `02148741ed8bc9fbec88709f69b10492e263514b2edcee76652485b399962235` | checkpoint、源码、严格加载、真实窗口 hook 闭环；不自动代表所有旧表行 |
| Circular P | 同上 | structured sparse MoE | `b2052a746fdaab60f83e65d1b91d038a9dded8255ce74f41e3f6fca4c9e717c5` | 同上 |
| Square S | SH overlap Candidate S | structured sparse MoE | `23c7b18fb44316fca28bef0aeac4458235c100b99b2b0c4503d7ae5770cc6fe9` | formal cache/checkpoint 绑定闭环 |
| Square H | SH/HP overlap Candidate H | stored sparse MoE；evaluator 走 data-only 输出路径 | `07af06ec5506e5b6a12d9d00f557a4e48fa722be3c2a183df752f0883ad40beb` | formal cache/checkpoint 绑定闭环 |
| Square P | HP overlap Candidate P | structured sparse MoE，固定组 | `a84f355ae7eba1faa9e988f56fd89029e0aa15e2bd56163543bf5e7c66a4d988` | formal cache/checkpoint 绑定闭环 |
| Pinball S | formal SH outer final-test Candidate 1 | `DeepFNNH3` B1 | `2ab91013c335645ec5424d2cd3d6396ddcf8ce9f9e08b77a347e1b13d0aaf3cf` | final-test seal 闭环；独立 autonomous 表行仍需逐行区分 |
| Pinball H | formal SH/HP outer final-test Candidate 2 | `DeepFNNH3` B1 | `8be7a5552105116c72fa892b01e3670453de04bd753036ff4080faa75942b3d4` | outer final-test 闭环；旧独立 summary 的 `heldout_loaded=false` 不可混用 |
| Pinball P | formal HP outer final-test Candidate 1；Periodic autonomous 数值链 | `DeepFNNH3` B1 | `c89a0cbf6cc0920664875f00234afaac49bc19b1c5c5f985cbba0b09e2dbdbda` | outer seal 闭环；K32/K56 autonomous 数值与 Deep-FNN-H3 summary 对应 |

重要限定：Pinball 的 formal seal 解决了 **outer overlap candidate identity**，并没有把所有早期独立 specialist 报表自动升级为同一结果链。结果身份必须按评价入口和 split 分开。

## 3. 跨案例通用架构总览

| 实例 | 家族 | `d_u/d_p` | 历史 | 输入 D | 隐藏表示 | 速度输出/推进 | 压力路径 |
|---|---|---:|---|---:|---|---|---|
| Circular H | sparse MoE | 32/32 | 3 states（当前+2 lag） | 493 | encoder `h=256` | Galerkin RHS + learned correction；RK4，各 stage 重算当前 feature/router | componentwise `gamma*b_PP + rho_p` |
| Circular P | sparse MoE | 32/32 | 3 states | 501 | `h=224` | 同类 additive hybrid RK4 | componentwise `gamma*b_PP + rho_p` |
| Square S | sparse MoE | 5/4 | 3 states | 91 | `h=224` | additive hybrid；velocity experts 在最终训练中冻结 | baseline `b_PP + rho_p`；stored confidence head 跳过/冻结 |
| Square H | stored sparse MoE | 11/11 | 3 states | 178 | `h=256` | evaluator 直接使用 learned velocity RHS；Galerkin RHS 只作 feature，不相加 | direct learned algebraic pressure；stored PPE/gate 不参与最终 overlap 输出 |
| Square P | sparse MoE | 28/26 | 3 states | 431 | `h=224` | additive hybrid；8 内部子步，`max_integrator_dt=0.5` | componentwise adaptive PPE gate |
| Pinball S | Deep-FNN-H3 | 4/3 | 3 states | 68 | shared trunk 512 | learned velocity derivative，经 RK4 推进；Galerkin 作为 feature | learned componentwise gate × PPE + learned residual |
| Pinball H | Deep-FNN-H3 | 5/5 | 3 states | 88 | shared trunk 512 | 同上 | 同上 |
| Pinball P | Deep-FNN-H3 | 17/16 | 3 states | 263 | shared trunk 512 | 同上 | 同上 |

Circular S 未进入总览数值行，因为尚未完成论文模型身份绑定。

## 4. sparse-MoE 的输入、encoder 与输出

### 4.1 Feature contract

五个确认实例的当前特征顺序为：

1. 标准化物理参数；
2. `Re_ref/max(Re,eps)`；
3. 可选 phase harmonics `sin(k theta), cos(k theta)`；
4. 当前 `a_n, b_n, g_n`；
5. 11 个 norm/energy/ratio 派生量；
6. 两个 lag 的 `[a,b,g,a_n-a_lag,b_n-b_lag,g_n-g_lag]`。

因此

```text
D = 13 + 2H + 5(2*d_u + d_p).
```

完整 `xi` 用各 chart 的 train split scaler 标准化；velocity RHS 与 pressure target 分别有输出 scaler，并在物理组合前反标准化。`dt` 供积分器使用，不是网络输入。

### 4.2 Shared encoder

```text
Linear(D,h)-LayerNorm-SiLU-Dropout(0.04)
Linear(h,h)-LayerNorm-SiLU-Dropout(0.04)
Linear(h,h)-LayerNorm-SiLU
```

其后是 2 个 refinement residual blocks：

```text
z <- z + Linear(h,h)(Dropout(SiLU(Linear(h,h)(LayerNorm(z)))))
```

配置 `num_blocks=3` 指三层主干并派生出 `num_blocks-1=2` 个 refinement blocks，不是“三个 refinement blocks”。速度和压力共享 `z`，但后续 router 和 expert 参数独立。

## 5. sparse-MoE 配置表

| 实例 | D/h | G/组策略 | Router input / 温度 | Routed/shared/Top-k | Expert B/展开 | Rank/kappa | Shared:routed | 压力模式 |
|---|---|---|---|---|---|---|---|---|
| Circular H | 493/256 | 1，单组，router frozen | `[z;xi]`; group 1.0, channel 0.8 | 每组每通道 6 routed +1 shared；Top-2 | 4/1024 | 4/0.05 | 1:0.75 | adaptive `gamma*b_PP+rho` |
| Circular P | 501/224 | 3，learned hard Top-1 | `[z;xi]`; 0.9/0.8 | 同上 | 3/768 | 4/0.05 | 1:0.75 | adaptive |
| Square S | 91/224 | 3，learned hard Top-1 | `[z;xi]`; 0.9/0.95 | 同上 | 3/768 | 4/0.05 | 1:0.85 | baseline PPE+residual |
| Square H | 178/256 | 1，单组，router frozen | `[z;xi]`; 1.0/0.8 | 同上 | 4/1024 | 4/0.05 | 1:0.75 | deployed data-only exception |
| Square P | 431/224 | 3，部署固定 index 1（0-based） | `[z;xi]`; 0.9/0.8 | 同上 | 3/768 | 4/0.05 | 1:0.75 | adaptive |

learned group router 使用 temperature softmax + 每样本 hard Top-1；训练使用 straight-through。channel router 在已选组的 6 个 routed experts 内独立为 u/p 选择 Top-2，再对保留概率归一化。shared expert 不参与竞争，按配置常量分配总质量：常规实例为 `4/7` shared、`3/7` routed；Square S 为 `1/1.85` 与 `0.85/1.85`。

## 6. 单个 structured expert

令 `s_u=a_std`，`s_p=[a_std;b_std]`。每个 shared/routed expert同构但不共享参数：

```text
y0 = GELU(Linear(h+dim(s), h)(LayerNorm([z;s])))
repeat B times:
    y <- y + rho * Linear(m,h)(Dropout(GELU(Linear(h,m)(LayerNorm(y)))))
N(z,s) = Linear(h,d_out)(LayerNorm(y))
E(z,s) = N(z,s) + L*s + kappa*Q(U,V,s)
```

- residual block scale `rho` 为可学习标量，初始 0.1；
- `L` 无 bias；
- `U,V` shape 都为 `[d_out,4,state_dim]`；
- quadratic contraction 是逐输出 rank-4 乘积，无额外 `W_o`；
- 分支先在单 expert 内相加，再进行 shared + Top-2 mixture。

关键 shapes：

| 实例 | u projection | p projection | u U/V | p U/V |
|---|---|---|---|---|
| Circular H | `256x288` | `256x320` | `32x4x32` | `32x4x64` |
| Circular P | `224x256` | `224x288` | `32x4x32` | `32x4x64` |
| Square S | `224x229` | `224x233` | `5x4x5` | `4x4x9` |
| Square H | `256x267` | `256x278` | `11x4x11` | `11x4x22` |
| Square P | `224x252` | `224x278` | `28x4x28` | `26x4x54` |

## 7. 实际时间推进和 pressure 例外

### 7.1 常规 hybrid RK stage

在一个 velocity macro-step 内，`b_n`、过去历史和物理参数固定；每个 RK trial velocity 都重新计算 Galerkin RHS、当前 feature、encoder、组路由、u Top-2 和 expert 输出。完成 velocity 更新后才计算 PPE candidate、pressure correction/gate，并移位历史。

这意味着“context frozen”不等于“RK 四阶段路由固定”。Circular P 的实际 stage 诊断观察到显著 pair/group 变化。

### 7.2 实例例外

- **Square S**：保存的 confidence head 不参与 baseline closure；velocity experts 被固定，但 encoder/router 等其他模块仍按训练入口设置更新。
- **Square H**：最终 overlap `make_ops` 直接输出 learned RHS 和 learned pressure。stored Galerkin/PPE/gate 结构不能算作最终预测的实际贡献。
- **Square P**：0-based fixed group `1`，并在 macro-step 内做 8 个子步。
- **Pinball B1**：每次 Deep-FNN forward 同时给 velocity derivative、pressure residual 和 pressure gate；RK4 以 learned derivative 推进，Galerkin RHS进入特征但不是加性 backbone。最终 pressure 为 `gate * b_PP + residual`。

## 8. Pinball Deep-FNN-H3 专表

| 实例 | 输入/历史 | Trunk | 激活/归一化 | Residual/dropout | Heads | 参数共享 | Target/推进 | Pressure |
|---|---|---|---|---|---|---|---|---|
| S | D=68，`d_u/d_p=4/3`，当前+2 lag | `LN(D) -> Linear(D,512)` + 5 个 `Linear(512,512)`，共 6 个 512-wide Linear | 每层 SiLU；后5层各接 LN | 无/无 | u `512-256-4`; p `512-256-3`; gate `512-128-3` | 三 head 共享 trunk，head 参数独立 | standardized velocity derivative；RK4 | sigmoid gate × PPE + residual |
| H | D=88，`5/5` | 同上 | 同上 | 同上 | u/p `512-256-5`; gate `512-128-5` | 同上 | 同上 | 同上 |
| P | D=263，`17/16` | 同上 | 同上 | 同上 | u `512-256-17`; p `512-256-16`; gate `512-128-16` | 同上 | 同上 | 同上 |

H3 的 shape contract 为：

```text
CURRENT_DIM = 13 + 2*d_u + d_p
history_block_dim = 4*d_u + 2*d_p
D = CURRENT_DIM + (history_len-1)*history_block_dim, history_len=3.
```

因此 H3 确实表示三个状态时刻，而不是“三层网络”。保存 parser 中的 token/head 参数属于其他 variant；B1 `DeepFNNH3` 没有 attention、KDA memory、MoE router 或 expert bank。

## 9. 参数量、冻结和活跃路径

| 实例 | 总 neural 参数 | 训练可更新 | 冻结/未贡献模块 | 单次 active 口径 | POD/物理算子 |
|---|---:|---:|---|---|---|
| Circular H | 32,004,147 | 31,810,392 | 193,755 frozen；训练辅助 head 与部署需分开 | prediction-active 14,165,816 | 排除 |
| Circular P | 48,617,547 | 48,617,547 | 无已记冻结 | prediction-active 8,308,959 | 排除 |
| Square S | 46,655,195 | 23,746,370 | 22,908,825 frozen，含 velocity expert 相关冻结；confidence 路径跳过 | 未完成真实 physical rollout trace | 排除 |
| Square H | 31,318,329 | 31,205,844 | group-router 112,485 frozen；stored PPE/gate 对最终 data-only 输出无贡献 | 未解析“执行”与“影响最终输出”两个口径 | 排除 |
| Square P | 48,209,967 | 48,209,967 | group 结构存储完整但部署固定 index 1 | 未完成真实 physical rollout trace | 排除 |
| Pinball S | 1,684,370 | 1,684,370 | 无 MoE；RK 中 pressure heads可被计算但中间结果不一定用于速度更新 | 每次完整 NN forward 全部 trunk+heads执行 | 排除 |
| Pinball H | 1,695,679 | 1,695,679 | 同上 | 同上 | 排除 |
| Pinball P | 1,792,959 | 1,792,959 | 同上 | 同上 | 排除 |

口径说明：

- 总 neural 参数按唯一 parameter tensor 计数；训练可更新依据最终训练源码的 freeze 设置，不把 state_dict 元素数直接当 trainable。
- Circular active 包括 encoder/refinement、实际执行 routers、选中组的 shared+u/p Top-2 和 pressure head，排除固定物理算子/冻结参数。
- Square 尚未按真实 physical rollout 对完整 active graph 做 hook；不能复制 Circular active 数值。
- 参数量不是 FLOPs、延迟或加速比。

## 10. Quadratic 分支的原版/修正版边界

| 项目 | 原版 Circular H/P | 修正版 |
|---|---|---|
| 构造器 | U/V 原生随机初始化 | 相同 |
| 训练入口重置 | U、V 同时置零 | 只置零一个因子，保留另一个随机 |
| 初始 quadratic 输出 | 0 | 0 |
| 初始梯度 | 两因子均为 0，梯度也为 0 | 零因子可从保留因子获得梯度 |
| checkpoint 分支激活 | 原版实际保持零 | 有非零分支证据 |
| 性能结论 | 不能声称 quadratic 带来收益 | P 平均未改善；H 仅 2/3 accepted，完整三种子稳健性不成立 |

论文应区分：存在参数化、实际激活、带来精度收益。三者不是同一个结论。修正版必须作为独立实验族，不能回写为原版 checkpoint 的结构效果。

## 11. 训练配置补充（与架构身份分开）

| 实例 | Optimizer/lr/WD | Budget/selected | Curriculum | Seed |
|---|---|---|---|---:|
| Circular H | AdamW `1e-3/1e-4` | 8000/7200 steps | 1,2,4,8 | 1248 |
| Circular P | AdamW `5.5e-4/1.5e-4` | 240/85 epochs | 4,8,12,16 | 1600 |
| Square S | AdamW `1.551637e-5/1e-4` | 4000/3200 steps | 1,4,8,16 | 202607281 |
| Square H | AdamW `1e-3/1e-4` | 8000/7200 steps | 1,2,4,8 | 1248 |
| Square P | AdamW `5.5e-4/1.5e-4` | 720/490 epochs | 4,8,12,16 | 1600 |
| Pinball S/H/P B1 | AdamW `1e-3/1e-4`，max 8000，clip1 | selected step 逐 checkpoint 查阅 | S: 4,8,16；H: 4,8,16,32,56；P: 4,8,16,24,32 | 1248 |

“预算/课程”不表示 selected checkpoint 已经历预算后的所有阶段；报告 checkpoint 的 selected step 时必须以保存字段为准。

## 12. 论文建议

### 12.1 主文 3.2 的适用范围

将统一断言改为：

> For the audited circular-cylinder Hopf/Periodic and centered-square specialists equipped with structured sparse routing, ...

随后把 Pinball 明确列为 Deep-FNN-H3 local predictors。除非 Circular S 身份被恢复，否则不要写“all nine specialists use the same sparse-MoE closure”。

### 12.2 附录建议四张表

1. 跨案例总览：本报告第 3 节。
2. sparse-MoE 逐实例设置：第 5 节。
3. Pinball Deep-FNN：第 8 节。
4. 参数/冻结/部署例外：第 9 节。

### 12.3 必须明说的例外

- Circular S identity unresolved；
- Square H data-only overlap path；
- Square S velocity freeze + baseline pressure；
- Square P fixed group 1 + substeps；
- Pinball formal overlap candidates are Deep-FNN, not MoE；
- original quadratic branch inactive；corrected branch does not establish robust gain。

## 13. 尚未闭环项目

1. Circular S 的论文行 -> evaluation JSON -> checkpoint -> model class -> source hash 链。
2. Pinball 独立 autonomous 表中 S/H 每一行的 held-out 身份；outer final-test 身份已闭环，不应重复查找。
3. Square S/H/P 的真实 held-out physical rollout active-parameter hook；只有在论文要报告 active counts 时才需要补做。
4. 原版/修正版 H/P 的逐 seed checkpoint 表和失败原因可另作实验稳健性附录，不应混入基础架构表。

## 14. 主要证据索引

- 总审计：`C:\Users\panxy1019\Documents\CHANNEL\RAL_MOE_ROM_ICLR2027\network_architecture_audit.md`
- 五个 sparse 模型结构：`C:\Users\panxy1019\Documents\CHANNEL\RAL_MOE_ROM_ICLR2027\build\network_architecture_audit_20260912\instantiated_models.json`
- Circular active 计数：`...\build\network_architecture_audit_20260912\parameter_counts.json`
- Pinball 实例化：`...\build\network_architecture_audit_20260912\pinball_instantiated.json`
- Pinball 候选身份：`...\build\network_architecture_audit_20260912\pinball_candidates.json`
- 内部 routing 实证：`C:\Users\panxy1019\Documents\CHANNEL\iclr_expert_routing_analysis\logs\hopf_routing_verification.json` 与 `periodic_routing_verification.json`
- RK stage：`C:\Users\panxy1019\Documents\CHANNEL\iclr_expert_routing_analysis\stats\rk4_stage_sensitivity.csv`
- Square candidate hashes：`C:\Users\panxy1019\Documents\CHANNEL\centeredsquare_fusion_v1\results\E2_T2C_K24_20260730_STRICT_V3\cache_sh\PREFLIGHT.json` 与 `cache_hp\PREFLIGHT.json`
- Pinball final-test seal：`.../Pinball/fluidic_pinball_fusion_v1/runs/E2_T2C_K24_20260806_STRICT_V1/full_rollout_20260807_V1/results/FINAL_TEST_SEAL.json`
- quadratic 修正版审计：`C:\Users\panxy1019\Documents\CHANNEL\round2_20260917\第二轮实验完整审计与论文整合建议_20260920.md`

## 15. 完成判定

跨案例结构、MoE/Deep-FNN 分族、部署例外、参数口径和 formal outer-candidate 身份已经整理完成。当前真正缺失的是 Circular S、Pinball 部分独立 autonomous 行以及 Square active trace；这些均已作为缺失项保留，没有为追求九格齐全而复制 Circular 配置或虚构 MoE 参数。
