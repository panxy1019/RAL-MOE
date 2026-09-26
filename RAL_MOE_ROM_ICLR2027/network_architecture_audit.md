# RAL-MoE-ROM 网络结构与训练配置审计

日期：2026-09-12。范围：提取、实例化、核验与生成独立表格；**没有修改论文、没有重新训练、没有修改 checkpoint**。

## 结论先读

1. **PAPER-CODE MISMATCH：centered-square T2-C 是 8 个历史描述通道 + 1 个参数，共 9 维输入，不是 7 维。** 原始 SH/HP checkpoint 第一层均为 `[64,9]`；12 个 full/mu-only 三种子 checkpoint 也全部如此。网络 `9→64→64→1`，每个 4,865 个 trainable parameters。
2. Circular Hopf / Periodic 已重新实例化并核对完整 state_dict shape，total trainable 分别为 **31,810,392 / 48,617,547**；预测路径 active 分别为 **14,165,816 / 8,308,959**，与 Table 13 一致。不是把 checkpoint 所有 tensor 的元素数直接当作 trainable。
3. 各 chart/benchmark **不共享统一配置或训练课程**。Circular Hopf 是 1 组、256 宽、4 个 1024 扩展专家块；Circular Periodic 是 3 组、224 宽、3 个 768 扩展块。两者训练课程分别为 `1,2,4,8` 与配置预算 `4,8,12,16`。
4. 审计还发现了比输入维数更重要的实现差异：pressure `adaptive_gate` 实际为 **gamma×PP + learned residual**，不是论文所写 `gamma×PP + (1−gamma)×data candidate`；centered-square Hopf 的最终融合 evaluator 使用 data-only 输出路径；centered-square Periodic 使用固定组 ID 1。这些均在修订清单中标为 **PAPER-CODE MISMATCH**，本轮未擅自修正文稿或重跑结果。
5. Pinball Periodic 的两条 Table 2 数值与发现的 **Deep-FNN-H3** 评估输出相符，而不是稀疏 MoE。记录中的 checkpoint 路径与本次检查路径相同，但该 summary 未提供历史 checkpoint hash，因此证据级别是“数值与路径关联”，不升级为历史字节级证明。
6. 不能可靠关联最终实验的 Circular Steady、Vanilla/DataOnly/Global MoE，以及未核验的外层 run-to-paper 映射，均明确为 **UNRESOLVED**，未用旧默认值填入确认表。

## A. Source of truth 与证据等级

工作区根目录：`C:/Users/panxy1019/Documents/CHANNEL`。
远端 `REMOTE`：`/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy`。
本次证据目录：`RAL_MOE_ROM_ICLR2027/build/network_architecture_audit_20260912`。

- **CHECKPOINT + EVALUATION HASH**：本次读取实际 checkpoint，重算 SHA-256，与既有最终评估/preflight 的 hash 相同；结构逐 tensor 核验。只能支撑被关联的评估，不自动证明同一 benchmark 的所有表格行。
- **CHECKPOINT + SOURCE**：实际 checkpoint 与源代码结构匹配，但最终论文行身份尚未闭环；只能列为候选或正式 run 的配置，不能混入已确认主表。
- **SOURCE ONLY**：找到了实现，但没有对应最终 checkpoint/config；具体最终数值标 `UNRESOLVED`。

### A1. 有评估 hash 关联的五个 specialist

| 对象 / 已关联范围 | REMOTE 下的 checkpoint | 配置与实现来源 |
|---|---|---|
| Circular Hopf，internal-routing / Appendix J | `particalMOE/Hopf/migrated_h4_expanded/final_evaluation/20260722_h4_expanded_final/HopfExpanded34_H4_NormalFormRadial_r32/final.pt` | checkpoint `args`；`train_h4_expanded.py:211` 的实际 model args；`train_hopf_moe_expanded.py:440` |
| Circular Periodic，internal-routing / Appendix J | `particalMOE/periodic_specialist_r32/checkpoint/FINAL_PERIODIC_SPECIALIST.pt` | checkpoint `args`；`train_periodic_moe.py` |
| Square Steady，SH 融合候选 | `particalMOE/centeredsquare_steady_specialist_v1/checkpoint/frozen_centeredsquare_steady_step3200.pt` | checkpoint `config`；`training_centeredsquare_steady_rank999.json`；`train_s2b_3090.py` |
| Square Hopf，SH/HP 融合候选 | `particalMOE/CenteredSquare_Hopf_H4_20260728/final_evaluation_mb46_dataonly/selected_validation_step7200.pt` | checkpoint `args`；`train_centeredsquare_h4.py`；最终 `build_boundary_cache.py:64–109` 调用链 |
| Square Periodic，HP 融合候选 | `particalMOE/centered_square_periodic_v1/runs/optimized_long_seed1600/best_validation.pt` | checkpoint `args`；`train_square_periodic_moe_optimized.py`；`build_boundary_cache.py:233–275` |

五个 checkpoint 的完整路径、SHA、所有 state key/shape/dtype、配置及 scheduler 状态已保存于 `remote_checkpoints.json` 与交付的 `resolved_architecture.json`。路径型配置此次保存了完整路径，不再仅保留 `PosixPath` 类型。

Circular 的 hash 关联来自 `iclr_expert_routing_analysis/logs/checkpoint_inventory.json` 与 routing verification；Square 来自 `centeredsquare_fusion_v1/results/E2_T2C_K24_20260730_STRICT_V3/cache_sh|cache_hp/PREFLIGHT.json`。因此本报告不把 Circular 的 routing checkpoint 擅自等同为尚未逐行恢复的 Table 4 全部训练身份。

Square 三个训练源码本地与远端 live SHA 一致。Square Hopf 依赖的 `periodic_moe_3090/train_periodic_moe.py` 与 Circular Periodic 源码 SHA 同为 `500c2179734e6c3a6db6469629d4727637ae4552843b4d3154ca7b281a102314`，不是仅凭同名替换依赖。

### A2. 外层路由与 gate

Center-square 原始路径：工作区 `centeredsquare_fusion_v1/results/E2_T2C_K24_20260730_STRICT_V3/{e2_router,gate_sh,gate_hp}/best.pt`。

- E2 SHA：`6d9e62729db1fd8e6c39686b582e7d451ceb79b58eb14637a34874c66aea1dac`；best step 3300，seed 20260730。
- SH gate SHA：`732059a2cf19acf6b4681590fa2a8a43ad4babdc2c5814638ed438c1dc5cafb0`；best step 5500，seed 42001。
- HP gate SHA：`fb154d2d3decb68c59b242a17c95c062e24f4443f5034692ec0abfaa55e9c887`；best step 3500，seed 42001。
- 三种子：论文目录 `experiments/iclr_seed_study_20260911/results_v1/centered_square/{sh,hp}/{full,mu_only}/seed_{42001,42002,42003}/best.pt`；各自 `config.json` 是实际解析配置，非默认值。

另检查了 Pinball 正式 outer run：`REMOTE/Pinball/fluidic_pinball_fusion_v1/runs/E2_T2C_K24_20260806_STRICT_V1/{e2_router,gate_sh,gate_hp}/best.pt`。E2 第一层 `[24,1]`、123 个 state elements；两个 gate 第一层 `[64,9]`、4,865 个 state elements，源类结构与 Square 相同。此 run 独立于 Square，不能理解为跨 benchmark 共享权重。与 Appendix H 各最终评估行的完整映射仍为 **UNRESOLVED**，不能直接把其训练设置推广到该表全部行。

### A3. 候选 / 未闭环模型

Pinball 三个 B1 checkpoint 的绝对路径和完整配置在 `pinball_candidates.json`：Steady `FluidicPinballV2_Steady_rank999_B1_seed1248/best_validation.pt`，Hopf `FluidicPinballV2_Hopf_rank999_B1_seed1248/best_validation.pt`，Periodic `FluidicPinballV2_B1_Deep_FNN_H3_seed1248/best_validation.pt`。它们均为正式 runs，未纳入 smoke，但“正式”不等于“已证明产生指定论文行”。

Circular Steady 的 `frozen_s4_validation_step_1200.pt` 不作为 Table 4 Steady 真值；没有用其配置替代论文模型。Vanilla/DataOnly 的 launcher 只在候选源码部分解释，Global MoE 没有估算参数量。无可核验 git commit，未编造 commit。

## B. Architecture extraction

### B1. 已核验结构汇总

`h` 为 encoder 宽度，`m` 为专家 FFN 扩展宽度，`B` 为专家 residual block 数；每个通道均有独立的专家集合。

| 模型 | ru/rp | 特征输入 D | h | 专家 m/B | groups | 每组每通道 routed/shared | total trainable |
|---|---:|---:|---:|---:|---:|---:|---:|
| Circular Hopf | 32/32 | 493 | 256 | 1024/4 | 1（router frozen） | 6/1 | 31,810,392 |
| Circular Periodic | 32/32 | 501 | 224 | 768/3 | 3 | 6/1 | 48,617,547 |
| Square Steady | 5/4 | 91 | 224 | 768/3 | 3 | 6/1；velocity experts frozen | 23,746,370 |
| Square Hopf | 11/11 | 178 | 256 | 1024/4 | 1（router frozen） | 6/1 | 31,205,844 |
| Square Periodic | 28/26 | 431 | 224 | 768/3 | 3，部署选择固定 ID 1 | 6/1 | 48,209,967 |

五个模型使用各自源码类实例化，**完整 state_dict key、shape、dtype 与 live checkpoint inventory 逐项一致**。这次本地实例化不载入大型远端预训练权重：用于结构/参数计数，不宣称重放了真实流场推理。Square trainable flags 按最终训练源码显式复原；freeze 不能从 state_dict 自身推断。

### B2. 输入、encoder 与 normalization

基础特征来自 `train_periodic_moe.py:1047–1100`，再由 `make_history_features_from_states_*` 添加历史：

- parameter：归一化 Re、倒数 Re 两槽；不是仅一个裸 scalar。
- 当前状态：a、b、projected Galerkin RHS。
- 11 个派生量：低/中/高模态范数、压力/RHS 范数、a/b/总能量、低/高能量占比、压力/速度范数比。
- 有 phase-harmonics=4 的模型保留 8 个正余弦槽；Hopf 设置 phase_harmonics=0，无这些槽。本轮不把“存在特征槽”推断成“使用未来物理真值”。
- history_len=3，除当前状态外两个 lag；各 lag 添加 a/b/RHS 和相对当前值的差。因此 D=`13+2H+5(2ru+rp)`，分别给出上表 493、501、91、178、431。
- 输入及目标使用训练集拟合的均值/尺度；Hopf checkpoint `norm_stats`，Periodic checkpoint `scalers`，Square Steady 的训练专属 scaler/配置。具体数组 shape 在 JSON；零尺度下限按各源码处理，不人为统一 epsilon。

共用的 encoder 是 **D→h→h→h**：三个 Linear 均带 bias，每层 LayerNorm、SiLU，前两层后 dropout=0.04；再接 **2 个** refinement residual blocks，每个 `LN(h)→Linear(h,h)→SiLU→Dropout(.04)→Linear(h,h)`，输出 `h + block(h)`。`num_blocks=3` 并不意味着 3 个 refinement blocks。

Encoder/普通隐藏 Linear 没有另一个统一自定义初始化覆盖时采用源码所调用的 PyTorch 层默认初始化；不能由训练后权重推回初始化样本。专家与压力输出有额外初始化，见下。

### B3. Structured expert：不是普通两层 MLP

实际源码 `PhysicsAwareExpert`（`train_periodic_moe.py:1430–1495`）。输入由编码 h 与**标准化输入中的局部 state slice**组成，不是所有 raw feature 都进入线性/二次分支：velocity state 维 ru；pressure state 维 ru+rp。

1. Nonlinear：`LN(h+s)→Linear(h+s,h)→GELU`；B 个 residual expanded FFN blocks；末端 `LN(h)→Linear(h,out)`。
2. 每个 expanded block：`z + rho · [Linear(m,h) Dropout(GELU(Linear(h,m) LN(z)))]`，按源码顺序在两处加 Dropout(.04)；rho 是可训练 scalar，初值 0.1。
3. Linear：`Linear(s,out,bias=False)`，没有独立 affine bias。velocity 权重 `[ru,ru]`；pressure 权重 `[rp,ru+rp]`。
4. Bilinear：`quad_left` 与 `quad_right` 均为 `[out,4,s]`；每个输出分别计算 `sum_j (U_oj state)(V_oj state)` 后乘 0.05。**没有独立的 W_o tensor 或 bilinear bias**。不能把论文的共享 rank-4 latent + W_o 写法直接当作这一参数化。

| 模型 | u 输入投影权重 | p 输入投影权重 | u 的 U/V 各自 shape | p 的 U/V 各自 shape |
|---|---|---|---|---|
| Circular Hopf | 256×288 | 256×320 | 32×4×32 | 32×4×64 |
| Circular Periodic | 224×256 | 224×288 | 32×4×32 | 32×4×64 |
| Square Steady | 224×229 | 224×233 | 5×4×5 | 4×4×9 |
| Square Hopf | 256×267 | 256×278 | 11×4×11 | 11×4×22 |
| Square Periodic | 224×252 | 224×278 | 28×4×28 | 26×4×54 |

Constructor 初始化 linear/head Xavier gain=.25，quad normal std=.015；但正式物理零残差初始化进一步把 linear weight、最终 head weight、quad U/V 置零，并设置去标准化补偿 bias。因此报告初始化时应写**最终训练入口的覆盖结果**，而非只抄 constructor。Square Steady 也把专家终端和 bilinear 清零；velocity 专家随后冻结。隐藏投影、FFN 未全部清零。

### B4. 内层 routing

各组/各 channel router 输入 `concat(h,x)`，维 D+h；网络 `LN(D+h)→Linear(D+h,h)→SiLU→Dropout(.04)→Linear(h,G 或 6)`。所有 Linear 带 bias，u/p routers 不共享参数。

| 模型 | router input | group 输出 | group temperature | channel temperature | group 行为 |
|---|---:|---:|---:|---:|---|
| Circular Hopf | 749 | 1 | 1.0 | .8 | 单组，参数冻结 |
| Circular Periodic | 725 | 3 | .9 | .8 | learned Top-1 |
| Square Steady | 315 | 3 | .9 | .95 | learned Top-1 |
| Square Hopf | 434 | 1 | 1.0 | .8 | 单组，参数冻结 |
| Square Periodic | 655 | 3 | .9 | .8 | fixed_regime_group=1，straight-through |

Channel 为每组 6 个 routed 中 Top-2；每组每 channel 1 个 shared。不是 velocity 与 pressure 共用“一个”专家。Circular Periodic 每 channel 共 18 routed + 3 shared，和 internal-routing 中 velocity 使用 18 个 ID、pressure 使用 14/18 个 ID 一致；未使用的 ID 不等于不存在。

`topk_router_from_logits`：softmax(logits/T)，保留 Top-k，按保留质量重新归一化；Top-1 使用 straight-through `hard - probs.detach() + probs`；这里 gate floor 均为 0。未发现额外 additive noisy-routing 实现；dropout 不应称为 noisy top-k。

Shared/routed 权重：常规 1.0/.75，经二者和归一化后为 4/7、3/7；Square Steady 1.0/.85，即约 .54054/.45946。普通模型即使组未选中也计算其 channel routers。Square optimized 模型有专门 fixed-group/batched 路径，不套用 Circular 的所有-group dispatch 计数。

Balance、entropy、diversity 正则并非统一：Periodic checkpoint 保存 lambda_router_balance=.06、lambda_router_entropy=−.002、lambda_group_balance=.04、lambda_group_entropy=0、router_smooth=.04、expert_diversity=.006。完整其他 lambda 在 JSON。Hopf H4 `common_loss` 使用 .02 balance、−.002 entropy、.01 diversity，另有动态梯度配比的 fluctuation/normal-form loss。

### B5. Pressure branch：必须区分“参数存在”和“输出实际使用”

常规 sparse MoE 共用同一 encoder/refinement，pressure 专家输出 rp，state 输入为 a,b。Adaptive gate head `h→64→rp`，GELU、最终 sigmoid、componentwise；最后一层 weight=0，bias=6，初始 sigmoid(6)。

- Circular H/P：pressure `gamma⊙b_PP + residual`。静态 PP 算子不是 NN trainable parameter；FiLM base 模块在 `pressure_base_mode=static` 下不实例化。
- Square Steady：`closure_mode=baseline`；存有 `224→56→2` confidence head，但冻结且 baseline closure 跳过它。pressure 使用物理 base 加残差；不能报告为一个训练中的 componentwise sigmoid head。
- Square Hopf：存有 `256→64→11` gate，但 `train_centeredsquare_h4.py:91–94` 的实际 `make_ops` 不用 PP/gamma 组装输出：velocity 是直接预测的 RHS，pressure 为直接 learned output。Galerkin 仍出现在输入特征，不等于作为加性 RHS backbone 使用。最终融合 `build_hopf_runtime` 调用的就是此 `make_ops`，而非 base trainer 里另一条 `g + residual` 实现。
- Square Periodic：`224→64→26`，实际采用 adaptive gate；除固定组之外还有 8 个 RK 子步、max_integrator_dt=.5 的实现设置。

**PAPER-CODE MISMATCH**：当前 Appendix C 的 convex pressure candidate 公式与上述 residual 参数化不同。理论上重新定义 candidate 可以产生某种代数表示，但没有源代码里的 `(1−gamma)` 缩放，不能在审计中把二者标成 MATCH。

### B6. E2 outer router

Center-square 严格 **scalar Re only**，先用训练 Re 均值/标准差标准化，`1→24→3`，隐藏 Tanh，预测后 softmax，123 trainable。没有读取 physical history 作为 E2 输入。

训练：AdamW lr=3e−4、weight decay=1e−4；CosineAnnealingLR T_max=8000、eta_min=1.5e−5；batch=110（DATA_AUDIT的train trajectory rows为110，代码取min(256,Ntrain)），有放回采样；clip norm=1；每100步验证（另含step1/final）；实际 history 到8000，best3300，seed20260730。

Loss=class-weighted cross entropy + .05 mean(pi_S pi_P)。选择分数为 validation balanced accuracy + macro F1 − .001×**当前训练 batch loss**；不是纯 validation CE，亦不是简单最小化 validation loss。没有据现有源码推断成另外的选择器。

Pinball 正式 outer run 独立 checkpoint 也确认 `1→24→3`、Tanh、123；源码是 Re-only；best8000、seed20260730。具体 Appendix H 全部结果身份及最终实际 batch 等仍未闭环。Circular E2 这轮未核验最终 checkpoint：UNRESOLVED。

### B7. T2-C gate 与八个 descriptor

源码 `centeredsquare_fusion_v1/common.py:109–147`；输入拼接 `build_boundary_cache.py:509`。令三个时间从旧到新为 t0,t1,t2，u_i 为速度，p_i 为 gauge-corrected pressure，Eu_i/Ep_i 表示此处的场平方范数（注意：这里的能量缩写不是论文百分比误差），A 为总面积。

| 顺序 | 实际通道（按源码操作） |
|---:|---|
| 1 | 0.5 log(Eu_2/A) |
| 2 | 0.5 log(Ep_2/A) |
| 3 | log1p(当前速度增量范数/Δt_2/当前速度范数) |
| 4 | log1p(前一速度增量范数/Δt_1/前一速度范数) |
| 5 | tanh(growth_2)，growth 为 log velocity-energy 的时间差分 |
| 6 | tanh(growth_2−growth_1) |
| 7 | log1p(当前压力增量范数/Δt_2/当前压力范数) |
| 8 | log1p(相邻两段速度变化率之差的范数/当前速度范数) |

表中省略了源码里的数值 clamp/EPS 以便阅读；精确定义以 common.py 对应函数为准。通道8没有再除以平均时间间隔，不能直接称为论文现有的二阶时间导数归一化公式。需要修正的不是仅 6→8，还包括前一速度增量、增长差和实际变换定义。

9维 `[Re; descriptors]` 用 development cache 中 train rows 的逐通道 mean/std 标准化，std<1e−8 置1；checkpoint 保存 mean/std。MLP `Linear(9,64)→SiLU→Linear(64,64)→SiLU→Linear(64,1)`，都有 bias，无 dropout、LayerNorm、residual。output weight/bias zero initialized。最终 `alpha=sigmoid(logit(pi_selected_first)+MLP(x))`，sigmoid 不在 MLP 内。初始输出等于 E2 选中 pair 的归一化首分量概率；代码先 clip 到[1e−6,1−1e−6]。

所有 checkpoint 第一层 key=`gate_state.correction.net.0.weight`，shape `[64,9]`。参数数目=9×64+64 +64×64+64 +64×1+1=**4865**。原始与三种子总共14个gate均逐个 strict load 成功。

训练：AdamW lr3e−4、weight decay1e−4；8,000 steps；SH batch56、HP batch32，均有放回；clip norm1；无 scheduler；每100步/step1/final验证；无 early-stop 分支。loss 为 K24 所有步与 batch 上 velocity、pressure 的**平方相对场误差之和**；selector 为 lexicographic `(validation worst_Re_mean, validation joint_mean)`，选择指标使用开方后的分量相对误差。一个 scalar alpha 固定整个窗口，u/p 共用；不把融合结果反馈入两个原生 rollout。

mu-only：完全同架构及参数数目，`features[:,1:]=0` 在标准化之后；matched seeds、optimizer/lr/weight decay、batch、steps、clip、验证和 selector 全部逐配置比较一致。Full 与 mu-only 不应标成“共享训练后权重”。

| overlap | seed | full best step | mu-only best step |
|---|---:|---:|---:|
| SH | 42001 | 5500 | 1 |
| SH | 42002 | 7100 | 1 |
| SH | 42003 | 5900 | 1 |
| HP | 42001 | 3500 | 400 |
| HP | 42002 | 3500 | 400 |
| HP | 42003 | 3200 | 400 |

### B8. Specialist training：预算与选中 step 必须分开

| 模型 | optimizer / initial lr / wd | batch | 配置预算 / selected | curriculum | validation |
|---|---|---|---|---|---|
| Circular H | AdamW / .001 / .0001 | 232×1 | 8000 steps / 7200 | 1,2,4,8 | 200；long400 |
| Circular P | AdamW / .00055 / .00015 | one-step256；rollout2 | 240 epochs /85 | 4,8,12,16（完整预算） | 每5epoch |
| Square S | AdamW /1.5516372391e−5 / .0001 | 16×4=64 | 4000 steps /3200 | 1,4,8,16 | 200 |
| Square H | AdamW / .001 / .0001 | 46×1 | 8000 steps /7200 | 1,2,4,8 | 200；long400 |
| Square P | AdamW / .00055 / .00015 | one-step1024；rollout56,56,40,28 | 720 epochs /490 | 4,8,12,16；阶段120,160,160,280epochs | 每10epoch |

**selected checkpoint 在预算结束前选出，不能据预算声称该权重已经历全部课程或全部训练轮数。** Square P 为 resumed run；checkpoint `resume_checkpoint` 指向 latest.pt；完整训练历史的重新拼接不在本次审计中。

- Scheduler：上述均 CosineAnnealingLR；H T_max8000 eta_min0；Circular P T_max240；Square P T_max720；Square S T_max4000 eta_min=7.758186e−7，另有 encoder/refine/router 0.1×lr 参数组。
- 所有五个训练入口 clip norm1。H 使用 CUDA float16 AMP + GradScaler，weights float32，TF32 allowed；Square P `amp_mode=bf16`；Square S 按设备 bf16 支持情况选择 bf16/fp16，checkpoint scaler为空，不把设备营销名称推成确认GPU型号。Circular P 当前训练入口无 AMP 上下文，state float32、CUDA、TF32 allowed。
- Seeds：Circular H1248，Circular P1600，Square S202607281，Square H1248，Square P1600。
- H teacher forcing：主状态 rollout 递归使用预测值；normal-form 辅助目标读取训练真值构造 loss，不等于推理 teacher forcing。Periodic scheduled_sampling_start=end=1，源码 `a_feed=p*a_next+(1-p)*target_a`，因此 p=1 是全预测反馈，而非全 teacher forcing。
- H selection：满足 hard safeguard 后，以 Hopf fluctuation/radial/growth score 选择，相对改善1e−3；early-stop patience8 evals，具体组合见 H4 `validate` / selector 调用。Square S validation pressure/fixed-point safeguard、patience4、early_stop_start3200、相对改善.01。Periodic score=`rhs_relative_l2 + alpha_head_relative_l2 + .35 pressure_head_relative_l2`，min delta=.001；patience Circular70/Square180，min_epochs180/561。不能把所有模型写成统一的“最小总场误差”。
- Loss：所有 checkpoint args/config 中的 lambda、scale-floor、sampling、AMP、optimizer/scheduler 状态均完整保存在 JSON。H4 `common_loss` 还显式含 normalized coefficient/dynamics、pressure、relative rollout、energy、router balance/entropy、diversity，加上按梯度比例自适应加权的 fluctuation/normal-form loss。论文仅用一个统一 rollout MSE 公式不能被读成完整实际目标。

## C. Baseline / Pinball：没有最终身份就不合并

### C1. Pinball B1 候选（实际 checkpoint shape 已核验）

| 模型 | ru/rp | 输入 | trunk | heads u / p / gate | 实例化 trainable |
|---|---|---:|---|---|---:|
| Steady B1 | 4/3 | 68 | 6×512，LayerNorm+SiLU | 512→256→4 /512→256→3 /512→128→3 | 1,684,370 |
| Hopf B1 | 5/5 | 88 | 6×512，LayerNorm+SiLU | 512→256→5 /512→256→5 /512→128→5 | 1,695,679 |
| Periodic B1 | 17/16 | 263 | 6×512，LayerNorm+SiLU | 512→256→17 /512→256→16 /512→128→16 | 1,792,959 |

无 MoE router/experts。Trunk 开头 LN(input)→Linear(input,512)→SiLU，随后5个 Linear512→LN512→SiLU；head 的隐藏 SiLU，gate 最终sigmoid，Linear均带bias，无dropout。velocity/pressure final weight/bias清零；gate final weight0、bias log(.99/.01)。

三者保存的 lr=.001、wd=.0001、AdamW、seed1248、max8000、clip1、AMP bfloat16；batch为256/2048/32。训练文件里 token_dim、num_heads 等属于其他 variant 的共用 parser 参数，**B1 模型并不因此具有 attention/KDA**。课程：Steady4,8,16；Hopf4,8,16,32,56；Periodic4,8,16,24,32。仅为检查到的 B1 配置，不宣称论文全部行已映射。

Periodic final-test summary 的 K32 `(Eu,Ep,joint)×100=(.28969518,.55180013,.84149532)`，K56 `(.62215413,1.04964835,1.67180243)`，分别舍入为当前 Table 2 对应两行；summary checkpoint 路径也是该 B1 文件。这是需要作者处理的 **PAPER-CODE MISMATCH / 方法身份差异**，不能通过把表中网络结构硬填成 MoE 掩盖。Hopf找到的 summary 明示 `heldout_loaded=false`，不能拿它冒充 Table 2 的 test 配置闭环。

### C2. Vanilla-FNN-MoE：SOURCE ONLY

`iclr_expert_routing_analysis/source/paper_experiments/code/run_vanilla_fnn_moe.py:78–136` 只替换专家类，保留原 trainer/router。专家输入 concat(h,state)，plain Linear→GELU→Dropout 重复 `max(2,num_blocks+1)` 次后输出 Linear；没有 residual/显式linear/bilinear 分支。宽度由 `matched_width` 按原专家目标参数数目求近似整数，不是简单把原 expert_hidden 原样填入 FNN。它是局部近似匹配意图，不足以证明最终全模型 parameter-matched。最终各 chart FNN width、参数量、checkpoint identity：**UNRESOLVED**。

### C3. DataOnly-MoE：SOURCE ONLY

`train_dataonly_moe.py:227–300,359–363,552` 的候选实现为直接离散 POD-state mapping，去掉 RK4/Galerkin/pressure physics；输出64，对应拼接32+32。Encoder 两层 Linear+LN+SiLU；group/local router 为简单 Linear；专家为 PlainExpert，u/p 并非沿用原来两个独立 structured expert 集合。源码即使给稀疏 gate 也计算 routed_stack 中所有专家，不能将 Top-2 当成执行参数稀疏。若确为 Table4 最终实现，差异远多于去掉 backbone；但本次未找齐最终 checkpoint/config，不宣布已确认论文行使用它。最终参数量：**UNRESOLVED**。

### C4. Global MoE / Circular Steady

Global POD rank、history、groups、expert size、total/active count：**UNRESOLVED**。Circular Steady Table4 的最终权重身份也 **UNRESOLVED**。这些字段在交付 LaTeX 中没有填成猜测数值。

## D. Paper-code consistency check

| Item | 当前论文 | 代码 / checkpoint | Status |
|---|---|---|---|
| T2-C 输入 / descriptor | 7 /6 | Square final 9 /8 | **PAPER-CODE MISMATCH** |
| T2-C hidden/activation/output | 64,64 /SiLU /1 | 相同 | MATCH |
| T2-C final zero init | zero | constructor先zero，训练权重不要求仍zero | MATCH |
| E2 history | parameter-only | Square及检查的Pinball outer均Re-only | MATCH（限定已查run） |
| history length | 3 states | 五个确认模型均3 | MATCH（限定范围） |
| q_b / kappa | 4 /.05 | 五个确认模型为4 /.05 | MATCH（不推广到FNN） |
| bilinear 参数化 | U,V,W_o 形式 | output-specific U/V `[out,4,state]`，无W_o | **PAPER-CODE MISMATCH** |
| group Top-1 | 统一learned selection | Hopf单组frozen；Square P固定组1 | **PAPER-CODE MISMATCH**（泛化措辞） |
| channel Top-2/shared | Top2+shared | 确认MoE模型每channel每组6 routed+1shared | MATCH |
| pressure assembly | gamma PP+(1−gamma)data | adaptive_gate 为gamma PP+residual；部分chart另有例外 | **PAPER-CODE MISMATCH** |
| universal projected+residual RHS | 各specialist统一 | Square H最终make_ops直接learned RHS | **PAPER-CODE MISMATCH** |
| universal curriculum | 4,8,12,16 | H:1,2,4,8；Square S:1,4,8,16等 | **PAPER-CODE MISMATCH** |
| Table13 H/P counts | 31.81M/14.17M；48.62M/8.31M | exact counts一致 | MATCH |
| Pinball Periodic模型身份 | RAL-MoE-ROM | 数值/路径关联到Deep-FNN-H3 | **PAPER-CODE MISMATCH**（历史hash链未完整） |

## E. 参数量核验的方法与边界

- 本次读取远端五个最终 checkpoint 的真实 SHA/state shape，并用真实源码类本地实例化。sum(p.numel() for p in model.parameters() if p.requires_grad) 重新计算。
- Hopf单组 router 参数虽存入 state_dict，却 frozen；Circular H 排除193,755个非trainable，Square H排除112,485。H4 normal_form 是独立 training-loss-only head，不并入部署模型；Circular H该辅助head为10个trainable。
- Square S 总 state elements46,655,195，其中22,908,825 frozen；trainable=23,746,370。不能把46.66M称为其trainable。
- Circular active 重跑 `scripts/count_trainable_parameters.py`：按参数对象 identity 去重，trace 实际 module forward 的直接参数；包括 encoder/refinement、所有实际执行的 router、selected group shared、u/p Top2、pressure head。不是把父模块所有 descendant 一概计入。
- 全部225 Hopf /675 Periodic可能 group/u-pair/p-pair support 的组合计数一致，synthetic dispatch覆盖每组与每个pair；得到14,165,816/8,308,959。native Hopf diagnostic stack为22,953,160，不能当预测路径计数。
- 本次 active probe 是源类 + 记录结构的合成输入/强制路由检查，不是再次运行已训练物理 rollout。真实权重不影响这些架构等大小支持集的参数数目，但不据此声称验证了新的路由频率或精度。
- Square 三个模型只复核 total/frozen/shape；尚未独立 trace 它们完整实际 physical rollout path，故其 active count **UNRESOLVED**。尤其 Square H gate虽计算却不参与最终输出，需先确定“执行”还是“影响最终预测”的计数定义，不能套用Circular。
- counts不是FLOPs、runtime或speedup；未以此宣称完整baseline容量匹配。

## F. 交付、复现与验收

交付文件：`network_architecture_audit.md`、`network_architecture_tables.tex`、`paper_revision_notes.md`、`resolved_architecture.json`。LaTeX为独立可插入片段，尚未自动加入main.tex；含确认结构/训练表及清晰的适用范围，避免把候选混成统一配置。

证据目录还包含：`local_checkpoints.json`、`remote_checkpoints.json`、`instantiated_models.json`、`parameter_counts.json`、`pinball_candidates.json`、`pinball_instantiated.json`、8份 `*_model.txt`、完整key/shape/dtype/模块参数列表及源码快照。配置中的所有lambda等长字段保存在机器文件，不因主表简洁而丢弃。

脚本：`extract_network_architecture.py`（真实checkpoint只读提取）、`instantiate_audited_architectures.py`（源码模型及shape验证）、`count_trainable_parameters.py`（active重算）、`collect_additional_architecture_sources.py`（补充Pinball证据）、`finalize_architecture_audit.py`（对照与不变性检查）。本地 PyTorch 使用项目已有 `build/parameter_count_runtime`，未另行训练。

自动检查34项覆盖：14gate输入/参数、paired配置一致、五个模型全部state shape、Circular total/active、live源码hash及31个论文上传文件hash保持不变。**PASS只针对这些明确检查，不代表UNRESOLVED项已解决。**

六张LaTeX表格片段已用5.5英寸正文宽度独立编译并逐页目视检查，3页预览，无overfull、缺字或未定义引用。预览只是表格质量检查，不是整合后的投稿PDF。审计使用项目执行管理与代码核验技能保持修改范围，PDF技能用于表格排版验证。

最终仍需恢复：Circular Steady/Table4各baseline完整模型身份；Pinball各论文行的历史checkpoint完整hash链；Circular E2；Square active完整物理路径；各run最终实际停止时刻/硬件型号中没有日志佐证的字段。没有把训练预算或parser默认值写成已执行事实。
