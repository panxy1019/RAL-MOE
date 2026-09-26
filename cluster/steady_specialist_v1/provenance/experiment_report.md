# V17 S4-Steady Anchor-Only 最终客观实验报告

## 执行与选择

训练按用户指令停止于最后记录 optimizer step 3480；最后完整 validation 为 step 3400。Heldout 在 validation 选择冻结后执行，成功指标产物仅生成一次，且未参与选择。

冻结 checkpoint：step 1200，SHA256 `bcab5661af39e3262103ed887ce9846415467452b453f3dcc14e133f0f8bd80d`。该 checkpoint 通过预注册保护门；选择规则为在所有保护门通过点中最小化 validation K16 physical pressure fixed-point residual。

## Validation 历史

执行审计：heldout evaluator 共启动 2 次，首次在全部 rollout 完成后的纯汇总阶段因浮点尺度精确匹配失败，未产生指标或选择；修正为容差匹配后，成功产物仅生成一次。两次均使用同一冻结 checkpoint 与同一 perturbation bank。

| step | 保护门 | K16 压力 FP worst | K56 压力 FP worst | K16 压力 gain P95 | 失败原因 |
|---|---|---|---|---|---|
| 0 | PASS | 17.3414% | 13.6524% | 5.344 | — |
| 200 | FAIL | 6.6569% | 18.7741% | 43.685 | k16_pressure_gain_p95_gt_5pct, k56_pressure_gain_p95_gt_5pct |
| 400 | FAIL | 9.9673% | 19.5416% | 6.004 | k16_pressure_gain_p95_gt_5pct, k56_pressure_gain_p95_gt_5pct |
| 600 | FAIL | 11.8017% | 14.1106% | 38.709 | k16_pressure_gain_p95_gt_5pct |
| 800 | FAIL | 13.3142% | 13.3372% | 15.723 | k16_pressure_gain_p95_gt_5pct, k56_pressure_gain_p95_gt_5pct |
| 1000 | FAIL | 13.9743% | 14.0505% | 5.235 | clean_k56_velocity_relative_l2_gt_5pct |
| 1200 | PASS | 14.0292% | 13.9000% | 5.168 | — |
| 1400 | FAIL | 13.6873% | 13.6950% | 5.808 | clean_k56_velocity_relative_l2_gt_5pct, k16_pressure_gain_p95_gt_5pct, k56_pressure_gain_p95_gt_5pct |
| 1600 | FAIL | 14.0239% | 14.0110% | 4.865 | clean_k56_velocity_relative_l2_gt_5pct |
| 1800 | FAIL | 14.2337% | 14.1657% | 5.697 | clean_k56_velocity_relative_l2_gt_5pct, k16_pressure_gain_p95_gt_5pct |
| 2000 | FAIL | 13.3666% | 13.3206% | 5.395 | clean_k56_velocity_relative_l2_gt_5pct |
| 2200 | FAIL | 13.7940% | 13.8729% | 5.013 | clean_k56_velocity_relative_l2_gt_5pct, k56_pressure_gain_p95_gt_5pct |
| 2400 | FAIL | 13.6116% | 13.6080% | 5.657 | clean_k56_velocity_relative_l2_gt_5pct, k16_pressure_gain_p95_gt_5pct |
| 2600 | FAIL | 13.4613% | 13.4246% | 5.417 | clean_k56_velocity_relative_l2_gt_5pct |
| 2800 | FAIL | 13.6968% | 13.6626% | 5.445 | clean_k56_velocity_relative_l2_gt_5pct, clean_k56_pressure_relative_l2_gt_5pct |
| 3000 | FAIL | 13.8339% | 13.7589% | 5.359 | clean_k56_velocity_relative_l2_gt_5pct |
| 3200 | FAIL | 13.8613% | 13.7924% | 5.308 | clean_k56_velocity_relative_l2_gt_5pct, clean_k56_pressure_relative_l2_gt_5pct |
| 3400 | FAIL | 13.8954% | 13.8413% | 5.963 | clean_k56_velocity_relative_l2_gt_5pct, k16_pressure_gain_p95_gt_5pct, k56_pressure_gain_p95_gt_5pct |

## Heldout clean autonomous rollout：全部测试集

所有 relative L2 均乘 100，以百分数表示。

| Heldout Re | Horizon | 窗口 | 物理速度 | 物理压力 | Raw POD 速度 | Raw POD 压力 | finite | 发散窗口 | 首次异常 | 状态范数倍数 | 压力 drift | 轨迹 FP residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Re_24p630436 | K1 | 61 | 0.0514% | 2.9973% | 0.0606% | 2.4091% | 100.0000% | 0 | — | 15.946 | 3.5787% | 3.5565% |
| Re_24p630436 | K4 | 58 | 0.0771% | 9.3747% | 0.1039% | 7.6061% | 100.0000% | 0 | — | 26.927 | 8.5959% | 10.9546% |
| Re_24p630436 | K8 | 54 | 0.1016% | 8.6490% | 0.1345% | 7.0232% | 100.0000% | 0 | — | 26.927 | 6.4204% | 10.9546% |
| Re_24p630436 | K16 | 46 | 0.1303% | 6.8131% | 0.1719% | 5.5395% | 100.0000% | 0 | — | 30.665 | 1.1380% | 10.9546% |
| Re_24p630436 | K56 | 6 | 0.3092% | 18.3157% | 0.4521% | 14.9068% | 100.0000% | 0 | — | 60.206 | 33.3203% | 33.1135% |
| Re_32p740068 | K1 | 61 | 0.0134% | 1.7287% | 0.0549% | 5.6073% | 100.0000% | 0 | — | 10.034 | 8.7505% | 8.4579% |
| Re_32p740068 | K4 | 58 | 0.0206% | 1.6709% | 0.1039% | 5.4141% | 100.0000% | 0 | — | 13.541 | 4.4337% | 14.0369% |
| Re_32p740068 | K8 | 54 | 0.0253% | 1.3142% | 0.1412% | 4.2557% | 100.0000% | 0 | — | 13.541 | 2.4300% | 13.0160% |
| Re_32p740068 | K16 | 46 | 0.0313% | 1.3667% | 0.1855% | 4.4278% | 100.0000% | 0 | — | 13.541 | 2.4983% | 13.0160% |
| Re_32p740068 | K56 | 6 | 0.0886% | 5.4462% | 0.5852% | 17.6518% | 100.0000% | 0 | — | 34.031 | 24.1755% | 29.2774% |
| Re_39p685479 | K1 | 61 | 0.0066% | 0.2823% | 0.0420% | 2.8621% | 100.0000% | 0 | — | 8.205 | 10.8866% | 9.5321% |
| Re_39p685479 | K4 | 58 | 0.0093% | 0.4128% | 0.0808% | 4.1994% | 100.0000% | 0 | — | 13.319 | 8.5697% | 12.1598% |
| Re_39p685479 | K8 | 54 | 0.0118% | 0.4840% | 0.1158% | 4.9457% | 100.0000% | 0 | — | 14.220 | 2.7678% | 5.5722% |
| Re_39p685479 | K16 | 46 | 0.0179% | 0.8811% | 0.1662% | 9.0185% | 100.0000% | 0 | — | 16.319 | 2.1559% | 3.4488% |
| Re_39p685479 | K56 | 6 | 0.0385% | 1.6975% | 0.4088% | 17.3796% | 100.0000% | 0 | — | 46.291 | 130.8982% | 111.7871% |
| Re_45p142703 | K1 | 61 | 0.0163% | 0.6963% | 0.0344% | 2.3362% | 100.0000% | 0 | — | 13.176 | 7.6822% | 7.6018% |
| Re_45p142703 | K4 | 58 | 0.0238% | 1.4659% | 0.0694% | 4.9067% | 100.0000% | 0 | — | 26.363 | 5.0914% | 12.4410% |
| Re_45p142703 | K8 | 54 | 0.0307% | 1.2927% | 0.0999% | 4.3229% | 100.0000% | 0 | — | 29.517 | 3.2449% | 10.7071% |
| Re_45p142703 | K16 | 46 | 0.0407% | 1.1934% | 0.1383% | 3.9833% | 100.0000% | 0 | — | 35.777 | 0.7777% | 6.2743% |
| Re_45p142703 | K56 | 6 | 0.0615% | 2.4706% | 0.2576% | 8.2754% | 100.0000% | 0 | — | 88.781 | 45.6744% | 59.0233% |

K128 按原数据连续窗口合同如实报告为不可用，不进行外推填补。

## Heldout 固定点残差：全部测试集

| Heldout Re | Horizon | 物理速度 FP | 物理压力 FP | Raw POD 速度 FP | Raw POD 压力 FP |
|---|---|---|---|---|---|
| Re_24p630436 | K1 | 0.0000% | 0.8863% | 0.0000% | 0.7323% |
| Re_24p630436 | K16 | 0.0000% | 28.1554% | 0.0000% | 23.2646% |
| Re_24p630436 | K56 | 0.0000% | 27.8651% | 0.0000% | 23.0248% |
| Re_32p740068 | K1 | 0.0000% | 0.7936% | 0.0000% | 2.6266% |
| Re_32p740068 | K16 | 0.0000% | 3.8234% | 0.0000% | 12.6549% |
| Re_32p740068 | K56 | 0.0000% | 3.8112% | 0.0000% | 12.6146% |
| Re_39p685479 | K1 | 0.0000% | 0.0778% | 0.0000% | 0.7504% |
| Re_39p685479 | K16 | 0.0000% | 2.8486% | 0.0000% | 27.4778% |
| Re_39p685479 | K56 | 0.0000% | 2.8630% | 0.0000% | 27.6174% |
| Re_45p142703 | K1 | 0.0000% | 0.3079% | 0.0000% | 1.0223% |
| Re_45p142703 | K16 | 0.0000% | 6.8101% | 0.0000% | 22.6121% |
| Re_45p142703 | K56 | 0.0000% | 6.4633% | 0.0000% | 21.4609% |

## 四空间严格 paired gain

以下统计使用 0.5%、1%、2%、5% 预注册选择尺度；分子为 perturbed prediction 与 clean prediction 的同空间距离，分母为同空间实际初始扰动范数。

| Horizon | 度量空间 | N | median | P90 | P95 | P99 | worst |
|---|---|---|---|---|---|---|---|
| K1 | Normalized POD state | 320 | 13.219 | 45.506 | 69.377 | 149.250 | 262.473 |
| K1 | Raw velocity POD | 192 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| K1 | Raw pressure POD | 272 | 0.239 | 2.868 | 4.590 | 23.457 | 152.611 |
| K1 | Area-weighted velocity field | 192 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| K1 | Gauged area-weighted pressure field | 272 | 0.239 | 2.868 | 4.591 | 23.461 | 152.589 |
| K4 | Normalized POD state | 320 | 18.360 | 93.336 | 132.415 | 235.286 | 434.674 |
| K4 | Raw velocity POD | 192 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| K4 | Raw pressure POD | 272 | 0.440 | 3.103 | 5.060 | 24.579 | 165.810 |
| K4 | Area-weighted velocity field | 192 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| K4 | Gauged area-weighted pressure field | 272 | 0.440 | 3.103 | 5.061 | 24.582 | 165.865 |
| K8 | Normalized POD state | 320 | 14.090 | 49.380 | 67.517 | 136.860 | 169.387 |
| K8 | Raw velocity POD | 192 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| K8 | Raw pressure POD | 272 | 0.321 | 2.764 | 4.773 | 22.863 | 144.030 |
| K8 | Area-weighted velocity field | 192 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| K8 | Gauged area-weighted pressure field | 272 | 0.321 | 2.764 | 4.773 | 22.862 | 144.010 |
| K16 | Normalized POD state | 320 | 14.508 | 49.592 | 70.564 | 165.285 | 217.287 |
| K16 | Raw velocity POD | 192 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| K16 | Raw pressure POD | 272 | 0.312 | 2.620 | 4.394 | 23.124 | 127.754 |
| K16 | Area-weighted velocity field | 192 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| K16 | Gauged area-weighted pressure field | 272 | 0.312 | 2.620 | 4.393 | 23.120 | 127.722 |
| K56 | Normalized POD state | 320 | 13.650 | 46.404 | 71.259 | 147.311 | 175.095 |
| K56 | Raw velocity POD | 192 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| K56 | Raw pressure POD | 272 | 0.313 | 2.776 | 4.444 | 23.638 | 145.818 |
| K56 | Area-weighted velocity field | 192 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| K56 | Gauged area-weighted pressure field | 272 | 0.313 | 2.775 | 4.444 | 23.637 | 145.778 |

### 每个 Heldout Re 的 physical pressure paired gain

| Heldout Re | Horizon | N | median | P90 | P95 | worst |
|---|---|---|---|---|---|---|
| Re_24p630436 | K16 | 68 | 0.537 | 2.161 | 3.102 | 7.246 |
| Re_24p630436 | K56 | 68 | 0.490 | 2.317 | 3.502 | 6.699 |
| Re_32p740068 | K16 | 68 | 0.184 | 1.995 | 3.972 | 8.992 |
| Re_32p740068 | K56 | 68 | 0.181 | 1.749 | 3.983 | 8.966 |
| Re_39p685479 | K16 | 68 | 0.191 | 2.456 | 3.768 | 43.260 |
| Re_39p685479 | K56 | 68 | 0.222 | 2.405 | 3.732 | 42.945 |
| Re_45p142703 | K16 | 68 | 0.513 | 3.552 | 11.328 | 127.722 |
| Re_45p142703 | K56 | 68 | 0.403 | 3.995 | 11.368 | 145.778 |

## 与 S3-A、S3-B 的同口径对照

| 模型 | Horizon | pressure gain median | P90 | P95 | worst |
|---|---|---|---|---|---|
| S3-A | K16 | 1.161 | 35.809 | 55.745 | 307.563 |
| S3-A | K56 | 1.226 | 10.236 | 27.631 | 143.654 |
| S3-B | K16 | 0.198 | 2.677 | 4.263 | 138.238 |
| S3-B | K56 | 0.243 | 2.927 | 4.628 | 183.314 |

### Clean physical worst 与 fixed-point 同口径对照

| 模型 | Horizon | clean 物理速度 worst | clean 物理压力 worst | 压力 FP worst |
|---|---|---|---|---|
| S3-A | K1 | 0.0514% | 8.3525% | 14.0249% |
| S3-A | K16 | 0.1308% | 7.0486% | 28.6270% |
| S3-A | K56 | 0.2690% | 16.4762% | 28.8197% |
| S3-B | K1 | 0.0514% | 8.2670% | 14.0708% |
| S3-B | K16 | 0.1303% | 7.0056% | 34.9616% |
| S3-B | K56 | 0.2995% | 18.5603% | 27.1317% |
| S4-Steady | K1 | 0.0514% | 2.9973% | 0.8863% |
| S4-Steady | K16 | 0.1303% | 6.8131% | 28.1554% |
| S4-Steady | K56 | 0.3092% | 18.3157% | 27.8651% |

## 预注册成功判定

| 检查项 | 结果 | 测量值 |
|---|---|---|
| clean_rollout_protection | PASS | all K1/K16/K56 physical errors within +5%, finite=100%, divergence=0 |
| heldout_fixed_point_improvement_ge_10pct | PASS | 19.47% (S4 28.1554% vs S3-B 34.9616%) |
| pressure_contraction_protection | PASS | K16/K56 median<1 and P95 within +5% of S3-B |
| validation_fixed_point_improvement_ge_10pct | PASS | 19.10% |

最终判定：**S4_SUCCESS_CANDIDATE**。S4 may be frozen only as the new Steady sub-MoE candidate; canonical best remains unchanged.

非强制目标：2/4 heldout Re below 5% (aspirational, not a mandatory gate)。该观察不覆盖预注册强制门槛。

## 三个概念的客观区分

1. 低误差有限时域自主预测：由 clean K1/K16/K56 physical relative L2 单独衡量。
2. 数值不发散：由 finite fraction、divergent windows 和首次异常 step 单独衡量；它不等价于局部扰动收缩。
3. 局部扰动收缩/固定点吸引性：由同空间 paired gain 与真实 Steady fixed-point residual 衡量；median<1 不代表 P95/worst 也收缩。

异常方向、低方差 pressure modes 的贡献、逐方向/尺度原始记录、分母分布与 raw/physical 一致性统计见 `paired_gain_four_space.json`。Heldout 未参与 checkpoint 选择，canonical best 未修改，也未启动其他训练。
