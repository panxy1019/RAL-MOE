# V17 扩展 Hopf 数据集与独立 POD/ROM 构建报告

生成日期：2026-07-21  
实验标识：`V17_HopfExpanded34_POD_20260721`

## 1. 目的与结论

本轮工作针对原 Hopf 数据集在临界区覆盖不足的问题，补充了 17 个指定 Reynolds 数的二维圆柱绕流模拟，并与原有 17 个 Hopf 案例合并。随后，使用 Reynolds 数互斥的训练/验证/测试划分，重新构建了独立的 Hopf POD、modal normalization、速度 Galerkin ROM 张量和压力 Poisson surrogate 张量。

最终验收为 **PASS**。新资产包含 34 个 Re、4800 个快照；POD 的均值、基、训练谱和 normalization 均严格只由 29 个训练 Re 拟合。验证集与测试集只在冻结 POD 后投影和报告，未参与拟合。

## 2. 数据集组成

### 2.1 原有 Hopf 案例（17 个）

```text
47.081355, 47.722947, 48.368688, 49.022357, 49.687640,
50.368054, 51.066785, 51.786450, 52.528767, 53.294175,
54.081508, 54.887950, 55.709610, 56.543246, 57.389970,
58.262636, 59.201432
```

### 2.2 本轮新增案例（17 个）

```text
45.5, 45.8, 46.1, 46.3, 46.5, 46.7, 46.9,
47.2, 47.4, 47.6, 47.8, 48.0, 48.3, 48.7,
49.3, 49.6, 50.0
```

新增案例严格沿用旧数据库的数值配置：同一 OpenFOAM 13 基准网格、`nu=1e-3`、`D=1`、`U_inlet=Re*nu/D`、同一输出格式和同一稳定性判定流程。

- `Re < 47`：先使用 `simpleFoam` 获得稳态初始化，再使用一致的数据输出流程；保留 63/64 帧。
- `Re >= 47`：使用 `pimpleFoam`，通过升力峰值的振幅/周期稳定性检测极限环；检测完成后保留 20 个周期，通常保留 161/162 帧。

重要说明：新增 `45.5–46.9` 在物理源标签中保持为 `pre_hopf_steady`，没有被伪装为周期解；但为了建立覆盖临界前后动力学的 specialist 数据资产，它们被**显式白名单**纳入本次的目标 Hopf 组。构建产物同时保存 `source_regime` 和 `target_regime=hopf`，因此该研究选择可追溯。

## 3. Reynolds 数切分合同

切分原则为 Reynolds-number-disjoint：同一 Re 不得同时出现在训练、验证或测试中。

| 划分 | 数量 | Reynolds 数 |
|---|---:|---|
| Train | 29 | 45.5, 45.8, 46.1, 46.3, 46.5, 46.9, 47.2, 47.4, 47.6, 47.722947, 47.8, 48.0, 48.3, 48.368688, 48.7, 49.3, 49.6, 49.687640, 50.0, 50.368054, 51.066785, 52.528767, 53.294175, 54.081508, 54.887950, 55.709610, 57.389970, 58.262636, 59.201432 |
| Validation | 2 | 46.7, 56.543246 |
| Test | 3 | 47.081355, 49.022357, 51.786450 |

选择逻辑：验证集使用一个临界前高密度样本和一个上侧 Hopf 样本；测试集保留原有的三个 canonical Hopf heldout，用于与此前 Hopf 专项实验直接对比。

## 4. POD 构建方法

构建环境：训练集群 `10.210.22.202:31283` 的 `pt_env`（Python 3.11.15、NumPy 2.4.4、SciPy 1.17.1）。

配置如下：

```text
centering:          single_train_only_regime_mean
pressure gauge:     subtract_area_mean_per_snapshot
POD inner product:  lumped point-area weighted L2
retained rank:      80
fair-control rank:  32
randomized POD:     oversampling=32, power_iterations=2, seed=20260721
```

训练拟合人口为 29 Re / 4092 snapshots。全部 34 Re 投影后，系数矩阵均为 `[4800, 80]`：

| 变量 | POD 基形状 | 系数形状 |
|---|---:|---:|
| Velocity | `[80, 194736]` | `[4800, 80]` |
| Pressure | `[80, 97368]` | `[4800, 80]` |

POD 的拟合采用训练快照的 pooled-snapshot 加权。因此临界前的 63/64 帧案例对协方差的权重低于 161/162 帧周期案例；这是沿用既有 POD 合同的结果。后续 MoE 训练不应按快照数随机采样，而应维持“先均匀选 Re、再选时间窗口”的 Re-balanced 采样规则。

## 5. 数据隔离与泄漏审计

最终审计结果：

```text
fit_rows:                    train only
basis_uses_validation:       false
basis_uses_heldout:          false
mean_uses_validation:        false
mean_uses_heldout:           false
normalization_uses_validation:false
normalization_uses_heldout:  false
```

新增 Re 不存在于旧 frozen global POD 的 Re-specific mean 表中，因此不为新增 Re 伪造或插值全局对照。其 frozen-global-r32 指标被标记为 unavailable；本报告中的 heldout 全局对照仍可报告，因为三个测试 Re 都是旧资产中已有的 canonical Re。

## 6. 投影诊断

测试集（3 个 canonical heldout）的 full-field relative L2 投影误差如下。

| 变量 | local r32 mean | frozen global r32 mean* | local r80 mean |
|---|---:|---:|---:|
| Velocity | `2.4602172e-06` | `1.8582597e-04` | `2.4584124e-07` |
| Pressure（gauge-fixed） | `8.5768202e-06` | `1.3495404e-03` | `1.5938141e-06` |

\* frozen global 对照的均值和基使用全体 100 Re 与 Re-specific mean，因此它只用于历史连续性比较，不是无泄漏的公平基线。

这些数值表明：重建的本地 Hopf 基在三个保留测试 Re 上具有远低于 ROM 目标误差尺度的截断误差；后续若 MoE 结果仍不理想，主要瓶颈不应归因于该 r80 POD 的表征误差，而应进一步检查闭环动力学、压力 closure、近临界振幅增长和训练分布。

## 7. 匹配的 Hopf ROM 资产

为避免“新 POD + 旧 ROM 张量”的坐标不一致，本轮与 POD 同步重建了：

```text
velocity_rom_hopf.npz
pressure_poisson_surrogate_hopf.npz
```

两套 ROM 张量均覆盖全部 34 个 Re，且数值有限。它们必须与本次 `velocity_pod_hopf.npz`、`pressure_pod_hopf.npz` 和 `normalization_hopf.npz` 成套使用；不得再与旧 17-Re Hopf POD/ROM 混用。

## 8. 最终资产与校验摘要

主目录：

```text
/cephfs/shared/V17_HopfExpanded34_POD_20260721
```

关键产物 SHA256：

| 产物 | SHA256 |
|---|---|
| velocity POD | `2f951192f6eeaa69e7d911aa21552454d1a99a49ab796166b4a13df16016a192` |
| pressure POD | `cc128a4361ab2a239165ac22a48407290de7d085b857bbff6726988457cfddf3` |
| velocity ROM | `7d16ca813efa0f92109689a686de0c3de02cdd269a4b31029c53c3ed1309d3e9` |
| pressure ROM | `a6646aa3eb3814d8648293ee3cd798a6abe268bf46a79ec0c0b8b17b30aaf06d` |

最终验收文件：

```text
/cephfs/shared/V17_HopfExpanded34_POD_20260721/artifacts/HOPF_EXPANDED_POD_ACCEPTANCE.json
```

验收状态：`PASS`。

## 9. 自动流水线的可追溯说明

初始自动流水线的 POD 和两套 ROM 计算均成功完成。之后最终验收脚本错误读取了 POD archive 中不存在的 `train_Re_labels` 字段，导致首次流水线退出码为 1。该问题仅存在于验收脚本；POD/ROM 文件没有失败、没有被覆盖、也没有被重新计算。

修复方式是从 POD archive 已有的 `Re_labels` 与 `split_by_Re` 推导训练标签，并重新执行验证。修复后的验收状态为 `PASS`；恢复记录位于：

```text
/cephfs/shared/V17_HopfExpanded34_POD_20260721/logs/RECOVERY_NOTE.txt
```

## 10. 供后续 Hopf MoE 训练使用的约束

1. 仅使用本轮 34-Re 独立 Hopf POD、normalization 与 ROM 张量。
2. Train/validation/test Re 严格按本报告第 3 节执行；测试集不得参与 checkpoint 选择。
3. 验证 checkpoint 时重点检查 `Re=46.7` 的近临界前侧行为与 `Re=56.543246` 的上侧 Hopf 行为。
4. 最终测试保留 `47.081355、49.022357、51.786450`，尤其关注近 onset 的 `Re=47.081355`。
5. r80 POD 已验证通过；后续首先应评估 ROM/MoE 的动力学泛化，而不要把模型误差直接归因于 POD 截断。
