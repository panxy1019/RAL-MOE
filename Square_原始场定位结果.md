# Square 原始全阶 CFD 场定位与完整性核查

- 核查日期：2026-09-20
- 几何：CenteredSquare（方柱绕流）
- 范围：只读定位与核验；未重跑 CFD，未重新训练，未搬迁或删除数据，未用 POD 重构场代替 CFD 原始场。

## 结论摘要

1. `192.168.232.130:22` 已恢复可达，2026-09-20 10:20 CST 成功以 `ray` 登录。
2. VM 上原路径 `/home/ray/Desktop/centeredSquare` 已不存在；因此原 100-case 与 31-case 整库不能在该路径直接使用。
3. VM 现存 **33 条 Periodic 全阶轨迹**：27 个训练 case + 6 个验证 case。所有 NPZ 已全量读取 `times/U/p`，shape 一致，无 NaN/Inf，时间严格递增。
4. 本机现存 54 个物理 NPZ 文件，去重后为 **42 个 Re**；其中 Hopf 域 34 Re 完整，Steady 固定 heldout 4 Re 完整，Periodic 验证/heldout 10 Re 完整。
5. S-H 接口固定测试 `Re=95.1, 95.3` 与 H-P 接口固定测试 `Re=100.5, 102` 的全轨迹 CFD NPZ 均在本机且已全量通过核验。
6. VM 恢复的 OpenFOAM `/home/ray/Desktop/centeredSquare_restore_20260827/Re100/100` **与正式 Re100 NPZ 不匹配**，不得将它当作正式数据的原生上游。

## 数据位置表

| 几何/流态 | 参数 | 数据划分及依据 | 主机 | 原始场绝对路径 | 格式/字段 | 时刻数与时间范围 | 网格/权重路径 | 检查状态 | 备注 |
|---|---:|---|---|---|---|---|---|---|---|
| Square / Hopf | 94--102，34 Re | 按冻结 `split_contract_resolved.json` | Windows 本机 | `C:\Users\panxy1019\Documents\CHANNEL\cluster_hopf_transfer\hopf_transfer_34\cases_npz` | NPZ: `Re,nu,times,U,p,regime_placeholder,metadata` | 126，0--500，Δt=4 | 见下文公共网格 | **可用，34/34 全量可读** | `regime_placeholder=UNLABELED`，流态标签以冻结 split 为准 |
| Square / Steady heldout | 60, 85, 95.1, 95.3 | 冻结 heldout 列表 | Windows 本机 | `C:\Users\panxy1019\Documents\CHANNEL\heldout_cases` | 同上 | 同上 | 见下文 | **可用，4/4 全量可读** | 包含 S-H 接口 Re95.1/95.3 |
| Square / Periodic validation+heldout | 99, 100.5, 101.5, 102, 110.344827586207, 120.689655172414, 125.862068965517, 141.379310344828, 144.827586206897, 150 | 冻结 validation/heldout 列表 | Windows 本机 | `C:\Users\panxy1019\Documents\CHANNEL\.transfer_square_periodic` | 同上 | 同上 | 见下文 | **可用，10/10 全量可读** | 包含 H-P 接口 Re100.5/102 |
| Square / Periodic train | 27 Re，见下文 | 冻结 Periodic train 列表 | VM `192.168.232.130:22` | `/home/ray/Desktop/centeredSquare_restore_20260827/three_regime_overlap_v1/subsets/periodic/cases_npz` | 同上 | 同上 | `/home/ray/Desktop/centeredSquare_restore_20260827/three_regime_overlap_v1/subsets/periodic/mesh/mesh_metadata.npz` | **可用，27/27 全量可读** | 实体文件，非失效链接 |
| Square / Periodic validation | 99, 101.5, 110.344827586207, 125.862068965517, 141.379310344828, 150 | `VALIDATION_RESTORE.json` | VM `192.168.232.130:22` | `/home/ray/Desktop/centeredSquare_validation_20260827` 下两个 dataset 的 `cases_npz` | 同上 | 同上 | 使用上述 Periodic mesh | **可用，6/6 全量可读** | 与本机副本 SHA-256 一致 |
| Square / S-H 接口 | 95.1, 95.3 | `cache_sh/PREFLIGHT.json`: heldout，history=3，horizon=24 | Windows 本机 | 上述 `heldout_cases` 及 Hopf 副本 | 同上 | 同上 | 见下文 | **可用，全量可读** | 每 Re 8 个固定评价窗口 |
| Square / H-P 接口 | 100.5, 102 | `cache_hp/PREFLIGHT.json`: heldout，history=3，horizon=24 | Windows 本机 | 上述 `.transfer_square_periodic` 及 Hopf 副本 | 同上 | 同上 | 见下文 | **可用，全量可读** | 每 Re 8 个固定评价窗口 |
| Square / Re100 OpenFOAM | 100 | 恢复的单帧原生场 | VM `192.168.232.130:22` | `/home/ray/Desktop/centeredSquare_restore_20260827/Re100/100/{U,p}` | OpenFOAM ASCII cell fields | 仅 t=100 | 同 case 的 `constant/polyMesh` | **可读但与正式 NPZ 不匹配** | 不可用作正式 Re100 真值上游 |
| Square / 原完整库 | 100 + 31 cases，合并后 120 唯一 Re | 原正式 sweep | VM `192.168.232.130:22` | 历史：`/home/ray/Desktop/centeredSquare/dataset_Re50_150_N100_npz` 与 `dataset_Re95_102_refined_N31_npz` | NPZ | 每 case 126 帧 | 原 dataset `mesh/mesh_metadata.npz` | **当前路径未找到** | 不能将历史记录写成当前可访问 |

## 推荐交接位置

### 可直接用于现有融合评价

- S-H 原始真值：`C:\Users\panxy1019\Documents\CHANNEL\heldout_cases\snapshots_Re095p100000.npz` 和 `snapshots_Re095p300000.npz`。
- H-P 原始真值：`C:\Users\panxy1019\Documents\CHANNEL\.transfer_square_periodic\snapshots_Re100p500000.npz` 和 `snapshots_Re102p000000.npz`。
- Hopf 域完整本地副本：`C:\Users\panxy1019\Documents\CHANNEL\cluster_hopf_transfer\hopf_transfer_34\cases_npz`。
- Periodic 训练副本：`192.168.232.130:/home/ray/Desktop/centeredSquare_restore_20260827/three_regime_overlap_v1/subsets/periodic/cases_npz`。

### 原生求解器文件

- 当前只找到恢复的 Re100 t=100 单帧和 `polyMesh`。
- 由于它与正式 NPZ 数值不匹配，**不推荐使用**。
- 正式 sweep 在 NPZ 验证通过后按设计删除了大多数 OpenFOAM 时间目录，因此 NPZ 是当前主要的无降维快照副本。

## 数据内容与时间

- `U`: `float32`, shape `[126, 9400, 2]`，轴为 `[time, cell, (ux,uy)]`，第三个方向在二维算例中为 0，NPZ 未保存。
- `p`: `float32`, shape `[126, 9400]`，轴为 `[time, cell]`。
- `times`: `float64`, shape `[126]`，从 0 到 500，固定间隔 4，严格递增，包含初始瞬态。
- 每个 NPZ 都是从 OpenFOAM cell-centered internal field 直接导出；未做 POD 截断，未减均值，未插值，未重采样，未空间裁剪。
- `p` 是 `icoFoam` 不可压求解的运动学压力（`p/rho`）；原设置为出口固定零 gauge。绝对物理单位/无量纲化尺度在现有元数据中未单独记录。
- POD 构建时的逐快照体积加权压力去均值是后处理，不是原 NPZ 的内容。

## 网格与积分权重

- 已实际读取 VM 网格元数据：
  - `cellCenters`: `(9400,3)`, `float64`, 全有限。
  - `cellVolumes`: `(9400,)`, `float64`, 全为正，范围 `1.6361617e-4`--`5.3638433e-3`。
  - `Nc=9400`，`volume_total=12.29999999479`。
- 积分权重应直接使用 `cellVolumes`；速度 POD 展平权重为 `repeat(sqrt(cellVolumes),2)`，压力为 `sqrt(cellVolumes)`。
- 本地参考 VTK：`C:\Users\panxy1019\Documents\CHANNEL\centeredsquare_fusion_v1\reference_plotting\centeredSquare_CN09_graded_Re100_internal_final_reference.vtk`，1,104,845 bytes，legacy binary unstructured grid，19,306 points / 9,400 cells，含 cell `U,p,cellID` 及 point data。它是单帧参考场，不是时间序列。
- 原生连接关系在 VM 恢复 Re100 的 `constant/polyMesh/{points,faces,owner,neighbour,boundary}`，但由于该 Re100 场与正式 NPZ 不匹配，使用前必须另行核实网格顺序。

## 冻结划分与接口轨迹

- 流态边界记录：Steady 上限 95.3，Hopf 上限 102，onset 估计 95.312。
- 专家重叠域：Steady 50--95.4，Hopf 94--102，Periodic 98.5--150。
- 训练计数：Steady 60 cases / 7560 snapshots，Hopf 23 / 2898，Periodic 27 / 3402。
- 验证：Steady `55,75,90,94.5,95.25`；Hopf `94.5,95.25,95.5,97.5,99,101.5`；Periodic `99,101.5,110.344827586207,125.862068965517,141.379310344828,150`。
- heldout：Steady `60,85,95.1,95.3`；Hopf `95.1,95.3,96.5,100.5,102`；Periodic `100.5,102,120.689655172414,144.827586206897`。
- Periodic VM 训练 Re：`98.5, 99.5, 100, 101, 101.724137931034, 103.448275862069, 105.172413793103, 106.896551724138, 108.620689655172, 112.068965517241, 113.793103448276, 115.517241379310, 117.241379310345, 118.965517241379, 122.413793103448, 124.137931034483, 127.586206896552, 129.310344827586, 131.034482758621, 132.758620689655, 134.48275862069, 136.206896551724, 137.931034482759, 139.655172413793, 143.103448275862, 146.551724137931, 148.275862068966`。

## 完整性证据

### VM 全量检查

- 33/33 NPZ 实际使用 `np.load(..., allow_pickle=False)` 打开并读取完整 `times/U/p` 数组，不是仅检查 ZIP 文件头。
- 累计大小 429,495,124 bytes；所有 shape 为 `times(126), U(126,9400,2), p(126,9400)`。
- 33/33 时间严格递增，Δt 唯一值为 4；`U/p` 的 NaN 和 Inf 计数均为 0。
- Re100 Periodic NPZ SHA-256: `5d846e7b35e0fa640f299192813ec37fd98bbfe208b572d86a266481ad55067e`。

### 本机全量检查

- 54 个物理文件均全量读取 `times/U/p`，无读取错误、NaN/Inf、重复或非递增时间。
- 重复 Re 的副本 SHA-256 一致；去重后 42 个 Re。
- 逐文件证据：`C:\Users\panxy1019\Documents\CHANNEL\square_local_raw_integrity_20260920.json`。
- 例：Re94 文件 12,932,822 bytes，SHA-256 `bf9764c2f6fa99992978e78648f9d99740bf26a49e41ddfdbaaec73b63484104`。

### Re100 原生场异常

- OpenFOAM t=100 文件可读：`U(9400,3)`、`p(9400)`，无 NaN/Inf。
- 与正式 Re100 NPZ 的 t=100 比较：`max_abs(Uxy)=0.7635657365`，`max_abs(p)=0.4302797360`。
- 扫描 NPZ 全部 126 帧后仍无匹配：U 最佳时刻 t=8，RMSE 0.12682；p 最佳时刻 t=48，RMSE 0.04125。
- 因此该 OpenFOAM 恢复目录应标记为“来源待核实/不得作为正式真值”，不能仅根据目录名 `Re100` 认定对应。

## 最小读取示例

以下代码已在 VM Python 3 + NumPy 环境成功运行：

```python
import numpy as np

path = "/home/ray/Desktop/centeredSquare_restore_20260827/three_regime_overlap_v1/subsets/periodic/cases_npz/snapshots_Re098p500000.npz"
with np.load(path, allow_pickle=False) as z:
    times = z["times"]
    U = z["U"]
    p = z["p"]
    print(times.shape, U.shape, p.shape)
    print(times[0], times[-1], np.isfinite(U).all(), np.isfinite(p).all())
```

实际输出对应：`(126,) (126, 9400, 2) (126, 9400)` 和 `0.0 500.0 True True`。

## 构建脚本与 POD/论文对应

- VM 导出脚本：`/home/ray/Desktop/centeredSquare_restore_20260827/scripts/dataset_tools.py`。
- 本地导出/运行脚本：`C:\Users\panxy1019\Documents\CHANNEL\remote_scripts\dataset_tools.py`、`run_formal_re50_150_n100_npz.sh`、`run_refined_re95_102_n29_npz.sh`。
- 三流态构建：`C:\Users\panxy1019\Documents\CHANNEL\remote_scripts\build_three_regime_centeredsquare.py`。
- ROM 张量组装：`C:\Users\panxy1019\Documents\CHANNEL\remote_scripts\build_centered_square_rom_tensors.py`。
- 冻结分割：`C:\Users\panxy1019\Documents\CHANNEL\cluster_hopf_transfer\hopf_transfer_34\split_contract_resolved.json`。
- S-H/H-P 固定窗口依据：`C:\Users\panxy1019\Documents\CHANNEL\centeredsquare_fusion_v1\results\E2_T2C_K24_20260730_STRICT_V3\heldout_evaluation_20260730_V1\cache_sh\PREFLIGHT.json` 与 `cache_hp\PREFLIGHT.json`。

## 缺失项与下一步

1. VM 当前不含原完整 120-Re 库；Steady 训练全集、Hopf 之外的部分原始轨迹不能仅从当前 VM 恢复。
2. 本机可支撑当前 S-H/H-P 接口评价与 Hopf 全域评价；如要对 Steady/Periodic 所有冻结训练参数做全量追溯，仍需定位原 100-case/31-case 的其他备份或从已知远程仓库恢复。
3. 在找到正式原生 Re100 上游前，不应使用 `/home/ray/Desktop/centeredSquare_restore_20260827/Re100` 验证点序或场值。
4. 新训练集群的 `/root/centeredSquare_three_regime_overlap_v1` 按原交付设计主要保留 POD/ROM/配置/网格/参考 VTK，不应假定其含有完整原始 `cases_npz`。
