# CenteredSquare 子数据集 POD 复现教程

本教程用于将完整数据集按参数点（推荐按 Reynolds 数/Re case）切分为多个子数据集，并在每个子数据集上独立复现全局、体积加权 L2 POD。POD 完成后，可继续构建 VTK 导数驱动的半侵入式 POD-Galerkin ROM。

当前完整数据集路径：

```text
/home/ray/Desktop/centeredSquare/dataset_Re50_150_N100_npz
```

POD 工具脚本：

```text
/home/ray/Desktop/centeredSquare/scripts/dataset_tools.py
```

## 1. 重要原则

1. 子数据集应优先按完整 Re case 切分，而不是改变单个 case 内的网格或单元顺序。
2. 每一个子数据集必须独立计算速度均值、压力均值、POD 模态、POD 系数和能量阶数。
3. 不要直接复用完整数据集的 POD 均值或 POD 模态。
4. `mesh_metadata.npz` 和 `reference_vtk/` 可从完整数据集直接复制，但不能改变网格 cell 排序。
5. VTK 不参与 POD 的 SVD；它用于后续 ROM 中的梯度、散度和 Laplacian 导数计算。

## 2. 子数据集目录结构

以 `subset_A` 为例，建议目录如下：

```text
/home/ray/Desktop/centeredSquare/subsets/subset_A/
  cases_npz/
    snapshots_ReXX_XXXXXX.npz
  manifest/
    re_points_subset.csv
  mesh/
    mesh_metadata.npz
  reference_vtk/
    centeredSquare_..._reference.vtk
    README.md
```

目录说明：

- `cases_npz/`：仅包含该子集选中的 Re case 文件。
- `manifest/re_points_subset.csv`：仅保留这些 Re 的记录；行中的 tag/文件名必须与 `cases_npz/` 精确对应。
- `mesh/mesh_metadata.npz`：从完整数据集复制，包含 cell centers、cell volumes 等网格信息。
- `reference_vtk/`：从完整数据集复制，供后续 ROM 导数计算和网格对齐校验使用。

每个 case NPZ 应至少包含：

```text
Re        标量 Reynolds 数
nu        运动黏度
times     (T,)
U         (T, Ncell, 2)
p         (T, Ncell)
```

## 3. 如何切分数据集

推荐按完整 Re case 切分。例如将部分 Re 放入 `subset_A`：

1. 新建 `subset_A/cases_npz`、`subset_A/manifest`、`subset_A/mesh`、`subset_A/reference_vtk`。
2. 将选定 Re 的 `snapshots_*.npz` 复制到 `subset_A/cases_npz/`。
3. 从完整 manifest 中筛选相同 Re，写入 `subset_A/manifest/re_points_subset.csv`。
4. 复制完整数据集的 `mesh/mesh_metadata.npz` 和整个 `reference_vtk/` 目录。

不要筛选网格 cell，也不要对 `U`、`p` 的 cell 维度重新排序。否则 POD 权重、VTK 导数和 ROM 内积将不再一致。

若必须按时间切分单个 Re case，需同步裁剪该 NPZ 内的 `times`、`U` 与 `p`，并确保三者的时间维度完全一致。此类切分不会改变 mesh/VTK，但得到的是时间窗口 POD，而非参数子区间 POD。

## 4. POD 的数学约定

对一个子数据集的全部快照，分别计算均值：

```text
U'(t) = U(t) - Ubar
p'(t) = p(t) - pbar
```

权重来自 `mesh_metadata.npz` 中的 `cellVolumes`。速度与压力的加权 L2 内积为：

```text
<Ui, Uj> = sum_c Vc * (ui,c dot uj,c)
<pi, pj> = sum_c Vc * pi,c * pj,c
```

实现中使用的 SVD 加权向量为：

```text
velocity weights = sqrt(cellVolumes) 对两个速度分量重复
pressure weights = sqrt(cellVolumes)
```

因此保存的 `modes` 是物理空间模态；`weighted_modes` 是已经乘以 `sqrt(weight)` 的模态表示。

## 5. 运行 POD

登录虚拟机后，运行：

```bash
python3 /home/ray/Desktop/centeredSquare/scripts/dataset_tools.py build-pod \
  --dataset /home/ray/Desktop/centeredSquare/subsets/subset_A \
  --max-modes 512
```

输出位于：

```text
subset_A/pod/
  weighted_pod_velocity.npz
  weighted_pod_pressure.npz
  pod_energy_report.csv
```

`max-modes=512` 是完整数据集的设置。若子集总快照数少于 512，应令 `max-modes` 不超过总快照数。

## 6. POD 输出内容

速度 POD：

```text
weighted_pod_velocity.npz
  modes:        (max_modes, 2*Ncell)
  coefficients: (Nsnapshot_total, max_modes)
  mean:         (Ncell, 2)
  weights:      (2*Ncell,)
  singular_values
  cumulative_energy
  weighted_modes
  snapshot_times
  snapshot_case_tags
  case_offsets
```

压力 POD：

```text
weighted_pod_pressure.npz
  modes:        (max_modes, Ncell)
  coefficients: (Nsnapshot_total, max_modes)
  mean:         (Ncell,)
  weights:      (Ncell,)
  singular_values
  cumulative_energy
  weighted_modes
  snapshot_times
  snapshot_case_tags
  case_offsets
```

## 7. 选择 99% 与 99.9% 能量阶数

读取：

```text
subset_A/pod/pod_energy_report.csv
```

为速度、压力分别记录：

```text
rank_99   累计能量首次达到 99% 的阶数
rank_999  累计能量首次达到 99.9% 的阶数
```

完整数据集仅供参考：

| POD 空间 | 99% | 99.9% |
| --- | ---: | ---: |
| 速度 | 12 | 26 |
| 压力 | 6 | 13 |

子数据集必须从自己的 `pod_energy_report.csv` 读取 rank，不能直接沿用上表数值。

## 8. POD 结果校验

每次 POD 完成后，至少确认：

1. `modes`、`coefficients`、`mean`、`weights` 中无 NaN 或 Inf。
2. `coefficients.shape[0]` 等于子集全部 case 的快照总数。
3. `cumulative_energy` 单调递增，最后接近 1。
4. `modes` 的加权 Gram 矩阵接近单位阵。
5. 速度 `mean` 的形状为 `(Ncell, 2)`，压力 `mean` 的形状为 `(Ncell,)`。

速度模态的加权正交性可写为：

```text
G_u[i,j] = sum_c Vc * (phi_i(c) dot phi_j(c)) ~= delta_ij
```

压力模态的加权正交性可写为：

```text
G_p[i,j] = sum_c Vc * psi_i(c) * psi_j(c) ~= delta_ij
```

## 9. POD 之后构建 ROM

保留 `reference_vtk/` 后，使用子集自身选出的阶数运行：

```bash
python3 /home/ray/Desktop/centeredSquare/subsets/subset_A/pod/build_centered_square_rom_tensors.py \
  --dataset /home/ray/Desktop/centeredSquare/subsets/subset_A \
  --rank rank99:<ru_99>:<rp_99> \
  --rank rank999:<ru_999>:<rp_999> \
  --derivative-backend vtk
```

其中：

- `ru_99`、`rp_99`：子集 99% 能量下的速度/压力阶数。
- `ru_999`、`rp_999`：子集 99.9% 能量下的速度/压力阶数。
- `--derivative-backend vtk`：通过 PyVista/VTK 的 `compute_derivative()` 计算 ROM 所需空间导数。

构建前需确认 VTK 的 cell centers 与 `mesh_metadata.npz` 中的 centers 对齐，且 cell 数与 `Ncell` 一致。当前完整数据集使用的容差为 `5e-6`。

## 10. 常见错误

| 现象 | 常见原因 | 处理方式 |
| --- | --- | --- |
| POD 或 ROM 结果明显异常 | cell 顺序改变 | 恢复原始 `U/p` cell 顺序，并复用对应 mesh/VTK。 |
| 找不到 case 或快照数不对 | manifest 和 `cases_npz` 不一致 | 检查 Re、tag、文件名和 manifest 行。 |
| 子集 rank 与完整数据集不同 | 这是正常现象 | 使用子集的能量报告，不要强行固定完整数据集 rank。 |
| POD 可运行、ROM 失败 | 缺少或不匹配 reference VTK | 复制 reference VTK，并检查 cell 数和 centers。 |
| ROM 导数质量不稳定 | 未使用 VTK 导数网格 | 对该数据集使用 `--derivative-backend vtk`。 |

## 11. 上传训练集群

训练集群必须从虚拟机进入：

```bash
ssh root@10.210.22.30 -p 30101
```

建议上传子集中的：

```text
pod/
pod/rom/
mesh/
manifest/
reference_vtk/
RUN_SUMMARY.md
```

上传后应比较源端与集群端的 SHA-256 校验和，重点确认 POD 文件、ROM 张量文件、manifest 和 VTK 文件一致。

## 12. 最短复现清单

```text
1. 按 Re case 建立子数据集，保持 cell 顺序不变。
2. 复制 mesh_metadata.npz、reference_vtk 和筛选后的 manifest。
3. 运行 build-pod。
4. 从 pod_energy_report.csv 读取该子集的 99%/99.9% rank。
5. 校验 POD 的快照数、有限性和加权正交性。
6. 使用 reference VTK 构建 rank99 与 rank999 ROM。
7. 从虚拟机上传到训练集群，并以 SHA-256 验证。
```
