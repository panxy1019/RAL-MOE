# Bottom-Mounted Square Obstacle Attractor ROM Dataset

本项目构造一个用于 ROM 验证的 OpenFOAM 数据集流水线，算例是
`2D channel flow past a bottom-mounted square obstacle`，即二维通道内底壁方形障碍物绕流。

项目目标是生成 **Physics-Generalizable Attractor Database**，用于观察不同 Reynolds number 下的最终稳定动力学状态，包括低 Re 的 steady attractor、高 Re 的 periodic attractor，以及可能存在的 near-onset / Hopf-like transition attractor。它不是 transition database，也不刻意训练初始启动瞬态或 Hopf transient amplitude dynamics。

这个几何是参考 wall-mounted square cylinder、low-Re square cylinder flow 和 OpenFOAM square obstacle 类算例后构造的二维可控 ROM benchmark。它不是官方固定 benchmark，也不能直接预设和圆柱绕流完全相同的 Hopf 临界 Re。steady / onset / periodic 区间应先通过 pilot sweep 的 forces 和 probes 信号判断，诊断结果只作为 metadata 和后续采样设计依据，不作为强监督标签。

## Layout

```text
config/sweep.yaml
templates/bottom_square_obstacle/
scripts/
data/npz/
data/manifest/
data/pod/
data/logs/
work/cases/
```

## OpenFOAM Environment

在当前虚拟机上先加载 OpenFOAM 13：

```bash
source /opt/openfoam13/etc/bashrc
```

如果脚本提示缺少 `blockMesh`、`checkMesh`、`decomposePar`、`simpleFoam`、`pimpleFoam`、`reconstructPar` 或 `foamToVTK`，说明当前 shell 还没有 source OpenFOAM。

## Generate Template

```bash
cd /home/ray/bottom_obstacle_attractor_rom
python3 scripts/00_make_template.py
```

template 使用 `blockMesh` 构造五个流体块，障碍物区域 `x in [0,1], y in [0,1]` 被挖掉，不属于流体域。patch 名称固定为：

```text
inlet outlet topWall bottomWall obstacle frontAndBack
```

OpenFOAM 13 使用 `physicalProperties` 和 `momentumTransport`；本项目也保留 `transportProperties` 和 `turbulenceProperties`，方便和旧管线兼容。runner 会同步写入 `nu = 1/Re`。

## Run One Re

默认使用 MPI，保留 OpenFOAM case，npz 验证成功后删除临时 VTK：

```bash
python3 scripts/01_run_single_re.py --re 80 --mode test
python3 scripts/01_run_single_re.py --re 80 --mode pilot
python3 scripts/01_run_single_re.py --re 80 --mode full
```

手动指定 MPI rank：

```bash
python3 scripts/01_run_single_re.py --re 80 --mode test --nprocs 8
```

调试串行运行：

```bash
python3 scripts/01_run_single_re.py --re 80 --mode test --serial
```

常用选项：

```text
--nprocs N       指定单个 case 的 MPI rank 数
--serial         禁用 MPI
--keep-case      保留 case；默认已经保留
--cleanup-case   npz 验证成功后删除该 Re 的 OpenFOAM case
--keep-vtk       保留 VTK；默认删除
--overwrite      已有有效 npz 时强制重跑
```

单个 Re 的 OpenFOAM MPI 流程是：

```text
blockMesh
checkMesh
decomposePar -force
mpirun -np N simpleFoam -parallel
mpirun -np N pimpleFoam -parallel   # spin-up
mpirun -np N pimpleFoam -parallel   # retained stage
reconstructPar
foamToVTK
scripts/03_vtk_to_npz.py
```

每个 shell 命令都会写入 `data/logs/` 下的独立日志，例如：

```text
data/logs/Re_000080_blockMesh.log
data/logs/Re_000080_simpleFoam_parallel.log
data/logs/Re_000080_pimpleFoam_retain_parallel.log
```

## Parallel Sweep

默认并行配置在 `config/sweep.yaml`：

```yaml
parallel:
  enabled: true
  total_cores: auto
  nProcs_per_case: 8
  concurrent_cases: 2
```

在 16 核虚拟机上，默认同时跑 2 个 Re，每个 Re 使用 8 个 MPI rank，总共 16 个 rank。脚本会检查：

```text
nProcs_per_case * concurrent_cases <= total_cores
```

如果超出 CPU 核数，会自动降低并发并给出 warning。也可以改成：

```yaml
nProcs_per_case: 16
concurrent_cases: 1
```

或：

```yaml
nProcs_per_case: 4
concurrent_cases: 4
```

运行 pilot sweep：

```bash
python3 scripts/02_run_sweep.py --config config/sweep.yaml --mode pilot
```

手动覆盖并行配置：

```bash
python3 scripts/02_run_sweep.py --config config/sweep.yaml --mode pilot --concurrent-cases 2 --nprocs-per-case 8
python3 scripts/02_run_sweep.py --config config/sweep.yaml --mode pilot --concurrent-cases 4 --nprocs-per-case 4
```

`pilot_diagnostics.csv` 会记录：

```text
Cd mean/std
Cl mean/std
probe v velocity std
periodic_hint
estimated_period
Strouhal
```

根据 `Cd/Cl` 和 probe 信号的 std、周期估计和主频，可以初步判断 onset 区间。这个判断只用于后续 Re 采样设计，不硬编码进训练标签。

full sweep 已写入配置，但不要自动运行。确认 pilot 后再手动执行：

```bash
python3 scripts/02_run_sweep.py --config config/sweep.yaml --mode full --concurrent-cases 2 --nprocs-per-case 8
```

## Restart Behavior

所有脚本都支持断点续跑。若 `data/npz/Re_XXXXXX.npz` 已存在且字段完整、无 NaN/Inf，默认跳过该 Re。需要重跑时加：

```bash
--overwrite
```

## NPZ Database

每个 Re 输出一个文件：

```text
data/npz/Re_000080.npz
```

字段包括：

```text
coords: [N, 2]
times: [T]
U: [T, N, 2]
p: [T, N]
Re
nu
metadata_json
```

验证数据库：

```bash
python3 scripts/04_verify_database.py
```

输出：

```text
data/manifest/database_summary.csv
```

如果不同 Re 的 `N` 或 `coords` 不一致，脚本会明确 warning，因为后续 POD 要求统一空间网格。

## Area Weights

面积加权 POD 使用 lumped nodal area：

```text
A_i = sum(cell_area / number_of_vertices_in_cell)
```

当前流体域面积为：

```text
40 * 8 - 1 * 1 = 319
```

构造面积权重：

```bash
python3 scripts/05_build_area_weights.py
```

输出：

```text
data/npz/point_area.npz
```

脚本会检查 `area > 0`，并检查 `sum(area)` 与 319 的相对误差。

## Weighted POD

速度内积：

```text
<q_a, q_b> = sum_i A_i (u_a,i u_b,i + v_a,i v_b,i)
```

压力内积：

```text
<p_a, p_b> = sum_i A_i p_a,i p_b,i
```

构造 area-weighted POD：

```bash
python3 scripts/06_build_weighted_pod_streaming.py \
  --ru 32 \
  --rp 32 \
  --pressure-gauge subtract_area_mean_per_snapshot
```

压力 gauge `subtract_area_mean_per_snapshot` 表示每个 pressure snapshot 在 POD 前减去面积加权空间平均压力，避免压力常数漂移污染 pressure POD。

输出：

```text
data/pod/velocity_pod.npz
data/pod/pressure_pod.npz
data/pod/modal_coefficients.npz
```

可用 CPU 选项：

```bash
python3 scripts/06_build_weighted_pod_streaming.py --ru 32 --rp 32 --blas-threads 8 --num-workers 1
```

不要把 OpenFOAM MPI sweep 和 Python POD 同时跑满 CPU。建议 sweep 完成后再单独运行 POD。

## Quick Figures

生成轻量诊断图：

```bash
python3 scripts/07_plot_quick_diagnostics.py
```

输出到：

```text
data/logs/figures/
```

包括 snapshot count、mean kinetic energy、pressure RMS、Cd/Cl std、probe v std、POD cumulative energy 和部分 modal coefficients。

## Dependencies

Ubuntu/OpenFOAM 侧需要：

```bash
python3 -m pip install pyyaml pyvista meshio matplotlib
```

当前脚本优先用 `pyvista` 读取 VTK，若不可用则尝试 `meshio`。两者都缺失时会明确报错，不会静默失败。

## Minimal Verification

实现或修改后，只跑最小流程：

```bash
source /opt/openfoam13/etc/bashrc
cd /home/ray/bottom_obstacle_attractor_rom
python3 scripts/00_make_template.py
python3 scripts/01_run_single_re.py --re 20 --mode test --nprocs 8
python3 scripts/04_verify_database.py
```

这不是正式数据库模拟，只验证 template、mesh、MPI、solver sequence、VTK 转 npz 和数据库校验链路。full sweep 必须等 pilot 结果确认后再手动运行。

## Operating Modes

`test` / smoke:

验证最小 OpenFOAM -> npz -> POD 链路。当前项目已经跑通，不需要作为日常任务重复执行。

`pilot_fast`:

forces/probes-only 快速扫描，不生成 VTK，不生成 npz，不做 POD。它只用于初步识别 unsteady candidates，不能直接作为最终 attractor database 的 regime 标签。

```bash
python3 scripts/02_run_sweep.py \
  --config config/sweep.yaml \
  --mode pilot_fast \
  --concurrent-cases 4 \
  --nprocs-per-case 4
```

Refine the short-scan diagnostics without rerunning OpenFOAM:

```bash
python3 scripts/08_refine_pilot_fast_diagnostics.py
```

Outputs:

```text
data/manifest/pilot_fast_refined_diagnostics.csv
data/manifest/pilot_fast_report.md
data/logs/figures/pilot_fast_refined_metrics.png
```

`pilot_viz`:

对少数代表性 Re 写少量场数据，运行 `reconstructPar` 和 `foamToVTK`，保留 VTK 用于 ParaView 查看流场状态。它不用于正式 ROM 数据库，默认不生成 npz，不做 POD。

```bash
python3 scripts/01_run_single_re.py \
  --re 80 \
  --mode pilot_viz \
  --nprocs 4 \
  --overwrite \
  --keep-vtk \
  --copy-vtk-preview
```

VTK 位置：

```text
work/cases/case_Re_000080/VTK/
data/vtk_preview/Re_000080/
```

Small visualization sweep:

```bash
python3 scripts/02_run_sweep.py \
  --config config/sweep.yaml \
  --mode pilot_viz \
  --concurrent-cases 2 \
  --nprocs-per-case 4 \
  --keep-vtk \
  --copy-vtk-preview
```

`probe_long`:

对少数候选 Re 延长 forces/probes 时间，用于确认周期性。默认不生成 npz，不做 POD，不生成 VTK。

```bash
python3 scripts/01_run_single_re.py --re 80 --mode probe_long --nprocs 4 --overwrite
```

`full` / production:

根据 pilot/probe_long 结果正式生成 attractor database，保存 retained snapshots、npz，最后做 area-weighted POD。不要在没有确认 pilot 结果前运行 full sweep。
