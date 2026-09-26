# Square CFD 运行时间核查

- 核查日期：2026-09-20
- 几何：CenteredSquare（方柱绕流）
- 范围：已有日志、终端记录、运行脚本和状态文件的只读核查。本轮未重跑 CFD。

## 结论

**现有记录可作为物理时间 0--500 的 FOM 数据生产管线端到端 wall-clock，但不能与 K24=96 的 ROM 时间直接相除。**

现有 `case_status.tsv` 的 START→DONE 时间戳能证明每个 case 正常计算到物理时间 500，并完成了 NPZ 转存和校验。2026-09-20 重新连接同一 VM 后，已补齐 CPU、内存、虚拟化、OpenFOAM 和 MPI 版本。按本次要求，5-case 并发仅作为历史运行条件记录，不作为时间证据的否决项。未找到的仍是求解器最后一行 `ExecutionTime/ClockTime`，因此不能把 START→DONE 声称为纯 solver 时间。

因此：

- 可使用现有 START→DONE 报告“物理时间 0--500 的历史 FOM 端到端成本”。只有当 ROM 也测量同一 Re、同一 0--500 物理区间时，才能直接代入 `Speedup=t_FOM/t_ROM,end-to-end`。
- 若论文保持当前 K24=96 的 ROM 口径，最小复跑建议仍为 Periodic held-out `Re=120.689655172414`、3 MPI ranks，但原因是匹配物理时间窗口，而不是历史并发数。
- 更完整的补充是再跑 Hopf held-out `Re=96.5`。

## 已找到的时间证据

| 几何 | 流态 | 物理参数 | 数据划分 | 网格 | 物理时间 / dt / 步数 | 求解器 | 硬件 | 并行规模 | Wall-clock | 计时范围 | 可否用于论文 |
|---|---|---:|---|---|---|---|---|---|---:|---|---|
| Square | Steady | Re=85 | held-out test | 19,306 points / 9,400 cells | 0--500 / 0.02 / 25,000 | OpenFOAM 13 `icoFoam`, CN0.9 | 16-vCPU VMware VM，Intel Core i5-14600K，14 GiB RAM | 1 node, 3 ranks/case | 1,347 s (22:27) | START 前于 case 复制；DONE 在分解、求解、重构、NPZ 转存/校验/清理后 | **是，作为 0--500 FOM 端到端时间** |
| Square | Hopf | Re=96.5 | held-out test | 同上 | 同上 | 同上 | 同上 | 同上 | 1,828 s (30:28) | 同上 | **是，作为 0--500 FOM 端到端时间** |
| Square | Periodic | Re=100 | train | 同上 | 同上 | 同上 | 同上 | 同上 | 2,178 s (36:18) | 同上 | **是，但不是 held-out** |
| Square | Periodic | Re=120.689655172414 | held-out test | 同上 | 同上 | 同上 | 同上 | 同上 | 2,757 s (45:57) | 同上 | **是，作为 0--500 FOM 端到端时间** |
| Square | Periodic | Re=144.827586206897 | held-out test | 同上 | 同上 | 同上 | 同上 | 同上 | 2,401 s (40:01) | 同上 | **是，作为 0--500 FOM 端到端时间** |
| Square | Periodic | Re=100 | pilot/classification | 非冻结 held-out | 同上 | 0--500 / 0.02 / 25,000 | 同上 | 同上 | 3 ranks | 246 s (04:06) | copy + decompose + solver + reconstruct + `foamToVTK` | **不用于 held-out 论文主表** |
| Square | Periodic | Re=120 | pilot/classification | 非冻结 held-out | 同上 | 同上 | 同上 | 同上 | 3 ranks，与 Re100 并发 | 310 s (05:10) | 同上 | **不可用，参数不等于 Re120.689655** |

正式 100-case 任务的整体批处理时间为 2026-07-08 01:01:49--10:49:42 CST，即 9:47:53；Re50 在此前单独 dry-run 已完成，正式批处理对其进行了 SKIPPED。该时间只能表示 5-case 并发完成 100-case 数据库的总吞吐，不能作为单轨迹 FOM 时间。

## 补充硬件与软件信息

2026-09-20 16:39 CST 在同一主机 `192.168.232.130` 上实际执行 `LC_ALL=C lscpu`、`free -h`、OpenFOAM 环境查询和 `mpirun --version`，得到：

| 项目 | 实测信息 |
|---|---|
| 主机 | `ray-virtual-machine`, `192.168.232.130` |
| 虚拟化 | VMware full virtualization，1 NUMA node |
| CPU | `Intel(R) Core(TM) i5-14600K` |
| vCPU | 16 online CPUs (`0-15`)，1 thread/core |
| 内存 | 15,223,208 kB，`free -h` 显示约 14 GiB |
| OS/kernel | Ubuntu 22.04 系，Linux `6.8.0-138-generic` x86_64 |
| OpenFOAM | OpenFOAM 13，`WM_OPTIONS=linux64GccDPInt32Opt` |
| MPI | Open MPI 4.1.2 |
| FOM GPU | 未使用；`icoFoam` 在 CPU/MPI 上运行 |

这是现存同一 VM 的当前实测配置。没有发现 VM 硬件配置在 2026-07-08 之后变更的记录；但因原始日志未嵌入 `lscpu`，严格说它是“同 VM 现时配置”，而不是历史日志自带的硬件快照。

## 日志与脚本位置

### 当前可读证据

- 原始 Codex 终端记录：`C:\Users\panxy1019\.codex\sessions\2026\07\07\rollout-2026-07-07T23-49-45-019f3d45-62a5-7fa3-ae4c-72fa8949a025.jsonl`。
  - JSONL 行 147：初始 CN09 graded sweep 的完整运行脚本和命令。
  - JSONL 行 267--289：pilot Re55--120 的 START/DONE 记录。
  - JSONL 行 552/564：正式 Re85 START/DONE。
  - JSONL 行 599/610：正式 Re96.5 START/DONE。
  - JSONL 行 610--634：正式 Re100 START/DONE。
  - JSONL 行 646--668：正式 Re120.689655 START/DONE。
  - JSONL 行 692--716：正式 Re144.827586 START/DONE。
- 正式 runner 副本：`C:\Users\panxy1019\Documents\CHANNEL\remote_scripts\run_formal_re50_150_n100_npz.sh`。
- 加密 runner 副本：`C:\Users\panxy1019\Documents\CHANNEL\remote_scripts\run_refined_re95_102_n29_npz.sh`。
- Periodic 冻结 held-out 依据：`C:\Users\panxy1019\Documents\CHANNEL\.transfer_square_periodic\CENTERED_SQUARE_PERIODIC_FINAL_REPORT.md`。
- K24 物理场评价口径：`C:\Users\panxy1019\Documents\CHANNEL\.transfer_square_periodic\PHYSICAL_FIELD_RECONSTRUCTION_EVAL.md`。

### 历史路径（当前 VM 已不存在）

- pilot 状态：`192.168.232.130:/home/ray/Desktop/centeredSquare/runs/cn09_graded_20260708_000142/status.tsv`。
- pilot solver 日志：各 case 的 `/home/ray/Desktop/centeredSquare/runs/cn09_graded_20260708_000142/Re*/run.log`。
- formal 状态：`192.168.232.130:/home/ray/Desktop/centeredSquare/dataset_Re50_150_N100_npz/manifest/case_status.tsv`。
- formal solver 日志：`192.168.232.130:/home/ray/Desktop/centeredSquare/dataset_Re50_150_N100_npz/logs/Re*_run.log`。
- formal 配置证据：`192.168.232.130:/home/ray/Desktop/centeredSquare/dataset_Re50_150_N100_npz/logs/case_configs/Re*/{0,constant,system}`。
- refined 状态/日志：`192.168.232.130:/home/ray/Desktop/centeredSquare/dataset_Re95_102_refined_N31_npz/{manifest/case_status.tsv,logs}`。

2026-09-20 16:39 CST 再次重连 VM 成功，但仍确认 `/home/ray/Desktop/centeredSquare` 不存在，全 `/home/ray` 搜索也未找到原 `case_status.tsv/status.tsv` 或 Square `Re*_run.log`。因此上述远程日志路径是历史证据，不是当前可直接读取的路径。本项目没有使用 Slurm/PBS 等调度器，任务是在单台 VM 上通过 `nohup`/shell 后台运行，因此没有调度器 job accounting 可以补齐计时。

## 关键日志摘录

### 正式 held-out case

```text
Re085p000000 START 2026-07-08T04:00:45+08:00 index=41 Re=85.000000000000
Re085p000000 DONE  2026-07-08T04:23:12+08:00 npz validated and work case removed

Re096p500000 START 2026-07-08T05:52:06+08:00 index=64 Re=96.500000000000
Re096p500000 DONE  2026-07-08T06:22:34+08:00 npz validated and work case removed

Re120p689655 START 2026-07-08T08:05:05+08:00 index=83 Re=120.689655172414
Re120p689655 DONE  2026-07-08T08:51:02+08:00 npz validated and work case removed

Re144p827586 START 2026-07-08T10:04:13+08:00 index=97 Re=144.827586206897
Re144p827586 DONE  2026-07-08T10:44:14+08:00 npz validated and work case removed
```

### 完成状态

```text
Re150p000000 DONE       2026-07-08T10:49:42+08:00 npz validated and work case removed
DATASET CASES_DONE      2026-07-08T10:49:42+08:00 case stage complete
DATASET DONE_NO_POD     2026-07-08T10:49:42+08:00 case data complete; POD intentionally skipped
snapshots=100  probes=100  active_cfd=0  failed=0  fatal_logs=0
```

### 求解配置

```text
application     icoFoam;
endTime         500;
deltaT          0.02;
writeControl    timeStep;
writeInterval   200;
mpirun -np 3 icoFoam -parallel
```

`500 / 0.02 = 25,000` 个 CFD 时间步；每 200 步写盘，包含 t=0 共 126 帧。每 case 的 probe 有 2,501 个样本，对应每 10 步记录一次并包含 t=0。

## 计时边界解释

正式 runner 的顺序是：

```text
START timestamp
copy template -> write nu -> preserve config
decomposePar
mpirun -np 3 icoFoam -parallel
reconstructPar
convert reconstructed U/p/probes to NPZ
validate shape/time/Nc/finite values
delete temporary OpenFOAM case
DONE timestamp
```

所以 START→DONE 是“CFD 数据产生管线端到端时间”，不是纯 `icoFoam` 时间。它包含结果写盘、重构和 NPZ 转存，不包含网格生成（网格每个 dataset 只构建一次），不包含 POD。记录中没有重启段；每个 DONE 均是单段计算完成后记录。

正式任务的运行配置是每 case 3 MPI ranks，数据集生产时最多同时运行 5 个 case。按本次要求，该条件只作为可复现性元数据，不在“可否用于论文”判断中扣除或否决这些实测 wall-clock。

## 论文可用性判断

| 必需信息 | 现状 | 判断 |
|---|---|---|
| 几何、Re、测试轨迹 | held-out Re 和冻结 split 可确认 | 通过 |
| 网格规模 | 19,306 points / 9,400 cells | 通过 |
| 物理时间与步数 | 0--500, dt=0.02, 25,000 steps | 通过 |
| 求解器与完成状态 | OpenFOAM 13 `icoFoam`，NPZ/probe 校验通过 | 通过 |
| 并行规模 | 3 ranks/case，1 node；历史数据集生产最多 5 case 同时运行 | 通过，不作否决项 |
| CPU 型号 | 同一 VM 当前实测：16-vCPU Intel Core i5-14600K / VMware / 14 GiB | 通过，注明为现时硬件快照 |
| FOM 端到端 wall-clock | START→DONE 边界可由 runner 和状态时间戳完整解释 | 通过 |
| 纯 solver ClockTime | 末行 `ExecutionTime/ClockTime` 日志已丢失 | 仅当声称“纯 solver time”时不通过 |
| 与 ROM 相同物理窗口 | 旧 CFD 计时是 0--500；ROM 主报告为 K24=96 物理时间 | **不通过** |

综合结论：**若 ROM 补测同一 Re 的 0--500 全区间，可直接使用现有 FOM START→DONE 端到端时间，无需复跑 CFD。若 ROM 仍只报告 K24=96，则需代表性 CFD 复跑以匹配物理窗口，不需重跑整个数据库。**

## 最小代表性复跑方案

### 方案 A：Periodic held-out（推荐）

| 项目 | 配置 |
|---|---|
| 参数 | `Re=120.689655172414`, `nu=1/Re=0.008285714285714285` |
| 选择依据 | 冻结 Periodic held-out；既有原始 CFD 快照；既有历史 START/DONE 证据；非接口边界点 |
| 原始快照 | `C:\Users\panxy1019\Documents\CHANNEL\.transfer_square_periodic\snapshots_Re120p689655.npz` |
| 原始配置路径 | 历史：`192.168.232.130:/home/ray/Desktop/centeredSquare/dataset_Re50_150_N100_npz/logs/case_configs/Re120p689655` |
| 必要前置 | 先从备份恢复该 `0/constant/system` 或正式 template，并与 `mesh_metadata.npz` 的 9,400 cell center/volume 校验一致 |
| 时间窗口 | 与 Periodic K24 评价窗口一致：24 个外部快照步，物理时长 96 |
| CFD 时间步 | 固定 `dt=0.02`，共 4,800 步 |
| 初始状态 | 从冻结评价器的某一个 K24 window start 对应原始 `U,p` 重启；窗口索引必须由评价器导出，不自行挑选 |
| ROM 历史 | 同一 window start 之前的 3 帧用于 history encoding；ROM 计时包含历史编码 |
| 并行 | 1 node，3 MPI ranks，单 case，无其他 CFD/POD 进程 |
| 写盘 | `writeInterval=200`，即每 4 物理时间写一帧；保留 probe；不运行 `foamToVTK` |
| 预计时间 | 隔离运行预计约 1--6 min/次，以实测为准；建议 1 次冷启动检查 + 5 次计时复现 |

注：当前 VM 恢复的 `/home/ray/Desktop/centeredSquare_restore_20260827/Re100` 已证实与正式 Re100 NPZ 不匹配，**不得用作复跑模板**。

### 方案 B：Hopf held-out（扩展）

- 参数：`Re=96.5`，冻结 Hopf held-out。
- 原始快照：`C:\Users\panxy1019\Documents\CHANNEL\cluster_hopf_transfer\hopf_transfer_34\cases_npz\snapshots_Re096p500000.npz`。
- 历史配置：`192.168.232.130:/home/ray/Desktop/centeredSquare/dataset_Re50_150_N100_npz/logs/case_configs/Re096p500000`。
- 主论文统一比较使用 K24：96 物理时间、4,800 CFD 步。
- 如同时报告 Hopf specialist native K48：192 物理时间、9,600 CFD 步，必须与 ROM K48 单独成对报告。
- 其他计时与隔离要求同方案 A。

## 复跑命令与计时记录

以下是建议命令，**本轮未执行**：

```bash
source /opt/openfoam13/etc/bashrc
cd /absolute/path/to/frozen_heldout_case

# 系统与软件证据
date -Ins > timing_environment.txt
lscpu >> timing_environment.txt
nproc >> timing_environment.txt
uname -a >> timing_environment.txt
mpirun --version >> timing_environment.txt 2>&1
icoFoam -help >> timing_environment.txt 2>&1

# 前处理单独计时
/usr/bin/time -v -o timing_decompose.txt decomposePar -force \
  > log.decomposePar 2>&1

# 纯 FOM solver，包含 solver 自身的快照/probe 写盘
/usr/bin/time -v -o timing_icoFoam.txt \
  mpirun -np 3 icoFoam -parallel > log.icoFoam 2>&1

# 重构单独计时
/usr/bin/time -v -o timing_reconstruct.txt reconstructPar \
  > log.reconstructPar 2>&1
```

另用一层脚本在 `decomposePar` 之前和 `reconstructPar` 之后记录 `date +%s.%N`，得到 FOM end-to-end；如果论文希望把 NPZ 转存算入 FOM 端到端，应将 conversion 单独计时并列出，不与 solver time 混在一个数里。

每次复跑必须保留：

1. `timing_environment.txt`、`timing_icoFoam.txt`、`log.icoFoam`。
2. START/END wall-clock 和退出码。
3. `Time = <end>`、末行 `ExecutionTime/ClockTime`、`End`。
4. `controlDict/fvSchemes/fvSolution/decomposeParDict/physicalProperties`的 SHA-256。
5. 网格文件的 SHA-256、`checkMesh` 的 points/cells 输出。
6. 每次复现的 wall-clock；报告平均值、标准差和原始重复值。

## ROM 时间对应要求

ROM 必须以 batch size 1，预热后重复测量，并在同一个 `t_ROM,end-to-end` 中包含：

1. 3 帧初始物理历史编码。
2. E2 参数路由和专家选择。
3. 专家 K24 自主滚动，包含 RK4 内部子步。
4. 速度 POD 物理场重构。
5. 如该参数使用双专家，包含 T2-C 物理空间融合。
6. 代数压力重构。

模型加载和训练时间单独报告。为避免将不同物理区间的耗时相除，计算加速比前必须确认 FOM 和 ROM 使用同一 Re、同一 window start、同一物理时长 96。
