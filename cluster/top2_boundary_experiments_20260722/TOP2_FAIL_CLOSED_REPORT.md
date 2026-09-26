# 相邻 Top-2 三路线执行报告

日期：2026-07-22  
最终状态：`FAIL_CLOSED`  
训练状态：三条路线均未启动训练，未创建候选 checkpoint，test 保持封存。

## 1. 已完成工作

三条路线已经按统一协议预注册：

1. `T2-C LearnedConvexCorrection_FieldBlend`
2. `RiskPredictionRouter`
3. `LookAheadShortRolloutRouter`

共同协议固定了：

- seed `42001`、8000 optimizer steps、batch size 256；
- complete-Re、complete-trajectory split；
- 只允许 S–H、H–P，禁止 S–P 和 Top-3；
- E2 和三个 specialist 全部冻结；
- 只做轨迹级、输出级物理场融合，不做 RHS 融合，不反馈融合状态；
- 相同边界阈值搜索空间、相同 native 输出缓存、相同物理场与吸引子指标；
- train-only 损失尺度，validation-only 阈值、权重与 checkpoint 选择；
- 三个 validation-selected checkpoint 全部冻结前，final evaluator 不得访问 test 状态或指标。

预注册配置 SHA256：

| 路线 | 配置 SHA256 |
|---|---|
| LearnedConvex | `1302da2e7b8d8539eab5bf04e9d8721e703da12207ef5c0e9d95bf91fa23eeb4` |
| RiskPrediction | `fe358483e7ead8cc6a6f71725234dd1ca8d654f2449a2820a108065762833c3c` |
| LookAhead | `40f13526087fedfd7aae60b00b016d26e341bf49d4def162029317c686d7c87e` |

## 2. 通过的前置门

- E2 与 S/H/P checkpoint SHA 均固定；
- 三套物理网格点顺序完全一致；
- cell-area 权重完全一致；
- pressure gauge 均为 `subtract_area_mean_per_snapshot`；
- Hopf 为 phase-free specialist；
- 邻接拓扑可以强制限制为 S–H、H–P；
- 审计没有加载 test 物理场、POD 状态或指标。为排除 test，只扫描了 index 中的 split 标签。

## 3. 硬失败原因

三条路线都依赖同一项共同资产：从同一个物理初始状态和历史出发，让相邻两个 frozen specialists 在同一查询时钟上独立 rollout，并缓存两个物理场输出。

该双路 native 合同当前不成立。

| Specialist | phase harmonics | CSV phase/period | native forward | 跨来源自主启动 |
|---|---:|---|---|---|
| Steady | 4 | 均无 | 读取 indexed `phase[current]` | FAIL |
| Hopf | 0 | 均无 | phase-free | PASS |
| Periodic | 4 | 均无 | 读取 indexed `phase[current]` | FAIL |

Steady 和 Periodic 的 loader 在没有 phase/period 字段时，使用整条数据库轨迹的最终时间把 progress 归一化为 phase。冻结 forward 随后从 `arrays['phase'][current]` 读取该值。这个合同只覆盖“模型在自己的数据库轨迹上运行”，不能为来自相邻 chart 的任意物理历史提供 phase。

因此：

- S–H：Hopf 可以自主启动，但 Steady 无认证的跨来源 phase，FAIL；
- H–P：Hopf 可以自主启动，但 Periodic 无认证的跨来源 phase，FAIL；
- 使用相邻数据库轨迹 endpoint、未来 phase 或真实中间状态补齐会违反本轮合同；
- 随意将 phase 置零或按查询长度归一化会改变 checkpoint 输入分布，也没有 validation 认证；
- phase 门未通过前，无法合法执行严格时间对齐的双路 rollout smoke。

预检 JSON SHA256：`82a77a981a19a3c878efa1768609497dabb700c11bcc505a286941a0c378caab`。

## 4. 为什么三条路线都必须停止

### LearnedConvex

没有合法的双专家物理场输出，就无法以真实场和吸引子损失训练融合权重。

### RiskPredictionRouter

候选风险标签需要每个 candidate 的长期 native rollout 误差和发散结果；缺少合法 cross-source rollout 时，风险标签本身无定义。

### LookAheadShortRolloutRouter

LookAhead 必须在不读取未来真值的情况下运行两个 candidate。给 Steady/Periodic 注入数据库 endpoint phase 会直接破坏这条核心定义。

所以不能只训练其中某一路，也不能用 E0/E2 历史结果代替本轮共享缓存。

## 5. Test 隔离说明

历史 E2/E3 工作已经报告过旧 test 指标，因此本项目不能宣称“人类和代理从未见过 test”。本轮执行采用机械隔离：预检只保留 train/validation index 行，未加载 test 物理/modal 状态或指标；由于 preflight 失败，final test evaluator 没有启动。

## 6. 恢复路线

若要继续并保持当前三个 specialist 主体不随机重训，需要先完成下面二选一：

1. **冻结模型外的 phase/time provider 认证**：仅使用当前三步物理历史和查询时钟，产生与 Steady/Periodic checkpoint 训练分布兼容的 phase；必须在 train/validation 上证明 feature 分布、K4/K8/K16/K56 和吸引子指标等价。
2. **warm-start phase 前端再认证**：保持 specialist 主干、Galerkin、MoE、pressure closure 与 POD 不变，只对 phase 输入前端或极小输出接口进行 warm-start 认证/微调。该路线会放宽“specialist 全部参数严格冻结”，需要新的明确授权。

此前自主 `atan2` phase estimator 对 Periodic 数据库 progress 的 validation phase MAE 约为 1.57 rad、频率相对误差约为 0.98，不能直接作为已认证方案。

在上述门通过前，唯一可执行的正式系统仍是已冻结 E2 trajectory-level Top-1 baseline。

## 7. 产物

- 三套预注册配置：`preregistration/`
- 结构化预检：`preflight_audit/TOP2_PREFLIGHT_AUDIT.json`
- 终止标志：`preflight_audit/FAIL_CLOSED.json`
- 远端根目录：`/root/panxy/particalMOE/top2_boundary_experiments_20260722`
