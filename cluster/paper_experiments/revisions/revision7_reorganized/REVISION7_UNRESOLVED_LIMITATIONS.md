# Revision 7 未解决限制与比较缺口

## 不可由本轮整理解决的缺口

1. **单固定 seed。** Vanilla、DataOnly 和 Proposed 的现有消融不支持跨 seed 方差或统计显著性结论。
2. **Global MoE† 的 transductive representation。** Global POD、mean 和 scaler 使用 all-Re 数据并包含 held-out Re；它不是严格 inductive baseline。
3. **不同原生时域。** Steady/Hopf/Global 为 K56，Periodic specialist 为 K48；而且数据库快照间隔随 Re 改变。尚无共同无量纲时间校准，不能构造统一全域平均。
4. **Global strict evaluator 缺口。** Global 已有 K56 场和系数误差，但未运行 Steady fixed-point continuation、Hopf strict 或 Periodic strict 的同合同 evaluator。
5. **S–H Global 精确对照缺口。** Global 没有 sealed Re=42.359071、43.50、43.90 的完整同 evaluator 结果；禁止用近邻 Re 填补。
6. **S–H 非全程 blind。** 本轮重组没有使用 test 调参，但历史 test 已被查看。
7. **H–P 没有共同 validation-supported 可行域。** Hopf 在 P-native validation 上为 0/6 finite K56；因此没有合法 H–P soft gate、cache 或 checkpoint。
8. **Hopf strict 泛化缺口。** Re=49.3、49.6、50.0 属于 train；正式 held-out strict count 是 0/3。
9. **Steady strict contraction 缺口。** Pressure fixed-point+drift 联合门 0/20，worst paired gain 仍大于 1；不能声称全方向收缩。
10. **Router 时间迁移缺口。** 数据没有经认证的 startup 或缓慢 Re(t) ramp 轨迹，当前只支持 temporal-consistent regime routing，不支持真实 S→H→P 迁移主张。
11. **部分指标没有统一冻结。** Proposed Steady 的全 split pressure scan 没有同一文件中的 tail increment、terminal increment 与 false-oscillation 二值汇总；revision 7 保留 N/A，未从其他合同拼接。
12. **Core 系统数值的输出同一性。** 表 3 在纯 core 区将 E2/Proposed 的确定性 Top-1 输出引用为 native specialist 结果；这不是独立重跑的 unified-wrapper test。若期刊要求“完整 wrapper 逐区域独立复测”，需另立新实验合同，不能由本轮文稿整理补出。

## 现阶段可以补充、但本轮明确未执行的工作

- 多 seed 重复及置信区间。
- 共同无量纲物理时间上的 KLong 认证。
- Global MoE† 的 train-only POD/mean/scaler 版本。
- Global 在 S–H exact sealed cache 上的同 evaluator 结果。
- Hopf held-out strict-preservation 改进实验。
- H–P 共同 development overlap 的 warm-start re-certification。
- startup 与缓慢参数 ramp CFD 轨迹，用于真实跨流态 routing。

这些项目均涉及新训练、重跑 evaluator、创建新数据或重新认证，超出 revision 7 的只读整理权限。
