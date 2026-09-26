# Global MoE 同口径单检查点复评

本目录保存 2026-09-26 Global MoE 复评的冻结窗口、逐步误差、聚合结果、独立校验和复现脚本。未覆盖旧实验，也未修改论文。

## 状态

- 1/1 个既有 Global MoE 检查点完成；
- Hopf 42/42、Periodic 32/32 个窗口完成；
- 3552/3552 个逐步记录有限；
- 目标时间键完全匹配；
- 独立聚合与主聚合最大差 `8.8818e-16`；
- 跨基 Gram 与直接物理解码抽查最大差 `1.3329e-11` 个百分点。

最终联合误差（%）：

| Regime | K=24 | K=48 |
|---|---:|---:|
| Hopf | 0.1314 | 0.1299 |
| Periodic | 2.7135 | 3.8990 |

这是单检查点结果，不带训练种子误差棒。完整模型身份、公平性边界和逐 Re 结果见 `Global_MoE_同口径复评报告_20260926.md`。

## 本地重算

```powershell
python .\aggregate_global_results.py `
  --input .\single_checkpoint_eval\per_step_errors.csv `
  --output-dir .\final_aggregate `
  --target-windows .\target_window_steps.csv

python .\independent_verify.py `
  --steps .\single_checkpoint_eval\per_step_errors.csv `
  --target .\target_window_steps.csv `
  --summary .\final_aggregate\final_summary.csv `
  --direct-audit .\single_checkpoint_eval\cross_basis_direct_decode_audit.json `
  --output .\independent_verification.json
```
