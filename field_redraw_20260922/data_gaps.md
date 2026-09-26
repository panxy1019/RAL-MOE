# 数据边界与未纳入材料

1. 本次总览三例齐全，但 Circular 行是局部 Periodic 预测，不是 T2-C。标题与图注没有把它写成统一融合实验。
2. H–P 融合图只包含完成同窗/E2/gate/候选核验的 Square、Pinball。没有建立 Circular 的相同 H–P 融合场数据链，因此未补造第三行。
3. **Circular S–H 历史融合数据确实有线索，不应说不存在。** 已定位：`particalMOE/paper_experiments/visualization/config_sh_fusion.json`、`PER_WINDOW_T2C_WEIGHTS.csv`，以及 `top2_boundary_experiments_20260722/one_sided_recovery_v2/resplit_20260723_v1/cache_final_test/G_SH_resplit_final_test_cache.npz`。该历史配置使用独立的 S/H 基、`time_offset=1` 和 resplit 协议；本交付未将其进一步纳入三案例 S–H 图，也未声称完成该资产的 E2/gate/checkpoint/seed 全链核验。本次 S–H 文件名明确只有 Square/Pinball。
4. 本次参考是 POD 重构；没有新增 raw CFD 对照，也不能用这些图评价 POD 截断误差。
5. Circular 资产报告为 raw physical POD、固定 `nu=0.001`。这与把全部图统一称为 `nu=1/Re` 无量纲幅值不兼容；本次保留原尺度并披露，不更改论文方法章节。若作者希望统一无量纲图，需要另行核实每例的 U、D、压力和时间变换后重导出。
6. 包内物理数组足以独立改图；重新选择任意其他时刻仍需服务器原缓存/检查点/POD 基，未在本包复制全部大型训练资产。
7. 这些代表帧不是平均性能证据。Square S–H、Pinball H–P 的该帧 T2-C 联合误差并非低于所有单候选；按预先规则保留，没有换帧。
