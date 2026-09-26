# Revision 8 changelog

- 新增独立 `revision8_detailed_per_re`，未覆盖 revision 6/7。
- 删除正文以跨 Re 平均值代替逐点结果的写法。
- 所有 Test Re 均逐行列出速度/压力物理场误差。
- 在冻结 evaluator 有保存时，逐行增加速度/压力 POD 系数误差。
- Steady、Hopf、Periodic 吸引子消融均展开到方法 × Re。
- 保留 Hopf 正式 held-out strict 0/3 与指定 train 案例 strict 3/3 的区别。
- 保留 Periodic DataOnly 3/3 strict PASS 负向消融结果。
- 为所有不可报告字段给出冻结合同级 N/A 原因，未反推或补算。
