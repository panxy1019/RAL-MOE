# 仅需作者确认的事项

1. **Figure 1 内嵌旧术语。** 用户提供的位图仍有 affine / low-rank bilinear 字样；正文已按第三轮要求称 linear / low-rank quadratic responses。请提供或授权修改图源。本轮没有把位图内容静默重绘。
2. **模型与结果身份。** Pinball Periodic 的 B1 / sparse-MoE 来源关联、Pinball H final-test 来源、Circular S 的 S4/S3-B 最终选择仍需证据确认。本轮未修改数值、方法标签或 checkpoint 身份。
3. **descriptor 定义与实现。** 按要求保留六维数学定义；此前八维实现差异仍需作者决定如何解释，不因术语一致而自动闭合。
4. **统一公式与 benchmark-specific 实现。** 保留当前公式；此前 Square H 等实现路径差异仍需作者核对。本轮没有重开架构修订。
5. **AI use statement。** 当前声明仍写 solely language polishing and LaTeX formatting；请核对是否准确覆盖实际使用的代码、分析、图示或实验协助。本轮不自行判断 venue policy，也不替作者确认该句真实。
6. **补充材料与最终 Overleaf 编译。** 请确认 supplementary implementation 实际随稿提供。上传后使用 main.tex 为主文件；本地已编译并核验，但 Overleaf 的 TeX Live 版本可能造成细微分页差异，建议上传后重编译检查。

这些事项不在纯语言/浮动体修改中推断解决。详细证据可参考上一轮的 unresolved_scientific_items.md；内部报告不放入本轮 Overleaf 源码包。
