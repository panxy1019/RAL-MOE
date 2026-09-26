# 恢复数据与复核

## 1. 克隆及按需恢复附件

```sh
git clone https://github.com/panxy1019/RAL-MOE.git
cd RAL-MOE
python tools/restore_archive.py --scope all
```

脚本仅使用 Python 标准库，从本仓库固定日期的 Release 下载附件，验证 SHA-256，恢复原路径并重建去重文件。`--scope local` 仅恢复本地资料，`--scope cluster` 仅恢复集群资产。文件已存在且哈希一致时跳过；内容不同则停止，避免覆盖修改。下载包逐个处理，默认保留在 `.archive_downloads/`；可加 `--remove-downloads` 在验证并解包后删除本次下载包。请预留清单所示文件总大小及至少一个附件的额外空间。

模型、POD 基、物理算子、原始场、压缩交付包和大文件通常存于附件。`archive/*-manifest.json` 每项的 `storage` 为 `git`、附件名或 `duplicate`；后一项通过 `duplicate_of` 指向相同内容的原路径。清单内 SHA 是实际归档字节；旧报告自带的哈希可能指向未脱敏的历史文件，应同时查看 `redactions`。

## 2. 不需要 GPU 的最新结果重算

安装 NumPy 后，在仓库根目录运行：

```sh
python global_moe_reeval_20260926/aggregate_global_results.py --input global_moe_reeval_20260926/single_checkpoint_eval/per_step_errors.csv --output-dir verification/global_aggregate --target-windows global_moe_reeval_20260926/target_window_steps.csv
python global_moe_reeval_20260926/independent_verify.py --steps global_moe_reeval_20260926/single_checkpoint_eval/per_step_errors.csv --target global_moe_reeval_20260926/target_window_steps.csv --summary verification/global_aggregate/final_summary.csv --direct-audit global_moe_reeval_20260926/single_checkpoint_eval/cross_basis_direct_decode_audit.json --output verification/global_independent.json
```

ql-ROM-style 的结果聚合入口是 `qlrom_20260923/build_report.py`；其 README 说明了运行位置和输出。建议先复制到新的工作目录，避免覆盖冻结报告。

## 3. 集群训练或模型推理

`cluster/` 对应原集群 `particalMOE/` 项目根目录；原机器上的运行脚本可能使用该根目录及其他数据目录的绝对路径。迁移时显式改为自己的路径，使用新输出目录，依照对应实验的冻结协议、配置和数据清单运行。

保留的环境记录包括各实验的配置、日志和计时元数据。完整训练依赖 PyTorch、NumPy、SciPy 等，某些工具另外依赖 OpenFOAM、Matplotlib 或 scikit-learn；以各实验文档为准。归档不包含 Conda/系统运行环境，也未在本次上传中重做昂贵训练。

集群附件保留正式/冻结/验证选择模型以及必要资产；中间检查点、筛选候选和大型可再生成数组的省略原因记录在清单 `omitted`。这不是一份无遗漏的机器镜像。原服务器 SSH 密码和私钥不在仓库中。
