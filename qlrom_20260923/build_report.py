"""Regenerate tables and audit report from immutable experiment output."""
from pathlib import Path
import csv,json,statistics as st,hashlib,difflib
from collections import defaultdict
BASE=Path(__file__).resolve().parent;R=BASE/'results'
def read(name):return list(csv.DictReader((R/name).open(encoding='utf-8')))
def js(name):return json.loads((R/name).read_text(encoding='utf-8'))
ORDER=['proposed','dense','structured','OpInf','ql-ROM-style (fixed POD)']
NAMES={'proposed':'Proposed, original','dense':'Dense, active-matched','structured':'Structured, original','OpInf':'OpInf','ql-ROM-style (fixed POD)':'ql-ROM-style（固定 POD）'}
def f(row,key):
    m=float(row[key+'_mean']);sd=row[key+'_sample_sd']
    return f'{m:.4f} ± {float(sd):.4f}' if sd!='' else f'{m:.4f}'
def table(data):
    index={(x['method'],int(x['horizon'])):x for x in data}
    lines=['| 方法 | K24 Eu (%) | K24 Ep (%) | K48 Eu (%) | K48 Ep (%) | 完成数/种子 |','|---|---:|---:|---:|---:|---|']
    for m in ORDER:
        a,b=index[m,24],index[m,48]
        lines.append('| '+NAMES[m]+' | '+' | '.join([f(a,'Eu'),f(a,'Ep'),f(b,'Eu'),f(b,'Ep')])+f" | 32/32 × {b['seed_count']} |")
    return '\n'.join(lines)
def latex(data,name):
    index={(x['method'],int(x['horizon'])):x for x in data};lines=[];joint=[]
    def cell(r,k):return '$'+f(r,k).replace(' ± ',r'\pm ' )+'$'
    for m in ORDER:
        a,b=index[m,24],index[m,48];label=NAMES[m] if m!='ql-ROM-style (fixed POD)' else 'ql-ROM-style (fixed POD space)'
        lines.append(label+' & '+' & '.join([cell(a,'Eu'),cell(a,'Ep'),cell(b,'Eu'),cell(b,'Ep')])+r' \\')
        joint.append(label+' & '+cell(a,'joint')+' & '+cell(b,'joint')+r' \\')
    caption=(f'Errors against {name} on the same 32 frozen Circular Periodic windows. '
        'A single autonomous 48-output rollout provides both horizons. Errors are averaged over output times, windows within each Reynolds number, '
        'and equally weighted Reynolds numbers, followed by the mean and sample standard deviation across three neural-training or clustering seeds. '
        'All 32 windows complete for every seed; OpInf is deterministic. Neural-training and clustering randomness are not paired. '
        'The ql-ROM-style baseline is independently implemented within the inherited fixed 32-dimensional POD space, with eight local charts of rank 16 and an inherited algebraic PPE; '
        'it is not a full-physical-space reproduction of ql-ROM. Pressure is independently area-mean centered for prediction and reference.')
    return '% Component rows\n'+'\n'.join(lines)+'\n\n% Joint-error rows\n'+'\n'.join(joint)+'\n\n% Suggested caption\n'+caption+'\n'
def verify_steps(paths,summary):
    groups=defaultdict(list);n=0
    for path in paths:
        for x in read(path):
            m=x['method'];m='ql-ROM-style (fixed POD)' if m.startswith('ql-ROM') else m
            groups[m,x['seed'],float(x['Re']),int(x['window'])].append(x);n+=1
    values=defaultdict(list)
    for (m,seed,re,w),rr in groups.items():
        assert len(rr)==48 and sorted(int(x['step']) for x in rr)==list(range(1,49))
        for K in [24,48]:
            use=[x for x in rr if int(x['step'])<=K]
            values[m,seed,re,K].append((st.mean(float(x['Eu']) for x in use),st.mean(float(x['Ep']) for x in use)))
    seedvalues=defaultdict(list)
    for (m,seed,re,K),vv in values.items():seedvalues[m,seed,K].append((st.mean(v[0] for v in vv),st.mean(v[1] for v in vv)))
    allseed=defaultdict(list)
    for (m,seed,K),vv in seedvalues.items():
        assert len(vv)==4
        u,p=st.mean(v[0] for v in vv),st.mean(v[1] for v in vv);allseed[m,K].append((u,p,u+p))
    worst=0.
    for row in summary:
        vv=allseed[row['method'],int(row['horizon'])]
        for i,k in enumerate(['Eu','Ep','joint']):
            worst=max(worst,abs(st.mean(v[i] for v in vv)-float(row[k+'_mean'])))
            if len(vv)>1:worst=max(worst,abs(st.stdev(v[i] for v in vv)-float(row[k+'_sample_sd'])))
    assert worst<1e-10
    return dict(step_rows=n,model_seed_windows=len(groups),max_summary_difference=worst,pass_check=True)

def main():
    pod=read('summary_pod_reference.csv');raw=read('summary_raw_cfd.csv');val=read('validation_search.csv')
    verified={'pod':verify_steps(['baseline_per_step_metrics.csv','per_step_metrics.csv'],pod),'raw':verify_steps(['raw_per_step_metrics.csv'],raw)}
    assert len(val)==96 and all(int(x['completed'])==48 for x in val)
    assert len(list((R/'predictions').glob('*.npz')))==96
    assert len(read('completion_all_methods.csv'))==832
    assert all(x['complete']=='True' for x in read('completion_all_methods.csv'))
    frozen=js('protocol_frozen.yaml');original=R/'code/qlrom_experiment_frozen.py';current=R/'code/qlrom_experiment.py'
    assert hashlib.sha256(original.read_bytes()).hexdigest()==frozen['script_sha256']
    assert hashlib.sha256((R/'validation_search.csv').read_bytes()).hexdigest()==js('selected_config.json')['validation_search_sha256']
    verified.update(validation_records=96,selected_test_predictions=96,completion_rows=832,all_complete=True,original_frozen_hash_matches=True)
    (R/'delivery_verification.json').write_text(json.dumps(verified,indent=2),encoding='utf-8')
    diff=''.join(difflib.unified_diff(original.read_text().splitlines(True),current.read_text().splitlines(True),fromfile='frozen_fit_validation',tofile='optional_physical_input_and_precomputed_ops'))
    (R/'implementation_interface_change.diff').write_text(diff,encoding='utf-8')
    (BASE/'table_rows_pod.tex').write_text(latex(pod,'the common training-only POD-reconstructed reference'),encoding='utf-8')
    (BASE/'table_rows_raw_cfd.tex').write_text(latex(raw,'the original CFD reference fields'),encoding='utf-8')
    t=js('timing.json')['results'];diag=js('deployment_and_switch_diagnostics.json');audit=js('raw_reference_audit.json');unit=js('unit_checks.json')
    validation=defaultdict(list)
    for x in val:validation[int(x['J']),int(x['r']),int(x['substeps'])].append(float(x['joint']))
    best=sorted((st.mean(v),k) for k,v in validation.items());bestJ1=min((a,k) for a,k in best if k[0]==1)
    vtable='| J | r | 子步/输出 | 验证联合误差 (%) |\n|---:|---:|---:|---:|\n'+'\n'.join(f'| {k[0]} | {k[1]} | {k[2]} | {a:.6f} |' for a,k in best[:8])
    time_table='| 方法 | 中位数 s | IQR s | 均值 s | 样本 SD s | GPU 峰值 MiB | 进程 RSS MiB |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for x in t:
        time_table+=f"| {x['method']} | {x['median']:.6f} | {x['iqr']:.6f} | {x['mean']:.6f} | {x['sample_sd']:.6f} | {x.get('peak_gpu_allocated_bytes',0)/2**20:.2f} | {x['peak_process_rss_bytes']/2**20:.2f} |\n"
    dt_table='| Re | 输出 Δt 范围 s | K48 物理时长范围 s |\n|---|---:|---:|\n'+'\n'.join(f"| {x['label']} | {x['output_dt_min']:.6f}–{x['output_dt_max']:.6f} | {x['K48_duration_min']:.6f}–{x['K48_duration_max']:.6f} |" for x in diag['physical_time'])
    off='| 聚类 seed | 聚类+局部 POD s | 单参数算子变换 s | 测试切换总次数 | 每输出区间最大累计投影跳变 |\n|---:|---:|---:|---:|---:|\n'+'\n'.join(f"| {x['seed']} | {x['fit_cluster_POD_seconds']:.6f} | {x['operator_transform_one_parameter_seconds']:.6f} | {x['total_switches']} | {x['max_jump_per_output_interval']:.6f} |" for x in diag['models'])
    projection=read('raw_projection_diagnostics.csv');pr='| K | 父 POD 速度投影误差 % | 父 POD 压力投影误差 % | 局部速度投影误差 % |\n|---:|---:|---:|---:|\n'
    for K in [24,48]:
        rr=[x for x in projection if int(x['horizon'])==K]
        pr+='| '+str(K)+' | '+' | '.join(f"{st.mean(float(x[k]) for x in rr):.6f}" for k in ['parent_velocity_projection_error_percent','parent_pressure_projection_error_percent','local_velocity_projection_error_percent'])+' |\n'
    report=f'''# Circular Periodic：ql-ROM-style 公平复现与预测对比报告

日期：2026-09-23。状态：本次受限实现的身份审计、训练/验证搜索、三聚类种子测试、双参考复评、配对计时与交付校验已完成。没有改动论文、旧 checkpoint 或旧实验结果。

## 1. 结论先读

本次依据论文独立实现了 **ql-ROM-style within a fixed POD space**，不是作者官方代码，也不是完整物理空间 ql-ROM。选型只使用验证集：J=8、局部速度秩 r=16、父速度/压力空间各 32 维、每输出间隔 4 个 RK4 内步。聚类种子 1248、1600、2026 均完成全部 32 个测试窗口，K24 是同一 K48 自主预测的前缀。

在当前受限设定中，ql-ROM-style 比三类神经模型延迟低，但 Eu/Ep 更大；Dense 的三种子平均精度最好，不能把该表写成 Proposed 全面领先。OpInf 的有限误差很大但没有非有限失败，必须保留，不能删作异常值。该实验仅比较 Periodic 局部预测，不检验 E2 或 T2-C。

最重要的限制：父 POD 只能表达 32 维；继承 PPE 忽略弱式边界项；63 个 Re 标签下的动量及 PPE 数组完全相同。单位测试证明了“对继承算子的变换正确”，**不证明继承算子完整表达了真实参数/边界效应**。现有结果可做受限补充对照，不宜包装成对完整 ql-ROM 的否定。

## 2. 来源、方法身份和原结果核对

来源：[Colanera 与 Magri, arXiv:2506.13738v1](https://arxiv.org/html/2506.13738v1)，方法与数值附录。详细公式、作者代码检索状态、适配边界见 [source_and_adaptation.md](source_and_adaptation.md)。未核实到官方代码，故官方 commit/代码许可为 N/A；本次独立实现的源码 SHA256 与既有模型代码 SHA256 均留档。

`results/model_identity_manifest.csv` 记录 9 个神经 checkpoint 及 1 个 OpInf 资产的路径/配置/hash。Proposed/Structured 为 original，不混入 repaired。直接从各自 modal_rollouts 与 frozen protocol 复评，而非从表格倒推行名，重现了用户清单中的 Proposed/Dense/Structured 数值对应关系（四位小数）。

新版逐步误差相对旧 evaluator 的最大绝对差为 {max(float(x['Eu_max_abs_diff']) for x in js('baseline_reaggregation_checks.json')):.7f} 个百分点（Eu）、{max(float(x['Ep_max_abs_diff']) for x in js('baseline_reaggregation_checks.json')):.7f} 个百分点（Ep），来源是统一 FP64 二次型计算与独立压力零均值处理，并非调参对齐。聚合值四位小数仍一致；逐模型差异见 `baseline_reaggregation_checks.json`。

## 3. 数据、窗口和泄漏核查

父速度/压力基元数据均为 train-only。训练 53 Re/8,539 快照，验证 6 Re/967 快照，测试 4 Re/644 快照；Re 组之间无交叉。聚类和局部 POD 只读取训练样本。验证 48 窗口，测试复用既有冻结 32 窗口、每 Re 8 个；历史和 48 步目标不跨 split/轨迹。

测试起始全局索引：976, 984, 1000, 1016, 1024, 1040, 1056, 1072；3874, 3882, 3898, 3914, 3922, 3938, 3954, 3970；7258, 7266, 7282, 7298, 7306, 7322, 7338, 7354；9513, 9521, 9537, 9553, 9561, 9577, 9593, 9609。对应轨迹、历史索引、真实时间与输出时间保存在 `windows.json`。

{dt_table}

继承 float32 时间缓存，随后转 FP64；表中的微小间隔起伏是既有时间精度。原始场保存的 float64 时刻与 provenance 精确一致。重叠窗口不是独立统计样本，没有进行普通显著性检验。没有生成新增训练 CFD。

## 4. 冻结搜索与选型

网格 J∈{{1,2,4,8}}，r∈{{16,32}}，子步∈{{1,2,4,8}}，每配置三个聚类种子，共 96 次验证评价；每次 48 个窗口完整且有限。r64 因父空间只有32维预先排除。k-means++，n_init=10，max_iter=300，tol=1e−4；所有簇有效秩32，没有空簇/秩不足拒绝。

排序先要求三种子全部验证窗口完成，再最小化三种子验证 Eu48+Ep48 均值；相对1e−8并列按 r/J/子步选择较小配置。冻结于 {js('selected_config.json')['selected_at']}，随后才测试。首轮最优位于4子步，预注册“边界8子步扩展”规则未触发。

{vtable}

J=1 最佳候选为 r={bestJ1[1][1]}、{bestJ1[1][2]} 子步，验证联合误差 {bestJ1[0]:.6f}%。这是验证 sanity baseline，未按测试结果筛选或另外报告测试最优值。

另在测试前对已选结构检查1/2/4/8/16子步趋势。4→8→16子步的聚合联合误差变化很小（约0.2%以内），但因离散切换时刻改变，逐点轨迹差并不单调缩小。不能称已经证明严格时间收敛。未因此修改选中配置；详见 `time_refinement_validation.json`。

## 5. 数学和实现检查

全部基本检查通过：加权正交性最大误差 {unit['weighted_orthogonality']:.3e}，局部 RHS 与直接投影差 {unit['local_vs_direct_rhs']:.3e}，压力变换差 {unit['pressure_transform']:.3e}，中心切换公式差 {unit['switch_formula']:.3e}，J1/r32 RK4退化差 {unit['J1_full_rank_RK4_degeneracy']:.3e}。没有漏掉中心诱导的常数/交叉项。当前 PPE 伪逆 Moore–Penrose 相对残差为 {diag['operator_checks']['L_pseudoinverse_Moore_Penrose_relative_residual']:.3e}。

完整 RK4 内步后根据预测速度选择新簇，并保存投影跳变；没有用真实未来标签。单次在线只推进一个局部模型。速度和代数压力保留非有限判定，不做真值重置/输出裁剪；本次没有测试失败，有限大误差全部纳入。

## 6. 表 A：共同 POD 重构参考

{table(pod)}

每个误差为逐步相对面积加权 L2 ×100；先时间平均、Re内窗口平均、Re等权平均，最后跨种子均值和样本标准差。压力预测/参考分别去面积均值。没有添加 epsilon；全部参考范数非零。联合误差先在每种子上 Eu+Ep，再求样本标准差，不相加两项SD。

## 7. 表 B：原始 CFD 参考——所有方法重新计算

{table(raw)}

四个公开原始场均已完整获取，逐文件 SHA256、URL、网格与时间核验见 `raw_reference_audit.json`。97,368 个点的点序完全一致；原始时刻与 provenance 最大差0。原始场重新投影与缓存系数最大差：速度 {max(x['u_cache_vs_raw_projection_max_abs'] for x in audit):.3e}、压力 {max(x['p_cache_vs_raw_projection_max_abs'] for x in audit):.3e}，符合缓存存储精度。

复评使用预测重构场相对原始场的完整加权范数二次型，而不是对旧总误差加减投影误差。抽样与显式物理场差分核对最大误差 {max(max(x['u_quadratic_vs_explicit_error_max_abs'],x['p_quadratic_vs_explicit_error_max_abs']) for x in audit):.3e} 个百分点。未重新拟合、改变测试输入或更换 rollout。

表 A 和 B 必须分开；若采用 B 为论文主表，必须整表替换，不能把 B 的 ql-ROM 行接入 A 的旧神经行。

## 8. 表示误差与切换诊断

{pr}

局部投影诊断用真实状态最近中心，仅用于离线表示误差分析，不参与自主预测。所有组都有8窗口，因此本表对窗口/种子的平均等于指定Re等权聚合。局部16维截断也引入误差；但当前预测误差还包含继承动力学、压力模型、积分和切换的影响，不能把差异单独归因于聚类或PPE。

{off}

跳变量为加权速度范数的绝对量（不是百分比），表中最大值是一个输出区间的累计值，详见逐步CSV。每窗口切换0–19次；说明存在真实自主切换，不能将其描述为固定路由模型。

## 9. 统一配对端到端计时

{time_table}

同一机器、Re70.314635、start976、seed1248（OpInf确定性）、batch1、K48，3次预热+10次重复，顺序运行。CPU：Intel Xeon Platinum8358P；GPU：RTX4090。BLAS/OpenMP和PyTorch intra-op为2线程；原神经程序 inter-op 实际为64，原样记录，未声称所有线程池都是2。ql-ROM/OpInf为CPU FP64；神经为FP32网络+CPU物理算子/FP64物理编码解码。共享宿主机未锁定CPU亲和性，也无连续负载轨迹；GPU前后快照为空闲，不能解释为绝对独占硬件。

边界：从内存中共同 POD 重构的物理初始历史开始，包含编码、历史RHS（神经需要）、初次选簇/切换、积分、压力恢复、全部48步场解码、压力规范和必要CPU/GPU传输；不含加载/磁盘IO、误差统计或离线算子构建。ql-ROM和OpInf只用当前速度，是方法所需输入差别，不强迫其计算多余历史特征。CUDA每次重复前后同步；不在各内部算子强制同步。

计时输出相对冻结缓存的最大系数差在神经模型中≤1.64e−7，ql-ROM约2.49e−14；OpInf约5.92e−8，来自物理编码及浮点路径，未替换模型或省略输出。逐次时间、均值/SD、IQR、设备信息见 `timing.json`。GPU数字为PyTorch峰值已分配内存；RSS为整个进程高水位，包含加载的验证资产和输出缓冲，**不是纯模型存储**。两者不要直接相加。

这些是各方法现有CPU/GPU实现的延迟，不是同FLOPs或同设备极限；也不是ROM/FOM加速比。约0.099秒与约2.183秒的差别必须和精度一起呈现，不用历史CFD耗时生成新加速比。

## 10. 离线成本、存储和预算

上表离线聚类/POD仅包括8539×32训练表示上的增量拟合，算子变换仅一组参数的8个局部模型。继承父POD/PPE的生成时间不可由本次追溯计入，明确不把0.1秒级增量时间称完整离线训练成本。96次验证窗口搜索的累计程序计时约 {sum(float(x['seconds']) for x in val):.2f} 秒，另有选中结构的时间加密诊断；验证计时含调用中的算子变换，不是部署延迟。

每种子中心/局部基+局部速度/压力张量数值存储873,472字节（FP64），每次只推进一个16维速度局部模型；压力输出32维。父场基另占37,389,312字节FP32，运行时转FP64约翻倍，均值/编码器/缓存另计。63个参数标签对应张量相同，可共享一份数值表；未宣称与神经active-matched参数量相等。

## 11. 交付、复算与验证

- `results/protocol_frozen.yaml`（JSON形式，合法YAML1.2）、`windows.json`、`model_identity_manifest.csv`：协议和来源。
- `results/unit_checks.json`、`time_refinement_validation.json`、`data_audit.json`：检查。
- `results/validation_search.csv`、`selected_config.json`：全部候选与唯一冻结选型。
- `results/models/`：24套训练聚类/POD模型和3套选中局部算子；`predictions/`：96个测试窗口预测NPZ，包含簇、切换、跳变及时间。
- `results/per_step_metrics.csv`：新增方法；`baseline_per_step_metrics.csv`：旧方法；`raw_per_step_metrics.csv`：全方法原始场误差。
- `results/completion.csv`：新增方法；`completion_all_methods.csv`：全部方法832条“模型/种子/窗口/长度”状态；`raw_completion.csv`：原始场参考。
- `results/summary_pod_reference.csv`、`summary_raw_cfd.csv`、`seed_metrics_*.csv`：全精度聚合。
- `results/timing.json`、`deployment_and_switch_diagnostics.json`、`raw_reference_audit.json`：成本、诊断与原始场证据。
- `table_rows_pod.tex`、`table_rows_raw_cfd.tex`：自动生成的组件/联合误差LaTeX行及caption，未写入论文。
- `results/code/`及本目录脚本：实现；`results/delivery_verification.json`：从全部逐步CSV独立重建表格的验证结果。

冻结拟合/验证脚本的SHA匹配最初协议。其后仅给rollout增加可选物理编码初值和预构算子入口供统一计时使用，默认行为未变；差分保存在 `implementation_interface_change.diff`。独立交付验证确认两套参考各19,968条逐步记录、416个模型种子窗口，可复算10行汇总，无缺失/重复步，表格最大数值差<1e−10。运行 `python build_report.py` 可从保存CSV重建本文和LaTeX表，不需要访问GPU或原始场。

## 12. 论文建议与尚未完成的完整复现范围

可在附录新增受限方法对照，采用完整名称或在行名附近加明确脚注；正文只陈述“本受限配置中，神经模型精度较高，ql-ROM-style推理延迟更低”。必须同时保留Dense优于Proposed的事实。

**尚未完成且未冒充完成：**完整物理场训练局部基；重新离散并保留参数/边界lifting的动量算子；与CFD一致的完整PPE边界处理；官方代码一致性验证；跨几何与长期稳定性比较。这些均超出本次已实施的受限路线。需要完整外部方法证据时，应另开冻结协议，不能因本次测试结果不佳而静默补调。
'''
    (BASE/'qlrom_comparison_report.md').write_text(report,encoding='utf-8')
    print(json.dumps(verified,indent=2));print('REPORT',BASE/'qlrom_comparison_report.md')

if __name__=='__main__':main()
