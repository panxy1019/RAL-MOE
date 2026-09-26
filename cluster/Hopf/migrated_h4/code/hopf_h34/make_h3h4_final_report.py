#!/usr/bin/env python3
"""Build the immutable H3/H4 validation-selected heldout report."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

NAMES=("HopfLocal32_H3_FluctuationNormalized","HopfLocal32_H4_NormalFormRadial")
SHORT={NAMES[0]:"H3 / FluctuationNormalized",NAMES[1]:"H4 / NormalFormRadial"}
HS=(1,2,4,8,16,24,48)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def pct(x):
 try:return "NaN" if x!=x or abs(x)==float('inf') else f"{100*x:.4f}%"
 except:return "—"
def num(x,n=4):
 try:return "NaN" if x!=x or abs(x)==float('inf') else f"{x:.{n}f}"
 except:return "—"
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |',*('| '+' | '.join(map(str,r))+' |' for r in rows)])
def atomic(obj,p):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=True),encoding='utf8');os.replace(q,p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--finalization-dir',type=Path,required=True);ap.add_argument('--trainer',type=Path,required=True);ap.add_argument('--evaluator',type=Path,required=True);ap.add_argument('--parent-b-metrics',type=Path);a=ap.parse_args();root=a.finalization_dir
 sel=json.loads((root/'checkpoint_selection_manifest.json').read_text());held=json.loads((root/'heldout_metrics.json').read_text());parent=json.loads(a.parent_b_metrics.read_text()) if a.parent_b_metrics else None
 val=[];k48=[];diag=[];multi=[];energy=[];rhs=[]
 for name in NAMES:
  s=sel['experiments'][name];val.append([SHORT[name],s['optimizer_step'],f"{s['validation_score']:.8f}",'PASS' if s['hard_gate'] else 'FAIL',s['sha256']])
  e=held['experiments'][name]
  for re,row in e['by_re'].items():
   for h in HS:
    m=row['horizons'][str(h)];multi.append([SHORT[name],re,f'K{h}',pct(m['velocity_area_weighted_physical_relative_l2']),pct(m['pressure_area_weighted_physical_relative_l2']),pct(m['velocity_modal_relative_l2']),pct(m['pressure_modal_relative_l2']),pct(m['finite_fraction']),m['divergent_windows'],m['first_divergence_step'] or '—'])
   m=row['horizons']['48'];d=row['hopf_diagnostics_k48'];k48.append([SHORT[name],re,pct(m['velocity_area_weighted_physical_relative_l2']),pct(m['pressure_area_weighted_physical_relative_l2']),pct(m['velocity_modal_relative_l2']),pct(m['pressure_modal_relative_l2']),pct(m['finite_fraction']),m['divergent_windows'],'PASS' if row['hopf_attractor_preserved'] else 'FAIL'])
   diag.append([SHORT[name],re,pct(d['rms_amplitude_error']),pct(d['peak_to_peak_amplitude_error']),pct(d['growth_sign_accuracy']),f"{d['true_local_log_growth']:.3e}",f"{d['predicted_local_log_growth']:.3e}" if d['predicted_local_log_growth']==d['predicted_local_log_growth'] else 'NaN',pct(d['frequency_relative_error']),num(d['terminal_phase_drift_cycles']),num(d['normalized_orbit_distance']),'是' if d['false_growth'] else '否'])
   energy.append([SHORT[name],re,pct(m['velocity_energy_drift']),pct(m['pressure_energy_drift'])]);rhs.append([SHORT[name],re,pct(d['one_step_rhs_relative_l2']),pct(d['one_step_pressure_closure_relative_l2'])])
 preserved={n:held['experiments'][n]['heldout_preserved_count'] for n in NAMES};improve=(sel['experiments'][NAMES[0]]['validation_score']-sel['experiments'][NAMES[1]]['validation_score'])/sel['experiments'][NAMES[0]]['validation_score'];h3r=held['experiments'][NAMES[0]]['by_re'];h4r=held['experiments'][NAMES[1]]['by_re']
 parent_rows=[]
 if parent:
  pb=parent['experiments']['HopfLocal32_V16ScaleAware_K1248']
  for re in ('47.081355','49.022357','51.786450'):
   b=pb['by_re'][re];h=h4r[re];parent_rows.append(['B / frozen parent',re,pct(b['hopf_diagnostics_k48']['rms_amplitude_error']),pct(b['hopf_diagnostics_k48']['peak_to_peak_amplitude_error']),pct(b['hopf_diagnostics_k48']['frequency_relative_error']),'PASS' if b['hopf_attractor_preserved'] else 'FAIL']);parent_rows.append([SHORT[NAMES[1]],re,pct(h['hopf_diagnostics_k48']['rms_amplitude_error']),pct(h['hopf_diagnostics_k48']['peak_to_peak_amplitude_error']),pct(h['hopf_diagnostics_k48']['frequency_relative_error']),'PASS' if h['hopf_attractor_preserved'] else 'FAIL'])
 lines=['# H3/H4 Hopf-local r32 最终实验报告','', '## 1. 最终结论','',f"H3 为 **{preserved[NAMES[0]]}/3**、H4 为 **{preserved[NAMES[1]]}/3** heldout attractor preserved；两组都没有达到预注册的至少 2/3 成功标准。H4 是明确更优版本，但仍不能判定整体 Hopf 闭环成功。",'',f"H4 的 validation 主分数相对 H3 改善 **{pct(improve)}**。H4 在 Re=51.786450 完整通过联合门，并把 Re=47.081355 从 H3 的 K48 首次 step {h3r['47.081355']['horizons']['48']['first_divergence_step']} 发散改为全程 finite、零 divergent window。",'',f"主要残余失败位于 near-onset Re=47.081355：H4 的 RMS 振幅误差为 {pct(h4r['47.081355']['hopf_diagnostics_k48']['rms_amplitude_error'])}、P2P 振幅误差为 {pct(h4r['47.081355']['hopf_diagnostics_k48']['peak_to_peak_amplitude_error'])}，且仍触发 false growth。Re=49.022357 虽保持数值稳定，但振幅联合门仍失败。",'', '## 2. Validation-only checkpoint 冻结','', 'heldout 在 checkpoint 选择和冻结完成后才首次解封。H3 的最低裸分数出现在 step 8000，但该点 Hopf 硬门失败，因此按合同冻结唯一合格的 step 6400；H4 冻结所有合格点中分数最低的 step 7600。','',table(['版本','冻结 step','validation score','硬门','SHA-256'],val),'','## 3. Heldout K48 主结果','',table(['版本','Re','u physical','p physical','u modal','p modal','finite','divergent','attractor'],k48),'','所有 relative error 均已乘 100，以百分数表示。physical-field 分母受均值场主导，数值很小不能单独证明小幅 Hopf 振荡被保留；必须结合振幅、增长、频率、相位和 orbit 指标判断。','','## 4. Hopf 吸引子诊断','',table(['版本','Re','RMS振幅误差','P2P振幅误差','growth-sign','真增长','预测增长','频率误差','末端相位/周期','orbit距离','false growth'],diag),'','H3 在 Re=51.786450 已非常接近成功：RMS、频率、相位、orbit 和能量均通过，但 P2P 振幅误差为 10.3412%，略高于 10% 门槛。H4 将该 P2P 误差降至 5.1703%，因而完整通过。','','## 5. 多 horizon 终评','',table(['版本','Re','horizon','u physical','p physical','u modal','p modal','finite','divergent','首次发散 step'],multi),'','## 6. 能量与一步动力学诊断','',table(['版本','Re','K48速度能量漂移','K48压力能量漂移'],energy),'',table(['版本','Re','one-step RHS误差','one-step pressure closure误差'],rhs),'','H4 的 attractor 指标优于 H3，但 one-step RHS 误差并未同步下降，说明 normal-form 辅助约束的主要收益来自闭环轨道尺度与长期稳定性，而不是局部 RHS 拟合精度。','','## 7. 与冻结父模型 B 的对照','']
 if parent_rows:lines += [table(['版本','Re','RMS振幅误差','P2P振幅误差','频率误差','attractor'],parent_rows),'','H4 相比父 B 在 Re=51.786450 将 RMS/P2P 振幅误差从 18.4821%/24.2607% 降至 1.6603%/5.1703%，并首次通过 attractor 门；near-onset 虽有改善，但 false growth 尚未消除。','']
 lines += ['## 8. 成功判据与数据隔离','', '单个 Re 的联合门要求：K48 全程 finite、零 divergent window、u/p physical relative L2≤5%、RMS/P2P 振幅误差≤10%、频率误差≤5%、末端相位漂移≤0.25 周期、orbit distance≤0.10、u/p 能量漂移≤10%，且无 false growth。','','POD、均值、scaler、Galerkin/pressure tensors、Hopf 主平面、振幅 floor 和增长统计均只由 12 个训练 Re 拟合；validation 仅用于 checkpoint 选择；三个 heldout Re 只在冻结完成后进入本 evaluator。','','## 9. SwanLab 与产物','', '- H3: https://swanlab.cn/@panxy1019/V17_HopfLocal32_H3H4/runs/438o46wf','- H4: https://swanlab.cn/@panxy1019/V17_HopfLocal32_H3H4/runs/6ma5nb9v','- `checkpoint_selection_manifest.json`：validation-only 选择证据与冻结 SHA。','- `heldout_metrics.json`：逐版本、逐 Re、逐 horizon 完整指标。','- 各实验目录中的 `final.pt`：只读冻结 checkpoint。','- `artifact_inventory.json`：最终产物大小与 SHA-256。','']
 (root/'FINAL_H3H4_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
 summary={'schema_version':1,'status':'FINAL_HELDOUT_COMPLETE','success_criterion':'at least 2/3 heldout attractors preserved','success':{n:preserved[n]>=2 for n in NAMES},'preserved_count':preserved,'validation_improvement_H4_vs_H3':improve,'selected_steps':{n:sel['experiments'][n]['optimizer_step'] for n in NAMES},'interpretation':'H4 improves long-horizon stability and preserves Re=51.786450, but near-onset false growth remains.'};atomic(summary,root/'FINAL_H3H4_SUMMARY.json')
 atomic({'status':'COMPLETE','selection_split':'validation_only','heldout_used_after_freeze_only':True,'report':'FINAL_H3H4_REPORT.md'},root/'FINALIZATION_COMPLETE.json')
 inv={'schema_version':1,'trainer':{'path':str(a.trainer),'sha256':sha(a.trainer)},'evaluator':{'path':str(a.evaluator),'sha256':sha(a.evaluator)},'artifacts':[]}
 for p in sorted(root.rglob('*')):
  if p.is_file() and p.name!='artifact_inventory.json':inv['artifacts'].append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)})
 atomic(inv,root/'artifact_inventory.json');print(json.dumps({'report':str(root/'FINAL_H3H4_REPORT.md'),'preserved':preserved,'success':summary['success']}))
if __name__=='__main__':main()
