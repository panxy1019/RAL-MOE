from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
inputs={'POD-Galerkin':'final_periodic_galerkin_v1','Dense correction':'final_periodic_dense_v1','Global MoE':'global_common_truth_v1','RAL local specialist':'final_periodic_full_v1'}
rows={m:list(csv.DictReader((root/p/'window_step_errors.csv').open())) for m,p in inputs.items()}
keys=[]
for m,rs in rows.items():
    keys.append({(round(float(r['Re']),4),int(r['start']),int(r['step'])) for r in rs})
assert all(k==keys[0] for k in keys)
out=root/'aligned_rollout_figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(10,3),constrained_layout=True)
records=[]
for (method,rs),color in zip(rows.items(),['#999999','#009E73','#D55E00','#0072B2']):
    for k in range(1,49):
        part=[r for r in rs if int(r['step'])==k]
        stats={m:float(np.mean([float(r[m]) for r in part])) for m in ['Eu_percent','Ep_percent','Ejoint_percent']}
        records.append(dict(method=method,step=k,windows=len(part),**stats))
    for ax,metric in zip(axs,['Eu_percent','Ep_percent','Ejoint_percent']):
        rr=[r for r in records if r['method']==method]
        ax.plot([r['step'] for r in rr],[r[metric] for r in rr],label=method,color=color)
        ax.set(xlabel='Rollout step k',ylabel=metric.replace('_percent','')+' (%)')
        ax.set_yscale('log')
axs[0].legend(fontsize=7)
fig.suptitle('Circular Periodic · 12 aligned held-out windows · common POD-reconstructed truth',fontsize=10)
fig.savefig(out/'four_method_error_growth.pdf')
fig.savefig(out/'four_method_error_growth.png',dpi=220)
with (out/'aggregate_curves.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
end=[r for r in records if r['step']==48]
(out/'K48_summary.json').write_text(json.dumps(end,indent=2))
print(json.dumps(end,indent=2))
plt.close(fig)
fig,axes=plt.subplots(4,3,figsize=(10,10),constrained_layout=True)
re_values=sorted(set(round(float(r['Re']),4) for r in next(iter(rows.values()))))
per_re=[]
for i,re in enumerate(re_values):
    for (method,rs),color in zip(rows.items(),['#999999','#009E73','#D55E00','#0072B2']):
        part=[r for r in rs if round(float(r['Re']),4)==re]
        terminal=[r for r in part if int(r['step'])==48]
        per_re.append(dict(Re=re,method=method,**{m:float(np.mean([float(r[m]) for r in terminal]))
            for m in ['Eu_percent','Ep_percent','Ejoint_percent']}))
        ts=[np.mean([float(r['time']) for r in part if int(r['step'])==k]) for k in range(1,49)]
        for j,metric in enumerate(['Eu_percent','Ep_percent','Ejoint_percent']):
            vals=[np.mean([float(r[metric]) for r in part if int(r['step'])==k]) for k in range(1,49)]
            axes[i,j].plot(ts,vals,label=method,color=color)
            axes[i,j].set(xlabel='Elapsed native time',ylabel=metric.replace('_percent','')+' (%)',title=f'Re = {re:g}')
            axes[i,j].set_yscale('log')
axes[0,0].legend(fontsize=6)
fig.suptitle('Common projected truth; pressure gauge aligned; three windows per Re',fontsize=11)
fig.savefig(out/'per_Re_physical_time_curves.pdf')
fig.savefig(out/'per_Re_physical_time_curves.png',dpi=170)
with (out/'per_Re_K48.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(per_re[0]));w.writeheader();w.writerows(per_re)
print(json.dumps(per_re))
