import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];ROOT=R.parents[1]
study=ROOT/'experiments/iclr_seed_study_20260911'
sys.path.insert(0,str(study/'code'))
from run_study import data_at,evaluate_gate
from common import sha256
root=ROOT/'centeredsquare_fusion_runs/E2_T2C_K24_20260730_STRICT_V3'
out=R/'fusion_seed_reevaluation';out.mkdir(exist_ok=False);rows=[]
for pair in ['sh','hp']:
 data=data_at(root/f'heldout_evaluation_20260730_V1/cache_{pair}/{pair}_heldout_cache.npz')
 dev=root/f'cache_{pair}/{pair}_development_cache.npz'
 for mode in ['full','mu_only']:
  for seed in [42001,42002,42003]:
   path=study/f'results_v1/centered_square/{pair}/{mode}/seed_{seed}/best.pt'
   ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['cache_sha256']==sha256(dev)
   result=evaluate_gate(path,root/'e2_router/best.pt',data);alpha=np.array(result['alpha_by_window'])[:,None]
   err=[]
   for name in ['quad_u','quad_p']:
    q=data[name];err.append(100*np.sqrt(np.maximum((alpha*alpha*q[:,:,0]+(1-alpha)**2*q[:,:,1]+2*alpha*(1-alpha)*q[:,:,2])/np.maximum(q[:,:,3],1e-12),0)))
   for re in np.unique(data['re']):
    mask=data['re']==re
    for K in [8,16,24]:
     row=dict(pair=pair,mode=mode,seed=seed,Re=float(re),K=K,checkpoint_sha256=sha256(path))
     for label,e in zip(['Eu','Ep','Ejoint'],[err[0],err[1],err[0]+err[1]]):
      row.update({label+'_window_mean':float(e[mask,:K].mean()),label+'_terminal':float(e[mask,K-1].mean()),label+'_window_max':float(e[mask,:K].max(1).mean())})
     rows.append(row)
with (out/'per_Re.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
summary=[]
for pair in ['sh','hp']:
 for mode in ['full','mu_only']:
  for K in [8,16,24]:
   values=[np.mean([r['Ejoint_window_mean'] for r in rows if r['pair']==pair and r['mode']==mode and r['K']==K and r['seed']==s]) for s in [42001,42002,42003]]
   summary.append(dict(pair=pair,mode=mode,K=K,seed_means=values,mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1))))
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
