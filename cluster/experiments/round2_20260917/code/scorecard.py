"""Build transparent macro-Re scorecards, never dropping failed windows or seeds."""
import csv
import json
from pathlib import Path
import numpy as np
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--fixed',action='store_true')
parser.add_argument('--h-evaluation-family',default=None);parser.add_argument('--output-suffix',default=None);args=parser.parse_args()
suffix=args.output_suffix or ('_fixed' if args.fixed else '_current')
R=Path(__file__).resolve().parents[1]
records=[]
def score(rows,regime,method,seed):
 for K in [8,16,24,48]:
  subset=[r for r in rows if int(r['step'])<=K]
  revalues=sorted(set(float(r['Re']) for r in subset));result=dict(regime=regime,method=method,seed=seed,K=K,reference='POD reconstructed fields, not FOM')
  for key in ['Eu_percent','Ep_percent','Ejoint_percent']:
   scores=[];term=[];window_max=[];tw=[];completed=total=0
   for re in revalues:
    rr=[r for r in subset if float(r['Re'])==re];windows=sorted(set(str(r.get('window',r.get('start'))) for r in rr));wm=[];tm=[];mx=[];tw_re=[]
    for window in windows:
     wr=sorted([r for r in rr if str(r.get('window',r.get('start')))==window],key=lambda r:int(r['step']))
     vals=np.array([float(r[key]) for r in wr]);ok=len(vals)==K and np.isfinite(vals).all();total+=1;completed+=int(ok)
     wm.append(float(np.mean(vals)) if ok else float('nan'));tm.append(float(vals[-1]) if ok else float('nan'));mx.append(float(np.max(vals)) if ok else float('nan'))
     times=np.asarray([float(r.get('elapsed',r.get('time',float('nan')))) for r in wr])
     dt=np.diff(np.r_[0.,times]);valid_time=len(times)==K and np.isfinite(times).all() and np.all(dt>0)
     tw_re.append(float(np.sum(vals*dt)/times[-1]) if ok and valid_time else float('nan'))
    scores.append(np.mean(wm));term.append(np.mean(tm));window_max.append(np.mean(mx))
    tw.append(np.mean(tw_re))
   result[key+'_window_mean']=float(np.mean(scores)) if np.isfinite(scores).all() else None
   result[key+'_terminal']=float(np.mean(term)) if np.isfinite(term).all() else None
   result[key+'_window_max']=float(np.mean(window_max)) if np.isfinite(window_max).all() else None
   result[key+'_physical_time_mean']=float(np.mean(tw)) if np.isfinite(tw).all() else None
   result['complete_windows']=completed;result['total_windows']=total
  records.append(result)

for kind in ['proposed','dense','structured']:
 for seed in [1248,1600,2026]:
  path=R/('P_evaluation_fixed' if args.fixed else 'P_evaluation_v2')/kind/str(seed)
  if (path/'COMPLETED.json').exists():
   with (path/'window_step_errors.csv').open() as f:score(list(csv.DictReader(f)),'P',kind,seed)
for reg in ['H','P']:
 path=R/f'OpInf_{reg}_heldout'
 if (path/'COMPLETED.json').exists():
  with (path/'window_step_errors.csv').open() as f:rr=list(csv.DictReader(f))
  initial={int(r['window']):float(np.load(path/f"window_{int(r['window']):03d}.npz")['initial_time']) for r in rr if int(r['step'])==1}
  for r in rr:r['elapsed']=float(r['time'])-initial[int(r['window'])]
  score(rr,reg,'OpInf','deterministic')
ROOT=R.parents[1]
u=np.load(ROOT/'Hopf/artifacts/hopf/velocity_pod_hopf.npz');p=np.load(ROOT/'Hopf/artifacts/hopf/pressure_pod_hopf.npz')
w=u['point_areas'];wu=np.r_[w,w]
def geometry(z,key,mean,weights):
 phi=z[key][:32].astype(float);m=z[mean].astype(float)
 return (phi*weights)@phi.T,(phi*weights)@m,float((m*m*weights).sum())
gu,cu,nu=geometry(u,'phi_uv','mean_uv_regime',wu);gp,cp,np0=geometry(p,'phi_p','mean_p_regime',w)
def errors(pred,true,g,c,n):
 d=pred-true
 return 100*np.sqrt(np.maximum(np.einsum('...i,ij,...j->...',d,g,d),0)/np.maximum(np.einsum('...i,ij,...j->...',true,g,true)+2*(true@c)+n,1e-12))
for kind in ['proposed','dense','structured']:
 for seed in [1248,1600,2026]:
  path=R/(args.h_evaluation_family or ('H_evaluation_fixed' if args.fixed else 'H_evaluation'))/kind/str(seed)
  if not (path/'COMPLETED.json').exists():continue
  rows=[]
  for file in sorted(path.glob('*_modal.npz')):
   z=np.load(file);rv=float(file.name.split('_')[1]);eu=errors(z['pred_a'],z['true_a'],gu,cu,nu);ep=errors(z['pred_b'],z['true_b'],gp,cp,np0)
   for j in range(eu.shape[0]):
    for k in range(eu.shape[1]):rows.append(dict(window=j,Re=rv,step=k+1,elapsed=float(z['times'][j,k]-z['initial_times'][j]),Eu_percent=float(eu[j,k]),Ep_percent=float(ep[j,k]),Ejoint_percent=float(eu[j,k]+ep[j,k])))
  score(rows,'H',kind,seed)
for record in records:record['experiment_family']='quadratic initialization repaired' if args.fixed else 'legacy initialization preserved'
(R/f'scorecard{suffix}.json').write_text(json.dumps(records,indent=2,allow_nan=False))
groups=[]
for reg in ['H','P']:
 for kind in ['proposed','dense','structured','OpInf']:
  for K in [24,48]:
   part=[r for r in records if r['regime']==reg and r['method']==kind and r['K']==K]
   if not part:continue
   row=dict(regime=reg,method=kind,K=K,evaluated_seeds=len(part),expected_seeds=1 if kind=='OpInf' else 3,reference='POD reconstructed, not FOM')
   for key in ['Eu_percent_window_mean','Ep_percent_window_mean','Ejoint_percent_window_mean']:
    values=[r[key] for r in part]
    row[key+'_mean']=float(np.mean(values)) if all(v is not None for v in values) else None
    row[key+'_sample_std']=float(np.std(values,ddof=1)) if len(values)>1 and all(v is not None for v in values) else None
   row['warning']='Incomplete/accepted-only seed summary; NOT a three-seed estimate' if len(part)!=row['expected_seeds'] else ''
   groups.append(row)
(R/f'seed_summary{suffix}.json').write_text(json.dumps(groups,indent=2,allow_nan=False))
print(json.dumps([r for r in groups if r['K']==24],indent=2))
