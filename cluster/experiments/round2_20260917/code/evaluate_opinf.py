"""Frozen OpInf evaluation on exactly the completed neural reference windows."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];ROOT=R.parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--regime',choices=['H','P'],required=True);args=p.parse_args();reg=args.regime
 dev=R/('OpInf_H_development_v2' if reg=='H' else 'OpInf_P_development')
 model=np.load(dev/'validation_selected.npz');out=R/f'OpInf_{reg}_heldout';out.mkdir(exist_ok=False)
 if reg=='H':
  data=np.load(ROOT/'Hopf/artifacts/hopf/velocity_pod_hopf.npz');press=np.load(ROOT/'Hopf/artifacts/hopf/pressure_pod_hopf.npz')
  with (ROOT/'Hopf/artifacts/hopf/projection_snapshots_velocity_hopf.csv').open() as f:index=list(csv.DictReader(f))
  ref=R/'H_evaluation/proposed/1248';assert (ref/'COMPLETED.json').exists()
  windows=[]
  for path in sorted(ref.glob('*_modal.npz')):
   z=np.load(path);re=float(path.name.split('_')[1])
   for j in range(len(z['starts'])):windows.append(dict(re=re,initial_time=float(z['initial_times'][j]),times=z['times'][j],true_a=z['true_a'][j],true_b=z['true_b'][j]))
 else:
  data=np.load(ROOT/'periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz')
  press=np.load(ROOT/'periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/global_pressure_pod_area_weighted_l2.npz')
  with (ROOT/'periodic_specialist_r32/assets/provenance/projection_snapshots_velocity_periodic.csv').open() as f:index=list(csv.DictReader(f))
  ref=R/'P_evaluation_v2/proposed/1248';assert (ref/'COMPLETED.json').exists()
  windows=[]
  for z in torch.load(ref/'modal_rollouts.pt',weights_only=False):
   values=z['values'];assert len(values)==48
   windows.append(dict(re=z['Re'],initial_time=z['initial_time'],times=np.array([v[4]+z['initial_time'] for v in values]),true_a=np.array([v[2] for v in values]),true_b=np.array([v[3] for v in values])))
 w=data['point_areas'];wu=np.r_[w,w];pu=data['phi_uv'][:32].astype(float);pp=press['phi_p'][:32].astype(float)
 mu=data['mean_uv_regime'].astype(float);mp=press['mean_p_regime'].astype(float)
 gu=(pu*wu)@pu.T;gp=(pp*w)@pp.T;cu=(pu*wu)@mu;cp=(pp*w)@mp
 nu=float((mu*mu*wu).sum());np0=float((mp*mp*w).sum())
 ri=np.array([float(r['Re']) for r in index]);ti=np.array([float(r['time']) for r in index])
 ii,jj=model['ii'],model['jj'];W,V=model['W'],model['V']
 def feat(x,r):return np.r_[1.,x,x[ii]*x[jj],r,r*x]
 def integrate(initial,times,re,substeps):
  x=(initial-model['mean'])/model['scale'];r=(1/re-model['inv_Re_mean'])/model['inv_Re_scale']
  pa=[];pb=[]
  def f(x):return feat(x,r)@W
  with np.errstate(over='ignore',invalid='ignore'):
   for dt0 in np.diff(times):
    dt=dt0/substeps
    for _ in range(substeps):
     k1=f(x);k2=f(x+dt*k1/2);k3=f(x+dt*k2/2);k4=f(x+dt*k3);x=x+dt*(k1+2*k2+2*k3+k4)/6
     if not np.isfinite(x).all() or abs(x).max()>1e6:break
    if not np.isfinite(x).all() or abs(x).max()>1e6:break
    pa.append(x*model['scale']+model['mean']);pb.append((feat(x,r)@V)*model['pressure_scale']+model['pressure_mean'])
  return np.asarray(pa),np.asarray(pb)
 def error(pred,true,g,c,n):
  d=pred-true;return 100*np.sqrt(max(float(d@g@d),0)/max(float(true@g@true+2*true@c+n),1e-12))
 rows=[];sensitivity=[]
 for wi,z in enumerate(windows):
  ids=np.flatnonzero(np.isclose(ri,z['re'],atol=2e-5,rtol=0)&np.isclose(ti,z['initial_time'],atol=2e-3,rtol=0));assert len(ids)==1
  initial=data['coeff_uv'][ids[0],:32];times=np.r_[z['initial_time'],z['times']]
  pa,pb=integrate(initial,times,z['re'],4)
  # Diagnostic only: never replaces primary 4-substep prediction after seeing test values.
  p8,b8=integrate(initial,times,z['re'],8)
  sensitivity.append(dict(window=wi,Re=z['re'],completed4=len(pa),completed8=len(p8),
    max_modal_difference=float(np.max(np.abs(pa-p8))) if len(pa)==len(p8)==48 else None))
  for k in range(48):
   ok=k<len(pa)
   row=dict(window=wi,Re=z['re'],step=k+1,time=float(z['times'][k]),finite=ok,Eu_percent=error(pa[k],z['true_a'][k],gu,cu,nu) if ok else float('nan'),Ep_percent=error(pb[k],z['true_b'][k],gp,cp,np0) if ok else float('nan'))
   row['Ejoint_percent']=row['Eu_percent']+row['Ep_percent'];rows.append(row)
  np.savez_compressed(out/f'window_{wi:03d}.npz',pred_a=pa,pred_b=pb,**z)
 with (out/'window_step_errors.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 (out/'integration_sensitivity.json').write_text(json.dumps(sensitivity,indent=2))
 (out/'protocol.json').write_text(json.dumps(dict(regime=reg,reference='same POD reconstructed fields as neural evaluator, not FOM',
  windows=len(windows),horizon=48,validation_selected_lambda=float(model['lambda_value']),
  integrator='fixed 4-substep RK4; 8-substep diagnostic not used for model selection',failure='retain incomplete windows and NaN errors; no survivor-only aggregate'),indent=2))
 (out/'COMPLETED.json').write_text('{"completed":true}')
if __name__=='__main__':main()
