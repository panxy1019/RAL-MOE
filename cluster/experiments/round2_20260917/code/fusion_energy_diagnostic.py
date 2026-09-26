"""Exact convex-assembly kinetic-energy deficit from saved quadratic statistics."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];ROOT=R.parents[1]
sys.path.insert(0,str(ROOT/'experiments/strengthening_20260915/code'))
import constant_alpha as old
run=ROOT/'centeredsquare_fusion_runs/E2_T2C_K24_20260730_STRICT_V3'
out=R/'fusion_energy_deficit';out.mkdir(exist_ok=False)
rows=[]
for pair in ['sh','hp']:
 d=old.load(run/f'heldout_evaluation_20260730_V1/cache_{pair}/{pair}_heldout_cache.npz')
 c=torch.load(run/f'gate_{pair}/best.pt',map_location='cpu',weights_only=False)
 base,full=old.weights(d,c,run/'e2_router/best.pt')
 q=d['quad_u'];distance=q[:,:,0]+q[:,:,1]-2*q[:,:,2]
 assert distance.min()>-1e-8
 distance=np.maximum(distance,0)
 for method,w in [('T2C',full),('Equal',np.full(len(full),.5)),('E2_probability',base)]:
  deficit=.5*w[:,None]*(1-w[:,None])*distance
  relative=deficit/(.5*np.maximum(q[:,:,3],1e-12))
  for re in np.unique(d['re']):
   mask=d['re']==re
   for k in range(relative.shape[1]):rows.append(dict(pair=pair,method=method,Re=float(re),step=k+1,
      mean_absolute_energy_deficit=float(deficit[mask,k].mean()),mean_deficit_over_truth_energy=float(relative[mask,k].mean())))
with (out/'per_Re_step.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'protocol.json').write_text(json.dumps(dict(identity='alpha*K(u1)+(1-alpha)*K(u2)-K(alpha*u1+(1-alpha)*u2)=0.5*alpha*(1-alpha)*||u1-u2||_M^2',
 source='||u1-u2||^2 recovered exactly as q11+q22-2*q12 from common-truth error quadratic statistics',
 meaning='kinetic-energy deficit relative to weighted candidate energies; NOT energy error vs CFD, and NOT momentum/PPE residual',
 scope='fixed cached K24 candidates and weights; no retraining'),indent=2))
