"""Evaluate frozen Global cached predictions against local POD-reconstructed physical truth.
This is explicitly a projected-truth comparison, not full-order snapshot error.
"""
import csv
import numpy as np
from dense_periodic import ROOT, write
from evaluate_periodic_dense import load
from types import SimpleNamespace
import torch

base=ROOT/'periodic_specialist_r32'
t=load('local_truth_trainer',base/'code/train_periodic_moe.py')
ck=torch.load(base/'checkpoint/FINAL_PERIODIC_SPECIALIST.pt',map_location='cpu',weights_only=False)
args=SimpleNamespace(**ck['args'])
args.data_root=base/'assets/Global_POD_AreaWeighted_L2'
args.tensor_path=base/'assets/velocity_rom_periodic.npz'
args.pressure_surrogate_path=base/'assets/pressure_poisson_surrogate_periodic.npz'
arrays,_=t.build_arrays(args)
groot=ROOT/'V16_1_SteadyPressureAnchor32/assets/common_global_data/Global_POD_AreaWeighted_L2'
cache=ROOT/'paper_experiments/revisions/revision10_missing_metric_completion_20260725/evaluations/global_periodic_k48_arrays_retry'
out=ROOT/'experiments/strengthening_20260915/global_common_truth_v1'
out.mkdir(exist_ok=True)
pods=[]
for name,phi,mean,vector in [('velocity','phi_uv','mean_uv',True),('pressure','phi_p','mean_p',False)]:
    l=np.load(args.data_root/f'global_{name}_pod_area_weighted_l2.npz')
    g=np.load(groot/f'global_{name}_pod_area_weighted_l2.npz')
    np.testing.assert_allclose(l['point_areas'],g['point_areas'],atol=1e-12,rtol=1e-8)
    if 'points' in l.files: np.testing.assert_allclose(l['points'],g['points'],atol=1e-10,rtol=1e-8)
    w=l['point_areas'].astype(np.float64)
    if vector:w=np.tile(w,2)
    lp=l[phi][:32].astype(np.float64); gp=g[phi][:32].astype(np.float64)
    lm=l[mean+'_regime'].astype(np.float64)
    if not vector:
        print('pressure gauge audit', 'local_mean',float(lm@w/w.sum()),
              'local_basis_max_mean',float(np.max(np.abs(lp@w/w.sum()))),
              'global_mean_range',np.min(g[mean+'_by_Re']@w/w.sum()),np.max(g[mean+'_by_Re']@w/w.sum()),
              'global_basis_max_mean',float(np.max(np.abs(gp@w/w.sum()))))
        lp=lp-(lp@w/w.sum())[:,None]
        gp=gp-(gp@w/w.sum())[:,None]
        lm=lm-float(lm@w/w.sum())
    pods.append((g,mean,w,lp,gp,lm,vector))
rows=[]
for path in sorted(cache.glob('*rollout_arrays.npz')):
    z=np.load(path); re=float(z['Re'][0])
    prepared=[]
    for g,mean,w,lp,gp,lm,vector in pods:
        ri=int(np.argmin(abs(g['Re_values']-re)))
        d=g[mean+'_by_Re'][ri].astype(np.float64)-lm
        if not vector: d=d-float(d@w/w.sum())
        mat=np.concatenate([gp,-lp,d[None,:]],axis=0)
        gram=(mat*w[None,:])@mat.T
        truth=np.concatenate([lp,lm[None,:]],axis=0)
        tg=(truth*w[None,:])@truth.T
        prepared.append((gram,tg))
    for j,times in enumerate(z['times']):
        future=[np.flatnonzero(np.isclose(arrays['re'],re,atol=2e-5,rtol=0)&np.isclose(arrays['time'],tt,atol=1e-3,rtol=0)) for tt in times]
        assert all(len(i)==1 for i in future)
        ids=np.array([i[0] for i in future]); previous=int(np.flatnonzero(arrays['next_idx']==ids[0])[0])
        errors=[]
        for c,(gram,tg) in zip(['a','b'],prepared):
            truth=arrays[c][ids].astype(np.float64)
            coeff=np.concatenate([z['pred_'+c][j].astype(np.float64),truth,np.ones((48,1))],axis=1)
            tc=np.concatenate([truth,np.ones((48,1))],axis=1)
            num=np.einsum('bi,ij,bj->b',coeff,gram,coeff)
            den=np.einsum('bi,ij,bj->b',tc,tg,tc)
            errors.append(100*np.sqrt(np.maximum(num,0)/den))
        for k in range(48):rows.append(dict(method='global_common_projected_truth',Re=re,start=previous,step=k+1,
            time=float(times[k]-arrays['time'][previous]),Eu_percent=float(errors[0][k]),
            Ep_percent=float(errors[1][k]),Ejoint_percent=float(errors[0][k]+errors[1][k])))
with (out/'window_step_errors.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write(out/'protocol.json',dict(reference='local Periodic POD-reconstructed physical truth, identical to aligned local/dense/Galerkin evaluation',
    comparison='cross-chart Gram matrix includes differences in bases and mean fields; physical mesh and weights asserted equal; pressure area-weighted mean removed from both fields',
    limitation='not full-order snapshot error; original global model uses supplied per-Re mean fields',windows=len(rows)//48))
print('common truth windows',len(rows)//48)
