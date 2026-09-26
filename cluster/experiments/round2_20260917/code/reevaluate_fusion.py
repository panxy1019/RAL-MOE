"""Frozen-cache full-window/macro-Re evaluation; no test-selected deployable control."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit

ROOT=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
sys.path.insert(0,str(ROOT/'experiments/strengthening_20260915/code'))
import constant_alpha as old

def squared(w,q):
    a=w[:,None]
    return np.maximum((a*a*q[:,:,0]+(1-a)**2*q[:,:,1]+2*a*(1-a)*q[:,:,2])/np.maximum(q[:,:,3],1e-12),0)

def objective(w,data,mask):
    return float(sum(squared(w[mask],data[k][mask]).mean() for k in ['quad_u','quad_p']))

def main():
    out=ROOT/'experiments/round2_20260917/fusion_reevaluation'
    out.mkdir(exist_ok=False)
    run=ROOT/'centeredsquare_fusion_runs/E2_T2C_K24_20260730_STRICT_V3'
    rows=[];curves=[];summary={}
    for B in ['SH','HP']:
        b=B.lower();dp=run/f'cache_{b}/{b}_development_cache.npz'
        tp=run/f'heldout_evaluation_20260730_V1/cache_{b}/{b}_heldout_cache.npz'
        d,t=old.load(dp),old.load(tp)
        tr,va=d['split']=='train',d['split']=='validation'
        assert set(t['split'])=={'heldout'} and not set(d['re'])&set(t['re'])
        gatepath=run/f'gate_{b}/best.pt';router=run/'e2_router/best.pt'
        c=torch.load(gatepath,map_location='cpu',weights_only=False)
        assert old.sha256(dp)==c['cache_sha256'] and old.sha256(router)==c['router_checkpoint_sha256']
        assert len(c['feature_mean'])==9
        db,_=old.weights(d,c,router);base,full=old.weights(t,c,router)
        logit=np.log(db/(1-db))
        scalar=minimize_scalar(lambda beta:objective(expit(logit+beta),d,tr),bounds=(-20,20),method='bounded',options={'xatol':1e-10})
        assert scalar.success
        alpha,_=old.fit(d)
        # Deterministic validation-only fixed candidate, same all-step squared objective.
        choice=min([0.,1.],key=lambda a:objective(np.full(len(db),a),d,va))
        ws={'candidate_1':np.ones(len(base)),'candidate_2':np.zeros(len(base)),
            'E2_Top1':(base>=.5).astype(float),'Equal':np.full(len(base),.5),
            'E2_probability':base,'Constant_alpha_train':np.full(len(base),alpha),
            'Constant_logit_train':expit(np.log(base/(1-base))+scalar.x),
            'Validation_fixed_candidate':np.full(len(base),choice),'T2C_frozen':full}
        summary[B]=dict(development_sha256=old.sha256(dp),heldout_sha256=old.sha256(tp),gate_sha256=old.sha256(gatepath),
            constant_alpha=alpha,constant_logit_beta=float(scalar.x),validation_fixed_candidate=1 if choice else 2,
            feature_input=9,candidate_1='Steady' if B=='SH' else 'Periodic',candidate_2='Hopf',
            cache_keys=list(t),candidate_finite={str(i):bool(t[f'candidate_{i}_finite'].all()) for i in [1,2]},
            candidate_divergent={str(i):int(t[f'candidate_{i}_divergent'].sum()) for i in [1,2]})
        for name,w in ws.items():
            eu,ep=100*np.sqrt(squared(w,t['quad_u'])),100*np.sqrt(squared(w,t['quad_p']))
            for rv in np.unique(t['re']):
                mask=t['re']==rv
                for K in [8,16,24]:
                    row=dict(overlap=B,method=name,Re=float(rv),K=K,windows=int(mask.sum()),
                        alpha_mean=float(w[mask].mean()),alpha_std=float(w[mask].std()),alpha_min=float(w[mask].min()),alpha_max=float(w[mask].max()))
                    for label,arr in [('Eu',eu),('Ep',ep),('Ejoint',eu+ep)]:
                        row.update({f'{label}_window_mean':float(arr[mask,:K].mean()),
                            f'{label}_terminal':float(arr[mask,K-1].mean()),
                            f'{label}_mean_window_max':float(arr[mask,:K].max(axis=1).mean())})
                    row['squared_objective']=float(np.mean((eu[mask,:K]/100)**2+(ep[mask,:K]/100)**2))
                    rows.append(row)
            for k in range(24):
                curves.append(dict(overlap=B,method=name,step=k+1,
                    Eu_pooled=float(eu[:,k].mean()),Ep_pooled=float(ep[:,k].mean()),
                    Eu_macro_Re=float(np.mean([eu[t['re']==r,k].mean() for r in np.unique(t['re'])])),
                    Ep_macro_Re=float(np.mean([ep[t['re']==r,k].mean() for r in np.unique(t['re'])]))))
        assert abs(float((eu+ep)[:,-1].mean())-{'SH':1.1958408357404098,'HP':.8465289562715951}[B])<1e-5
    for file,data in [('per_Re_scores.csv',rows),('curves.csv',curves)]:
        with (out/file).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    macro=[]
    for B in ['SH','HP']:
        for method in sorted({r['method'] for r in rows}):
            for K in [8,16,24]:
                part=[r for r in rows if r['overlap']==B and r['method']==method and r['K']==K]
                macro.append(dict(overlap=B,method=method,K=K,re_count=len(part),
                    **{key:float(np.mean([r[key] for r in part])) for key in part[0] if key.startswith(('Eu_','Ep_','Ejoint_','squared_'))}))
    (out/'macro_Re_scores.json').write_text(json.dumps(macro,indent=2))
    (out/'provenance.json').write_text(json.dumps(dict(status='completed_cached_metrics',interfaces=summary,
        aggregation='time-step average, then window average within Re, then equal Re; pooled curves separate',
        exclusions='No windows excluded. Physical residuals, energy, mu-only replay and end-to-end timing are not provided by this script.',
        reference='archived physical-field quadratic cache; FOM lineage audit remains separate'),indent=2))
    print(out)

if __name__=='__main__':main()
