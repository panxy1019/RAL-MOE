"""Train-only closed-form scalar fusion; same-cache controls and full K24 curves."""
import sys, json, csv, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
sys.path.insert(0,str(ROOT/'experiments/iclr_seed_study_20260911/code'))
from common import ConvexGate, router_probabilities, sha256, atomic_json
from evaluate_heldout_rollout import errors,metrics,oracle_weights

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def fit(d):
    mask=d['split']=='train'; numer=0.;denom=0.
    for key in ['quad_u','quad_p']:
        a,b,c,t=np.moveaxis(d[key][mask],-1,0);t=np.maximum(t,1e-12)
        numer+=np.mean((b-c)/t);denom+=np.mean((a+b-2*c)/t)
    assert denom>=-1e-10
    alpha=float(np.clip(numer/denom,0,1)) if denom>1e-14 else .5
    return alpha,{'quadratic_curvature':float(denom),'linear_numerator':float(numer),'unconstrained_alpha':float(numer/denom) if denom>1e-14 else None}

def weights(d,checkpoint,router):
    x=((d['features']-checkpoint['feature_mean'])/checkpoint['feature_std']).astype('float32')
    pr=router_probabilities(router,d['re'])[:,checkpoint['pair_indices']];pr/=pr.sum(1,keepdims=True)
    base=np.clip(pr[:,0],1e-6,1-1e-6).astype('float32')
    gate=ConvexGate(x.shape[1]);gate.load_state_dict(checkpoint['gate_state']);gate.eval()
    with torch.inference_mode():alpha=gate(torch.tensor(x),torch.tensor(np.log(base/(1-base)))).numpy()
    return base,alpha

def main():
    run=ROOT/'centeredsquare_fusion_runs/E2_T2C_K24_20260730_STRICT_V3'
    out=ROOT/'experiments/strengthening_20260915/constant_alpha_v1';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'protocol.json',dict(created=time.time(),fit_split='train',fit_loss='mean(all-step Eu_squared+Ep_squared)',primary_fit='analytic scalar convex minimizer',sensitivity='validation grid selected by worst_Re_mean then pooled mean',alpha_order={'SH':'Steady weight','HP':'Periodic weight'},seed='not applicable: deterministic one-dimensional optimum',test_selection=False,oracle='future-information diagnostic, not deployable'))
    results={};curves=[];window_rows=[]
    for B in ['SH','HP']:
        b=B.lower(); dp=run/f'cache_{b}/{b}_development_cache.npz';tp=run/f'heldout_evaluation_20260730_V1/cache_{b}/{b}_heldout_cache.npz'
        d=load(dp);t=load(tp); tr=d['split']=='train';va=d['split']=='validation'
        assert set(d['split'])=={'train','validation'} and set(t['split'])=={'heldout'}
        assert not set(d['re'][tr])&set(d['re'][va]) and not set(d['re'])&set(t['re'])
        assert np.array_equal(d['pair_indices'][0],t['pair_indices'][0])
        for data in [d,t]:
            for i in [1,2]:assert data[f'candidate_{i}_finite'].all() and not data[f'candidate_{i}_divergent'].any()
        alpha,coef=fit(d)
        # Test independence: fitting ignores every nontraining quadratic value.
        perturbed={k:v.copy() for k,v in d.items()}
        for k in ['quad_u','quad_p']:perturbed[k][~tr]=12345
        assert fit(perturbed)[0]==alpha
        grid=np.linspace(0,1,10001);scores=[]
        for a in grid:
            eu=errors(np.full(va.sum(),a),d['quad_u'][va]);ep=errors(np.full(va.sum(),a),d['quad_p'][va]);j=eu+ep
            scores.append((max(j[d['re'][va]==r].mean() for r in np.unique(d['re'][va])),j.mean()))
        idx=min(range(len(grid)),key=lambda i:scores[i]);valalpha=float(grid[idx])
        c=torch.load(run/f'gate_{b}/best.pt',map_location='cpu',weights_only=False);router=run/'e2_router/best.pt'
        assert sha256(dp)==c['cache_sha256'] and sha256(router)==c['router_checkpoint_sha256']
        base,full=weights(t,c,router)
        ws={'Constant-alpha_train':np.full(len(t['re']),alpha),'Constant-alpha_validation_sensitivity':np.full(len(t['re']),valalpha),'Equal':np.full(len(t['re']),.5),'E2_probability':base,'T2-C':full,'Convex_oracle_diagnostic':oracle_weights(t['quad_u'],t['quad_p'])}
        res={'train_alpha':alpha,'validation_sensitivity_alpha':valalpha,'coefficients':coef,'hashes':{'development':sha256(dp),'heldout':sha256(tp)},'methods':{}}
        for name,w in ws.items():
            res['methods'][name]=metrics(w,t['quad_u'],t['quad_p'],t['re'])
            eu=100*errors(w,t['quad_u']);ep=100*errors(w,t['quad_p'])
            for k in range(eu.shape[1]):curves.append(dict(overlap=B,method=name,step=k+1,Eu_percent=float(eu[:,k].mean()),Ep_percent=float(ep[:,k].mean()),Ejoint_percent=float((eu+ep)[:,k].mean())))
        for split,data in [('development',d),('heldout',t)]:
            prob,learned=weights(data,c,router)
            for i,r in enumerate(data['re']):window_rows.append(dict(overlap=B,split=str(data['split'][i]),window=i,Re=float(r),E2_pair_probability=float(prob[i]),T2C_alpha=float(learned[i]),constant_alpha=alpha,validation_scalar=valalpha))
        got=100*res['methods']['T2-C']['by_horizon']['K24']['joint_mean'];assert abs(got-{'SH':1.1958408357404098,'HP':.8465289562715951}[B])<1e-5
        results[B]=res;print(B,'alpha',alpha,'validation_alpha',valalpha,'K24', {k:100*v['by_horizon']['K24']['joint_mean'] for k,v in res['methods'].items()},flush=True)
    atomic_json(out/'results.json',results)
    for name,rows in [('fusion_error_curves.csv',curves),('router_fusion_windows.csv',window_rows)]:
        with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(out/'status.json',{'status':'complete','original_T2C_replay':'passed','fit_ignores_validation_values':'passed','split_disjoint':'passed'})

if __name__=='__main__':main()
