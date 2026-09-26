"""Mechanical final-test evaluator for already frozen S-H routes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m
def sha256(path):
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()
def atomic_json(path,value):
    t=path.with_suffix(path.suffix+'.tmp'); t.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)); os.replace(t,path)


def fixed_point(alpha,data):
    root=Path('/root/panxy/particalMOE'); sr=root/'steady_specialist_v1/source_artifacts/steady'; hr=root/'Hopf/artifacts/hopf'
    vals=[]
    with np.load(sr/'velocity_pod_steady.npz') as su, np.load(hr/'velocity_pod_hopf.npz') as hu:
        w=np.r_[su['point_areas'],su['point_areas']].astype(np.float64); ps=su['phi_uv'][:32].astype(np.float64); ph=hu['phi_uv'][:32].astype(np.float64)
        gs=(ps*w)@ps.T; gh=(ph*w)@ph.T; c=(ps*w)@ph.T; ms=su['mean_uv_regime'].astype(np.float64); psms=(ps*w)@ms; ms2=float((ms*w)@ms)
        for i,a in enumerate(alpha):
            ds=data['s_a'][i,-1].astype(float)-data['s_a'][i,-2].astype(float); dh=data['h_a'][i,-1].astype(float)-data['h_a'][i,-2].astype(float)
            q=a*a*(ds@gs@ds)+(1-a)**2*(dh@gh@dh)+2*a*(1-a)*(ds@c@dh)
            y=data['true_a'][i,-1].astype(float); den=ms2+2*y@psms+y@gs@y
            vals.append(float(np.sqrt(max(q,0)/max(den,1e-12))))
    return {'mean':float(np.mean(vals)),'worst':float(np.max(vals)),'definition':'area-weighted final-step field drift relative to final true field norm'}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--training-dir',type=Path,required=True); p.add_argument('--decision',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
    if not (a.training_dir/'ALL_ROUTES_VALIDATION_FROZEN.json').exists(): raise RuntimeError('routes not frozen')
    decision=json.loads(a.decision.read_text());
    if decision['test_used_for_selection'] is not False: raise RuntimeError('decision contract')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    routes=load_module('final_routes',Path(__file__).with_name('train_sh_routes.py'))
    with np.load(a.cache,allow_pickle=False) as z: data={k:z[k] for k in z.files}
    if set(data['split'].tolist())!={'heldout'}: raise RuntimeError('not a heldout-only cache')
    e2a=routes.e2_pair_probability(data['re']); results={}; inference={}
    for method in routes.METHODS:
        ck=torch.load(a.training_dir/method/'best.pt',map_location='cpu',weights_only=False)
        x=((data['features']-ck['feature_mean'])/ck['feature_std']).astype(np.float32); tx=torch.as_tensor(x); base=np.log(np.maximum(e2a,1e-6)/np.maximum(1-e2a,1e-6)).astype(np.float32); tb=torch.as_tensor(base)
        gate=routes.ConvexGate(x.shape[1]); gate.load_state_dict(ck['gate_state']); gate.eval(); scorer=None; sx=tx
        if method=='RiskPredictionRouter': scorer=routes.MLP(x.shape[1],2)
        if method=='LookAheadShortRolloutRouter':
            la=routes.lookahead_features(data); xla=np.concatenate((x,(la-ck['lookahead_mean'])/ck['lookahead_std']),axis=1).astype(np.float32); sx=torch.as_tensor(xla); scorer=routes.MLP(xla.shape[1],2)
        if scorer: scorer.load_state_dict(ck['scorer_state']); scorer.eval()
        t0=time.perf_counter()
        with torch.inference_mode():
            alpha=gate(tx,tb).numpy(); predicted=scorer(sx).numpy() if scorer else None
        inference[method]={'gate_seconds_per_trajectory':(time.perf_counter()-t0)/len(alpha),'relative_native_rollout_calls':2}
        active=np.ones(len(alpha),bool)
        risk={}
        if predicted is not None:
            margin=np.abs(predicted[:,0]-predicted[:,1]); active=margin<=float(ck['threshold']); top=np.argmin(predicted,axis=1); alpha[~active]=(top[~active]==0).astype(np.float32)
            true_risk=routes.risk_targets(data['quad_u'],data['quad_p']); risk={'ranking_accuracy':float(np.mean(np.argmin(predicted,axis=1)==np.argmin(true_risk,axis=1))),'risk_mse':float(np.mean((predicted-true_risk)**2))}
        metric=routes.metrics(alpha,data['quad_u'],data['quad_p'],data['re'],active); metric['steady_fixed_point']=fixed_point(alpha,data); metric['risk']=risk; metric['alpha_S']={'mean':float(alpha.mean()),'min':float(alpha.min()),'max':float(alpha.max())}; results[method]=metric
    baselines={}
    for name,alpha,active in [('E2_pair_Top1',(e2a>=.5).astype(np.float32),np.zeros(len(e2a),bool)),('fixed_0.5',np.full(len(e2a),.5,np.float32),np.ones(len(e2a),bool)),('E2_pair_probability_blend',e2a,np.ones(len(e2a),bool))]:
        baselines[name]=routes.metrics(alpha,data['quad_u'],data['quad_p'],data['re'],active); baselines[name]['steady_fixed_point']=fixed_point(alpha,data)
    qu,qp=data['quad_u'],data['quad_p']; grid=np.linspace(0,1,10001,dtype=np.float64)[:,None]; oracle=[]
    for i in range(len(qu)):
        eu=(grid*grid*qu[i,:,0]+(1-grid)**2*qu[i,:,1]+2*grid*(1-grid)*qu[i,:,2])/np.maximum(qu[i,:,3],1e-12); ep=(grid*grid*qp[i,:,0]+(1-grid)**2*qp[i,:,1]+2*grid*(1-grid)*qp[i,:,2])/np.maximum(qp[i,:,3],1e-12)
        oracle.append(float(grid[np.argmin(np.mean(np.sqrt(np.maximum(eu,0))+np.sqrt(np.maximum(ep,0)),axis=1)),0]))
    oracle=np.asarray(oracle,dtype=np.float32); oracle_m=routes.metrics(oracle,data['quad_u'],data['quad_p'],data['re']); baselines['per_window_convex_oracle']=oracle_m
    oracle_score=oracle_m['joint_all_mean']
    for metric in results.values(): metric['relative_oracle_gap']=metric['joint_all_mean']-oracle_score
    report={'status':'FINAL_TEST_COMPLETE','protocol':'one-sided S-native S-H output-only Top-2','test_cache_sha256':sha256(a.cache),'decision_sha256':sha256(a.decision),'test_Re':sorted(set(data['re'].tolist())),'windows':len(data['re']),'routes':results,'baselines':baselines,'inference':inference,'promoted_method_validation_frozen':decision['promoted_method'],'test_used_for_selection':False,'not_blind_test_statement':'Legacy test results were known before this recovery; this recovery did not use test fields or metrics for tuning, thresholding, checkpoint selection, or promotion.'}
    atomic_json(a.output_dir/'FINAL_TEST_REPORT.json',report); print(json.dumps({'status':report['status'],'promoted':report['promoted_method_validation_frozen'],'test_Re':report['test_Re'],'route_joint':{k:v['joint_all_mean'] for k,v in results.items()}},indent=2))


if __name__=='__main__': main()
