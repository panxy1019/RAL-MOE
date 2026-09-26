"""Re-evaluate cached predictions against a shared gauge-fixed POD reference."""
import csv,json,time
import numpy as np
import torch
from qlrom_experiment import R,OUT,SEEDS,load_data,windows,errors,csvwrite,write,aggregate,sha

def cached(d):
    _,starts=windows(d)
    for kind in ['proposed','dense','structured']:
        for seed in SEEDS:
            file=R/f'P_evaluation_v2/{kind}/{seed}/modal_rollouts.pt'
            records=torch.load(file,weights_only=False)
            assert [z['start'] for z in records]==starts
            for z in records:
                s=z['start'];vv=z['values'];n=len(vv)
                a=np.full((48,32),np.nan);b=a.copy()
                if n:
                    a[:n]=np.array([v[0] for v in vv]);b[:n]=np.array([v[1] for v in vv])
                    assert np.array_equal(np.array([v[2] for v in vv]),d['a'][s+1:s+n+1])
                    assert np.array_equal(np.array([v[3] for v in vv]),d['b'][s+1:s+n+1])
                    assert np.max(np.abs(np.array([v[4] for v in vv])+z['initial_time']-d['time'][s+1:s+n+1]))<1e-9
                yield kind,seed,s,a,b,str(file)
    for i,s in enumerate(starts):
        path=R/f'OpInf_P_heldout/window_{i:03d}.npz';z=np.load(path)
        assert abs(z['initial_time']-d['time'][s])<1e-9
        a=np.full((48,32),np.nan);b=a.copy();n=len(z['pred_a']);a[:n]=z['pred_a'];b[:n]=z['pred_b']
        yield 'OpInf','deterministic',s,a,b,str(path)

def summarize(rows):
    summary=[];seedrows=[]
    for method in sorted(set(r['method'] for r in rows)):
        seeds=sorted(set(str(r['seed']) for r in rows if r['method']==method))
        for K in [24,48]:
            ss=[]
            for seed in seeds:
                rr=[r for r in rows if r['method']==method and str(r['seed'])==seed and r['horizon']==K]
                stats=aggregate(rr,K);ss.append(stats);seedrows.append(dict(method=method,seed=seed,horizon=K,**stats))
            ok=all(v['completed']==v['total'] and np.isfinite(v['joint']) for v in ss)
            row=dict(method=method,horizon=K,seed_count=len(ss),complete_seeds=sum(v['completed']==v['total'] for v in ss),
                     complete_windows=sum(v['completed'] for v in ss),total_windows=sum(v['total'] for v in ss))
            for metric in ['Eu','Ep','joint']:
                row[metric+'_mean']=float(np.mean([v[metric] for v in ss])) if ok else float('nan')
                row[metric+'_sample_sd']=float(np.std([v[metric] for v in ss],ddof=1)) if ok and len(ss)>1 else ''
            summary.append(row)
    return summary,seedrows

def main():
    d=load_data();rows=[];steps=[];checks=[]
    for kind,seed,s,a,b,source in cached(d):
        eu=errors(a,d['a'][s+1:s+49],d['gu']);ep=errors(b,d['b'][s+1:s+49],d['gp'])
        for K in [24,48]:
            ok=bool(np.isfinite(a[:K]).all() and np.isfinite(b[:K]).all())
            rows.append(dict(method=kind,seed=seed,Re=float(d['idx'][s]['Re']),start=s,horizon=K,complete=ok,
                Eu=float(np.mean(eu[:K])) if ok else float('nan'),Ep=float(np.mean(ep[:K])) if ok else float('nan'),
                failure='' if ok else 'cached_incomplete_rollout',source=source))
        for k in range(48):steps.append(dict(method=kind,seed=seed,Re=float(d['idx'][s]['Re']),window=s,step=k+1,
            time=float(d['time'][s+k+1]),Eu=float(eu[k]),Ep=float(ep[k]),reference='common_POD_r32_gauge_fixed',source=source))
    csvwrite(OUT/'baseline_completion.csv',rows);csvwrite(OUT/'baseline_per_step_metrics.csv',steps)
    for kind in ['proposed','dense','structured']:
        for seed in SEEDS:
            old=list(csv.DictReader((R/f'P_evaluation_v2/{kind}/{seed}/window_step_errors.csv').open()))
            new={(r['window'],r['step']):r for r in steps if r['method']==kind and r['seed']==seed}
            diffs={k:[] for k in ['Eu','Ep']}
            for r in old:
                n=new[(int(r['start']),int(r['step']))]
                for key in diffs:diffs[key].append(abs(n[key]-float(r[key+'_percent'])))
            checks.append(dict(method=kind,seed=seed,Eu_max_abs_diff=max(diffs['Eu']),Ep_max_abs_diff=max(diffs['Ep']),
                note='recomputed FP64 quadratic norm and independently centered pressure gauge; original uses its native arithmetic'))
    write(OUT/'baseline_reaggregation_checks.json',checks)
    if (OUT/'completion.csv').exists():
        for r in csv.DictReader((OUT/'completion.csv').open()):
            rows.append(dict(method='ql-ROM-style (fixed POD)',seed=int(r['seed']),Re=float(r['Re']),start=int(r['start']),
                horizon=int(r['horizon']),complete=r['complete']=='True',Eu=float(r['Eu']),Ep=float(r['Ep']),failure=r['failure'],source='predictions/'))
    summary,seedrows=summarize(rows);csvwrite(OUT/'summary_pod_reference.csv',summary);csvwrite(OUT/'seed_metrics_pod.csv',seedrows)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
