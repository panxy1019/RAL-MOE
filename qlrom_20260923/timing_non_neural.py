"""Physical-state-to-fields timing for ql-ROM-style and frozen OpInf."""
import argparse,resource,platform,time,json,os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import numpy as np
from qlrom_experiment import OUT,R,load_data,load_model,local_ops,rollout,write,sha
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['qlrom','opinf'],required=True);kind=p.parse_args().kind
d=load_data();s=976;phi_u=d['u']['phi_uv'].astype(float);phi_p=d['p']['phi_p'].astype(float)
wu=np.r_[d['u']['point_areas'],d['u']['point_areas']].astype(float);wp=d['p']['point_areas'].astype(float)
mu=d['u']['mean_uv_regime'].astype(float);mp=d['p']['mean_p_regime'].astype(float)
phi_p-=(phi_p@wp/wp.sum())[:,None];mp-=mp@wp/wp.sum()
encoder=(phi_u*wu).T@np.linalg.inv(d['G'])
input_history_u=d['a'][[s,s-1,s-2]]@phi_u+mu
input_history_p=d['b'][[s,s-1,s-2]]@phi_p+mp
if kind=='qlrom':
    cfg=json.loads((OUT/'selected_config.json').read_text());m=load_model(cfg['J'],cfg['r'],1248)
    operators=local_ops(d,m,d['labels'][s])
    def predict(a):
        rr=rollout(d,m,d['labels'][s],[s],cfg['substeps'],initial_a=a[None],prepared_ops=operators)
        return rr['pred_a'][0],rr['pred_b'][0]
    ref=np.load(OUT/f'predictions/qlrom_s1248_start{s}.npz')
else:
    model=np.load(R/'OpInf_P_development/validation_selected.npz')
    ii,jj=model['ii'],model['jj'];W,V=model['W'],model['V']
    re=float(np.float32(d['idx'][s]['Re']));rn=(1/re-model['inv_Re_mean'])/model['inv_Re_scale']
    def feat(x):return np.r_[1.,x,x[ii]*x[jj],rn,rn*x]
    def predict(a):
        x=(a-model['mean'])/model['scale'];pa=[];pb=[]
        for h0 in np.diff(d['time'][s:s+49]):
            h=h0/4
            for _ in range(4):
                k1=feat(x)@W;k2=feat(x+h*k1/2)@W;k3=feat(x+h*k2/2)@W;k4=feat(x+h*k3)@W
                x=x+h*(k1+2*k2+2*k3+k4)/6
            pa.append(x*model['scale']+model['mean']);pb.append((feat(x)@V)*model['pressure_scale']+model['pressure_mean'])
        return np.array(pa),np.array(pb)
    ref=np.load(R/'OpInf_P_heldout/window_000.npz')
def run():
    a=(input_history_u[0]-mu)@encoder
    pa,pb=predict(a);u=pa@phi_u+mu;p=pb@phi_p+mp;p-=(p@wp/wp.sum())[:,None]
    return u,p,pa,pb
for _ in range(3):run()
samples=[]
for _ in range(10):
    t=time.perf_counter();u,p,pa,pb=run();samples.append(time.perf_counter()-t)
    assert len(pa)==48 and np.isfinite(u).all() and np.isfinite(p).all()
delta=max(float(np.max(np.abs(pa-ref['pred_a']))),float(np.max(np.abs(pb-ref['pred_b']))))
output=OUT/'timing'/kind;output.mkdir(parents=True,exist_ok=False)
write(output/'timing.json',dict(method=kind,seed=1248 if kind=='qlrom' else 'deterministic',start=s,
    seconds=samples,median=float(np.median(samples)),iqr=float(np.percentile(samples,75)-np.percentile(samples,25)),
    warmups=3,repeats=10,batch=1,K=48,dtype='FP64',cpu=platform.processor(),gpu='not used',cpu_threads=2,
    peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    peak_rss_scope='whole process high-water mark including loaded reference assets',max_modal_difference_from_cached=delta,
    scope='in-memory common POD-reconstructed physical history -> all decoded gauge-fixed fields; uses current velocity only, history available but unused; operator construction/model loading/disk IO excluded',
    synchronization='CPU synchronous',code_sha256=sha(__file__)))
print(kind,float(np.median(samples)),delta,flush=True)
