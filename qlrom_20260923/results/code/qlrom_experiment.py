"""Independent fixed-POD-space ql-ROM-style experiment, train/validation first.

No neural closure or inferred dynamical operator is used. Existing projected
momentum and algebraic PPE tensors are transformed to training-only local bases.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import argparse, csv, hashlib, json, time, platform, sys
from pathlib import Path
sys.path.insert(0,'/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/experiments/qlrom_20260923/deps')
import numpy as np
from sklearn.cluster import KMeans
import torch

ROOT=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
B=ROOT/'periodic_specialist_r32/assets'
R=ROOT/'experiments/round2_20260917'
OUT=ROOT/'experiments/qlrom_20260923'
SEEDS=[1248,1600,2026]
METHOD='ql-ROM-style within a fixed POD space'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()

def write(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=True),encoding='utf-8')

def csvwrite(path,rows):
    if not rows: return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)

def geometry(phi,mean,w,gauge=False):
    phi=phi.astype(float);mean=mean.astype(float)
    if gauge:
        phi=phi-(phi@w/w.sum())[:,None];mean=mean-mean@w/w.sum()
    return (phi*w)@phi.T,(phi*w)@mean,float((mean*mean)@w)

def errors(pred,true,geom):
    G,c,n=geom;delta=pred-true
    num=np.einsum('...i,ij,...j->...',delta,G,delta)
    den=np.einsum('...i,ij,...j->...',true,G,true)+2*true@c+n
    if np.any(den<=0): raise ValueError('Undefined reference norm; no epsilon is allowed')
    return 100*np.sqrt(np.maximum(num,0)/den)

def load_data():
    u=np.load(B/'Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz')
    p=np.load(B/'Global_POD_AreaWeighted_L2/global_pressure_pod_area_weighted_l2.npz')
    v=np.load(B/'velocity_rom_periodic.npz');q=np.load(B/'pressure_poisson_surrogate_periodic.npz')
    idx=list(csv.DictReader((B/'provenance/projection_snapshots_velocity_periodic.csv').open()))
    a=u['coeff_uv'].astype(float);b=p['coeff_p'].astype(float)
    labels=np.asarray([r['Re_label'] for r in idx]);split=np.asarray([r['split'] for r in idx])
    times=np.asarray([r['time'] for r in idx],float).astype(np.float32).astype(float)
    w=u['point_areas'].astype(float);pu=u['phi_uv'].astype(float);pp=p['phi_p'].astype(float)
    G=(pu*np.r_[w,w])@pu.T;S=np.linalg.cholesky(G).T
    return dict(u=u,p=p,v=v,q=q,idx=idx,a=a,b=b,labels=labels,split=split,time=times,
                G=G,S=S,Si=np.linalg.inv(S),y=a@S.T,
                gu=geometry(pu,u['mean_uv_regime'],np.r_[w,w]),
                gp=geometry(pp,p['mean_p_regime'],w,True))

def polynomial(c,A,H,x):
    return c+x@A.T+np.einsum('ijk,nj,nk->ni',H,x,x,optimize=True)

def affine_transform(c,A,H,center,basis,project):
    constant=project@(c+A@center+np.einsum('ijk,j,k->i',H,center,center))
    jac=A+np.einsum('ijk,j->ik',H,center)+np.einsum('ijk,k->ij',H,center)
    linear=project@jac@basis
    quadratic=np.einsum('oi,ijk,jb,kc->obc',project,H,basis,basis,optimize=True)
    return constant,linear,quadratic

def ambient_ops(d,label):
    v,q=d['v'],d['q']
    cp,Ap,Hp=q[label+'_c_tilde'],q[label+'_A_tilde'],q['H_tilde']
    c=v[label+'_c']+v['P']@cp
    A=v[label+'_A']+v['P']@Ap
    H=v['H']+np.einsum('ip,pjk->ijk',v['P'],Hp)
    return (c,A,H),(cp,Ap,Hp)

def build(d,J,r,seed):
    t0=time.perf_counter();Y=d['y'][d['split']=='train']
    km=KMeans(n_clusters=J,init='k-means++',n_init=10,max_iter=300,tol=1e-4,
              random_state=seed,algorithm='lloyd').fit(Y)
    centers=[];bases=[];ranks=[];pop=[]
    for j in range(J):
        x=Y[km.labels_==j];c=x.mean(0);_,s,V=np.linalg.svd(x-c,full_matrices=False)
        rank=int(np.sum(s>s[0]*1e-12));ranks.append(rank);pop.append(len(x))
        if rank<r: raise ValueError(f'cluster {j} rank {rank} < requested {r}')
        centers.append(c);bases.append(V[:r].T)
    return dict(centers=np.array(centers),bases=np.array(bases),ranks=ranks,population=pop,
                offline_seconds=time.perf_counter()-t0,J=J,r=r,seed=seed,inertia=float(km.inertia_))

def local_ops(d,model,label):
    vel,pressure=ambient_ops(d,label);vels=[];ps=[]
    for c,U in zip(model['centers'],model['bases']):
        center=d['Si']@c;basis=d['Si']@U
        vels.append(affine_transform(*vel,center,basis,U.T@d['S']))
        ps.append(affine_transform(*pressure,center,basis,np.eye(32)))
    return tuple(np.stack([v[i] for v in vels]) for i in range(3)),tuple(np.stack([v[i] for v in ps]) for i in range(3))

def batch_poly(ops,z,j):
    c,A,H=ops
    return c[j]+np.einsum('nij,nj->ni',A[j],z)+np.einsum('nijk,nj,nk->ni',H[j],z,z)

def rollout(d,model,label,starts,substeps,initial_a=None,prepared_ops=None):
    ops,pops=local_ops(d,model,label) if prepared_ops is None else prepared_ops
    C,U=model['centers'],model['bases']
    starts=np.asarray(starts)
    y=d['y'][starts].copy() if initial_a is None else np.asarray(initial_a)@d['S'].T
    j=((y[:,None,:]-C[None,:,:])**2).sum(2).argmin(1)
    z=np.einsum('nir,ni->nr',U[j],y-C[j]);n=len(starts)
    initial_jump=np.linalg.norm(y-(C[j]+np.einsum('nir,nr->ni',U[j],z)),axis=1)
    pa=np.full((n,48,32),np.nan);pb=pa.copy();assign=np.full((n,48),-1,int)
    switches=np.zeros((n,48),int);jump=np.zeros((n,48));alive=np.ones(n,bool)
    failures=['']*n;failed_step=np.full(n,-1,int)
    with np.errstate(over='ignore',invalid='ignore'):
        for k in range(48):
            dt=(d['time'][starts+k+1]-d['time'][starts+k])/substeps
            for _ in range(substeps):
                ids=np.flatnonzero(alive)
                if not len(ids): break
                zz=z[ids];jj=j[ids];h=dt[ids,None]
                k1=batch_poly(ops,zz,jj);k2=batch_poly(ops,zz+h*k1/2,jj)
                k3=batch_poly(ops,zz+h*k2/2,jj);k4=batch_poly(ops,zz+h*k3,jj)
                zz=zz+h*(k1+2*k2+2*k3+k4)/6
                yy=C[jj]+np.einsum('nir,nr->ni',U[jj],zz)
                good=np.isfinite(yy).all(1)
                for ii in ids[~good]: alive[ii]=False;failures[ii]='nonfinite_rollout';failed_step[ii]=k+1
                ids=ids[good];yy=yy[good];zz=zz[good];jj=jj[good]
                if not len(ids): continue
                newj=((yy[:,None,:]-C[None,:,:])**2).sum(2).argmin(1)
                changed=newj!=jj
                if np.any(changed):
                    newz=np.einsum('nir,ni->nr',U[newj[changed]],yy[changed]-C[newj[changed]])
                    projected=C[newj[changed]]+np.einsum('nir,nr->ni',U[newj[changed]],newz)
                    jump[ids[changed],k]+=np.linalg.norm(yy[changed]-projected,axis=1)
                    switches[ids[changed],k]+=1;zz[changed]=newz
                z[ids]=zz;j[ids]=newj
            ids=np.flatnonzero(alive)
            if len(ids):
                yy=C[j[ids]]+np.einsum('nir,nr->ni',U[j[ids]],z[ids])
                pa[ids,k]=yy@d['Si'].T;pb[ids,k]=batch_poly(pops,z[ids],j[ids]);assign[ids,k]=j[ids]
                bad=~np.isfinite(pb[ids,k]).all(1)
                for ii in ids[bad]: alive[ii]=False;failures[ii]='nonfinite_pressure';failed_step[ii]=k+1
    return dict(pred_a=pa,pred_b=pb,cluster=assign,switches=switches,jump=jump,
                initial_projection_jump=initial_jump,failures=np.array(failures),failed_step=failed_step)

def windows(d):
    protocol=json.loads((R/'P_evaluation_v2/proposed/1248/protocol.json').read_text())
    test=protocol['starts'];val=[]
    for label in sorted(set(d['labels'][d['split']=='validation'])):
        ids=np.flatnonzero(d['labels']==label);candidates=ids[np.arange(2,len(ids)-49,8)]
        val.extend(candidates[np.linspace(0,len(candidates)-1,8).astype(int)].tolist())
    return val,test

def evaluate(d,model,starts,substeps):
    rows=[];results={}
    for label in sorted(set(d['labels'][starts])):
        ss=[s for s in starts if d['labels'][s]==label]
        rr=rollout(d,model,label,ss,substeps)
        for i,s in enumerate(ss):
            a,b=rr['pred_a'][i],rr['pred_b'][i]
            eu=errors(a,d['a'][s+1:s+49],d['gu']);ep=errors(b,d['b'][s+1:s+49],d['gp'])
            for horizon in [24,48]:
                ok=bool(np.isfinite(a[:horizon]).all() and np.isfinite(b[:horizon]).all())
                rows.append(dict(seed=model['seed'],J=model['J'],r=model['r'],substeps=substeps,
                    Re=float(d['idx'][s]['Re']),start=s,horizon=horizon,complete=ok,
                    Eu=float(np.mean(eu[:horizon])) if ok else float('nan'),
                    Ep=float(np.mean(ep[:horizon])) if ok else float('nan'),
                    failure=str(rr['failures'][i]),failed_step=int(rr['failed_step'][i])))
            results[s]={key:val[i] for key,val in rr.items()};results[s].update(Eu=eu,Ep=ep)
    return rows,results

def aggregate(rows,horizon):
    rows=[r for r in rows if r['horizon']==horizon]
    complete=sum(r['complete'] for r in rows)
    if complete<len(rows): return dict(completed=complete,total=len(rows),Eu=float('nan'),Ep=float('nan'),joint=float('nan'))
    re=sorted(set(r['Re'] for r in rows))
    u=np.mean([np.mean([r['Eu'] for r in rows if r['Re']==x]) for x in re])
    p=np.mean([np.mean([r['Ep'] for r in rows if r['Re']==x]) for x in re])
    return dict(completed=complete,total=len(rows),Eu=float(u),Ep=float(p),joint=float(u+p))

def prepare(d):
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol_frozen.yaml').exists(): raise FileExistsError('Protocol already frozen')
    val,test=windows(d);split_sets={x:set(d['labels'][d['split']==x]) for x in ['train','validation','heldout']}
    assert all(not split_sets[x]&split_sets[y] for x in split_sets for y in split_sets if x!=y)
    assert str(d['u']['fit_split'])=='train' and str(d['p']['fit_split'])=='train'
    assert all(d['split'][s]=='heldout' and len(set(d['labels'][s-2:s+49]))==1 for s in test)
    assert all(len(set(d['split'][s-2:s+49]))==1 for s in val+test)
    pathlist=[B/'Global_POD_AreaWeighted_L2'/x for x in ['global_velocity_pod_area_weighted_l2.npz','global_pressure_pod_area_weighted_l2.npz']]
    pathlist += [B/'velocity_rom_periodic.npz',B/'pressure_poisson_surrogate_periodic.npz',B/'provenance/projection_snapshots_velocity_periodic.csv']
    manifest=[]
    for kind in ['proposed','dense','structured']:
        for seed in SEEDS:
            ev=R/f'P_evaluation_v2/{kind}/{seed}'
            p=json.loads((ev/'protocol.json').read_text());cp=Path(p['checkpoint'])
            assert p['starts']==test and sha(cp)==p['checkpoint_sha256']
            manifest.append(dict(model=kind,identity='original',seed=seed,checkpoint=str(cp),sha256=sha(cp),
              config=str(cp.parent/'protocol.json'),config_sha256=sha(cp.parent/'protocol.json'),
              predictions=str(ev/'modal_rollouts.pt'),prediction_sha256=sha(ev/'modal_rollouts.pt'),
              evaluation_script=str(R/'code/evaluate_periodic_round2.py'),reference='common periodic r32 POD reconstruction'))
    cp=R/'OpInf_P_development/validation_selected.npz'
    manifest.append(dict(model='OpInf',identity='deterministic fitted polynomial',seed='N/A',checkpoint=str(cp),sha256=sha(cp),
        config=str(R/'OpInf_P_development/protocol.json'),config_sha256=sha(R/'OpInf_P_development/protocol.json'),
        predictions=str(R/'OpInf_P_heldout/window_*.npz'),prediction_sha256='per-file manifest to follow',
        evaluation_script=str(R/'code/evaluate_opinf.py'),reference='common periodic r32 POD reconstruction'))
    csvwrite(OUT/'model_identity_manifest.csv',manifest)
    wr=[]
    for split,starts in [('validation',val),('heldout',test)]:
        for s in starts:
            wr.append(dict(split=split,start=s,trajectory=d['labels'][s],Re=float(d['idx'][s]['Re']),
                initial_time=float(d['time'][s]),times=d['time'][s+1:s+49].tolist(),history_indices=[s-2,s-1,s],
                target_indices=list(range(s+1,s+49)),local_start=int(d['idx'][s]['local_snapshot_index'])))
    write(OUT/'windows.json',wr)
    config=dict(frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),method=METHOD,
        scope='Circular Periodic only; independent paper-based restricted implementation',seeds=SEEDS,J=[1,2,4,8],rank=[16,32],
        excluded_rank64='verified parent basis and physical operators have dimension 32',substeps=[1,2,4,8],
        kmeans=dict(package='scikit-learn 1.7.2',init='k-means++',n_init=10,max_iter=300,tol=1e-4,algorithm='lloyd',empty_cluster='sklearn relocation; rank-deficient candidate rejected'),
        precision='CPU float64; parent artifacts stored float32/float64',fit='8539 training snapshots from 53 Re only',
        selection='all 48 validation windows complete in all 3 seeds, minimum mean Eu48+Ep48; relative ties 1e-8 choose r,J,substeps lexicographically',
        pressure='inherited algebraic PPE composed into momentum at every RK4 stage; no learned pressure correction',
        switch='after each full internal RK4 step, nearest predicted state; no intra-stage switching',
        failure='nonfinite outputs retained; finite large errors retained; incomplete full-window metric undefined',
        aggregate='time mean -> window mean within Re -> equal Re mean -> seed mean and sample sd',
        gauge='area mean removed separately from prediction and reference; no epsilon',
        validation_extension='if no finite candidate or best upper-bound 8-step candidate changes joint metric >1% versus same configuration at 4, expand all configurations once to 16 before testing',
        timing='3 warmups,10 repeats; physical in-memory history -> all decoded fields; exclude loading/I/O; CPU threads2; FP64 ql-ROM vs FP32 neural',
        reference='same r32 train-only POD-reconstructed reference for all methods',
        files={str(p):sha(p) for p in pathlist},windows_sha256=sha(OUT/'windows.json'),script_sha256=sha(__file__))
    # JSON is a valid YAML 1.2 document.
    write(OUT/'protocol_frozen.yaml',config)
    write(OUT/'data_audit.json',dict(splits={k:len(v) for k,v in split_sets.items()},
        snapshot_counts={k:int(np.sum(d['split']==k)) for k in split_sets},
        split_overlap=False,train_only_parent_basis=True,
        operator_Re_c_max_difference=float(np.max(np.abs(d['v']['c_all']-d['v']['c_all'][0]))),
        operator_Re_A_max_difference=float(np.max(np.abs(d['v']['A_all']-d['v']['A_all'][0]))),
        gram_error=float(np.max(np.abs(d['G']-np.eye(32)))),
        operator_metadata=json.loads(str(d['v']['metadata_json'])),ppe_metadata=json.loads(str(d['q']['metadata_json']))))
    print('PREPARED',len(val),len(test),flush=True)

def unit_checks(d):
    rng=np.random.default_rng(913);label=d['labels'][0]
    m=build(d,2,16,1248);ops,pops=local_ops(d,m,label);vel,pressure=ambient_ops(d,label)
    checks={}
    checks['weighted_orthogonality']=max(float(np.max(np.abs(U.T@U-np.eye(16)))) for U in m['bases'])
    checks['local_vs_direct_rhs']=0.;checks['pressure_transform']=0.;checks['switch_formula']=0.
    for j in range(2):
        z=rng.normal(size=(7,16))*.01;c=m['centers'][j];U=m['bases'][j]
        a=(c+z@U.T)@d['Si'].T
        direct=polynomial(*vel,a)@d['S'].T@U
        local=polynomial(*(v[j] for v in ops),z)
        checks['local_vs_direct_rhs']=max(checks['local_vs_direct_rhs'],float(np.max(np.abs(direct-local))))
        bp=polynomial(*(v[j] for v in pops),z)
        checks['pressure_transform']=max(checks['pressure_transform'],float(np.max(np.abs(bp-polynomial(*pressure,a)))))
        U2=m['bases'][1-j];c2=m['centers'][1-j]
        formula=z@U.T@U2+(c-c2)@U2
        direct_switch=(a@d['S'].T-c2)@U2
        checks['switch_formula']=max(checks['switch_formula'],float(np.max(np.abs(formula-direct_switch))))
    full=build(d,1,32,1248);c=full['centers'][0];U=full['bases'][0]
    test=d['y'][d['split']=='train'][:32]
    checks['full_rank_encode_decode']=float(np.max(np.abs(c+(test-c)@U@U.T-test)))
    op,_=local_ops(d,full,label);a=d['a'][2].copy();z=(d['S']@a-c)@U;h=.01
    def step(x,f):
        k1=f(x);k2=f(x+h*k1/2);k3=f(x+h*k2/2);k4=f(x+h*k3)
        return x+h*(k1+2*k2+2*k3+k4)/6
    a1=step(a,lambda x:polynomial(*vel,x[None])[0])
    z1=step(z,lambda x:polynomial(*(v[0] for v in op),x[None])[0])
    checks['J1_full_rank_RK4_degeneracy']=float(np.max(np.abs(a1-d['Si']@(c+U@z1))))
    checks['constant_translation_term']=float(np.max(np.abs(op[0][0]-U.T@d['S']@polynomial(*vel,(d['Si']@c)[None])[0])))
    checks['all_pass']=all(v<1e-9 for v in checks.values())
    write(OUT/'unit_checks.json',checks);assert checks['all_pass'],checks
    print('UNIT',checks,flush=True)

def validate(d):
    assert (OUT/'unit_checks.json').exists()
    val,_=windows(d);outrows=[];modeldir=OUT/'models';modeldir.mkdir(exist_ok=True)
    config=json.loads((OUT/'protocol_frozen.yaml').read_text())
    for J in config['J']:
        for r in config['rank']:
            for seed in SEEDS:
                m=build(d,J,r,seed);path=modeldir/f'J{J}_r{r}_s{seed}.npz'
                np.savez_compressed(path,centers=m['centers'],bases=m['bases'],J=J,r=r,seed=seed,
                    population=m['population'],effective_ranks=m['ranks'],offline_seconds=m['offline_seconds'])
                for sub in config['substeps']:
                    t=time.perf_counter();rows,_=evaluate(d,m,val,sub)
                    row=dict(J=J,r=r,seed=seed,substeps=sub,seconds=time.perf_counter()-t,**aggregate(rows,48))
                    outrows.append(row);csvwrite(OUT/'validation_search.csv',outrows)
                    print('VALIDATION',row,flush=True)
    choose(d,outrows,val)

def load_model(J,r,seed):
    z=np.load(OUT/f'models/J{J}_r{r}_s{seed}.npz')
    return dict(centers=z['centers'],bases=z['bases'],J=J,r=r,seed=seed)

def choose(d,rows,val):
    def candidates(rows):
        found=[]
        for J,r,sub in sorted(set((v['J'],v['r'],v['substeps']) for v in rows)):
            rr=[v for v in rows if (v['J'],v['r'],v['substeps'])==(J,r,sub)]
            if len(rr)==3 and all(v['completed']==v['total'] and np.isfinite(v['joint']) for v in rr):
                found.append(dict(J=J,r=r,substeps=sub,validation_joint=float(np.mean([v['joint'] for v in rr]))))
        return found
    choices=candidates(rows);extend=not choices
    if choices:
        best=min(choices,key=lambda v:v['validation_joint'])
        if best['substeps']==8:
            prev=[v for v in choices if (v['J'],v['r'],v['substeps'])==(best['J'],best['r'],4)]
            extend=not prev or abs(prev[0]['validation_joint']/best['validation_joint']-1)>.01
    if extend:
        write(OUT/'validation_extension.json',dict(reason='frozen upper-bound rule triggered',added_substeps=16,time=time.strftime('%FT%T%z')))
        for J,r,seed in sorted(set((v['J'],v['r'],v['seed']) for v in rows)):
            m=load_model(J,r,seed);t=time.perf_counter();rr,_=evaluate(d,m,val,16)
            row=dict(J=J,r=r,seed=seed,substeps=16,seconds=time.perf_counter()-t,**aggregate(rr,48))
            rows.append(row);csvwrite(OUT/'validation_search.csv',rows);print('EXTENDED',row,flush=True)
        choices=candidates(rows)
    if not choices:
        write(OUT/'selected_config.json',dict(status='validation_failed',test_authorized=False));return
    minimum=min(v['validation_joint'] for v in choices)
    tied=[v for v in choices if v['validation_joint']<=minimum+max(abs(minimum),1)*1e-8]
    best=min(tied,key=lambda v:(v['r'],v['J'],v['substeps']))
    best.update(status='selected',selected_at=time.strftime('%FT%T%z'),seeds=SEEDS,
                test_seen=False,validation_search_sha256=sha(OUT/'validation_search.csv'))
    write(OUT/'selected_config.json',best);print('SELECTED',best,flush=True)

def test(d):
    selected=json.loads((OUT/'selected_config.json').read_text())
    assert selected['status']=='selected'
    _,starts=windows(d);allrows=[];perstep=[]
    pred=OUT/'predictions';pred.mkdir(exist_ok=True)
    for seed in SEEDS:
        m=load_model(selected['J'],selected['r'],seed)
        rows,res=evaluate(d,m,starts,selected['substeps']);allrows.extend(rows)
        for s,z in res.items():
            np.savez_compressed(pred/f'qlrom_s{seed}_start{s}.npz',**z,start=s,times=d['time'][s+1:s+49])
            for k in range(48):perstep.append(dict(method=METHOD,seed=seed,Re=float(d['idx'][s]['Re']),window=s,
                step=k+1,time=float(d['time'][s+k+1]),reference='common_POD_r32_gauge_fixed',
                Eu=float(z['Eu'][k]),Ep=float(z['Ep'][k]),cluster=int(z['cluster'][k]),
                switches=int(z['switches'][k]),switch_projection_jump=float(z['jump'][k])))
        print('TEST',seed,aggregate(rows,24),aggregate(rows,48),flush=True)
    csvwrite(OUT/'completion.csv',allrows);csvwrite(OUT/'per_step_metrics.csv',perstep)
    write(OUT/'TEST_COMPLETED.json',dict(completed=True,selected_sha256=sha(OUT/'selected_config.json')))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','unit','validate','test']);stage=p.parse_args().stage
    d=load_data()
    {'prepare':prepare,'unit':unit_checks,'validate':validate,'test':test}[stage](d)

if __name__=='__main__':main()
