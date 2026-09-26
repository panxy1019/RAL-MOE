"""Direct raw-CFD reference errors via exact weighted quadratic forms."""
from qlrom_experiment import *
from reaggregate import cached,summarize

def main():
    d=load_data();_,starts=windows(d);lookup={};audit=[];proj=[]
    w=d['u']['point_areas'].astype(float);wu=np.r_[w,w]
    pu=d['u']['phi_uv'].astype(float);pp=d['p']['phi_p'].astype(float)
    mu=d['u']['mean_uv_regime'].astype(float);mp=d['p']['mean_p_regime'].astype(float)
    pp-=(pp@w/w.sum())[:,None];mp-=mp@w/w.sum()
    for label in sorted(set(d['labels'][starts])):
        path=OUT/'raw_reference'/f'{label}_uvp_pointData.npz';z=np.load(path)
        assert np.array_equal(z['points'],d['u']['points']), 'Grid point ordering mismatch'
        ids=np.flatnonzero(d['labels']==label);li=np.array([int(d['idx'][s]['local_snapshot_index']) for s in ids])
        exact_time=np.array([float(d['idx'][s]['time']) for s in ids])
        assert np.max(np.abs(z['times'][li]-exact_time))<1e-6,'Raw time mismatch'
        rawu=np.c_[z['u'][li],z['v'][li]].astype(float);rawp=z['p'][li].astype(float)
        rawp-=(rawp@w/w.sum())[:,None]
        info=dict(label=label,path=str(path),sha256=sha(path),bytes=path.stat().st_size,grid_exact=True,
            raw_time_max_abs_difference=float(np.max(np.abs(z['times'][li]-exact_time))),
            float32_evaluation_time_max_difference=float(np.max(np.abs(d['time'][ids]-exact_time))),
            source_url=f'https://huggingface.co/datasets/panxy1019/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re/resolve/main/{path.name}')
        geoms=[]
        for name,x,phi,mean,weights,coef in [('u',rawu,pu,mu,wu,d['a'][ids]),('p',rawp,pp,mp,w,d['b'][ids])]:
            delta=mean-x;G=(phi*weights)@phi.T
            linear=(delta*weights)@phi.T;constant=(delta*delta)@weights;den=(x*x)@weights
            assert np.all(den>0)
            encoded=((x-mean)*weights)@phi.T@np.linalg.inv(G)
            info[name+'_cache_vs_raw_projection_max_abs']=float(np.max(np.abs(encoded-coef)))
            geoms.append((G,linear,constant,den))
            # Validate expanded norm against explicit reconstructed fields, not a projection-error subtraction.
            pred=coef[[0,len(ids)//2,-1]]+.01
            residual=pred@phi+mean-x[[0,len(ids)//2,-1]]
            direct=100*np.sqrt((residual*residual)@weights/den[[0,len(ids)//2,-1]])
            quad=100*np.sqrt((np.einsum('ni,ij,nj->n',pred,G,pred)+2*np.sum(pred*linear[[0,len(ids)//2,-1]],1)+constant[[0,len(ids)//2,-1]])/den[[0,len(ids)//2,-1]])
            info[name+'_quadratic_vs_explicit_error_max_abs']=float(np.max(np.abs(direct-quad)))
            assert np.max(np.abs(direct-quad))<1e-8
        for k,s in enumerate(ids):lookup[s]=[(G,L[k],C[k],D[k]) for G,L,C,D in geoms]
        audit.append(info);print('RAW_ALIGNED',label,flush=True)
    def error(a,ids,component):
        v=[]
        for x,s in zip(a,ids):
            G,L,C,D=lookup[s][component];v.append(100*np.sqrt(max(float(x@G@x+2*x@L+C),0)/D))
        return np.array(v)
    predictions=list(cached(d))
    for seed in SEEDS:
        for s in starts:
            path=OUT/f'predictions/qlrom_s{seed}_start{s}.npz';z=np.load(path)
            predictions.append(('ql-ROM-style (fixed POD)',seed,s,z['pred_a'],z['pred_b'],str(path)))
    rows=[];steps=[]
    for kind,seed,s,a,b,source in predictions:
        ids=list(range(s+1,s+49));eu=error(a,ids,0);ep=error(b,ids,1)
        for K in [24,48]:
            ok=bool(np.isfinite(a[:K]).all() and np.isfinite(b[:K]).all())
            rows.append(dict(method=kind,seed=seed,Re=float(d['idx'][s]['Re']),start=s,horizon=K,complete=ok,
                Eu=float(np.mean(eu[:K])) if ok else float('nan'),Ep=float(np.mean(ep[:K])) if ok else float('nan'),failure='' if ok else 'incomplete'))
        for k in range(48):steps.append(dict(method=kind,seed=seed,Re=float(d['idx'][s]['Re']),window=s,step=k+1,
            time=float(d['time'][s+k+1]),Eu=float(eu[k]),Ep=float(ep[k]),reference='raw_CFD_individually_area_gauge_fixed',source=source))
    summary,seedrows=summarize(rows)
    csvwrite(OUT/'summary_raw_cfd.csv',summary);csvwrite(OUT/'seed_metrics_raw.csv',seedrows)
    csvwrite(OUT/'raw_per_step_metrics.csv',steps);csvwrite(OUT/'raw_completion.csv',rows)
    for seed in SEEDS:
        model=load_model(8,16,seed)
        for s in starts:
            ids=np.arange(s+1,s+49);y=d['y'][ids];C,U=model['centers'],model['bases']
            j=((y[:,None]-C[None])**2).sum(2).argmin(1)
            zz=np.einsum('nir,ni->nr',U[j],y-C[j]);aa=(C[j]+np.einsum('nir,nr->ni',U[j],zz))@d['Si'].T
            eu=error(aa,ids,0);parent=error(d['a'][ids],ids,0);parentp=error(d['b'][ids],ids,1)
            for K in [24,48]:proj.append(dict(seed=seed,Re=float(d['idx'][s]['Re']),window=s,horizon=K,
                local_velocity_projection_error_percent=float(eu[:K].mean()),parent_velocity_projection_error_percent=float(parent[:K].mean()),
                parent_pressure_projection_error_percent=float(parentp[:K].mean()),note='diagnostic true-state nearest center; never used in autonomous rollout'))
    csvwrite(OUT/'raw_projection_diagnostics.csv',proj)
    write(OUT/'raw_reference_audit.json',audit)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
