"""Post-test descriptive diagnostics; no selection or tuning."""
from qlrom_experiment import *
import subprocess

def main():
    d=load_data();_,starts=windows(d);cfg=json.loads((OUT/'selected_config.json').read_text())
    rows=[];physical=[]
    for label in sorted(set(d['labels'][starts])):
        ss=[s for s in starts if d['labels'][s]==label]
        delta=np.concatenate([np.diff(d['time'][s:s+49]) for s in ss])
        physical.append(dict(label=label,output_dt_min=float(delta.min()),output_dt_max=float(delta.max()),
            K24_duration_min=min(d['time'][s+24]-d['time'][s] for s in ss),K24_duration_max=max(d['time'][s+24]-d['time'][s] for s in ss),
            K48_duration_min=min(d['time'][s+48]-d['time'][s] for s in ss),K48_duration_max=max(d['time'][s+48]-d['time'][s] for s in ss)))
    for seed in SEEDS:
        path=OUT/f'models/J8_r16_s{seed}.npz';z=np.load(path);m=load_model(8,16,seed)
        t=time.perf_counter();ops=local_ops(d,m,d['labels'][starts[0]]);elapsed=time.perf_counter()-t
        arrays={f'{part}_{key}':v for part,group in zip(['velocity','pressure'],ops) for key,v in zip(['constant','linear','quadratic'],group)}
        deploy=OUT/f'models/selected_local_operators_seed{seed}.npz';np.savez_compressed(deploy,**arrays)
        switches=[];jumps=[];initial=[];pressure_diffs=[];weights=d['p']['point_areas'].astype(float)
        for s in starts:
            p=np.load(OUT/f'predictions/qlrom_s{seed}_start{s}.npz')
            switches.append(int(p['switches'].sum()));jumps.extend(p['jump'].tolist());initial.append(float(p['initial_projection_jump']))
            _,po=ambient_ops(d,d['labels'][s]);pressure_diffs.append(float(np.max(np.abs(polynomial(*po,p['pred_a'])-p['pred_b']))))
        rows.append(dict(seed=seed,J=8,local_r=16,parent_r=32,pressure_r=32,cluster_population=z['population'].tolist(),
            effective_ranks=z['effective_ranks'].tolist(),fit_cluster_POD_seconds=float(z['offline_seconds']),
            operator_transform_one_parameter_seconds=elapsed,local_model_numeric_storage_bytes=int(sum(v.nbytes for v in arrays.values())+m['centers'].nbytes+m['bases'].nbytes),
            local_operators_file_bytes=deploy.stat().st_size,centers_bases_file_bytes=path.stat().st_size,
            active_dynamical_charts=1,total_switches=sum(switches),switches_per_window_min=min(switches),switches_per_window_max=max(switches),
            sum_switch_projection_jump=float(sum(jumps)),max_jump_per_output_interval=float(max(jumps)),initial_projection_jump_max=float(max(initial)),
            pressure_direct_vs_local_max_abs=max(pressure_diffs)))
    labels=sorted(set(d['labels']))
    base=ambient_ops(d,labels[0]);maxdiff=max(float(np.max(np.abs(a-b))) for lab in labels for ga,gb in zip(ambient_ops(d,lab),base) for a,b in zip(ga,gb))
    q=d['q'];L=q['L'];singular=np.linalg.svd(L,compute_uv=False)
    rhs=q[labels[0]+'_c_p'] if labels[0]+'_c_p' in q.files else None
    unit=dict(composed_velocity_and_pressure_all_Re_max_difference=maxdiff,L_singular_values=singular.tolist(),
        L_pseudoinverse_Moore_Penrose_relative_residual=float(np.linalg.norm(L@q['L_pinv']@L-L)/np.linalg.norm(L)),
        parent_velocity_field_basis_bytes=int(d['u']['phi_uv'].nbytes),parent_pressure_field_basis_bytes=int(d['p']['phi_p'].nbytes),
        scope='Local operator timing excludes inherited parent POD/PPE generation and file IO. All 63 parameter-labelled tensors are equal; one numerical tensor table can be shared but labels are preserved.',
        boundary='Inherited projected PPE neglects boundary terms; this verifies numerical inheritance, not boundary consistency with the CFD solver.')
    write(OUT/'deployment_and_switch_diagnostics.json',dict(models=rows,operator_checks=unit,physical_time=physical))
    times=[json.loads(p.read_text()) for p in sorted((OUT/'timing').glob('*/timing.json'))]
    for t in times:
        t['mean']=float(np.mean(t['seconds']));t['sample_sd']=float(np.std(t['seconds'],ddof=1))
    env=dict(cpu_model=subprocess.check_output(['lscpu'],text=True),
        GPU_after_timing=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,utilization.gpu,memory.used','--format=csv'],text=True),
        threadpools=__import__('threadpoolctl').threadpool_info(),note='Five timed processes ran sequentially. Shared host; no CPU exclusivity or affinity guarantee. GPU snapshots before/after idle, not a continuous utilization trace.')
    write(OUT/'timing.json',dict(results=times,environment=env))
    versions=[]
    paths=[R/'code/evaluate_periodic_round2.py',ROOT/'experiments/strengthening_20260915/code/evaluate_periodic_dense.py',
        R/'code/periodic_controls.py',R/'code/evaluate_opinf.py',ROOT/'periodic_specialist_r32/code/train_periodic_moe.py']
    for p in paths:
        versions.append(dict(path=str(p),exists=p.exists(),sha256=sha(p) if p.exists() else None))
    for p in sorted((R/'OpInf_P_heldout').glob('window_*.npz')):versions.append(dict(path=str(p),sha256=sha(p)))
    write(OUT/'source_and_prediction_hashes.json',versions)
    complete=[]
    for name in ['baseline_completion.csv','completion.csv']:
        for r in csv.DictReader((OUT/name).open()):
            complete.append(dict(method=r.get('method',METHOD),seed=r['seed'],Re=r['Re'],start=r['start'],horizon=r['horizon'],
                complete=r['complete'],failure=r['failure'],failed_step=r.get('failed_step',''),
                last_finite_time=float(d['time'][int(r['start'])+int(r['horizon'])]) if r['complete']=='True' else 'see prediction',source=name))
    csvwrite(OUT/'completion_all_methods.csv',complete)
    print(json.dumps(dict(models=rows,operator_checks=unit,physical_time=physical),indent=2),flush=True)

if __name__=='__main__':main()
