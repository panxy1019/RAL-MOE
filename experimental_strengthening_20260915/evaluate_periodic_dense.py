"""Common-window K48 physical-space errors; standalone benchmark only on idle GPU."""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import shlex
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from dense_periodic import ROOT, patch, write

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['dense', 'full', 'galerkin', 'global'], required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--global-windows', action='store_true')
    cli = p.parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    base = ROOT/'periodic_specialist_r32'
    global_data = ROOT/'V16_1_SteadyPressureAnchor32/assets/common_global_data'
    global_source = ROOT/'paper_experiments/revisions/revision10_missing_metric_completion_20260725/code/global_eval_revision10.py'
    trainer = load('native_eval', global_source if cli.method=='global' else base/'code/train_periodic_moe.py')
    helper = load('native_helper', base/'code/evaluate_periodic_r32_portable.py')
    if cli.method == 'dense':
        patch(trainer, cli.output/'architecture.json')
    checkpoint_path = cli.checkpoint or base/'checkpoint/FINAL_PERIODIC_SPECIALIST.pt'
    if cli.method=='global' and cli.checkpoint is None:
        checkpoint_path = ROOT/'paper_experiments/revisions/revision5_supplemental_evaluation_20260725/recovery/global_k56_valid_history_windows_v5/V16_1_SteadyPressureAnchor32_ru32_rp32_Re_24p630436_checkpoint.pt'
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if cli.method=='global':
        argv = sys.argv
        sys.argv = ['global_eval', *shlex.split('--r-u 32 --r-p 32 --num-blocks 3 --num-regime-groups 3 --experts-per-group 6 --num-experts 6 --num-shared-experts 1 --top-k 2 --group-top-k 1 --hidden-dim 224 --expert-hidden 768 --expert-blocks 3 --quadratic-rank 4 --quadratic-scale 0.05 --dropout 0.04 --temperature 0.95 --gate-floor 0 --group-temperature 0.9 --group-gate-floor 0 --shared-scale 1 --routed-scale 0.85 --rhs-target residual --pressure-target closure --pressure-input-mode pressure_only --closure-mode adaptive_gate --pressure-base-mode static --attractor-balanced-sampling --test-re-selection regime_default --swanlab-mode offline --allow-tf32')]
        saved = trainer.parse_args()
        sys.argv = argv
    else:
        saved = ck['args']
    args = SimpleNamespace(**(vars(saved) if hasattr(saved, '__dict__') else saved))
    args.data_root = base/'assets/Global_POD_AreaWeighted_L2'
    args.tensor_path = base/'assets/velocity_rom_periodic.npz'
    args.pressure_surrogate_path = base/'assets/pressure_poisson_surrogate_periodic.npz'
    if cli.method=='global':
        args.data_root = global_data/'Global_POD_AreaWeighted_L2'
        args.tensor_path = global_data/'semi_intrusive_galerkin_tensors_allRe100_areaWeightedL2_ru80_rp80_compact.npz'
        args.pressure_surrogate_path = global_data/'pressure_poisson_surrogate_tensors_allRe100_areaWeightedL2_ru80_rp80.npz'
    arrays, _ = trainer.build_arrays(args)
    tensors, pressure = np.load(args.tensor_path), np.load(args.pressure_surrogate_path)
    scalers = {k: trainer.Standardizer(mean=np.asarray(v['mean'], np.float32),
              scale=np.asarray(v['scale'], np.float32)) for k, v in ck['scalers'].items()}
    device = torch.device('cuda')
    model = None if cli.method == 'galerkin' else helper.instantiate_model(trainer, args, arrays, ck['model_state'], device)
    del ck
    vel = np.load(args.data_root/'global_velocity_pod_area_weighted_l2.npz')
    pre = np.load(args.data_root/'global_pressure_pod_area_weighted_l2.npz')
    gu = gp = None
    if cli.method!='global':
        gu = helper.weighted_geometry(vel['phi_uv'][:args.r_u], vel['mean_uv_regime'], vel['point_areas'], True)
        gp = helper.weighted_geometry(pre['phi_p'][:args.r_p], pre['mean_p_regime'], pre['point_areas'], False)
    valid = set(arrays['sample_ids'].tolist())
    starts = []
    for re in helper.HELDOUT_RE:
        ids = np.flatnonzero(np.isclose(arrays['re'], re, atol=2e-5, rtol=0))
        ids = ids[np.argsort(arrays['time'][ids])]
        candidates = [int(ids[j]) for j in range(1, len(ids)-49, 8) if int(ids[j]) in valid]
        assert candidates, re
        # Eight evenly spaced legal native windows, fixed identically for all methods.
        chosen = np.unique(np.linspace(0, len(candidates)-1, min(8,len(candidates))).astype(int))
        starts.extend(candidates[j] for j in chosen)
    if cli.global_windows:
        starts = []
        cache = ROOT/'paper_experiments/revisions/revision10_missing_metric_completion_20260725/evaluations/global_periodic_k48_arrays_retry'
        for path in sorted(cache.glob('*rollout_arrays.npz')):
            z = np.load(path)
            re = float(z['Re'][0])
            for ts in z['times']:
                future = np.flatnonzero(np.isclose(arrays['re'], re, atol=2e-5, rtol=0)
                    & np.isclose(arrays['time'], ts[0], atol=1e-3, rtol=0))
                assert len(future)==1
                previous = np.flatnonzero(arrays['next_idx']==future[0])
                assert len(previous)==1 and int(previous[0]) in valid, (re,ts[0],previous)
                starts.append(int(previous[0]))

    def rollout(start):
        cur = start
        a, b = arrays['a'][cur].copy(), arrays['b'][cur].copy()
        ah, bh, rh = trainer.init_history_states_np(cur, arrays)
        result = []
        for k in range(48):
            nxt = int(arrays['next_idx'][cur])
            dt = float(arrays['time'][nxt]-arrays['time'][cur])
            if cli.method == 'galerkin':
                def rhs(v):
                    return trainer.galerkin_rhs_by_label(tensors, v[None,:], b[None,:],
                        arrays['label_id'][cur:cur+1], arrays['labels'], args.r_u, args.r_p)[0]
                g = rhs(a)
                k2, k3 = None, None
                if args.integrator == 'rk4':
                    k2 = rhs(a+.5*dt*g)
                    k3 = rhs(a+.5*dt*k2)
                    an = a+dt/6*(g+2*k2+2*k3+rhs(a+dt*k3))
                else:
                    an = a+dt*g
                bn = trainer.pressure_surrogate_by_label(pressure, an[None,:],
                    arrays['label_id'][cur:cur+1], arrays['labels'], args.r_u, args.r_p)[0]
            else:
                an, bn, g = trainer.integrate_autonomous_step_np(model,a,b,cur,dt,ah,bh,rh,
                    arrays,scalers,tensors,pressure,args,device)
            if not (np.isfinite(an).all() and np.isfinite(bn).all()):
                break
            result.append((an.copy(), bn.copy(), arrays['a'][nxt], arrays['b'][nxt], float(arrays['time'][nxt]-arrays['time'][start])))
            ah = np.concatenate([an[None,None,:], ah[:,:-1,:]],axis=1)
            bh = np.concatenate([bn[None,None,:], bh[:,:-1,:]],axis=1)
            rh = np.concatenate([g[None,None,:], rh[:,:-1,:]],axis=1)
            a,b,cur = an,bn,nxt
        return result

    results = []
    with torch.inference_mode():
        for start in starts:
            if cli.method=='global':
                ri = int(np.argmin(np.abs(vel['Re_values']-arrays['re'][start])))
                gu = helper.weighted_geometry(vel['phi_uv'][:args.r_u], vel['mean_uv_by_Re'][ri],vel['point_areas'],True)
                gp = helper.weighted_geometry(pre['phi_p'][:args.r_p], pre['mean_p_by_Re'][ri],pre['point_areas'],False)
            result = rollout(start)
            for k in range(48):
                row = dict(method=cli.method,start=start,Re=float(arrays['re'][start]),step=k+1,finite=k<len(result))
                if k < len(result):
                    pa,pb,ta,tb,t = result[k]
                    eu = float(np.sqrt(max(0,helper.field_error_energy(ta,pa,gu))/max(helper.field_energy(ta,gu),1e-30)))
                    ep = float(np.sqrt(max(0,helper.field_error_energy(tb,pb,gp))/max(helper.field_energy(tb,gp),1e-30)))
                    row.update(time=t,Eu_percent=100*eu,Ep_percent=100*ep,Ejoint_percent=100*(eu+ep))
                else:
                    row.update(time='',Eu_percent=float('nan'),Ep_percent=float('nan'),Ejoint_percent=float('nan'))
                results.append(row)
    with (cli.output/'window_step_errors.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    summary = []
    for k in range(1,49):
        rs = [r for r in results if r['step']==k]
        summary.append(dict(step=k,windows=len(rs),finite_windows=sum(r['finite'] for r in rs),
            **{m:float(np.mean([r[m] for r in rs])) for m in ['Eu_percent','Ep_percent','Ejoint_percent']}))
    write(cli.output/'summary.json',summary)
    write(cli.output/'protocol.json',dict(method=cli.method,starts=starts,horizon=48,
        checkpoint=str(checkpoint_path),checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        metric='mean per-window relative physical L2 against POD-reconstructed truth, original pressure gauge; Ejoint=Eu+Ep',
        nonfinite_policy='retain all windows and report NaN aggregate after any divergence; no survivor-only mean',
        scope='Circular Periodic local chart; full denotes frozen RAL Periodic specialist, not two-specialist fusion'))
    if cli.method=='global':
        write(cli.output/'ACCURACY_USAGE_WARNING.json', {
            'warning':'This native interface metric uses global projected truth and original pressure gauge. Do not use it in cross-chart accuracy comparison.',
            'use_instead':'global_common_truth_v1/window_step_errors.csv, common local projected truth and centered pressure',
            'runtime':'Runtime remains valid for the explicitly stated native modal implementation.'})
    if cli.benchmark:
        with torch.inference_mode():
            for _ in range(3): rollout(starts[0])
            samples=[]
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            for _ in range(10):
                torch.cuda.synchronize(); t=time.perf_counter()
                rr=rollout(starts[0]); torch.cuda.synchronize()
                samples.append(time.perf_counter()-t)
                assert len(rr)==48, 'incomplete rollout cannot be used for speed comparison'
        write(cli.output/'runtime.json',dict(seconds=samples,median_seconds=float(np.median(samples)),
            gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),warmups=3,repeats=10,batch=1,horizon=48,
            includes='native modal rollout, feature building, host/device transfers, pressure reconstruction in modal coordinates',
            excludes='checkpoint/data loading, physical-field decoding, error computation',
            caveat='Galerkin native NumPy path runs on CPU; this is native implementation timing, not GPU-kernel parity'))
    write(cli.output/'COMPLETED.json',{'completed':True})

if __name__=='__main__': main()
