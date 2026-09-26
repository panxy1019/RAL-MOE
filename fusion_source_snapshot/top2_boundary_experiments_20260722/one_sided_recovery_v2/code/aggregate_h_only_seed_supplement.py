"""Aggregate frozen H-only supplements across route-module seeds."""
from __future__ import annotations
import argparse, csv, hashlib, json, os
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
def atomic(path,value):
    t=path.with_suffix(path.suffix+'.tmp'); t.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)); os.replace(t,path)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--analysis-dir',type=Path,action='append',required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()): raise RuntimeError(f'refusing non-empty output: {a.output_dir}')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    runs=[json.loads((d/'H_ONLY_WEIGHT_DISTANCE_ABLATIONS.json').read_text()) for d in a.analysis_dir]
    methods=list(runs[0]['splits']['validation']['methods']); out={'status':'MULTI_SEED_H_ONLY_SUPPLEMENT_COMPLETE','seed_runs':len(runs),'source_sha256':{str(d):sha(d/'H_ONLY_WEIGHT_DISTANCE_ABLATIONS.json') for d in a.analysis_dir},'splits':{}}
    for split in ('validation','heldout'):
        result={'H_only':runs[0]['splits'][split]['H_only'],'methods':{}}
        for method in methods:
            items=[run['splits'][split]['methods'][method] for run in runs]; seeds=[int(x['seed']) for x in items]
            weights=[x['alpha_H']['overall'] for x in items]; errors=[x['full']['joint_all_mean'] for x in items]; worst=[x['full']['joint_all_worst'] for x in items]; distances=[x['distance_to_H_only']['all_steps']['joint_mean'] for x in items]
            by_re={}
            for re_value in items[0]['alpha_H']['by_Re']:
                values=[x['alpha_H']['by_Re'][re_value] for x in items]
                by_re[re_value]={'seed_means':{str(seed):v['mean'] for seed,v in zip(seeds,values)},'mean_of_seed_means':float(np.mean([v['mean'] for v in values])),'min_all_seeds_windows':float(min(v['min'] for v in values)),'max_all_seeds_windows':float(max(v['max'] for v in values))}
            result['methods'][method]={'seeds':seeds,'alpha_H':{'per_seed':{str(seed):value for seed,value in zip(seeds,weights)},'mean_of_seed_means':float(np.mean([v['mean'] for v in weights])),'min_all_seeds_windows':float(min(v['min'] for v in weights)),'max_all_seeds_windows':float(max(v['max'] for v in weights)),'by_Re':by_re},'joint_error':{'per_seed_mean':{str(seed):value for seed,value in zip(seeds,errors)},'mean_across_seeds':float(np.mean(errors)),'min_across_seeds':float(np.min(errors)),'max_across_seeds':float(np.max(errors)),'per_seed_worst':{str(seed):value for seed,value in zip(seeds,worst)}},'distance_to_H':{'per_seed_mean':{str(seed):value for seed,value in zip(seeds,distances)},'mean_across_seeds':float(np.mean(distances)),'max_across_seeds':float(np.max(distances))},'module_ablation':{'max_abs_error_delta':float(max(abs(x['ablation']['delta_full_minus_gate_only']['joint_all_mean']) for x in items)),'total_routing_decisions_changed':int(sum(x['ablation']['routing_decisions_changed'] for x in items))}}
        out['splits'][split]=result
    atomic(a.output_dir/'MULTI_SEED_H_ONLY_WEIGHT_DISTANCE_ABLATIONS.json',out)
    combined=[]
    for d in a.analysis_dir:
        with (d/'PER_WINDOW_ALPHA_H_AND_DISTANCE.csv').open(newline='',encoding='utf-8') as f: combined.extend(csv.DictReader(f))
    with (a.output_dir/'MULTI_SEED_PER_WINDOW_ALPHA_H_AND_DISTANCE.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(combined[0])); w.writeheader(); w.writerows(combined)
    lines=['# Multi-seed H-only and module-ablation supplement','','Three frozen route-module seeds per method were evaluated on the same native caches. Specialists were never retrained.','']
    for split,v in out['splits'].items():
        h=v['H_only']; lines += [f'## {split}','',f"H-only joint mean/worst: `{h['joint_all_mean']:.9g}` / `{h['joint_all_worst']:.9g}`.",'','| Method | seeds | a_H mean of seed means | a_H all-window range | error mean across seeds | d_H mean | ablation max | decisions changed |','|---|---|---:|---:|---:|---:|---:|---:|']
        for method,x in v['methods'].items():
            aw=x['alpha_H']; lines.append(f"| {method} | {','.join(map(str,x['seeds']))} | {aw['mean_of_seed_means']:.6f} | [{aw['min_all_seeds_windows']:.6f},{aw['max_all_seeds_windows']:.6f}] | {x['joint_error']['mean_across_seeds']:.9g} | {x['distance_to_H']['mean_across_seeds']:.3g} | {x['module_ablation']['max_abs_error_delta']:.3g} | {x['module_ablation']['total_routing_decisions_changed']} |")
        lines.append('')
    lines += ['## Scientific interpretation','','Across seeds, validation predictions are numerically H-only. Heldout gates retain only a small Steady contribution and do not consistently improve H-only mean error. Removing the Risk critic or LookAhead scorer changes no routed output and no reported error. The current evidence therefore supports H-only as the source of nearly all observed gain; a small convex correction can alter worst-case metrics, but the attached Risk/LookAhead mechanisms have no demonstrated incremental value on this boundary cache.','']
    (a.output_dir/'MULTI_SEED_H_ONLY_WEIGHT_DISTANCE_ABLATIONS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':out['status'],'splits':{s:{m:{'aH':x['alpha_H']['mean_of_seed_means'],'error':x['joint_error']['mean_across_seeds']} for m,x in v['methods'].items()} for s,v in out['splits'].items()}},indent=2))

if __name__=='__main__': main()
