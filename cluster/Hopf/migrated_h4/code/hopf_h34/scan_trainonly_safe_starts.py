#!/usr/bin/env python3
"""Fail-closed warm-start gate using the formal trainer's exact forward path."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch

def load(path):
 s=importlib.util.spec_from_file_location('h34',path);m=importlib.util.module_from_spec(s);assert s.loader;sys.modules[s.name]=m;s.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--trainer',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--coeff',type=Path,required=True);p.add_argument('--galerkin',type=Path,required=True);p.add_argument('--pressure',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=256);p.add_argument('--allow-tf32',action=argparse.BooleanOptionalAction,default=True);a=p.parse_args()
 H=load(a.trainer);B=H.load_module(a.baseline);device=torch.device('cuda');H.seed_all(1248);H.configure_math(a.allow_tf32)
 base=SimpleNamespace(coefficient_view=a.coeff,galerkin_path=a.galerkin,pressure_path=a.pressure,asset_manifest=a.manifest,history_len=3,hidden_dim=256,expert_hidden=1024,num_blocks=3,experts=6,top_k=2,expert_blocks=4,quadratic_rank=4,dropout=.04,temperature=.8,adaptive_gate_initial_logit=6.,lambda_scale_amplitude=1.,lambda_scale_growth=.5,lambda_scale_sign=.1,scale_floor_quantile=.1)
 data=B.load_coefficients(base);romnp=B.load_train_rom(base);norm,_=B.fit_stats(data,romnp,base);stats={k:torch.as_tensor(v,device=device) for k,v in asdict(norm).items()};rom={k:torch.as_tensor(v,device=device) for k,v in romnp.items()};fc=H.FluctuationContract(a.contract,device);state=B.state_features;probe=B.batch_from_ids(data,data['train_ids'][:2],device);rh=B.galerkin(probe['ah'].reshape(-1,32),probe['bh'].reshape(-1,32),probe['re'][:,None].expand(-1,3).reshape(-1),rom).reshape_as(probe['ah']);raw_dim=state(probe['a'],probe['b'],probe['re'],probe['ah'],probe['bh'],rh,rom,stats)[0].shape[1];dim=raw_dim
 model=B.build_model(dim,base,stats,device);parent,parent_sha,parent_keys=H.load_parent_weights(model,H.PARENT_B,H.PARENT_B_SHA);model.eval();_,roll=H.make_ops(B,fc,state)
 # The zero-extended H3/H4 backbone must reproduce the frozen parent before new losses.
 parent_model=B.build_model(raw_dim,base,stats,device);parent_model.load_state_dict(parent['model_state'],strict=True);parent_model.eval();eq_starts=fc.schedule[0][:min(a.batch,len(fc.schedule[0]))]
 with torch.no_grad():
  old=B.rollout_batch(parent_model,data,eq_starts,8,rom,stats,device);new=roll(model,data,eq_starts,8,rom,stats,device)
  diffs={name:max(float((x-y).abs().max().cpu()) for x,y in zip(old[name],new[name])) for name in ('pred_a','pred_b','pred_rhs')}
 equivalence={'batch_size':int(len(eq_starts)),'horizon':8,'max_abs':diffs,'atol':2e-4,'passed':max(diffs.values())<=2e-4}
 unsafe=set();counts={};by_re={}
 with torch.no_grad():
  for k in (1,2,4,8):
   starts=B.legal_starts(data,data['train_ids'],k);kbad=set();re_counts={}
   # Preserve the formal batch shape and use the exact shared safety function.
   for lo in range(0,len(starts),a.batch):
    part=starts[lo:lo+a.batch];out=roll(model,data,part,k,rom,stats,device);bad,_=H.rollout_safety(out)
    for i in part[bad.cpu().numpy()]:kbad.add(int(i));unsafe.add(int(i))
   for rv in H.TRAIN:
    ids=starts[np.isclose(data['re'][starts],rv,atol=5e-6)];nb=sum(int(i) in kbad for i in ids);re_counts[f'{rv:.6f}']={'candidate':int(len(ids)),'safe':int(len(ids)-nb),'unsafe':int(nb)}
   counts[str(k)]={'candidate':int(len(starts)),'unsafe_this_horizon':len(kbad),'unsafe_cumulative':len(unsafe)};by_re[str(k)]=re_counts
  schedule_gate={}
  for step,k in ((0,1),(1200,2),(2800,4),(4800,8)):
   starts=fc.schedule[step];out=roll(model,data,starts,k,rom,stats,device);bad,diag=H.rollout_safety(out);schedule_gate[str(step)]={'horizon':k,**diag,'bad_starts':[int(x) for x in starts[bad.cpu().numpy()]]}
 all_re_safe=all(x['safe']>0 for rows in by_re.values() for x in rows.values());schedule_passed=all(x['bad_count']==0 for x in schedule_gate.values())
 report={'schema':'H3H4_EXACT_FORMAL_FORWARD_GATE_V1','criterion':'formal trainer make_ops + rollout_safety; all rollout steps finite and modal norm <=10x truth','math_policy':{'allow_tf32':a.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision()},'batch':a.batch,'parent_sha256':parent_sha,'parent_copied_keys':parent_keys,'parent_output_equivalence':equivalence,'unsafe_starts':sorted(unsafe),'counts':counts,'by_re':by_re,'stage_first_schedule_batches':schedule_gate,'all_re_have_safe_windows':all_re_safe,'passed':bool(equivalence['passed'] and all_re_safe and schedule_passed and not unsafe)}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({'passed':report['passed'],'unsafe':len(unsafe),'equivalence':equivalence,'schedule':schedule_gate,'parent_sha256':parent_sha}))
 if not report['passed']:raise SystemExit(2)
if __name__=='__main__':main()
