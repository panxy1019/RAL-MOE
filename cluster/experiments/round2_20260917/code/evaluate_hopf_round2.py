"""Frozen H controls/proposed evaluation on identical native heldout windows."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np
import torch
from torch import nn
import periodic_controls as controls
R=Path(__file__).resolve().parents[1]
BASE=controls.ROOT/'Hopf/migrated_h4_expanded'

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['proposed','dense','structured'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--quadratic-fixed',action='store_true')
 ap.add_argument('--checkpoint-family',default=None);ap.add_argument('--evaluation-family',default=None);a=ap.parse_args()
 default_eval='H_evaluation_fixed' if a.quadratic_fixed else 'H_evaluation'
 out=R/(a.evaluation_family or default_eval)/a.kind/str(a.seed)
 if out.exists() and any(out.iterdir()):raise FileExistsError(out)
 out.mkdir(parents=True,exist_ok=True)
 tag='dense_fp32' if a.kind=='dense' else a.kind
 checkpoint=R/f'H_{tag}'/str(a.seed)/'HopfExpanded34_H4_NormalFormRadial_r32/best_validation.pt'
 if a.quadratic_fixed:
  checkpoint=R/(a.checkpoint_family or 'quadratic_fixed')/'H'/a.kind/str(a.seed)/'HopfExpanded34_H4_NormalFormRadial_r32/best_validation.pt'
 if not checkpoint.exists():
  final=checkpoint.parent/'final_training.pt'
  if not final.exists():raise RuntimeError('Missing accepted checkpoint and no completed training record')
  ck=torch.load(final,map_location='cpu',weights_only=False)
  assert ck['best_step']<0
  (out/'VALIDATION_REJECTED.json').write_text(json.dumps(dict(kind=a.kind,seed=a.seed,best_step=ck['best_step'],last_step=ck['optimizer_step'],reason='No checkpoint passed original validation gate; no heldout evaluation',completed_training=True),indent=2))
  return
 B=load('h_base_eval',BASE/'code/train_hopf_moe_expanded.py')
 if a.kind!='proposed':
  controls.TARGET=14165816;controls.patch(B.v16,a.kind,out)
  original=B.v16.OperatorSpaceMoEROM
  class Compatible(original):
   def __init__(self,*args,**kw):super().__init__(*args,**kw);self.group_router=nn.Identity()
  B.v16.OperatorSpaceMoEROM=Compatible
 source=BASE/'code/evaluate_h4_expanded.py';text=source.read_text()
 anchor='        times_np = np.stack(times, axis=1)'
 assert text.count(anchor)==1
 text=text.replace(anchor,anchor+'\n        np.savez_compressed(OUTPUT / f"Re_{value:.6f}_modal.npz", pred_a=pa,pred_b=pb,true_a=ta,true_b=tb,times=times_np,starts=starts,initial_times=data["time"][starts])')
 m=types.ModuleType('h_round2_eval');m.__file__=str(source);m.OUTPUT=out;sys.modules[m.__name__]=m;exec(compile(text,str(source),'exec'),m.__dict__)
 with torch.inference_mode():result=m.evaluate_one(B,checkpoint,controls.ROOT/'Hopf/artifacts/hopf',BASE/'trainonly_contract/trainonly_fluctuation_contract.npz',8)
 (out/'metrics.json').write_text(json.dumps(result,indent=2))
 (out/'protocol.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
   reference='native POD reconstructed fields; not full FOM',kind=a.kind,seed=a.seed,window_stride=8,horizon=48,
   checkpoint_family=a.checkpoint_family,evaluation_family=a.evaluation_family,
   modified_source_sha256=hashlib.sha256(text.encode()).hexdigest(),timing='not measured during concurrent training'),indent=2))
 (out/'COMPLETED.json').write_text('{"completed":true}')
if __name__=='__main__':main()
