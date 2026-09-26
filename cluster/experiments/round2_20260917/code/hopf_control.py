"""H4 controls using native sealed data, losses, curriculum and checkpoint selector."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import torch
from torch import nn
import periodic_controls as controls

ROOT=controls.ROOT
BASE=ROOT/'Hopf/migrated_h4_expanded'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['proposed','dense','structured'],required=True)
 ap.add_argument('--seed',type=int,required=True);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--smoke',action='store_true');ap.add_argument('--quadratic-fix',action='store_true')
 ap.add_argument('--fp32',action='store_true',help='Disable autocast and GradScaler for stability diagnostics or training.')
 ap.add_argument('--preflight-per-re',action='store_true',help='Exercise every training Re and curriculum horizon without updating weights.')
 args=ap.parse_args()
 args.output.mkdir(parents=True,exist_ok=False)
 src=BASE/'code/train_h4_expanded.py'
 spec=importlib.util.spec_from_file_location('h4_control_native',src);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 original_loader=m.load_module
 def load(path):
  b=original_loader(path)
  if args.kind=='proposed':
   if args.quadratic_fix:
    from quadratic_init_fix import install
    install(b.v16,args.output)
   return b
  controls.TARGET=14165816
  controls.patch(b.v16,args.kind,args.output)
  # Native H builder freezes an inert group-router attribute; controls remove it.
  baseclass=b.v16.OperatorSpaceMoEROM
  class Compatible(baseclass):
   def __init__(self,*a,**kw):
    super().__init__(*a,**kw);self.group_router=nn.Identity()
  b.v16.OperatorSpaceMoEROM=Compatible
  if args.quadratic_fix:
   from quadratic_init_fix import install
   install(b.v16,args.output)
  # Single constant gate has no router regularization; empty expert stack has no diversity.
  b.v16.router_regularization=lambda gates:(gates[0].new_zeros(()),gates[0].new_zeros(()),{})
  b.v16.expert_diversity_loss=lambda stack:stack.new_zeros(())
  return b
 m.load_module=load
 cfg=json.loads((BASE/'runs/HopfExpanded34_H4_NormalFormRadial_r32/config.json').read_text())
 argv=[sys.argv[0],'--variant','h4','--experiment-name',cfg['experiment_name'],'--seed',str(args.seed),'--output-root',str(args.output),'--swanlab-mode','disabled']
 for key in ['baseline_trainer','coefficient_view','galerkin_path','pressure_path','asset_manifest','contract']:
  argv+=['--'+key.replace('_','-'),str(BASE/cfg[key])]
 for key in ['micro_batch','grad_accum','max_steps','lr','weight_decay','grad_clip','eval_every','long_eval_every','validation_windows_per_re','early_stop_patience_evals']:
  argv+=['--'+key.replace('_','-'),str(cfg[key])]
 if args.smoke:argv+=['--benchmark-steps','8','--benchmark-horizon','1']
 if args.kind=='dense' or args.fp32:argv+=['--no-amp']
 if args.preflight_per_re:argv+=['--preflight-per-re']
 (args.output/'control_protocol.json').write_text(json.dumps(dict(kind=args.kind,seed=args.seed,smoke=args.smoke,
  fp32=args.kind=='dense' or args.fp32,preflight_per_re=args.preflight_per_re,command=argv,
  matching='reference per-sample active parameters; no runtime equivalence',
  native_losses='all except inapplicable router/diversity penalties',
  initialization=('left quadratic factor zero; native random right factor preserved' if args.quadratic_fix else 'native physical-zero policy')),
  indent=2))
 sys.argv=argv
 m.main()
 (args.output/'CONTROL_COMPLETED.json').write_text(json.dumps(dict(completed=True,smoke=args.smoke)))

if __name__=='__main__':main()
