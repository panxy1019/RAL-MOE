"""Native Periodic training; stop after saving selected checkpoint, before test evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types
import torch

ROOT=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--seed',type=int,required=True)
 ap.add_argument('--smoke',action='store_true')
 ap.add_argument('--architecture',choices=['proposed','dense','structured'],default='proposed')
 ap.add_argument('--quadratic-fix',action='store_true')
 args=ap.parse_args()
 args.output.mkdir(parents=True,exist_ok=False)
 base=ROOT/'periodic_specialist_r32'
 source=base/'code/train_periodic_moe.py'
 text=source.read_text()
 anchor='    train_metrics = evaluate_model(\n'
 assert text.count(anchor)==1
 # Native train_one_split has already restored and saved validation-selected weights.
 # Stop before any posttraining train/heldout analyses. Training and selection unchanged.
 modified=text.replace(anchor,'    raise TrainingFinished()\n'+anchor)
 class TrainingFinished(Exception):pass
 module=types.ModuleType('periodic_round2_native')
 module.__file__=str(source);module.TrainingFinished=TrainingFinished
 sys.modules[module.__name__]=module
 exec(compile(modified,str(source),'exec'),module.__dict__)
 if args.architecture!='proposed':
  from periodic_controls import patch
  patch(module,args.architecture,args.output)
 if args.quadratic_fix:
  from quadratic_init_fix import install
  install(module,args.output)
 ck=torch.load(base/'checkpoint/FINAL_PERIODIC_SPECIALIST.pt',map_location='cpu',weights_only=False)
 saved=ck['args'];cfg=dict(vars(saved) if hasattr(saved,'__dict__') else saved)
 original=dict(cfg)
 cfg.update(data_root=base/'assets/Global_POD_AreaWeighted_L2',
   tensor_path=base/'assets/velocity_rom_periodic.npz',
   pressure_surrogate_path=base/'assets/pressure_poisson_surrogate_periodic.npz',
   output_dir=args.output,experiment_name='Circular_P_'+args.architecture+'_round2',seed=args.seed,
   resume_checkpoint=None,eval_only_checkpoint=None,swanlab_mode='disabled',
   swanlab_required=False,swanlab_log_dir=args.output/'swanlog')
 if args.smoke:cfg.update(epochs=1,min_epochs=1)
 if args.architecture!='proposed':
  for key in cfg:
   if key.startswith('lambda_') and any(t in key for t in ('gate','router','balance','entropy','diversity','group')):cfg[key]=0.
 (args.output/'protocol.json').write_text(json.dumps(dict(config=cfg,
   changes={k:dict(old=original.get(k),new=v) for k,v in cfg.items() if str(v)!=str(original.get(k))},
   original_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
   executed_source_sha256=hashlib.sha256(modified.encode()).hexdigest(),
   test_evaluation_disabled=True,smoke_only=args.smoke),indent=2,default=str))
 module.parse_args=lambda:argparse.Namespace(**cfg)
 try:module.main()
 except TrainingFinished:
  (args.output/'TRAINING_COMPLETED.json').write_text(json.dumps(dict(completed=True,smoke_only=args.smoke,heldout_evaluated=False)))
 else:raise RuntimeError('Did not reach the selected-checkpoint boundary')

if __name__=='__main__':main()
