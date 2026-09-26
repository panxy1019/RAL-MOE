"""Check selected checkpoints, low-rank branch activity and seed acceptance."""
import json
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1]
rows=[]
for reg in ['H','P']:
 for method in ['proposed','dense','structured']:
  for seed in [1248,1600,2026]:
   tag='dense_fp32' if reg=='H' and method=='dense' else method
   folder=R/f'{reg}_{tag}'/str(seed)
   if reg=='H':folder=folder/'HopfExpanded34_H4_NormalFormRadial_r32'
   path=folder/'best_validation.pt'
   row=dict(regime=reg,method=method,seed=seed,accepted_checkpoint=path.exists())
   if path.exists():
    ck=torch.load(path,map_location='cpu',weights_only=False)
    sd=ck['model_state'];q={k:v for k,v in sd.items() if 'quad_left' in k or 'quad_right' in k}
    row.update(selected_step=ck.get('optimizer_step',ck.get('epoch')),quadratic_tensors=len(q),
      quadratic_nonzero_tensors=sum(bool(torch.count_nonzero(v)) for v in q.values()),
      quadratic_max_abs=max([float(v.abs().max()) for v in q.values()] or [0.]))
   elif (folder/'final_training.pt').exists():
    ck=torch.load(folder/'final_training.pt',map_location='cpu',weights_only=False)
    row.update(completed_training=True,best_step=ck.get('best_step'),validation_rejected=ck.get('best_step',0)<0)
   else:row['state']='pending_or_failed_see_campaign_status'
   rows.append(row)
(R/'checkpoint_branch_audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
