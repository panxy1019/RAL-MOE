import json
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1]
records=[]
for reg in ['H','P']:
 for kind in ['proposed','structured']:
  folder=R/'quadratic_fixed'/reg/kind/'smoke'
  if reg=='H':path=folder/'HopfExpanded34_H4_NormalFormRadial_r32/final_training.pt'
  else:path=folder/'final_training.pt'
  if not path.exists():continue
  ck=torch.load(path,map_location='cpu',weights_only=False);state=ck['model_state']
  left=[v for k,v in state.items() if k.endswith('quad_left')]
  right=[v for k,v in state.items() if k.endswith('quad_right')]
  active=sum(int(torch.count_nonzero(v)>0) for v in left)
  assert active>0,'No quadratic left factor changed after optimization'
  records.append(dict(regime=reg,kind=kind,quadratic_pairs=len(left),nonzero_left_factors=active,
    largest_left_magnitude=max(float(v.abs().max()) for v in left),largest_right_magnitude=max(float(v.abs().max()) for v in right)))
(R/'fixed_smoke_branch_audit.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
