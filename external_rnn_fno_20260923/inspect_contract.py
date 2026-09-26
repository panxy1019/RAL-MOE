import numpy as np,torch,json,csv
from pathlib import Path
R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
for reg,p in [('P',R/'experiments/round2_20260917/P_proposed/1248/Circular_P_proposed_round2_Re_70p314635_checkpoint.pt'),('H',R/'experiments/round2_20260917/H_proposed/1248/HopfExpanded34_H4_NormalFormRadial_r32/best_validation.pt')]:
 c=torch.load(p,map_location='cpu',weights_only=False);print(reg,list(c))
 for key in ['scalers','norm_stats','stats']:
  if key in c:
   print(key,{k:({kk:np.shape(vv) for kk,vv in v.items()} if isinstance(v,dict) else np.shape(v)) for k,v in c[key].items()})
z=np.load(R/'Hopf/migrated_h4_expanded/assets_r32/expanded_h4_trainval_r32.npz');u=np.load(R/'Hopf/artifacts/hopf/velocity_pod_hopf.npz')
print('expanded vs native',np.max(abs(z['coeff_uv']-u['coeff_uv'][z['source_row'],:32])),np.max(abs(z['phi_uv']-u['phi_uv'][:32])))
print('domain',u['points'].min(0),u['points'].max(0))
for name in ['H','P']:
 path=R/('Hopf/artifacts/hopf/projection_snapshots_velocity_hopf.csv' if name=='H' else 'periodic_specialist_r32/assets/provenance/projection_snapshots_velocity_periodic.csv')
 rows=list(csv.DictReader(path.open()));print(name,'validation',sorted(set(r['Re_label'] for r in rows if r['split']=='validation')))
