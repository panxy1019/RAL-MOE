"""Read-only inventory for ql-ROM adaptation; no model fitting."""
from pathlib import Path
import csv, json, numpy as np, torch
R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
B=R/'periodic_specialist_r32/assets'
for name in ['Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz',
             'Global_POD_AreaWeighted_L2/global_pressure_pod_area_weighted_l2.npz',
             'velocity_rom_periodic.npz','pressure_poisson_surrogate_periodic.npz']:
    z=np.load(B/name,allow_pickle=False)
    print('FILE',name,'bytes',(B/name).stat().st_size)
    for key in z.files:
        a=z[key]
        print(key,a.shape,str(a.dtype),str(a.tolist())[:180] if a.size<15 else '')
idx=list(csv.DictReader((B/'provenance/projection_snapshots_velocity_periodic.csv').open()))
print('INDEX_FIRST',idx[0])
for split in sorted(set(r['split'] for r in idx)):
    rows=[r for r in idx if r['split']==split]
    print('SPLIT',split,len(rows),sorted(set(r['Re'] for r in rows)))
for k in ['proposed','dense','structured']:
    p=R/f'experiments/round2_20260917/P_evaluation_v2/{k}/1248'
    print('EVAL',k,(p/'protocol.json').read_text())
    z=torch.load(p/'modal_rollouts.pt',weights_only=False)
    print('ROLLOUT',len(z),list(z[0]),{key:str(val)[:180] for key,val in z[0].items() if key!='values'})
print('PROVENANCE_FILES',[p.name for p in (B/'provenance').iterdir()])
