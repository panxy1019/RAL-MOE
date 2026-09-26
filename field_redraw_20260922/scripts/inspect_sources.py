from pathlib import Path
import numpy as np
import torch
R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
P=R.parent/'Pinball'
for p in [R/'periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz',R/'periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/global_pressure_pod_area_weighted_l2.npz',P/'fluidicPinball_v2/eval_pod_coefficients_v2/periodic/final_test']:
    if p.is_dir():
        print('FILES',p,list(p.glob('*'))[:10]);continue
    z=np.load(p);print('NPZ',p,[(k,z[k].shape,str(z[k].dtype)) for k in z.files])
    for k in ['points','coordinates','point_areas']:
        if k in z:print(k,z[k].min(axis=0),z[k].max(axis=0))
for p in [R/'experiments/round2_20260917/P_evaluation_v2/proposed/1248/modal_rollouts.pt',R/'experiments/round2_20260917/P_proposed/1248/Circular_P_proposed_round2_Re_70p314635_checkpoint.pt']:
    z=torch.load(p,map_location='cpu',weights_only=False);print('PT',p,type(z))
    if isinstance(z,list):print('COUNT',len(z),'FIRST', {k:v for k,v in z[0].items() if k!='values'},'values',len(z[0]['values']),[(np.shape(x)) for x in z[0]['values'][0]])
    else: print('KEYS',list(z))
for p in [R/'centeredsquare_fusion_runs/E2_T2C_K24_20260730_STRICT_V3/gate_hp/best.pt',P/'fluidic_pinball_fusion_v1/runs/E2_T2C_K24_20260806_STRICT_V1/gate_hp/best.pt']:
    z=torch.load(p,map_location='cpu',weights_only=False);print('GATE',p, {k:v for k,v in z.items() if k not in ['gate_state','optimizer_state','feature_mean','feature_std']})
