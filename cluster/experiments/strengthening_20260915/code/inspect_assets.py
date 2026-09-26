from pathlib import Path
import numpy as np
from dense_periodic import ROOT
import torch
ck=torch.load(ROOT/'paper_experiments/revisions/revision5_supplemental_evaluation_20260725/recovery/global_k56_valid_history_windows_v5/V16_1_SteadyPressureAnchor32_ru32_rp32_Re_24p630436_checkpoint.pt',map_location='cpu',weights_only=False)
print('checkpoint keys',list(ck))
for k,v in ck.items():
    if isinstance(v,dict): print(k,list(v)[:35])
paths = [ROOT/'paper_experiments/visualization/cache/periodic_proposed/proposed_rollouts.npz']
paths += sorted((ROOT/'paper_experiments/revisions/revision10_missing_metric_completion_20260725/evaluations/global_periodic_k48_arrays_retry').glob('*rollout_arrays.npz'))[:1]
for p in paths:
    z=np.load(p,allow_pickle=False)
    print(str(p))
    print({k:z[k].shape for k in z.files})
    for k in z.files:
        if 'time' in k.lower() or k.lower()=='re':
            print(k,z[k].reshape(-1)[:15])
    print('first future time per window', z['times'][:,0])
for p in (ROOT/'V16_1_SteadyPressureAnchor32/assets/common_global_data/Global_POD_AreaWeighted_L2').glob('*.npz'):
    z=np.load(p)
    print(p.name, {k:z[k].shape for k in z.files if any(s in k for s in ('mean','phi','area'))})
    print('keys',z.files)
