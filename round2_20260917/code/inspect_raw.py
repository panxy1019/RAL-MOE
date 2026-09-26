from pathlib import Path
import numpy as np
R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
for path in [R/'Hopf/source_database/Re_47p081355_uvp_pointData.npz',R/'periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz']:
 print(path)
 z=np.load(path,allow_pickle=False)
 for k in z.files:
  a=z[k];print(k,a.shape,a.dtype)
