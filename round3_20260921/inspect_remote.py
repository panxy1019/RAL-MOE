from pathlib import Path
import numpy as np
R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
run=R/'centeredsquare_fusion_runs/E2_T2C_K24_20260730_STRICT_V3'
with (run/'paper_boundary_figures_20260730_V1/inputs/centeredSquare_CN09_graded_Re100_internal_final_reference.vtk').open('rb') as f:
 for _ in range(4):f.readline()
 n=int(f.readline().split()[1]);points=np.frombuffer(f.read(n*12),dtype='>f4').reshape(n,3)
 print('mesh bounds',points.min(0),points.max(0))
for pair in ['sh','hp']:
 z=np.load(run/f'heldout_evaluation_20260730_V1/cache_{pair}/{pair}_heldout_cache.npz')
 print(pair,[(k,z[k].shape,str(z[k].dtype)) for k in z.files]);print(z['timestamps'][0],z['re'],z['split'])
z=np.load(R/'periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz')
print('circular keys',[(k,z[k].shape) for k in z.files])
