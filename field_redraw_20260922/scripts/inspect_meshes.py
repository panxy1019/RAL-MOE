import sys
from pathlib import Path
R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
sys.path.append(str(R/'.runtime/plot_env_v1/lib/python3.11/site-packages'))
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
from vtkmodules.util.numpy_support import vtk_to_numpy
import numpy as np
paths=[R/'steady_specialist_v1/expanded_validation_20260723/visualization/Re_43p500000_last_snapshot_uvp.vtk',R/'centeredsquare_fusion_runs/E2_T2C_K24_20260730_STRICT_V3/paper_boundary_figures_20260730_V1/inputs/centeredSquare_CN09_graded_Re100_internal_final_reference.vtk',R.parent/'Pinball/fluidicPinball_v2/rom_assets_v2/reference_vtk/fluidicPinball_production_reference.vtk']
for p in paths:
 r=vtkUnstructuredGridReader();r.SetFileName(str(p));r.Update();m=r.GetOutput();points=vtk_to_numpy(m.GetPoints().GetData());print(str(p),m.GetNumberOfPoints(),m.GetNumberOfCells(),m.GetBounds())
 if 'Re_43' in p.name:
  z=np.load(R/'periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz');print('POINT_MATCH',np.max(abs(points-z['points'])))
 else:
  print('POINTS',points[:3])
