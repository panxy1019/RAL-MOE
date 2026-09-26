"""Export original VTK triangles and their explicit data association."""
import json
import sys
from pathlib import Path
import numpy as np
R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
sys.path.append(str(R/'.runtime/plot_env_v1/lib/python3.11/site-packages'))
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
from vtkmodules.vtkCommonDataModel import vtkPlane
from vtkmodules.vtkFiltersCore import vtkCutter, vtkTriangleFilter, vtkCellCenters
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
from vtkmodules.util.numpy_support import vtk_to_numpy,numpy_to_vtk
O=R/'experiments/field_redraw_20260922/data'
for key in ['circular_p','square_sh','square_hp','pinball_sh','pinball_hp']:
 meta=json.loads((O/f'{key}.json').read_text());z=np.load(O/f'{key}.npz')
 r=vtkUnstructuredGridReader();r.SetFileName(meta['mesh']['path']);r.Update();m=r.GetOutput();pts=vtk_to_numpy(m.GetPoints().GetData())
 if meta['data_association']=='cell':
  ids=numpy_to_vtk(np.arange(m.GetNumberOfCells()),deep=True);ids.SetName('source_cell');m.GetCellData().AddArray(ids)
  plane=vtkPlane();plane.SetNormal(0,0,1);plane.SetOrigin(0,0,(m.GetBounds()[4]+m.GetBounds()[5])/2)
  cut=vtkCutter();cut.SetInputData(m);cut.SetCutFunction(plane);cut.Update();t=vtkTriangleFilter();t.SetInputConnection(cut.GetOutputPort());t.Update();out=t.GetOutput()
  xy=vtk_to_numpy(out.GetPoints().GetData())[:,:2];faces=vtk_to_numpy(out.GetPolys().GetData()).reshape(-1,4);tri=faces[:,1:];value_ids=vtk_to_numpy(out.GetCellData().GetArray('source_cell')).astype('int64')
  assert np.unique(value_ids).size==len(z['areas'])==m.GetNumberOfCells()
  centers=vtkCellCenters();centers.SetInputData(m);centers.Update();cc=vtk_to_numpy(centers.GetOutput().GetPoints().GetData())
  if len(z['points']):
   delta=float(np.max(abs(cc-z['points'])));assert delta<1e-4;meta['mesh_alignment_max_abs']=delta
  else:
   # Verify against the original POD geometry (not merely matching cell counts).
   expert='steady' if key.endswith('sh') else 'periodic'
   pod=R.parent/f'Pinball/fluidicPinball_v2/rom_assets_v2/{expert}/pod/weighted_pod_velocity.npz'
   pz=np.load(pod);print('POD spatial keys',key,[(k,pz[k].shape) for k in pz.files if any(s in k for s in ['point','center','mesh','coord','weight'])])
   meshpath=R.parent/'Pinball/fluidicPinball_v2/data_npz/mesh/mesh_production.npz'
   mz=np.load(meshpath);print('MESH KEYS',[(k,mz[k].shape) for k in mz.files],flush=True)
   coords=mz['cell_centres']
   if coords is None:raise RuntimeError('Cannot verify Pinball spatial ordering')
   delta=float(np.max(abs(cc[:,:2]-coords[:,:2])));assert delta<1e-4
   meta['mesh_alignment_max_abs']=delta;meta['geometry_coordinate_source']=str(meshpath)
   raw=np.load(meta['reference_path']['path']);assert str(raw['mesh_hash'])==str(pz['mesh_hash'])
   assert str(raw['mesh_hash'])==str(mz['mesh_hash'])
   assert np.allclose(z['areas']/z['areas'].sum(),mz['cell_volumes']/mz['cell_volumes'].sum(),rtol=1e-5,atol=1e-10)
   meta['geometry_metadata']=str(mz['geometry_metadata'])
   meta['mesh_hash_contract']=str(raw['mesh_hash'])
 else:
  assert np.array_equal(pts,z['points'])
  surf=vtkDataSetSurfaceFilter();surf.SetInputData(m);surf.PassThroughPointIdsOn();surf.Update();t=vtkTriangleFilter();t.SetInputConnection(surf.GetOutputPort());t.Update();out=t.GetOutput()
  xyz=vtk_to_numpy(out.GetPoints().GetData());faces=vtk_to_numpy(out.GetPolys().GetData()).reshape(-1,4);tri=faces[:,1:]
  # One original extrusion plane. No averaging between planes, no interpolation across holes.
  bottom=np.all(abs(xyz[tri,2]-xyz[:,2].min())<1e-7,axis=1);tri=tri[bottom]
  original=vtk_to_numpy(out.GetPointData().GetArray('vtkOriginalPointIds'));used=np.unique(tri);mapping=np.full(len(xyz),-1);mapping[used]=np.arange(len(used));tri=mapping[tri];xy=xyz[used,:2];value_ids=original[used].astype('int64');meta['mesh_alignment_max_abs']=0.0
 assert np.isfinite(xy).all() and len(tri)>0
 np.savez_compressed(O/f'{key}_geometry.npz',xy=xy,triangles=tri,value_ids=value_ids)
 meta['plot_geometry']=f'{key}_geometry.npz';meta['plot_triangles']=len(tri);meta['plot_vertices']=len(xy)
 (O/f'{key}.json').write_text(json.dumps(meta,indent=2,allow_nan=False))
 print(key,len(xy),len(tri),'match',meta['mesh_alignment_max_abs'],flush=True)
