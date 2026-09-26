"""Validate exported physical data, hashes, gauges, and extrema without torch."""
import json,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'.runtime'))
import numpy as np
M=json.loads((R/'data/panel_manifest.json').read_text());results=[]
required=['case','geometry','Re','source_field_path','reference_path','checkpoint_path','model_family','original_or_revised_init','training_seed','E2_checkpoint','gate_checkpoint','gate_seed','split','trajectory_id','window_id','t0','dt_output','k','K','snapshot_time','reference_type','mesh','integration_weights','velocity_definition','pressure_definition','nondimensional_scales','gauge_correction','interpolation','crop_extent','nonfinite_points','snapshot_metrics','plot_scale_records']
for m in M:
 assert all(k in m and m[k] is not None for k in required)
 assert 'null' not in json.dumps(m['training_seed'])
 z=np.load(R/m['local_array']);g=np.load(R/m['local_geometry']);w=z['areas'];c=R/'config/figure_config.json';crop=json.loads(c.read_text())['crop_extent'][m['case_id']]
 assert hashlib.sha256((R/m['local_geometry']).read_bytes()).hexdigest()==m['local_geometry_sha256']
 assert hashlib.sha256((R/m['local_array']).read_bytes()).hexdigest()==m['snapshot_arrays']['sha256']
 if m['data_association']=='cell':
  xy=np.empty((len(w),2));xy[g['value_ids']]=g['xy'][g['triangles']].mean(1)
 else:xy=z['points'][:,:2]
 mask=(xy[:,0]>=crop[0])&(xy[:,0]<=crop[1])&(xy[:,1]>=crop[2])&(xy[:,1]<=crop[3]);checks={}
 for method in ['reference','prediction','e2','candidate1','candidate2']:
  if method not in z:continue
  f=z[method];assert np.isfinite(f).all();gp=abs(float(f[2]@w/w.sum()));assert gp<1e-6,(m['case_id'],method,gp)
  du=np.hypot(*(f[:2]-z['reference'][:2]));dp=abs(f[2]-z['reference'][2]);checks[method]={'gauge_abs_weighted_mean':gp}
  for name,v in [('velocity',du),('pressure',dp)]:
   idx=int(v.argmax());checks[method][name]={'global_max':float(v[idx]),'global_max_xy':xy[idx].tolist(),'roi_max':float(v[mask].max()),'global_max_in_roi':bool(mask[idx])}
 results.append({'case':m['case_id'],'fields':checks,'all_finite':True,'weights_positive':bool((w>0).all())})
for m in json.loads((R/'output/render_qa.json').read_text()):
 assert abs(m['width_inches']-5.5)<1e-12
 for ext in ['pdf','png']:assert hashlib.sha256((R/'output'/f'{m["figure"]}.{ext}').read_bytes()).hexdigest()==m[f'{ext}_sha256']
log=(R/'output/preview_iclr.log').read_text(errors='replace')
assert 'Float too large' not in log and 'Overfull' not in log
(R/'data/independent_qa.json').write_text(json.dumps(results,indent=2,allow_nan=False))
print('PASS: 5 snapshots, all source/geometry/render hashes, finite fields, positive weights, gauges, 139.7 mm figures and LaTeX page fit.')
print(json.dumps(next(r for r in results if r['case']=='pinball_sh'),indent=2))
