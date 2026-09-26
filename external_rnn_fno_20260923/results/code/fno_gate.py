"""Validation-only FV/Cartesian roundtrip gate, before any FNO fitting."""
from common import *
from scipy.spatial import Delaunay,cKDTree
from scipy.sparse import csr_matrix

def mapping(points,nx,ny):
 xy=points[:,:2].astype(float);unique,inverse,counts=np.unique(xy,axis=0,return_inverse=True,return_counts=True)
 averaging=csr_matrix((1/counts[inverse],(inverse,np.arange(len(xy)))),shape=(len(unique),len(xy)))
 x=np.linspace(xy[:,0].min(),xy[:,0].max(),nx);y=np.linspace(xy[:,1].min(),xy[:,1].max(),ny)
 X,Y=np.meshgrid(x,y);grid=np.c_[X.ravel(),Y.ravel()];mask=(grid**2).sum(1)>=.5**2
 tri=Delaunay(unique);simp=tri.find_simplex(grid);inside=simp>=0;safe=np.maximum(simp,0)
 delta=grid-tri.transform[safe,2];bc=np.einsum('nij,nj->ni',tri.transform[safe,:2],delta);weights=np.c_[bc,1-bc.sum(1)]
 ids=tri.simplices[safe];nearest=cKDTree(unique).query(grid[~inside])[1]
 ids[~inside]=nearest[:,None];weights[~inside]=[1,0,0];weights[~mask]=0
 fwd=csr_matrix((weights.ravel(),(np.repeat(np.arange(len(grid)),3),ids.ravel())),shape=(len(grid),len(unique)))@averaging
 ix=np.clip(np.searchsorted(x,xy[:,0])-1,0,nx-2);iy=np.clip(np.searchsorted(y,xy[:,1])-1,0,ny-2)
 tx=(xy[:,0]-x[ix])/(x[ix+1]-x[ix]);ty=(xy[:,1]-y[iy])/(y[iy+1]-y[iy])
 backids=np.stack([iy*nx+ix,iy*nx+ix+1,(iy+1)*nx+ix,(iy+1)*nx+ix+1],1)
 bw=np.stack([(1-tx)*(1-ty),tx*(1-ty),(1-tx)*ty,tx*ty],1);bw*=mask[backids]
 missing=bw.sum(1)<=0
 if missing.any():
  fluid=np.flatnonzero(mask);near=fluid[cKDTree(grid[mask]).query(xy[missing])[1]];backids[missing]=near[:,None];bw[missing]=[1,0,0,0]
 bw/=bw.sum(1)[:,None]
 back=csr_matrix((bw.ravel(),(np.repeat(np.arange(len(xy)),4),backids.ravel())),shape=(len(xy),len(grid)))
 assert np.max(abs(np.asarray(back.sum(1)).ravel()-1))<1e-12
 assert np.max(abs(np.asarray(fwd.sum(1)).ravel()[mask]-1))<1e-12
 return fwd,back,mask,x,y,dict(unique_xy=len(unique),points=len(xy),duplicate_xy_reduction='equal average along extrusion',outside_hull_grid_points=int((~inside&mask).sum()),masked_inverse_fallback_points=int(missing.sum()))
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--regime',choices=['H','P'],required=True);regimes=[ap.parse_args().regime]
 out=OUT/'fno_gate'/regimes[0];out.mkdir(parents=True,exist_ok=True);manifest=json.loads((OUT/'split_manifest.json').read_text());protocol=json.loads((OUT/'protocol_frozen.json').read_text())
 scores=json.loads((OLD/'seed_summary_current.json').read_text());threshold={}
 for reg in regimes:
  okay=[x for x in scores if x['regime']==reg and x['K']==48 and x['evaluated_seeds']==x['expected_seeds']==3 and x['Ejoint_percent_window_mean_mean'] is not None]
  best=min(okay,key=lambda x:x['Ejoint_percent_window_mean_mean'])
  threshold[reg]=dict(Eu=.2*best['Eu_percent_window_mean_mean'],Ep=.2*best['Ep_percent_window_mean_mean'],reference_model=best['method'],source_sha256=sha(OLD/'seed_summary_current.json'))
 write(out/'gate_protocol.json',dict(threshold=threshold,frames_per_validation_trajectory=17,seed=None,grids=protocol['fno']['grids'],
   geometry='Delaunay barycentric FV->grid; masked bilinear grid->FV; nearest geometric fallback only when no valid stencil. Cylinder radius0.5 centered0,0; 2D duplicate coordinates averaged.',
   rule='Every validation Re mean Eu and Ep must pass; no test frames used; largest registered grid is cap, not proof no other method can pass',
   training='No training or new test evaluation until geometry gate passes. Memory smoke and visual QA additionally required for passing grid.'))
 samples={}
 for reg in regimes:
  for tr in manifest[reg]['trajectories']:
   if tr['split']!='validation':continue
   recovered=OUT/'raw_validation'/f"{tr['label']}_uvp_pointData.npz"
   path=recovered if recovered.exists() else Path(tr['raw_path'])
   z=np.load(path);ids=np.unique(np.linspace(0,len(z['times'])-1,17,dtype=int));u=z['u'][ids].astype(float);v=z['v'][ids].astype(float);p=z['p'][ids].astype(float)
   geom=np.load(OUT/f'{reg}_geometry.npz');w=geom['areas'];assert np.array_equal(z['points'],geom['points'])
   p-=(p@w/w.sum())[:,None]
   samples[reg,tr['label']]=dict(field=np.stack([u,v,p],2),w=w,path=str(path),sha256=sha(path),frames=ids.tolist(),points=z['points'])
 write(out/'raw_validation_sources.json',[dict(regime=k[0],label=k[1],path=v['path'],sha256=v['sha256'],frames=v['frames']) for k,v in samples.items()])
 points=next(iter(samples.values()))['points'];rows=[];decisions={};mapdir=out/'maps';mapdir.mkdir(exist_ok=True)
 for nx,ny in protocol['fno']['grids']:
  started=time.perf_counter();fwd,back,mask,x,y,meta=mapping(points,nx,ny)
  from scipy.sparse import save_npz
  save_npz(mapdir/f'fv_to_grid_{nx}x{ny}.npz',fwd);save_npz(mapdir/f'grid_to_fv_{nx}x{ny}.npz',back)
  np.savez_compressed(mapdir/f'grid_{nx}x{ny}.npz',x=x,y=y,mask=mask.reshape(ny,nx))
  for (reg,label),sample in samples.items():
   field=sample['field'];w=sample['w'];es=[]
   for fi,z in enumerate(field):
    grid=fwd@z;restored=back@grid;restored[:,2]-=restored[:,2]@w/w.sum();delta=restored-z
    eu=100*np.sqrt((w[:,None]*delta[:,:2]**2).sum()/(w[:,None]*z[:,:2]**2).sum());ep=100*np.sqrt((w*delta[:,2]**2).sum()/(w*z[:,2]**2).sum());es.append((eu,ep))
    if fi==len(field)//2 and label==sorted(k[1] for k in samples if k[0]==reg)[0]:
     np.savez_compressed(out/f'visual_{nx}x{ny}.npz',grid=grid.reshape(ny,nx,3),original=z,restored=restored,points=points,x=x,y=y,mask=mask.reshape(ny,nx))
   eu,ep=np.mean(es,0);rows.append(dict(regime=reg,label=label,nx=nx,ny=ny,Eu=float(eu),Ep=float(ep),Eu_max=max(e[0] for e in es),Ep_max=max(e[1] for e in es),
    Eu_threshold=threshold[reg]['Eu'],Ep_threshold=threshold[reg]['Ep'],passed=bool(eu<=threshold[reg]['Eu'] and ep<=threshold[reg]['Ep']),frames=len(es)))
  csvwrite(out/'interpolation_by_Re.csv',rows);write(out/f'mapping_{nx}x{ny}.json',dict(**meta,seconds=time.perf_counter()-started,forward_sha256=sha(mapdir/f'fv_to_grid_{nx}x{ny}.npz'),backward_sha256=sha(mapdir/f'grid_to_fv_{nx}x{ny}.npz')))
  for reg in regimes:
   rr=[r for r in rows if r['regime']==reg and r['nx']==nx];passed=all(r['passed'] for r in rr)
   if reg not in decisions and passed:decisions[reg]=dict(status='interpolation_passed_memory_visual_pending',nx=nx,ny=ny)
  print('GRID_COMPLETE',nx,ny,[r for r in rows if r['nx']==nx],flush=True)
 for reg in regimes:
  if reg not in decisions:decisions[reg]=dict(status='STOP_GEOMETRY_DISCRETIZATION_MISMATCH',reason='No registered uniform grid passed both fixed roundtrip error thresholds on all validation Re; do not train/report confounded FNO scores.')
 write(out/'decision.json',dict(decisions=decisions,threshold=threshold,all_candidates_evaluated=True));print('FNO_GATE',decisions,flush=True)
if __name__=='__main__':main()
