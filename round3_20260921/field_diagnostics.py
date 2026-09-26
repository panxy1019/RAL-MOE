"""Offline common-grid diagnostics, NOT original finite-volume residuals."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.signal import detrend, hilbert
from types import SimpleNamespace
R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
P=R/'paper_consistency_20260920'
O=R/'experiments/round3_20260921/field_diagnostics';O.mkdir(parents=True,exist_ok=True)
def load_basis(spec):
 v=np.load(spec['velocity']);p=np.load(spec['pressure']);area=v['point_areas'].astype(float)
 phip=p['phi_p'][:spec['r_p']].astype(float);mp=p[spec.get('mean_p_key','mean_p_regime')].astype(float)
 phip-=(phip@area/area.sum())[:,None];mp-=mp@area/area.sum()
 return SimpleNamespace(phi_u=v['phi_uv'][:spec['r_u']].astype(float),phi_p=phip,mean_u=v[spec.get('mean_u_key','mean_uv_regime')].astype(float),mean_p=mp,areas=area,velocity_path=Path(spec['velocity']),pressure_path=Path(spec['pressure']))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,data): (O/name).write_text(json.dumps(data,indent=2,allow_nan=False))
def csvout(name,rows):
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def centers(path):
 with Path(path).open('rb') as f:
  for _ in range(4):f.readline()
  n=int(f.readline().split()[1]);pts=np.frombuffer(f.read(n*12),dtype='>f4').reshape(n,3).astype(float)
  line=f.readline()
  while not line.strip():line=f.readline()
  _,nc,sz=line.split();raw=np.frombuffer(f.read(int(sz)*4),dtype='>i4');cs=[];j=0
  for _ in range(int(nc)):
   k=int(raw[j]);cs.append(pts[raw[j+1:j+1+k]].mean(0));j+=k+1
 return np.array(cs)[:,:2]
def operators(xy,k):
 dist,ix=cKDTree(xy).query(xy,k=384);rr=[];cc=[];vals=[[],[],[]];used=[]
 for i in range(len(xy)):
  for kk in range(k,385,k):
   ids=ix[i,:kk];d=xy[ids]-xy[i];s=np.maximum(np.max(abs(d),axis=0),1e-12);x,y=(d/s).T
   M=np.column_stack([np.ones(kk),x,y,x*x,x*y,y*y])
   if np.linalg.matrix_rank(M)==6 and np.linalg.cond(M)<1e6:break
  assert np.linalg.matrix_rank(M)==6,(i,kk)
  inv=np.linalg.pinv(M);used.append(kk)
  rr.extend([i]*kk);cc.extend(ids);vals[0].extend(inv[1]/s[0]);vals[1].extend(inv[2]/s[1]);vals[2].extend(2*(inv[3]/s[0]**2+inv[5]/s[1]**2))
 mats=[csr_matrix((v,(rr,cc)),shape=(len(xy),len(xy))) for v in vals]
 x,y=xy.T;f=1+2*x-3*y+4*x*x+5*x*y+6*y*y
 err=[float(np.max(abs(m@f-t))) for m,t in zip(mats,[2+8*x+5*y,-3+5*x+12*y,np.full(len(xy),20.)])]
 assert max(err)<1e-6,err
 return mats,{'errors':err,'adaptive_neighbor_min':min(used),'adaptive_neighbor_max':max(used),'expanded_points':sum(v>k for v in used)}
def reconstruct(z,a,b,basis):
 uv=np.asarray(z[a],float)@basis.phi_u+basis.mean_u
 pp=np.asarray(z[b],float)@basis.phi_p+basis.mean_p
 pp-=((pp@basis.areas)/basis.areas.sum())[:,None]
 return np.stack([uv[:,0::2],uv[:,1::2],pp],axis=1)
def diagnostic(f,t,re,ops):
 dx,dy,lap=ops;u,v,p=f[:,0],f[:,1],f[:,2]
 d=lambda m,x:(m@x.T).T
 conv=np.stack([u*d(dx,u)+v*d(dy,u),u*d(dx,v)+v*d(dy,v)],axis=1)
 mom=np.gradient(f[:,:2],t,axis=0,edge_order=2)+conv+np.stack([d(dx,p),d(dy,p)],1)-np.stack([d(lap,u),d(lap,v)],1)/re
 ppe=d(lap,p)+d(dx,conv[:,0])+d(dy,conv[:,1])
 div=d(dx,u)+d(dy,v)
 return mom,ppe,div,conv
vtk=R/'centeredsquare_fusion_runs/E2_T2C_K24_20260730_STRICT_V3/paper_boundary_figures_20260730_V1/inputs/centeredSquare_CN09_graded_Re100_internal_final_reference.vtk'
xy=centers(vtk);mask=(xy[:,0]>=6)&(xy[:,0]<=16)&(xy[:,1]>=1)&(xy[:,1]<=3)
probe=int(np.argmin(np.sum((xy-[8,2.5])**2,axis=1)))
audit={'scope':'POD-reconstructed fields, offline LS differential operators, not OpenFOAM FV residuals','roi':[6,16,1,3],'probe_target':[8,2.5],'probe_actual':xy[probe].tolist(),'mesh_sha256':sha(vtk),'operators':{},'sources':{}}
rows=[];phase_rows=[];checks=[]
for kn in [24,36]:
 ops,err=operators(xy,kn);audit['operators'][str(kn)]={'manufactured_polynomial_max_abs_error':err,'points':len(xy),'roi_points':int(mask.sum())}
 for pair in ['sh','hp']:
  cfg=json.loads((P/f'config_{pair}_boundary_figure.json').read_text());z=np.load(cfg['bundle']);bases={k:load_basis(v) for k,v in cfg['bases'].items()};basis=bases[cfg['truth_basis']];area=basis.areas
  assert len(area)==len(xy) and len(z['re'])==16 and np.all(z['split']=='heldout')
  zpos=np.load(cfg['bases'][cfg['truth_basis']]['velocity'])
  if 'points' in zpos:
   delta=float(np.max(np.abs(zpos['points'][:,:2]-xy)));assert delta<1e-4; audit['operators'][str(kn)]['pod_mesh_max_delta']=delta
  weights=list(csv.DictReader(open(cfg['weights_csv'])));wm={(round(float(v['Re']),6),int(v['start'])):float(v[cfg['weight_alpha_column']]) for v in weights}
  audit['sources'][pair]={'config':cfg,'bundle_sha256':sha(cfg['bundle']),'weights_sha256':sha(cfg['weights_csv']),'basis_hashes':{str(v):sha(v) for b in bases.values() for v in [b.velocity_path,b.pressure_path]}}
  def norm(x):
   q=x*x if x.ndim==2 else np.sum(x*x,axis=1)
   return np.sqrt(np.sum(q[:,mask]*area[mask],axis=-1)/area[mask].sum())
  mcfg=cfg['methods']['t2c']
  for w in range(16):
   zz={k:z[k][w] for k in ['truth_a','truth_b','candidate_1_a','candidate_1_b','candidate_2_a','candidate_2_b']};t=z['timestamps'][w].astype(float);re=float(z['re'][w]);start=int(z['starts'][w]);alpha=wm[(round(re,6),start)]
   fs={'reference':reconstruct(zz,'truth_a','truth_b',basis),'candidate1':reconstruct(zz,'candidate_1_a','candidate_1_b',bases[mcfg['candidate_1_basis']]),'candidate2':reconstruct(zz,'candidate_2_a','candidate_2_b',bases[mcfg['candidate_2_basis']])}
   fs['fusion']=alpha*fs['candidate1']+(1-alpha)*fs['candidate2'];ds={m:diagnostic(f,t,re,ops) for m,f in fs.items()}
   for name,weight in [('candidate1',1.),('candidate2',0.),('fusion',alpha)]:
    delta=fs[name]-fs['reference']
    for channel,inds in [('u',slice(0,2)),('p',slice(2,3))]:
     num=np.sum(np.sum(delta[:,inds]**2,axis=1)*area,axis=1)
     den=np.sum(np.sum(fs['reference'][:,inds]**2,axis=1)*area,axis=1)
     q=z['quad_'+channel][w];cache=(weight**2*q[:,0]+(1-weight)**2*q[:,1]+2*weight*(1-weight)*q[:,2])/q[:,3]
     assert np.max(abs(num/den-cache))<1e-8,(pair,w,name,channel,float(np.max(abs(num/den-cache))))
   energy={m:.5*np.sum(np.sum(f[:,:2]**2,axis=1)*area,axis=1) for m,f in fs.items()}
   deficit=alpha*energy['candidate1']+(1-alpha)*energy['candidate2']-energy['fusion']
   du=fs['candidate1']-fs['candidate2'];expected=.5*alpha*(1-alpha)*np.sum(np.sum(du[:,:2]**2,axis=1)*area,axis=1)
   dc=diagnostic(du,t,re,ops)[3];cross=-alpha*(1-alpha)*dc
   excess=ds['fusion'][0]-alpha*ds['candidate1'][0]-(1-alpha)*ds['candidate2'][0]
   eppe=ds['fusion'][1]-alpha*ds['candidate1'][1]-(1-alpha)*ds['candidate2'][1]
   crossppe=(ops[0]@cross[:,0].T+ops[1]@cross[:,1].T).T
   checks.append({'neighbors':kn,'pair':pair,'window':w,'energy_identity_max_abs':float(np.max(abs(deficit-expected))),'momentum_identity_max_abs':float(np.max(abs(excess-cross))),'ppe_identity_max_abs':float(np.max(abs(eppe-crossppe)))})
   for method,f in fs.items():
    mom,ppe,div,_=ds[method]
    for k in range(24):rows.append(dict(neighbors=kn,pair=pair,window=w,Re=re,start=start,step=k+1,time=float(t[k]),lead_time=float((k+1)*(t[1]-t[0])),method=method,alpha=alpha,energy=float(energy[method][k]),energy_relative_error=float(abs(energy[method][k]-energy['reference'][k])/energy['reference'][k]),deficit_over_reference=float(deficit[k]/energy['reference'][k]),momentum_rms=float(norm(mom)[k]),ppe_rms=float(norm(ppe)[k]),divergence_rms=float(norm(div)[k]),momentum_excess_rms=float(norm(excess)[k]),ppe_excess_rms=float(norm(eppe)[k]),probe_v=float(f[k,1,probe]),time_derivative_interior=bool(0<k<23)))
   if kn==24:
    sig={m:detrend(f[:,1,probe]) for m,f in fs.items()};freq=np.fft.rfftfreq(24,d=t[1]-t[0]);power=abs(np.fft.rfft(sig['reference']))**2;j=int(np.argmax(power[1:])+1);cycles=float(freq[j]*(t[-1]-t[0]));amplitude=float(np.std(sig['reference']));valid=cycles>=2 and amplitude>1e-5 and j<12
    phases={m:np.angle(hilbert(s)) for m,s in sig.items()}
    for m in fs:
     for k in range(24):phase_rows.append(dict(pair=pair,window=w,Re=re,step=k+1,lead_time=float((k+1)*(t[1]-t[0])),method=m,reference_cycles=cycles,reference_std=amplitude,phase_resolved=bool(valid),phase_difference_rad=float(np.angle(np.exp(1j*(phases[m][k]-phases['reference'][k])))),probe_v=float(fs[m][k,1,probe])))
  print('done',kn,pair,flush=True)
csvout('field_curves.csv',rows);csvout('phase_curves.csv',phase_rows);csvout('identity_checks.csv',checks);dump('provenance.json',audit)
assert max(v['energy_identity_max_abs'] for v in checks)<1e-8
assert max(v['momentum_identity_max_abs'] for v in checks)<1e-8
assert max(v['ppe_identity_max_abs'] for v in checks)<1e-7
# Fixed-window, all-case diagnostic plots. No outcome-based frame selection.
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
except ImportError:
 plt=None
palette={'reference':'#333333','candidate1':'#3366AA','candidate2':'#BB7722','fusion':'#9A4488'};styles={'reference':'--','candidate1':':','candidate2':'-.','fusion':'-'}
summary=[]
for pair in ['sh','hp']:
 if plt:fig,axs=plt.subplots(2,3,figsize=(12,6.4),layout='constrained')
 for method in palette:
  rr=[r for r in rows if r['neighbors']==24 and r['pair']==pair and r['method']==method]
  for ax,key,label in zip(axs.flat[:5] if plt else [],['energy_relative_error','momentum_rms','ppe_rms','divergence_rms','probe_v'],['Energy relative error','Interior momentum RMS','Interior PPE RMS','Interior divergence RMS','Probe transverse velocity']):
   use=rr if key!='probe_v' else [r for r in rr if r['window']==0]
   steps=range(2,24) if 'rms' in key else range(1,25)
   vals=[float(np.mean([r[key] for r in use if r['step']==k])) for k in steps]
   ax.plot(np.array(list(steps))*4,vals,label=method,color=palette[method],ls=styles[method]);ax.set(xlabel='Prediction time',ylabel=label);ax.grid(alpha=.2)
  for kn in [24,36]:
   subset=[r for r in rows if r['neighbors']==kn and r['pair']==pair and r['method']==method and r['time_derivative_interior']]
   summary.append(dict(pair=pair,neighbors=kn,method=method,**{k:float(np.mean([r[k] for r in subset])) for k in ['energy_relative_error','momentum_rms','ppe_rms','divergence_rms','deficit_over_reference']}))
 if not plt:continue
 pr=[r for r in phase_rows if r['pair']==pair and r['window']==0 and r['method']=='fusion']
 axs[1,2].plot([r['lead_time'] for r in pr],[r['phase_difference_rad'] for r in pr],color=palette['fusion']);axs[1,2].set(xlabel='Prediction time',ylabel='Fusion phase difference (rad)',title='Window 0; '+('resolved' if pr[0]['phase_resolved'] else 'phase unresolved'))
 axs[0,0].legend(fontsize=8);axs[1,1].set_title('Window 0, geometric probe (8, 2.5)')
 fig.suptitle(pair.upper()+' | all 16 held-out windows; offline reconstructed-field diagnostics')
 fig.savefig(O/f'{pair}_diagnostics.pdf');fig.savefig(O/f'{pair}_diagnostics.png',dpi=180);plt.close(fig)
dump('summary.json',summary);dump('COMPLETED.json',{'rows':len(rows),'windows':32,'phase_resolved_windows':sum(r['phase_resolved'] for r in phase_rows if r['method']=='fusion' and r['step']==1)})
