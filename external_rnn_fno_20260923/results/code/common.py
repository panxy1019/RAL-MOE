import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import numpy as np,torch,json,csv,hashlib,time
ROOT=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
OLD=ROOT/'experiments/round2_20260917';OUT=ROOT/'experiments/external_rnn_fno_20260923'
SEEDS=[1248,1600,2026]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=True));os.replace(tmp,p)
def csvwrite(p,rows):
 if not rows:return
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def geometry(phi,mean,w,gauge=False):
 phi=phi.astype(float);mean=mean.astype(float)
 if gauge:phi-=(phi@w/w.sum())[:,None];mean-=mean@w/w.sum()
 return (phi*w)@phi.T,(phi*w)@mean,float((mean*mean)@w)
def error(a,b,g,c,n):
 delta=a-b;den=np.einsum('...i,ij,...j->...',b,g,b)+2*b@c+n
 assert np.all(den>0)
 return 100*np.sqrt(np.maximum(np.einsum('...i,ij,...j->...',delta,g,delta),0)/den)
def paths(reg):
 base=ROOT/('Hopf/artifacts/hopf' if reg=='H' else 'periodic_specialist_r32/assets')
 if reg=='H':return base/'velocity_pod_hopf.npz',base/'pressure_pod_hopf.npz',base/'projection_snapshots_velocity_hopf.csv'
 return base/'Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz',base/'Global_POD_AreaWeighted_L2/global_pressure_pod_area_weighted_l2.npz',base/'provenance/projection_snapshots_velocity_periodic.csv'
def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 assert not (OUT/'protocol_frozen.json').exists()
 manifest={};audit={}
 for reg in ['H','P']:
  pu,pp,index=paths(reg);u=np.load(pu);p=np.load(pp);rows=list(csv.DictReader(index.open()))
  a=u['coeff_uv'][:,:32];b=p['coeff_p'][:,:32];re=np.array([float(r['Re']) for r in rows]);ts=np.array([float(r['time']) for r in rows]).astype(np.float32)
  split=np.array([r['split'] for r in rows]);labels=np.array([r['Re_label'] for r in rows]);w=u['point_areas'].astype(float)
  assert str(u['fit_split'])==str(p['fit_split'])=='train'
  groups={s:set(labels[split==s]) for s in set(split)}
  assert all(not groups[s]&groups[t] for s in groups for t in groups if s!=t)
  trajectories=[];train=[];val=[];test=[]
  for label in sorted(set(labels)):
   ids=np.flatnonzero(labels==label);ids=ids[np.argsort(ts[ids])];assert np.all(np.diff(ts[ids])>0)
   assert np.all(np.diff(ids)==1),'Noncontiguous raw ordering requires explicit next links'
   sp=str(split[ids[0]])
   raw=ROOT/'Hopf/source_database'/f'{label}_uvp_pointData.npz'
   trajectories.append(dict(label=label,Re=float(re[ids[0]]),split=sp,indices=ids.tolist(),times=ts[ids].astype(float).tolist(),
       coefficient_source_sha256=sha(pu),raw_path=str(raw) if raw.exists() else None,raw_sha256=sha(raw) if raw.exists() else None))
   if sp=='train':train.extend(ids[2:-16].tolist())
   if sp=='validation':
    eligible=ids[2:-48];assert len(eligible)>0
    val.extend(eligible[np.unique(np.linspace(0,len(eligible)-1,8,dtype=int))].tolist())
  if reg=='P':test=json.loads((OLD/'P_evaluation_v2/proposed/1248/protocol.json').read_text())['starts']
  else:
   for path in sorted((OLD/'H_evaluation/proposed/1248').glob('*_modal.npz')):
    z=np.load(path);rv=float(path.name.split('_')[1])
    for i,t in enumerate(z['initial_times']):
     s=np.flatnonzero(np.isclose(re,rv,atol=2e-5,rtol=0)&np.isclose(ts,t,atol=1e-5,rtol=0));assert len(s)==1;s=int(s[0]);test.append(s)
     assert np.array_equal(a[s+1:s+49],z['true_a'][i]);assert np.array_equal(ts[s+1:s+49],z['times'][i])
  assert len(test)==(42 if reg=='H' else 32)
  for s in val+test:assert len(set(labels[s-2:s+49]))==1 and len(set(split[s-2:s+49]))==1
  checkpoint=Path(json.loads((OLD/('H_evaluation' if reg=='H' else 'P_evaluation_v2')/'proposed/1248/protocol.json').read_text())['checkpoint'])
  ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
  if reg=='H':mean=np.asarray(ck['norm_stats']['x_mean'])[2:66];scale=np.asarray(ck['norm_stats']['x_scale'])[2:66]
  else:mean=np.r_[ck['scalers']['alpha_next']['mean'],ck['scalers']['pressure_state']['mean']];scale=np.r_[ck['scalers']['alpha_next']['scale'],ck['scalers']['pressure_state']['scale']]
  assert mean.shape==scale.shape==(64,) and np.all(scale>0)
  normsource='H raw a/b feature standardizers x[2:66]' if reg=='H' else 'P alpha_next and pressure_state checkpoint standardizers'
  gu,cu,nu=geometry(u['phi_uv'][:32],u['mean_uv_regime'],np.r_[w,w]);gp,cp,np0=geometry(p['phi_p'][:32],p['mean_p_regime'],w,True)
  state=np.c_[a,b];tr=split=='train';inv=1/re;param_mean=float(inv[tr].mean());param_scale=float(inv[tr].std())
  dt=np.zeros(len(ts));dt[:-1]=np.diff(ts);dt_train=dt[np.array(train)];dt_mean=float(dt_train.mean());dt_scale=float(dt_train.std());dt_scale=dt_scale if dt_scale>1e-8 else 1.
  constants=dict(mean=mean,scale=scale,param_mean=param_mean,param_scale=param_scale,dt_mean=dt_mean,dt_scale=dt_scale,
    gu=gu,cu=cu,nu=nu,gp=gp,cp=cp,np0=np0,
    loss_u_scale=float(np.mean(np.einsum('ni,ij,nj->n',a[tr]-a[tr].mean(0),gu,a[tr]-a[tr].mean(0)))),
    loss_p_scale=float(np.mean(np.einsum('ni,ij,nj->n',b[tr]-b[tr].mean(0),gp,b[tr]-b[tr].mean(0)))))
  for part,keep,starts in [('development',split!='heldout',train+val),('sealed_test',split=='heldout',test)]:
   ids=np.flatnonzero(keep);mapping={int(s):i for i,s in enumerate(ids)}
   np.savez_compressed(OUT/f'{reg}_{part}.npz',state=state[ids],Re=re[ids],time=ts[ids],split=split[ids],global_ids=ids,
      train_starts=[mapping[s] for s in train] if part=='development' else [],validation_starts=[mapping[s] for s in val] if part=='development' else [],
      test_starts=[mapping[s] for s in test] if part=='sealed_test' else [],**constants)
  np.savez_compressed(OUT/f'{reg}_geometry.npz',phi_u=u['phi_uv'][:32],phi_p=p['phi_p'][:32],mean_u=u['mean_uv_regime'],mean_p=p['mean_p_regime'],areas=w,points=u['points'])
  manifest[reg]=dict(trajectories=trajectories,train_starts=train,validation_starts=val,test_starts=test,history=3,horizon=48,
     files={str(f):sha(f) for f in [pu,pp,index]},standardization_checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),standardization_identity=normsource)
  audit[reg]=dict(split_counts={s:len(g) for s,g in groups.items()},snapshot_counts={s:int(np.sum(split==s)) for s in groups},
    no_split_overlap=True,history_future_same_trajectory=True,train_only_basis=True,normalization=normsource)
  print('PREPARED',reg,len(train),len(val),len(test),flush=True)
 write(OUT/'split_manifest.json',manifest);write(OUT/'data_audit.json',audit)
 candidates=[]
 # Balanced reduced grid, fixed before evaluation, identical budget for both cells/regimes.
 for i in range(12):candidates.append(dict(id=i,width=[32,64,128][i%3],layers=1+(i//3)%2,dropout=[0.,.1][(i//6)],lr=[1e-3,3e-4][(i//3+i)%2],weight_decay=[0.,1e-5,1e-4][(i//3+i)%3]))
 write(OUT/'protocol_frozen.json',dict(created=time.strftime('%FT%T%z'),seeds=SEEDS,cells=['LSTM','GRU'],regimes=['H','P'],
  history=3,screen_seed=1248,candidates=candidates,screen_steps=800,formal_steps=4000,batch=64,eval_every=200,
  curriculum='equal quarters:1,4,8,16; fully closed-loop, no teacher forcing',optimizer='AdamW, cosine to 10% initial LR, global clip1, FP32',
  loss='standardized a MSE + standardized b MSE + weighted velocity error energy/train fluctuation energy + weighted pressure error energy/train fluctuation energy, weights all1',
  selector='complete finite 48-step validation in all windows; lexicographic min worst-Re joint then macro-Re joint; lowest id on exact ties',
  test='once after both cell configs for both regimes are frozen; formal seed1248 retrained from initialization with full budget, not screening checkpoint',
  failure='nonfinite loss/gradient terminates seed; no replacement. Finite large errors retained. Divergence >10x true-window max modal norm diagnosed after rollout, never used to reset',
  fno=dict(grids=[[192,128],[384,256],[768,512],[1536,1024]],gate='both Eu/Ep roundtrip <=20% frozen best complete existing K48 model, all validation trajectories; smallest passing resolution',
           validation_only=True,memory_cap_GiB=22,layers=4,modes=[12,16,24],width=[32,48,64],teacher_forcing=False),
  manifest_sha256=sha(OUT/'split_manifest.json'),code_sha256=sha(__file__)))
if __name__=='__main__':prepare()
