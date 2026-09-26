"""Gated full-field Periodic FNO. Raw CFD only; truncated BPTT, no ROM decoder."""
from common import *
from fno_model import *
from scipy.sparse import load_npz
import argparse,gc,subprocess,zipfile
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
FOUT=OUT/'fno_P'

def freeze():
 FOUT.mkdir(exist_ok=True)
 if (FOUT/'protocol_frozen.json').exists():return
 gate=json.loads((OUT/'fno_gate/P/decision.json').read_text());assert gate['decisions']['P']['nx']==1536
 assert json.loads((OUT/'fno_gate/P/memory_smoke.json').read_text())['records'][-1]['within_22GiB']
 candidates=[dict(id=0,width=32,modes=12,increment=True,lr=1e-3,weight_decay=1e-5),dict(id=1,width=32,modes=12,increment=False,lr=1e-3,weight_decay=1e-5),
  dict(id=2,width=32,modes=16,increment=True,lr=3e-4,weight_decay=1e-4),dict(id=3,width=48,modes=12,increment=True,lr=1e-3,weight_decay=1e-4),
  dict(id=4,width=64,modes=12,increment=True,lr=3e-4,weight_decay=1e-5),dict(id=5,width=32,modes=24,increment=True,lr=1e-3,weight_decay=1e-4)]
 write(FOUT/'protocol_frozen.json',dict(created=time.strftime('%FT%T%z'),regime='P',grid=[1536,1024],history=3,layers=4,seeds=SEEDS,
   candidates=candidates,screen_seed=1248,screen_steps=400,screen_eval_every=200,formal_steps=4000,formal_eval_every=1000,batch=1,
   curriculum='equal quarters:1,4,8,16 output steps per sampled window',TBPTT=1,
   optimizer='AdamW; average detached-step gradients over one closed-loop window; clip1; cosine LR to10%; FP32',
   loss='equal normalized u_x/u_y/p masked MSE; normalization is training-only physical area weighted channel mean/std',
   selector='all 48 validation windows finite at fullK48; minimize worst-Re common-FV joint, then macro-Re joint; no test use',
   scope='full raw physical fields on selected uniform grid, no POD reconstruction inputs or learned specialist components',
   acceptance='finite training + at least one full validation checkpoint; no numerical clipping or true-frame refresh',
   pressure='zero fluid-grid mean during autonomous advance; independent area-mean gauge after mapping back for common-FV metrics',
   padding='8 cells on high x/y side each block stack, crop before projection',source_sha256=sha(__file__),manifest_sha256=sha(OUT/'split_manifest.json'),
   source_method='https://arxiv.org/abs/2010.08895',visual_gate='roundtrip_visual_P.png inspected: wake retained; local interpolation error quantified separately'))
def source_path(tr):
 return OUT/('raw_training' if tr['split']=='train' else 'raw_validation')/f"{tr['label']}_uvp_pointData.npz"
def prepare():
 freeze()
 if (FOUT/'data_prepared.json').exists():return
 manifest=json.loads((OUT/'split_manifest.json').read_text())['P'];dev=np.load(OUT/'P_development.npz');ids=dev['global_ids'];lookup={int(s):i for i,s in enumerate(ids)}
 g=np.load(OUT/'P_geometry.npz');w=g['areas'];N=len(w);dest=FOUT/'raw_development.npy'
 cache=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(len(ids),N,3));sources=[];sumv=np.zeros(3);sumv2=np.zeros(3);frames=0
 for tr in manifest['trajectories']:
  if tr['split']=='heldout':continue
  path=source_path(tr);z=np.load(path);assert np.array_equal(z['points'],g['points'])
  indices=np.array(tr['indices']);times=np.array(tr['times']);assert len(times)==len(z['times']) and np.max(abs(times-z['times']))<2e-3
  field=np.stack([z['u'],z['v'],z['p']],2).astype(np.float32);field[:,:,2]-=(field[:,:,2].astype(float)@w/w.sum()).astype(np.float32)[:,None]
  cache[[lookup[int(s)] for s in indices]]=field
  if tr['split']=='train':
   sumv+=np.einsum('tnc,n->c',field.astype(float),w)/w.sum();sumv2+=np.einsum('tnc,n->c',field.astype(float)**2,w)/w.sum();frames+=len(field)
  sources.append(dict(label=tr['label'],split=tr['split'],path=str(path),sha256=sha(path),frames=len(field)))
  print('FNO_RAW_PREPARED',tr['label'],flush=True)
 cache.flush();mean=sumv/frames;scale=np.sqrt(sumv2/frames-mean**2);mean[2]=0.;assert np.all(scale>0)
 np.savez(FOUT/'normalization.npz',mean=mean,scale=scale)
 write(FOUT/'raw_sources.json',sources);write(FOUT/'data_prepared.json',dict(shape=list(cache.shape),dtype='float32',train_frames=frames,train_only_normalization=True,
    cache_sha256=sha(dest),normalization_sha256=sha(FOUT/'normalization.npz'),raw_sources_sha256=sha(FOUT/'raw_sources.json')))

class Data:
 def __init__(self,test=False):
  self.test=test;self.d={k:v for k,v in np.load(OUT/f"P_{'sealed_test' if test else 'development'}.npz").items()}
  geom=np.load(OUT/'P_geometry.npz');self.w=torch.tensor(geom['areas'],device='cuda',dtype=torch.float32);self.norm=np.load(FOUT/'normalization.npz')
  self.mean=torch.tensor(self.norm['mean'],device='cuda',dtype=torch.float32)[None,:,None,None];self.scale=torch.tensor(self.norm['scale'],device='cuda',dtype=torch.float32)[None,:,None,None]
  grid=np.load(OUT/'fno_gate/P/maps/grid_1536x1024.npz');self.mask=torch.tensor(grid['mask'],device='cuda',dtype=torch.float32)[None,None]
  self.coords=torch.tensor(np.stack(np.meshgrid((grid['x']-5)/15,grid['y']/10)),device='cuda',dtype=torch.float32)[None]
  def sparse(path):
   a=load_npz(path).tocoo();return torch.sparse_coo_tensor(torch.tensor(np.stack([a.row,a.col]),device='cuda'),torch.tensor(a.data,device='cuda',dtype=torch.float32),a.shape).coalesce()
  self.forward=sparse(OUT/'fno_gate/P/maps/fv_to_grid_1536x1024.npz');self.back=sparse(OUT/'fno_gate/P/maps/grid_to_fv_1536x1024.npz')
  if not test:self.raw=np.load(FOUT/'raw_development.npy',mmap_mode='r')
  else:
   manifest=json.loads((OUT/'split_manifest.json').read_text())['P'];lookup={int(s):i for i,s in enumerate(self.d['global_ids'])};self.raw=np.empty((len(lookup),len(self.w),3),np.float32)
   for tr in manifest['trajectories']:
    if tr['split']!='heldout':continue
    path=ROOT/'experiments/qlrom_20260923/raw_reference'/f"{tr['label']}_uvp_pointData.npz";z=np.load(path);assert np.array_equal(z['points'],geom['points'])
    field=np.stack([z['u'],z['v'],z['p']],2);field[:,:,2]-=(field[:,:,2].astype(float)@geom['areas']/geom['areas'].sum()).astype(np.float32)[:,None]
    self.raw[[lookup[int(s)] for s in tr['indices']]]=field
 def fv(self,i):return torch.tensor(np.asarray(self.raw[i]),device='cuda',dtype=torch.float32)
 def grid(self,i):
  x=torch.sparse.mm(self.forward,self.fv(i)).T.reshape(1,3,1024,1536)
  p=x[:,2:3];p=p-(p*self.mask).sum()/self.mask.sum();x=torch.cat([x[:,:2],p],1)*self.mask
  return (x-self.mean)/self.scale*self.mask
 def normalize_output(self,x):
  p=x[:,2:3];p=p-(p*self.mask).sum()/self.mask.sum();return torch.cat([x[:,:2],p],1)*self.mask
 def physical(self,x):return (x*self.scale+self.mean)*self.mask
 def inverse(self,x):
  raw=torch.sparse.mm(self.back,self.physical(x)[0].reshape(3,-1).T);p=raw[:,2:3];p=p-(p[:,0]*self.w).sum()/self.w.sum();return torch.cat([raw[:,:2],p],1)
 def condition(self,s,k):
  d=self.d;mu=torch.tensor([(1/d['Re'][s]-d['param_mean'])/d['param_scale']],device='cuda',dtype=torch.float32)
  dt=torch.tensor([((d['time'][s+k+1]-d['time'][s+k])-d['dt_mean'])/d['dt_scale']],device='cuda',dtype=torch.float32);return mu,dt
 def initial(self,s):return torch.stack([self.grid(s-2),self.grid(s-1),self.grid(s)],1)

def errors_fv(pred,true,w):
 p=true[:,2:3];p=p-(p[:,0]*w).sum()/w.sum();true=torch.cat([true[:,:2],p],1)
 du=(w[:,None]*true[:,:2]**2).sum();dp=(w*true[:,2]**2).sum();assert du>0 and dp>0
 return 100*torch.sqrt((w[:,None]*(pred[:,:2]-true[:,:2])**2).sum()/du),100*torch.sqrt((w*(pred[:,2]-true[:,2])**2).sum()/dp)
def evaluate(m,data,starts,output=None):
 m.eval();records=[];hashes=[];predcache=None
 if output:
  predcache=np.lib.format.open_memmap(output/'fv_predictions.npy',mode='w+',dtype=np.float32,shape=(len(starts),48,len(data.w),3))
 with torch.inference_mode():
  for wi,s in enumerate(starts):
   s=int(s);hist=data.initial(s);rows=[]
   for k in range(48):
    mu,dt=data.condition(s,k);pred=data.normalize_output(m(hist,data.coords,data.mask,mu,dt));finite=bool(torch.isfinite(pred).all())
    if finite:
     back=data.inverse(pred);eu,ep=errors_fv(back,data.fv(s+k+1),data.w)
     truth=data.physical(data.grid(s+k+1));physical=data.physical(pred);mask=data.mask
     nu=100*torch.sqrt(((physical[:,:2]-truth[:,:2])**2*mask).sum()/(truth[:,:2]**2*mask).sum());np0=100*torch.sqrt(((physical[:,2:]-truth[:,2:])**2*mask).sum()/(truth[:,2:]**2*mask).sum())
     row=dict(window=int(data.d['global_ids'][s]),Re=float(data.d['Re'][s]),step=k+1,time=float(data.d['time'][s+k+1]),finite=True,Eu=float(eu),Ep=float(ep),native_Eu=float(nu),native_Ep=float(np0))
     if output:
      predcache[wi,k]=back.cpu().numpy();hashes.append(dict(window=wi,step=k+1,native_grid_sha256=hashlib.sha256(physical.cpu().numpy().tobytes()).hexdigest()))
    else:
     row=dict(window=int(data.d['global_ids'][s]),Re=float(data.d['Re'][s]),step=k+1,time=float(data.d['time'][s+k+1]),finite=False,Eu=float('nan'),Ep=float('nan'),native_Eu=float('nan'),native_Ep=float('nan'))
     if output:predcache[wi,k]=np.nan
    rows.append(row);hist=torch.cat([hist[:,1:],pred[:,None]],1)
   records.extend(rows)
   if output:csvwrite(output/'per_step_metrics.csv',records)
 re_scores=[]
 for rv in sorted(set(r['Re'] for r in records)):
  rr=[r for r in records if r['Re']==rv];re_scores.append(np.mean([r['Eu']+r['Ep'] for r in rr]))
 finite=all(r['finite'] for r in records);result=dict(complete=finite,worst=float(max(re_scores)) if finite else float('inf'),macro=float(np.mean(re_scores)) if finite else float('inf'))
 if output:
  predcache.flush();write(output/'test_predictions_or_hashes.json',dict(fv_file=str(output/'fv_predictions.npy'),fv_sha256=sha(output/'fv_predictions.npy'),native_grid_hashes=hashes))
 return result,records

def train(data,cfg,seed,phase,budget,every):
 output=FOUT/phase/(f"candidate_{cfg['id']:02d}" if phase=='screen' else str(seed));output.mkdir(parents=True,exist_ok=True)
 if (output/'summary.json').exists():return json.loads((output/'summary.json').read_text())
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
 m=FNO2d(cfg['width'],cfg['modes'],cfg['increment']).cuda();opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,budget,eta_min=cfg['lr']*.1)
 config=dict(seed=seed,phase=phase,budget=budget,architecture=cfg,code_sha256=sha(__file__),real_parameter_count=sum(p.numel()*(2 if p.is_complex() else 1) for p in m.parameters()),protocol_sha256=sha(FOUT/'protocol_frozen.json'))
 write(output/'config.yaml',config);best=(float('inf'),float('inf'));best_step=-1;history=[];started=time.perf_counter();failure='';torch.cuda.reset_peak_memory_stats()
 for step in range(1,budget+1):
  m.train();K=[1,4,8,16][min(3,(step-1)*4//budget)];s=int(rng.choice(data.d['train_starts']));hist=data.initial(s);opt.zero_grad(set_to_none=True);lossvalue=0.
  for k in range(K):
   mu,dt=data.condition(s,k);pred=data.normalize_output(m(hist,data.coords,data.mask,mu,dt));target=data.grid(s+k+1)
   loss=((pred-target)**2*data.mask).sum()/(3*data.mask.sum()*K)
   if not bool(torch.isfinite(loss)):failure='nonfinite_training_loss';break
   loss.backward();lossvalue+=float(loss.detach());hist=torch.cat([hist[:,1:],pred.detach()[:,None]],1).detach()
  if failure:break
  grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
  if not bool(torch.isfinite(grad)):failure='nonfinite_gradient';break
  opt.step();schedule.step()
  if step%every==0 or step==budget:
   validation,_=evaluate(m,data,data.d['validation_starts']);score=(validation['worst'],validation['macro'])
   row=dict(step=step,K=K,loss=lossvalue,gradient_norm=float(grad),validation_worst=score[0],validation_macro=score[1],complete=validation['complete'],elapsed=time.perf_counter()-started);history.append(row);csvwrite(output/'train_history.csv',history)
   if validation['complete'] and score<best:
    best=score;best_step=step;torch.save(dict(model=m.state_dict(),config=config,validation=validation,step=step),output/'best_checkpoint.pt');write(output/'checkpoint_selection.json',dict(step=step,validation=validation))
   print('FNO_TRAIN',phase,cfg['id'],seed,row,flush=True)
  if step%20==0:write(FOUT/'progress.json',dict(phase=phase,candidate=cfg['id'],seed=seed,step=step,budget=budget,K=K,elapsed=time.perf_counter()-started,updated=time.strftime('%FT%T%z')))
 summary=dict(**config,accepted=best_step>=0 and not failure,best_step=best_step,validation_worst=best[0],validation_macro=best[1],training_seconds=time.perf_counter()-started,
    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),failure=failure,last_step=step,heldout_evaluated=False)
 write(output/'summary.json',summary);del m,opt;gc.collect();torch.cuda.empty_cache();return summary

def campaign():
 prepare();protocol=json.loads((FOUT/'protocol_frozen.json').read_text());data=Data()
 screen=[train(data,c,1248,'screen',protocol['screen_steps'],protocol['screen_eval_every']) for c in protocol['candidates']]
 okay=[r for r in screen if r['accepted']]
 if not okay:write(FOUT/'STOP.json',dict(reason='all_screening_candidates_validation_rejected',test_run=False));return
 best=min(okay,key=lambda r:(r['validation_worst'],r['validation_macro'],r['architecture']['id']))
 if not (FOUT/'selected_config.json').exists():write(FOUT/'selected_config.json',dict(frozen_at=time.strftime('%FT%T%z'),architecture=best['architecture'],test_seen=False))
 for seed in SEEDS:train(data,best['architecture'],seed,'formal',protocol['formal_steps'],protocol['formal_eval_every'])
 write(FOUT/'TRAINING_COMPLETED.json',dict(time=time.strftime('%FT%T%z')))
 del data;gc.collect();torch.cuda.empty_cache()
 test=Data(test=True)
 for seed in SEEDS:
  out=FOUT/'formal'/str(seed);summary=json.loads((out/'summary.json').read_text())
  if not summary['accepted']:continue
  ck=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);cfg=ck['config']['architecture'];m=FNO2d(cfg['width'],cfg['modes'],cfg['increment']).cuda();m.load_state_dict(ck['model']);m.eval()
  score,rows=evaluate(m,test,test.d['test_starts'],out);summary.update(heldout_evaluated=True,test_score=score);write(out/'summary.json',summary);del m;torch.cuda.empty_cache()
 write(FOUT/'PREDICTIONS_COMPLETED.json',dict(time=time.strftime('%FT%T%z'),remaining='final aggregation and paired timing'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','prepare','campaign']);args=p.parse_args();globals()[args.stage]()
