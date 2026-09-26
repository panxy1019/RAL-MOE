"""Independent history-3 POD RNN, no Galerkin/PPE/learned specialist calls."""
from common import *
from torch import nn
import argparse,random,math,resource
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

class PODRNN(nn.Module):
 def __init__(self,cell,cfg):
  super().__init__();self.rnn=getattr(nn,cell)(66,cfg['width'],cfg['layers'],batch_first=True,
      dropout=cfg['dropout'] if cfg['layers']>1 else 0.)
  self.drop=nn.Dropout(cfg['dropout']);self.head=nn.Linear(cfg['width'],64)
  nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,history,param,dt):
  # Every advance encodes the same three available frames with reset hidden state.
  x=torch.cat([history,param[:,None,None].expand(-1,3,1),dt[:,None,None].expand(-1,3,1)],2)
  y,_=self.rnn(x);return history[:,-1]+self.head(self.drop(y[:,-1]))

def load(reg,part='development'):
 d={k:v for k,v in np.load(OUT/f'{reg}_{part}.npz').items()}
 for key in ['state','mean','scale']:
  assert np.isfinite(d[key]).all()
 d['norm']=(d['state']-d['mean'])/d['scale'];return d
def tensors(d):
 return {k:torch.as_tensor(d[k],device='cuda',dtype=torch.float32) for k in ['norm','mean','scale','gu','gp','loss_u_scale','loss_p_scale']}
def predict(model,d,starts,K,tt=None,initial=None):
 tt=tensors(d) if tt is None else tt;s=torch.as_tensor(starts,device='cuda',dtype=torch.long)
 history=tt['norm'][s[:,None]+torch.arange(-2,1,device='cuda')] if initial is None else initial
 param=torch.tensor((1/d['Re'][starts]-d['param_mean'])/d['param_scale'],device='cuda',dtype=torch.float32)
 outputs=[]
 for k in range(K):
  dt=torch.tensor(((d['time'][np.asarray(starts)+k+1]-d['time'][np.asarray(starts)+k])-d['dt_mean'])/d['dt_scale'],device='cuda',dtype=torch.float32)
  out=model(history,param,dt);outputs.append(out);history=torch.cat([history[:,1:],out[:,None]],1)
 return torch.stack(outputs,1)
def metrics(pred,d,starts,K=48):
 true=d['state'][np.asarray(starts)[:,None]+np.arange(1,K+1)]
 eu=error(pred[:,:,:32],true[:,:,:32],d['gu'],d['cu'],float(d['nu']))
 ep=error(pred[:,:,32:],true[:,:,32:],d['gp'],d['cp'],float(d['np0']))
 finite=np.isfinite(pred).all(2);complete=bool(finite.all());by=[]
 for re in sorted(set(d['Re'][starts])):
  mask=d['Re'][starts]==re;by.append(dict(Re=float(re),Eu=float(np.mean(eu[mask])),Ep=float(np.mean(ep[mask])),joint=float(np.mean(eu[mask]+ep[mask]))))
 return dict(complete=complete,worst=max(x['joint'] for x in by) if complete else float('inf'),
   macro=float(np.mean([x['joint'] for x in by])) if complete else float('inf'),by_Re=by),eu,ep
def evaluate(model,d,starts,tt):
 model.eval()
 with torch.inference_mode():p=predict(model,d,starts,48,tt)*tt['scale']+tt['mean']
 return metrics(p.double().cpu().numpy(),d,starts)[0]
def train(reg,cell,cfg,seed,phase,steps):
 output=OUT/phase/reg/cell/(f"candidate_{cfg['id']:02d}" if phase=='screen' else str(seed))
 if (output/'summary.json').exists():return json.loads((output/'summary.json').read_text())
 output.mkdir(parents=True,exist_ok=True);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 d=load(reg);tt=tensors(d);m=PODRNN(cell,cfg).cuda();opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps,eta_min=cfg['lr']*.1)
 protocol=json.loads((OUT/'protocol_frozen.json').read_text());conf=dict(regime=reg,cell=cell,seed=seed,phase=phase,steps=steps,config=cfg,
    code_sha256=sha(__file__),manifest_sha256=sha(OUT/'split_manifest.json'),protocol_sha256=sha(OUT/'protocol_frozen.json'),parameter_count=sum(p.numel() for p in m.parameters()))
 write(output/'config.yaml',conf);history=[];best=(float('inf'),float('inf'));best_step=-1;start=time.perf_counter();failure=''
 torch.cuda.reset_peak_memory_stats();rng=np.random.default_rng(seed)
 for step in range(1,steps+1):
  m.train();K=[1,4,8,16][min(3,(step-1)*4//steps)];ids=rng.choice(d['train_starts'],size=64,replace=True)
  out=predict(m,d,ids,K,tt);target=tt['norm'][torch.tensor(ids,device='cuda')[:,None]+torch.arange(1,K+1,device='cuda')]
  delta=out-target;da=delta[:,:,:32]*tt['scale'][:32];db=delta[:,:,32:]*tt['scale'][32:]
  loss=delta[:,:,:32].square().mean()+delta[:,:,32:].square().mean()+torch.einsum('bki,ij,bkj->bk',da,tt['gu'],da).mean()/tt['loss_u_scale']+torch.einsum('bki,ij,bkj->bk',db,tt['gp'],db).mean()/tt['loss_p_scale']
  opt.zero_grad(set_to_none=True)
  if not bool(torch.isfinite(loss)):failure='nonfinite_loss';break
  loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
  if not bool(torch.isfinite(grad)):failure='nonfinite_gradient';break
  opt.step();scheduler.step()
  if step%200==0 or step==steps:
   met=evaluate(m,d,d['validation_starts'],tt);score=(met['worst'],met['macro'])
   row=dict(step=step,K=K,loss=float(loss.detach()),gradient_norm=float(grad),worst_Re_validation_joint=score[0],macro_validation_joint=score[1],complete=met['complete'],elapsed_seconds=time.perf_counter()-start)
   history.append(row);csvwrite(output/'train_history.csv',history)
   if met['complete'] and score<best:
    best=score;best_step=step;torch.save(dict(model=m.state_dict(),config=conf,best_step=step,validation=met),output/'best_checkpoint.pt')
    write(output/'checkpoint_selection.json',dict(step=step,validation=met,rule=protocol['selector']))
   print(reg,cell,phase,cfg['id'],seed,row,flush=True)
 elapsed=time.perf_counter()-start
 summary=dict(**conf,accepted=best_step>=0 and not failure,best_step=best_step,validation_worst=best[0],validation_macro=best[1],
  training_seconds=elapsed,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),failure=failure,last_step=step,heldout_evaluated=False)
 write(output/'summary.json',summary);del m,opt,tt;torch.cuda.empty_cache();return summary

def smoke():
 checks=[]
 for reg in ['H','P']:
  d=load(reg);tt=tensors(d);cfg=dict(width=32,layers=1,dropout=0.,id=-1)
  for cell in ['LSTM','GRU']:
   m=PODRNN(cell,cfg).cuda();ids=d['train_starts'][:4];p=predict(m,d,ids,4,tt)
   expected=tt['norm'][torch.tensor(ids,device='cuda')][:,None].expand(-1,4,-1)
   identity=float((p-expected).abs().max());assert identity==0
   target=tt['norm'][torch.tensor(ids,device='cuda')[:,None]+torch.arange(1,5,device='cuda')];loss=(p-target).square().mean();loss.backward()
   assert all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in m.parameters())
   optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3);optimizer.step();m.eval()
   with torch.no_grad():after=predict(m,d,ids,4,tt)
   assert bool(torch.isfinite(after).all())
   # Alter unavailable future targets: rollout must remain unchanged.
   altered=dict(d);altered['norm']=d['norm'].copy();future=np.unique(np.r_[tuple(np.arange(i+1,i+49) for i in ids)])
   # Use disjoint single window to avoid changing another window's input history.
   sid=ids[:1];altered['norm'][sid[0]+1:sid[0]+49]+=777
   with torch.no_grad():a=predict(m,d,sid,4,tt);b=predict(m,altered,sid,4,tensors(altered))
   assert torch.equal(a,b)
   checks.append(dict(regime=reg,cell=cell,identity_initialization=identity,finite_forward_backward=True,no_future_state_leakage=True))
 write(OUT/'smoke_tests.json',checks);print('SMOKE_PASS',flush=True)
def campaign():
 protocol=json.loads((OUT/'protocol_frozen.json').read_text());smoke();selections={}
 for reg in ['H','P']:
  selections[reg]={}
  for cell in ['LSTM','GRU']:
   results=[train(reg,cell,c,1248,'screen',protocol['screen_steps']) for c in protocol['candidates']]
   okay=[x for x in results if x['accepted']]
   if not okay:selections[reg][cell]=dict(status='all_validation_failed');continue
   best=min(okay,key=lambda x:(x['validation_worst'],x['validation_macro'],x['config']['id']))
   selections[reg][cell]=dict(status='selected',config=best['config'],validation_worst=best['validation_worst'],validation_macro=best['validation_macro'])
   write(OUT/'screen_progress.json',selections)
 frozen=OUT/'selected_configs.json'
 if frozen.exists():assert json.loads(frozen.read_text())['selections']==selections
 else:write(frozen,dict(frozen_at=time.strftime('%FT%T%z'),test_seen=False,selections=selections))
 for reg in ['H','P']:
  for cell in ['LSTM','GRU']:
   choice=selections[reg][cell]
   if choice['status']!='selected':continue
   for seed in SEEDS:train(reg,cell,choice['config'],seed,'formal',protocol['formal_steps'])
 write(OUT/'TRAINING_COMPLETED.json',dict(time=time.strftime('%FT%T%z'),test_started=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','campaign']);args=ap.parse_args();globals()[args.stage]()
