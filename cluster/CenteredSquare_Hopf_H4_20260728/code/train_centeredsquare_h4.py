#!/usr/bin/env python3
"""H3/H4 Hopf-only training: fluctuation normalization and optional radial normal form.

This program consumes only the sealed CenteredSquare train+validation r11 view. It rejects
heldout rows before model construction; heldout evaluation is deliberately not
implemented here.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, importlib.util, json, math, os, random, sys, time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RANK=11
TRAIN=np.asarray([94.0,95.0,95.05,95.15,95.2,95.35,95.4,95.45,95.55,95.6,95.75,96.0,96.25,96.75,97.0,97.25,97.75,98.0,98.5,99.5,100.0,101.0,101.724137931034])
VAL=np.asarray([94.5,95.25,95.5,97.5,99.0,101.5]); HELD=np.asarray([95.1,95.3,96.5,100.5,102.0]); STAGES=((0,1200,1),(1200,2800,2),(2800,4800,4),(4800,8000,8)); EPS=1e-12

def sha(p:Path):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load_parent_weights(model,parent:Path,expected_sha:str):
 actual=sha(parent)
 if actual.lower()!=expected_sha.lower():raise RuntimeError(f'parent checkpoint SHA mismatch: {actual}')
 ck=torch.load(parent,map_location='cpu',weights_only=False)
 if ck.get('variant')!='scale_aware' or int(ck.get('optimizer_step',-1))!=7200:raise RuntimeError('parent is not frozen B step7200')
 src=ck['model_state'];dst=model.state_dict();copied=[]
 for k,v in src.items():
  if k not in dst:continue
  if dst[k].shape==v.shape:dst[k]=v;copied.append(k)
  elif v.ndim==1 and dst[k].shape[0]>v.shape[0]:
   x=torch.zeros_like(dst[k]);x[:v.shape[0]]=v;dst[k]=x;copied.append(k+'[zero_extended_columns]')
  elif v.ndim==2 and dst[k].shape[0]==v.shape[0] and dst[k].shape[1]>v.shape[1]:
   x=torch.zeros_like(dst[k]);x[:,:v.shape[1]]=v;dst[k]=x;copied.append(k+'[zero_extended_columns]')
  else:raise RuntimeError(f'unsupported parent shape mismatch {k}: {v.shape}->{dst[k].shape}')
 model.load_state_dict(dst,strict=True);return ck,actual,copied
def atomic_json(x,p):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=True,default=str),encoding='utf8');os.replace(q,p)
def atomic_torch(x,p):
 q=p.with_suffix(p.suffix+'.tmp');torch.save(x,q);os.replace(q,p)
def load_module(path):
 spec=importlib.util.spec_from_file_location('baseline_hopf',path);m=importlib.util.module_from_spec(spec);assert spec.loader;sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def configure_math(allow_tf32:bool):
 """Single precision-policy entry point shared by preflight and formal training."""
 torch.backends.cuda.matmul.allow_tf32=allow_tf32;torch.backends.cudnn.allow_tf32=allow_tf32
 torch.set_float32_matmul_precision('high' if allow_tf32 else 'highest')
def stage(step):
 for lo,hi,k in STAGES:
  if lo<=step<hi:return k,lo
 return 8,4800

class FluctuationContract:
 def __init__(self,path:Path,device):
  z=np.load(path,allow_pickle=False);self.path=path;self.sha=sha(path)
  self.nodes=torch.as_tensor(z['nodes'],device=device,dtype=torch.float32);self.ma=torch.as_tensor(z['mean_a'],device=device,dtype=torch.float32);self.mb=torch.as_tensor(z['mean_b'],device=device,dtype=torch.float32);self.sa=torch.as_tensor(z['scale_a'],device=device,dtype=torch.float32);self.sb=torch.as_tensor(z['scale_b'],device=device,dtype=torch.float32);self.plane=torch.as_tensor(z['plane'],device=device,dtype=torch.float32);self.rs=torch.as_tensor(z['radial_scale'],device=device,dtype=torch.float32);self.rf=float(z['radial_floor']);self.gt=float(z['growth_tolerance']);self.ts=float(z['time_scale']);self.schedule=z['schedule']
 def interp(self,x,values):
  hi=torch.searchsorted(self.nodes,x).clamp(1,self.nodes.numel()-1);lo=hi-1;w=((x-self.nodes[lo])/(self.nodes[hi]-self.nodes[lo]).clamp_min(1e-8))
  return torch.lerp(values[lo],values[hi],w.view(-1,*([1]*(values.ndim-1))) )
 def components(self,a,b,re):
  ma,mb=self.interp(re,self.ma),self.interp(re,self.mb);sa,sb=self.interp(re,self.sa),self.interp(re,self.sb)
  ap=(a-ma)/sa;bp=(b-mb)/sb;z=(a-ma)@self.plane.T;rho=torch.sqrt(torch.sum(z.float().square(),dim=1)+1e-12)/self.interp(re,self.rs)
  return ap,bp,z,rho,ma,mb,sa,sb

class NormalForm(nn.Module):
 """Nine-parameter smooth Re map plus zero-initialized bounded correction."""
 def __init__(self,nodes):
  super().__init__();self.register_buffer('lo',nodes.min());self.register_buffer('hi',nodes.max());self.mu=nn.Parameter(torch.tensor([0.,0.,0.]));self.beta_raw=nn.Parameter(torch.tensor([-40.,0.,0.]));self.eps=nn.Parameter(torch.zeros(4))
 def forward(self,re,rho,z):
  with torch.autocast(device_type=re.device.type,enabled=False):
   re=re.float();rho=rho.float();z=z.float();x=(2*(re-self.lo.float())/(self.hi.float()-self.lo.float()).clamp_min(1e-8)-1).clamp(-1.5,1.5);basis=torch.stack((torch.ones_like(x),x,2*x*x-1),1);mu=basis@self.mu;beta=F.softplus(basis@self.beta_raw);ctx=torch.stack((rho,z[:,0],z[:,1],torch.ones_like(rho)),1);eps=.05*torch.tanh(ctx@self.eps);return mu,beta,eps
 def derivative(self,re,rho,z):
  mu,beta,eps=self(re,rho,z);return mu*rho-beta*rho.pow(3)+eps,mu,beta

def parse():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=('h3','h4'),required=True);p.add_argument('--baseline-trainer',type=Path,required=True);p.add_argument('--coefficient-view',type=Path,required=True);p.add_argument('--galerkin-path',type=Path,required=True);p.add_argument('--pressure-path',type=Path,required=True);p.add_argument('--asset-manifest',type=Path,required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--experiment-name',required=True);p.add_argument('--seed',type=int,default=1248);p.add_argument('--micro-batch',type=int,default=92);p.add_argument('--grad-accum',type=int,default=1);p.add_argument('--max-steps',type=int,default=8000);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--grad-clip',type=float,default=1.);p.add_argument('--hopf-grad-min',type=float,default=.10);p.add_argument('--hopf-grad-max',type=float,default=.25);p.add_argument('--hopf-grad-target',type=float,default=.18);p.add_argument('--nf-grad-min',type=float,default=.05);p.add_argument('--nf-grad-max',type=float,default=.12);p.add_argument('--nf-grad-target',type=float,default=.08);p.add_argument('--device',default='cuda');p.add_argument('--gpu-memory-fraction',type=float,default=.42);p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--allow-tf32',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--fused-adamw',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--eval-every',type=int,default=200);p.add_argument('--long-eval-every',type=int,default=400);p.add_argument('--validation-windows-per-re',type=int,default=24);p.add_argument('--early-stop-patience-evals',type=int,default=8);p.add_argument('--validation-lock',type=Path,default=Path('/tmp/centeredsquare_hopf_h4_validation_gpu.lock'));p.add_argument('--swanlab-mode',choices=('disabled','online','offline','local'),default='online');p.add_argument('--swanlab-project',default='CenteredSquare_Hopf_H4');p.add_argument('--swanlab-group',default='CenteredSquare_Hopf_r11');p.add_argument('--swanlab-log-every',type=int,default=20);p.add_argument('--smoke-only',action='store_true');p.add_argument('--preflight-per-re',action='store_true');p.add_argument('--preflight-windows',type=int,default=8);p.add_argument('--benchmark-steps',type=int,default=0);p.add_argument('--benchmark-horizon',type=int,default=8);p.add_argument('--resume',type=Path);return p.parse_args()

def expected_name(v):return 'CenteredSquareHopf34_H3_FluctuationNormalized_r11' if v=='h3' else 'CenteredSquareHopf34_H4_NormalFormRadial_r11'
def build_state(B,fc,raw_state):
 def state(a,b,re,ah,bh,rh,rom,stats):
  x,g=raw_state(a,b,re,ah,bh,rh,rom,stats);ap,bp,z,rho,*_=fc.components(a,b,re);extra=torch.cat((ap,bp,z,rho[:,None]),1);extra=torch.nan_to_num(extra,nan=0.,posinf=20.,neginf=-20.).clamp(-20.,20.);return torch.cat((x,extra),1),g
 return state
def make_ops(B,fc,state):
 def outputs(model,a,b,re,ah,bh,rh,rom,stats,stack=False):
  x,g=state(a,b,re,ah,bh,rh,rom,stats);rs,ps,gates,st,closure=model(x,return_expert_stack=stack,return_closure_params=True);return rs*stats['rhs_scale']+stats['rhs_mean'],ps*stats['pressure_scale']+stats['pressure_mean'],gates,st,closure,g
 def step(model,a,b,re,dt,ah,bh,rh,rom,stats):
  k1,pres,gates,st,closure,g=outputs(model,a,b,re,ah,bh,rh,rom,stats,True);k2=outputs(model,a+.5*dt*k1,b,re,ah,bh,rh,rom,stats)[0];k3=outputs(model,a+.5*dt*k2,b,re,ah,bh,rh,rom,stats)[0];k4=outputs(model,a+dt*k3,b,re,ah,bh,rh,rom,stats)[0];an=a+dt/6*(k1+2*k2+2*k3+k4);bn=pres;return an,bn,g,k1,gates,st
 def rollout(model,data,starts,h,rom,stats,device):
  q=B.batch_from_ids(data,starts,device);a,b,re,ah,bh=q['a'],q['b'],q['re'],q['ah'],q['bh'];start_a,start_b=a,b;rhs=B.galerkin(ah.reshape(-1,RANK),bh.reshape(-1,RANK),re[:,None].expand(-1,ah.shape[1]).reshape(-1),rom).reshape_as(ah);cur=starts.copy();out={k:[] for k in ('pred_a','pred_b','true_a','true_b','pred_rhs','true_rhs','gates','stacks','dt')}
  for _ in range(h):
   nxt=data['next'][cur];dt=torch.as_tensor((data['time'][nxt]-data['time'][cur])[:,None],device=device);before=a;a,b,g,rhs1,gates,st=step(model,a,b,re,dt,ah,bh,rhs,rom,stats);ta=torch.as_tensor(data['a'][nxt],device=device);tb=torch.as_tensor(data['b'][nxt],device=device)
   for k,v in [('pred_a',a),('pred_b',b),('true_a',ta),('true_b',tb),('pred_rhs',rhs1),('true_rhs',(ta-before)/dt.clamp_min(1e-8)),('gates',gates),('stacks',st),('dt',dt)]:out[k].append(v)
   ah=torch.cat((a[:,None],ah[:,:-1]),1);bh=torch.cat((b[:,None],bh[:,:-1]),1);rhs=torch.cat((g[:,None],rhs[:,:-1]),1);cur=nxt
  out['start_a']=start_a;out['start_b']=start_b;out['re']=re;return out
 return outputs,rollout

def rollout_safety(out,threshold=100.):
 """Authoritative rollout gate used unchanged by scanner and training."""
 pa=torch.stack(out['pred_a'],1);pb=torch.stack(out['pred_b'],1);ta=torch.stack(out['true_a'],1);tb=torch.stack(out['true_b'],1)
 ru=torch.linalg.norm(pa,dim=2)/torch.linalg.norm(ta,dim=2).clamp_min(EPS);rp=torch.linalg.norm(pb,dim=2)/torch.linalg.norm(tb,dim=2).clamp_min(EPS);ratio=torch.maximum(ru,rp)
 finite=torch.isfinite(pa).all((1,2))&torch.isfinite(pb).all((1,2))&torch.isfinite(ratio).all(1);bad=(~finite)|(ratio.max(1).values>threshold)
 return bad,{'finite_fraction':float(finite.float().mean().detach().cpu()),'max_norm_ratio':float(torch.nan_to_num(ratio,nan=float('inf'),posinf=float('inf')).max().detach().cpu()),'bad_count':int(bad.sum().detach().cpu())}

def geometry(data,device):
 au=np.asarray(data['areas']);pu=np.asarray(data['phi_u']);pp=np.asarray(data['phi_p']);wu=np.r_[au,au];return torch.as_tensor((pu*wu[None])@pu.T,device=device,dtype=torch.float32),torch.as_tensor((pp*au[None])@pp.T,device=device,dtype=torch.float32)
def field_ratio(d,scale,gram):
 # Area-weighted Gram contractions overflow in FP16 for normalized fluctuations.
 with torch.autocast(device_type=d.device.type,enabled=False):
  x=d.float();s=scale.float();g=gram.float();return torch.mean(torch.einsum('bi,ij,bj->b',x,g,x)/torch.einsum('bi,ij,bj->b',s,g,s).clamp_min(1e-12))
def fluc_losses(fc,out,gu,gp):
 pa=torch.stack(out['pred_a'],1);pb=torch.stack(out['pred_b'],1);ta=torch.stack(out['true_a'],1);tb=torch.stack(out['true_b'],1);n,h,_=pa.shape;re=out['re'][:,None].expand(-1,h).reshape(-1);P=pa.reshape(-1,RANK);Q=pb.reshape(-1,RANK);T=ta.reshape(-1,RANK);U=tb.reshape(-1,RANK);ap,bp,z,rho,_,_,sa,sb=fc.components(P,Q,re);at,bt,zt,trho,_,_,_,_=fc.components(T,U,re);modal_u=F.smooth_l1_loss(ap,at);modal_p=F.smooth_l1_loss(bp,bt);field_u=field_ratio(P-T,sa,gu);field_p=field_ratio(Q-U,sb,gp);rad=F.smooth_l1_loss(torch.log(rho+fc.rf),torch.log(trho+fc.rf));pl=torch.log(rho.reshape(n,h)+fc.rf);tl=torch.log(trho.reshape(n,h)+fc.rf)
 if h>1:
  dg=pl[:,1:]-pl[:,:-1];tg=tl[:,1:]-tl[:,:-1];growth=F.smooth_l1_loss(dg,tg)+.1*torch.mean(F.relu(-dg*tg)/(tg.abs()+fc.gt))
 else:growth=rad*0
 total=modal_u+modal_p+field_u+field_p+rad+growth
 return total,{'fluc_modal_u':modal_u,'fluc_modal_p':modal_p,'fluc_field_u':field_u,'fluc_field_p':field_p,'radial':rad,'growth_normalized':growth},(rho.reshape(n,h),trho.reshape(n,h),z.reshape(n,h,2),zt.reshape(n,h,2))
def nf_losses(head,fc,out,rad):
 rho,trho,z,tz=rad;n,h=rho.shape;re=out['re'][:,None].expand(-1,h);dt=torch.cat(out['dt'],1)/fc.ts;_,_,z0,r0,*_=fc.components(out['start_a'],out['start_b'],out['re']);true0=torch.cat((r0[:,None],trho),1);pred0=torch.cat((r0[:,None],rho),1);ctx_r=true0[:,:-1];ctx_z=torch.cat((z0[:,None],tz[:,:-1]),1);tg=(true0[:,1:]-true0[:,:-1])/dt;der,mu,beta=head.derivative(re.reshape(-1),ctx_r.reshape(-1),ctx_z.reshape(-1,2));der=der.reshape(n,h);mu=mu.reshape(n,h);beta=beta.reshape(n,h);derloss=F.smooth_l1_loss(der,tg)
 sim=r0[:,None];steps=[]
 for j in range(h):
  d,_,_=head.derivative(out['re'],sim[:,0],ctx_z[:,j]);sim=(sim+dt[:,j:j+1]*d[:,None]).clamp_min(0);steps.append(sim[:,0])
 sim=torch.stack(steps,1);steploss=F.smooth_l1_loss(sim,trho);target=torch.tanh(tg/fc.gt);signloss=F.smooth_l1_loss(torch.tanh(der/fc.gt),target);mask=(trho>1.)&(tg.abs()<fc.gt)
 eq=torch.sqrt(torch.relu(mu)/(beta+1e-8)+1e-8);sat=F.smooth_l1_loss(eq[mask],trho[mask]) if mask.any() else derloss*0;pg=(pred0[:,1:]-pred0[:,:-1])/dt;cons=F.smooth_l1_loss(pg,der);total=derloss+steploss+signloss+sat+cons
 return total,{'nf_derivative':derloss,'nf_step':steploss,'nf_growth_sign':signloss,'nf_saturation':sat,'nf_consistency':cons,'mu_mean':mu.mean(),'beta_mean':beta.mean()}

def gradnorm(loss,params,retain=True):
 # Mirror loss scaling during AMP audits. An unweighted fluctuation-field term can
 # overflow an FP16 backward even though the weighted formal objective is finite.
 audit_scale=1./4096.
 gs=torch.autograd.grad(loss*audit_scale,params,retain_graph=retain,allow_unused=True)
 x=[(g.double()*g.double()).sum() for g in gs if g is not None]
 return (torch.sqrt(torch.stack(x).sum())/audit_scale).float() if x else loss.new_zeros(())
def grad_diagnostics(loss,params):
 gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
 finite=[g.detach().float() for g in gs if g is not None]
 return {'nonfinite_tensors':sum(int(not torch.isfinite(g).all()) for g in finite),
         'max_abs':max((float(torch.nan_to_num(g.abs(),nan=float('inf'),posinf=float('inf')).max()) for g in finite),default=0.)}
def audit(model,head,common,h3,nf,variant,step,args):
 params=[p for p in model.parameters() if p.requires_grad];cg=gradnorm(common,params,True);hg=gradnorm(h3,params,True);raw_ratio=float((hg/cg.clamp_min(EPS)).detach())
 if not math.isfinite(raw_ratio) or raw_ratio<=EPS:raise RuntimeError(f'H3 raw gradient invalid common={float(cg.detach())} h3={float(hg.detach())}')
 hm=args.hopf_grad_target/raw_ratio;hr=raw_ratio*hm
 rec={'step':step,'common_grad_norm':float(cg.detach()),'h3_raw_grad_norm':float(hg.detach()),'h3_weight':hm,'h3_weighted_ratio':hr,'h3_raw':float(h3.detach()),'h3_weighted':float((h3*hm).detach())}
 if not args.hopf_grad_min<=hr<=args.hopf_grad_max:raise RuntimeError(f'H3 gradient fail closed ratio={hr} raw_ratio={raw_ratio}')
 if variant=='h4':
  np_=[p for p in model.parameters() if p.requires_grad]+[p for p in head.parameters() if p.requires_grad];ng=gradnorm(nf,np_,True); # compare full normal-form gradient against base model norm
  raw_nf=float((ng/cg.clamp_min(EPS)).detach())
  if not math.isfinite(raw_nf) or raw_nf<=EPS:raise RuntimeError(f'normal-form raw gradient invalid common={float(cg.detach())} nf={float(ng.detach())}')
  nm=args.nf_grad_target/raw_nf;nr=raw_nf*nm;rec.update({'nf_raw_grad_norm':float(ng.detach()),'nf_weight':nm,'nf_weighted_ratio':nr,'nf_raw':float(nf.detach()),'nf_weighted':float((nf*nm).detach())})
  if not args.nf_grad_min<=nr<=args.nf_grad_max:raise RuntimeError(f'normal-form gradient fail closed ratio={nr}')
 else:nm=0.
 return hm,nm,rec

@contextlib.contextmanager
def lock(path):
 path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a+');import fcntl
 try:fcntl.flock(f,fcntl.LOCK_EX);yield
 finally:fcntl.flock(f,fcntl.LOCK_UN);f.close()
@torch.no_grad()
def validate(B,fc,model,roll,data,rom,stats,gu,gp,device,args,step):
 hs=[1,2,4,8,16]+([24,48] if step>=4800 and step%args.long_eval_every==0 else []);report={'step':step,'by_re':{},'numeric_gate':True,'hopf_gate':True}
 for rv in VAL:
  ids=data['val_ids'][abs(data['re'][data['val_ids']]-rv)<5e-6];row={}
  for h in hs:
   starts=B.legal_starts(data,ids,h);starts=starts[np.linspace(0,len(starts)-1,min(len(starts),args.validation_windows_per_re),dtype=int)];out=roll(model,data,starts,h,rom,stats,device);pa,pb,ta,tb=out['pred_a'][-1],out['pred_b'][-1],out['true_a'][-1],out['true_b'][-1];finite=torch.isfinite(pa).all(1)&torch.isfinite(pb).all(1);ru=torch.linalg.norm(pa,dim=1)/torch.linalg.norm(ta,dim=1).clamp_min(EPS);rp=torch.linalg.norm(pb,dim=1)/torch.linalg.norm(tb,dim=1).clamp_min(EPS);bad=(~finite)|(ru>10)|(rp>10);_,_,z,rho,_,_,sa,sb=fc.components(pa,pb,out['re']);_,_,tz,trho,*_=fc.components(ta,tb,out['re']);fu=field_ratio(pa-ta,sa,gu).sqrt();fp=field_ratio(pb-tb,sb,gp).sqrt();total_u=B.physical_relative(pa,ta,torch.as_tensor(data['phi_u'],device=device),torch.as_tensor(data['mean_u'],device=device),torch.cat((torch.sqrt(torch.as_tensor(data['areas'],device=device)),)*2));total_p=B.physical_relative(pb,tb,torch.as_tensor(data['phi_p'],device=device),torch.as_tensor(data['mean_p'],device=device),torch.sqrt(torch.as_tensor(data['areas'],device=device)))
   inflation=torch.mean(rho/(trho+fc.rf));raderr=torch.sqrt(torch.mean((rho-trho)**2)/(torch.mean(trho**2)+EPS));growth=float('nan');
   if h>1:
    R=torch.stack(out['pred_a'],1);TT=torch.stack(out['true_a'],1);re=out['re'][:,None].expand(-1,h).reshape(-1);_,_,_,rr,*_=fc.components(R.reshape(-1,RANK),torch.stack(out['pred_b'],1).reshape(-1,RANK),re);_,_,_,trr,*_=fc.components(TT.reshape(-1,RANK),torch.stack(out['true_b'],1).reshape(-1,RANK),re);dg=torch.diff(torch.log(rr.reshape(-1,h)+fc.rf),1);tg=torch.diff(torch.log(trr.reshape(-1,h)+fc.rf),1);mask=tg.abs()>fc.gt;growth=float((torch.sign(dg[mask])==torch.sign(tg[mask])).float().mean()) if mask.any() else 1.
   row[str(h)]={'fluc_u':float(fu),'fluc_p':float(fp),'total_u':float(torch.nanmean(total_u)),'total_p':float(torch.nanmean(total_p)),'radial_rms':float(raderr),'scale_inflation':float(inflation),'growth_sign_accuracy':growth,'finite_fraction':float(finite.float().mean()),'divergent_windows':int(bad.sum()),'max_norm_ratio':float(torch.maximum(ru,rp).max())}
   if h in (8,16) and (bad.any() or inflation>3 or (h>1 and growth<.55)):report['numeric_gate']=False if bad.any() else report['numeric_gate'];report['hopf_gate']=False if inflation>3 or (h>1 and growth<.55) else report['hopf_gate']
  report['by_re'][f'{rv:.6f}']=row
 scores=[]
 for r in report['by_re'].values():
  for h in ('8','16'):
   x=r[h];scores.append(x['fluc_u']+x['fluc_p']+.3*x['radial_rms']+.1*(1-x['growth_sign_accuracy'])+.05*(x['total_u']+x['total_p']))
 report['score']=max(scores);report['hard_gate']=report['numeric_gate'] and report['hopf_gate'];return report

def per_re_forward_backward_smoke(B,fc,model,head,roll,data,rom,stats,scale_t,gu,gp,device,args,base_args):
 """Run the formal AMP/loss/backward path for every train Re and curriculum horizon."""
 rows=[];params=[p for p in list(model.parameters())+(list(head.parameters()) if head else []) if p.requires_grad]
 for k in (1,2,4,8):
  for rv in TRAIN:
   ids=data['train_ids'][np.isclose(data['re'][data['train_ids']],rv,atol=5e-6)];legal=B.legal_starts(data,ids,k)
   if not len(legal):raise RuntimeError(f'preflight has no legal Re={rv:.6f} K{k}')
   starts=legal[np.linspace(0,len(legal)-1,min(len(legal),args.preflight_windows),dtype=int)];model.train();model.zero_grad(set_to_none=True)
   if head:head.train();head.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16,enabled=args.amp):
    common,cparts,out=common_loss(B,model,data,starts,k,rom,stats,device,roll,base_args);bad,safety=rollout_safety(out);sl,sparts=B.scale_loss(out['pred_a'],out['true_a'],scale_t,base_args);fl,fparts,rad=fluc_losses(fc,out,gu,gp);h3=sl+fl;nf,nparts=nf_losses(head,fc,out,rad) if head else (h3*0,{});loss=common+args.hopf_grad_target*h3+(args.nf_grad_target*nf if head else 0)
   if bad.any() or not torch.isfinite(loss):
    diag={n:float(v.detach()) for n,v in {'common':common,'scale':sl,'fluc':fl,'nf':nf,**cparts,**sparts,**fparts,**nparts}.items()};raise RuntimeError(f'preflight forward failed Re={rv:.6f} K{k} safety={safety} loss={float(loss.detach())} parts={diag}')
   h3g=gradnorm(h3,[p for p in model.parameters() if p.requires_grad],True);nfg=gradnorm(nf,params,True) if head else loss.new_zeros(())
   gd={name:grad_diagnostics(value,[p for p in model.parameters() if p.requires_grad]) for name,value in {'scale':sl,'fluc':fl,**fparts}.items()} if not torch.isfinite(h3g) else None
   # Formal GradScaler starts at 2^-10; exercise the same scaled backward path.
   (loss*(1./1024.)).backward();bad_grad=[n for n,p in list(model.named_parameters())+([] if not head else [('normal_form.'+n,p) for n,p in head.named_parameters()]) if p.grad is not None and not torch.isfinite(p.grad).all()]
   if bad_grad or not torch.isfinite(h3g) or (head and (not torch.isfinite(nfg) or float(nfg.detach())<=0)):
    raise RuntimeError(f'preflight backward failed Re={rv:.6f} K{k} bad_grad={bad_grad[:8]} h3g={float(h3g.detach())} nfg={float(nfg.detach())} grad_parts={gd}')
   rows.append({'Re':float(rv),'horizon':k,'windows':int(len(starts)),'loss':float(loss.detach()),'backward_scale':1./1024.,'h3_grad_norm':float(h3g.detach()),'nf_grad_norm':float(nfg.detach()),**safety})
 return {'schema':'H3H4_PER_RE_FORMAL_PATH_SMOKE_V1','variant':args.variant,'amp':args.amp,'allow_tf32':args.allow_tf32,'rows':rows,'passed':True}

def main():
 args=parse();
 if args.experiment_name!=expected_name(args.variant):raise ValueError('exact experiment name required')
 if args.max_steps!=8000 and not(args.smoke_only or args.benchmark_steps):raise ValueError('formal budget fixed at 8000')
 B=load_module(args.baseline_trainer);seed_all(args.seed);device=torch.device(args.device);configure_math(args.allow_tf32)
 if args.gpu_memory_fraction<1:torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
 audit_base=B.audit_assets(SimpleNamespace(coefficient_view=args.coefficient_view,galerkin_path=args.galerkin_path,pressure_path=args.pressure_path,asset_manifest=args.asset_manifest));data=B.load_coefficients(SimpleNamespace(coefficient_view=args.coefficient_view,history_len=3));romnp=B.load_train_rom(SimpleNamespace(galerkin_path=args.galerkin_path,pressure_path=args.pressure_path));base_args=SimpleNamespace(**{**vars(args),'history_len':3,'hidden_dim':256,'expert_hidden':1024,'num_blocks':3,'experts':6,'top_k':2,'expert_blocks':4,'quadratic_rank':4,'dropout':.04,'temperature':.8,'adaptive_gate_initial_logit':6.,'lambda_scale_amplitude':1.,'lambda_scale_growth':.5,'lambda_scale_sign':.1,'scale_floor_quantile':.1})
 norms,scale=B.fit_stats(data,romnp,base_args);stats={k:torch.as_tensor(v,device=device) for k,v in asdict(norms).items()};scale_t={k:torch.as_tensor(v,device=device) for k,v in asdict(scale).items()};rom={k:torch.as_tensor(v,device=device) for k,v in romnp.items()};fc=FluctuationContract(args.contract,device);state=B.state_features;probe=B.batch_from_ids(data,data['train_ids'][:2],device);prh=B.galerkin(probe['ah'].reshape(-1,RANK),probe['bh'].reshape(-1,RANK),probe['re'][:,None].expand(-1,3).reshape(-1),rom).reshape_as(probe['ah']);in_dim=state(probe['a'],probe['b'],probe['re'],probe['ah'],probe['bh'],prh,rom,stats)[0].shape[1]
 model=B.build_model(in_dim,base_args,stats,device);head=NormalForm(fc.nodes).to(device) if args.variant=='h4' else None;outputs,roll=make_ops(B,fc,state);gu,gp=geometry(data,device);params=list(model.parameters())+(list(head.parameters()) if head else []);kw={'lr':args.lr,'weight_decay':args.weight_decay};
 try:opt=torch.optim.AdamW(params,fused=args.fused_adamw,**kw)
 except:opt=torch.optim.AdamW(params,**kw)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.max_steps);scaler=torch.amp.GradScaler('cuda',enabled=args.amp,init_scale=1/1024,growth_interval=2000);outdir=args.output_root/args.experiment_name;outdir.mkdir(parents=True,exist_ok=True);contract_json=json.loads(args.contract.with_name('TRAINONLY_H3H4_CONTRACT.json').read_text());config={**vars(args),'train_Re':TRAIN.tolist(),'validation_Re':VAL.tolist(),'heldout_hard_disabled':HELD.tolist(),'baseline_code_sha256':sha(args.baseline_trainer),'trainer_code_sha256':sha(Path(__file__)),'contract_sha256':fc.sha,'asset_audit':audit_base,'model_input_semantics':'Galerkin and pressure ROM operators are input features; learned finite-difference dynamics are integrated without directly adding the unstable continuous-time ROM backbone','initialization':{'model':'stable_zero_update_with_operator_features','normal_form':'strict neutral/zero-output','old_checkpoint_loaded':False},'selector':'Hopf-aware fluctuation/radial/growth primary; total-field safeguard secondary'};atomic_json(config,outdir/'config.json');atomic_json(contract_json,outdir/'trainonly_manifest.json')
 if args.preflight_per_re:
  result=per_re_forward_backward_smoke(B,fc,model,head,roll,data,rom,stats,scale_t,gu,gp,device,args,base_args);atomic_json(result,outdir/'per_re_forward_backward_smoke.json');print(json.dumps({'passed':True,'variant':args.variant,'rows':len(result['rows'])}));return
 run=None
 if args.swanlab_mode!='disabled':
  import swanlab;run=swanlab.init(project=args.swanlab_project,group=args.swanlab_group,mode=args.swanlab_mode,name=args.experiment_name,config=config,reinit=True,parallel='shared')
 steps=8 if args.smoke_only else args.benchmark_steps if args.benchmark_steps else args.max_steps;hist=[];audits=[];best=float('inf');beststep=-1;k8eval=0;bad=0;last=None;h3w=nfw=0.;t0=time.perf_counter();running={};rn=0
 try:
  for stepn in range(steps):
   k,start=stage(stepn) if not args.smoke_only else ((1,1,2,2,4,4,8,8)[stepn],stepn-(stepn%2));k=args.benchmark_horizon if args.benchmark_steps else k;changed=k!=last;bench_base={1:0,2:1200,4:2800,8:4800}[k] if args.benchmark_steps else 0;schedule_step=bench_base+stepn if args.benchmark_steps else stepn;starts=fc.schedule[schedule_step if schedule_step<fc.schedule.shape[0] else -1];opt.zero_grad(set_to_none=True)
   if changed:
    model.eval();out=roll(model,data,starts,k,rom,stats,device);bad_mask,safety=rollout_safety(out);unsafe=torch.where(bad_mask)[0].detach().cpu().tolist()
    if unsafe:raise RuntimeError(f'physical-backbone unsafe sampler starts={[int(starts[i]) for i in unsafe[:32]]}')
    common_tmp,debug_parts,_=common_loss(B,model,data,starts,k,rom,stats,device,roll,base_args);sl,_=B.scale_loss(out['pred_a'],out['true_a'],scale_t,base_args);fl,debug_fluc,rad=fluc_losses(fc,out,gu,gp);h3=sl+fl;nf=nf_losses(head,fc,out,rad)[0] if head else h3*0
    if not torch.isfinite(common_tmp+h3+nf):
     badidx=torch.where(~torch.isfinite(out['pred_a'][-1]).all(1)|~torch.isfinite(out['pred_b'][-1]).all(1))[0].detach().cpu().tolist()
     badre=[(int(starts[i]),float(data['re'][starts[i]])) for i in badidx[:16]]
     raise RuntimeError(f'nonfinite pre-audit common={float(common_tmp)} scale={float(sl)} fluc={float(fl)} bad_starts={badre} details={ {k:float(v) for k,v in {**debug_parts,**debug_fluc}.items()} }')
    h3w,nfw,rec=audit(model,head,common_tmp,h3,nf,args.variant,stepn,args);audits.append(rec);atomic_json({'audits':audits},outdir/'gradient_audits.json');model.train()
   with torch.autocast('cuda',dtype=torch.float16,enabled=args.amp):
    common,parts,out=common_loss(B,model,data,starts,k,rom,stats,device,roll,base_args);sl,_=B.scale_loss(out['pred_a'],out['true_a'],scale_t,base_args);fl,fparts,rad=fluc_losses(fc,out,gu,gp);h3=sl+fl;nf,nparts=nf_losses(head,fc,out,rad) if head else (h3*0,{}) ;ramp=min(1.,max(0.,(stepn-start)/200));loss=common+h3*h3w*ramp+nf*nfw*ramp
   scaler.scale(loss).backward();scaler.unscale_(opt);gn=float(torch.nn.utils.clip_grad_norm_(params,args.grad_clip));
   if not math.isfinite(gn):raise FloatingPointError('nonfinite gradient')
   scaler.step(opt);scaler.update();sched.step();last=k
   for n,v in {**parts,**fparts,**nparts,'scale_raw':sl,'h3_raw':h3,'nf_raw':nf,'total':loss}.items():running[n]=running.get(n,0)+v.detach();rn+=1 if n=='total' else 0
   actual=stepn+1
   if actual==1 or actual%args.swanlab_log_every==0:atomic_json({'status':'running','optimizer_step':actual,'horizon':k,'last_update_unix':time.time(),'pid':os.getpid()},outdir/'runtime_status.json')
   if actual%args.swanlab_log_every==0 and run:
    payload={f'train/{n}':float((v/max(rn,1)).detach().cpu()) for n,v in running.items()};payload.update({'train/horizon':k,'train/grad_norm':gn,'train/lr':sched.get_last_lr()[0],'train/ramp':ramp});swanlab.log(payload,step=actual);running={};rn=0
   if actual%args.eval_every==0 or actual==steps:
    with lock(args.validation_lock):val=validate(B,fc,model,roll,data,rom,stats,gu,gp,device,args,actual)
    hist.append(val);atomic_json({'history':hist},outdir/'validation_history.json');
    if run:swanlab.log({'validation/score':val['score'],'validation/hard_gate':float(val['hard_gate'])},step=actual)
    if actual>=4800:k8eval+=1
    if actual>=6400 and k8eval>=6 and val['hard_gate']:
     if val['score']<best*(1-1e-3):best=float(val['score']);beststep=actual;bad=0;atomic_torch(checkpoint(model,head,opt,sched,scaler,args,actual,best,beststep,norms,fc,audits,hist),outdir/'best_validation.pt')
     else:bad+=1
    atomic_torch(checkpoint(model,head,opt,sched,scaler,args,actual,best,beststep,norms,fc,audits,hist),outdir/'latest.pt')
    if actual>=6400 and k8eval>=6 and bad>=8:break
  final=checkpoint(model,head,opt,sched,scaler,args,actual,best,beststep,norms,fc,audits,hist);final['status']='training_complete_heldout_not_run';final['performance']={'elapsed_seconds':time.perf_counter()-t0,'steps_per_min':60*actual/max(time.perf_counter()-t0,EPS)};atomic_torch(final,outdir/'final_training.pt');atomic_json(final['performance'],outdir/'throughput.json');atomic_torch({'status':'PENDING_USER_FREEZE_AND_HELDOUT','best_validation':'best_validation.pt' if beststep>=0 else None,'heldout_evaluation_performed':False},outdir/'final.pt')
  if head:
   with torch.no_grad():
    r=torch.as_tensor(TRAIN,dtype=torch.float32,device=device);rho=torch.ones_like(r);z=torch.zeros(len(r),2,device=device);mu,beta,_=head(r,rho,z);atomic_json({'Re':TRAIN.tolist(),'mu':mu.cpu().tolist(),'beta':beta.cpu().tolist(),'equilibrium_radius':torch.sqrt(torch.relu(mu)/(beta+EPS)).cpu().tolist()},outdir/'normal_form_train_nodes.json')
 finally:
  if run:
   import swanlab;swanlab.finish()

def common_loss(B,model,data,starts,k,rom,stats,device,roll,args):
 out=roll(model,data,starts,k,rom,stats,device);pa,pb,ta,tb=out['pred_a'],out['pred_b'],out['true_a'],out['true_b'];ar=torch.stack([B.relative_loss(x,y,stats['a_rel_floor']) for x,y in zip(pa,ta)]).mean();br=torch.stack([B.relative_loss(x,y,stats['b_rel_floor']) for x,y in zip(pb,tb)]).mean();coeff=torch.stack([F.mse_loss((x-y)/stats['a_scale'],torch.zeros_like(x)) for x,y in zip(pa,ta)]).mean();pressure=torch.stack([F.mse_loss((x-y)/stats['b_state_scale'],torch.zeros_like(x)) for x,y in zip(pb,tb)]).mean();energy=torch.stack([F.smooth_l1_loss((x*x).sum(1),(y*y).sum(1)) for x,y in zip(pa,ta)]).mean();gates=[g for x in out['gates'] for g in x];bal,ent,_=B.v16.router_regularization(gates);div=torch.stack([B.v16.expert_diversity_loss(s) for s in out['stacks']]).mean();dyn=torch.stack([F.mse_loss((x-y)/stats['rhs_scale'],torch.zeros_like(x)) for x,y in zip(out['pred_rhs'],out['true_rhs'])]).mean();traj=ar+.25*br;loss=coeff+dyn+.55*pressure+.08*ar+.30*(ar+.25*br)+.15*traj+.10*traj+.02*energy+.02*bal-.002*ent+.01*div;return loss,{'common':loss,'coeff':coeff,'dynamics':dyn,'pressure':pressure,'rollout_u':ar,'rollout_p':br,'energy':energy},out
def checkpoint(model,head,opt,sched,scaler,args,step,best,beststep,norms,fc,audits,hist):
 return {'schema_version':1,'contract':'H3H4_strict_train_only','experiment_name':args.experiment_name,'variant':args.variant,'optimizer_step':step,'model_state':model.state_dict(),'normal_form_state':head.state_dict() if head else None,'optimizer_state':opt.state_dict(),'scheduler_state':sched.state_dict(),'grad_scaler_state':scaler.state_dict(),'rng':torch.get_rng_state(),'best_score':best,'best_step':beststep,'norm_stats':asdict(norms),'fluctuation_contract_sha256':fc.sha,'gradient_audits':audits,'history':hist,'args':vars(args)}
if __name__=='__main__':main()
