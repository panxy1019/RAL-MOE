from __future__ import annotations

import argparse, copy, fcntl, hashlib, importlib.util, json, math, os, random, shutil, socket, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

EPS = 1e-8


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)); tmp.replace(path)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def emit(event,**values): print(json.dumps({'event':event,**values},sort_keys=True),flush=True)


class S3:
    def __init__(self,args):
        self.args=args; self.kind=args.experiment; self.run=Path(args.run_dir); self.run.mkdir(parents=True,exist_ok=True); self.ckpt=self.run/'checkpoints'; self.ckpt.mkdir(exist_ok=True)
        cfg=json.loads(Path(args.config).read_text()); cfg=copy.deepcopy(cfg); cfg['training']['learning_rate']=args.learning_rate; cfg['training']['max_steps']=3600; cfg['training']['base_budget_end']=3600; cfg['training']['curriculum']=[{'name':'K16','start':1,'end':3600,'k':16,'use_rollout':True}]
        cfg['checkpoint_root']=str(self.ckpt); self.cfg=cfg
        trainer=load_module('s3_base_trainer',args.trainer); self.trainer=trainer
        self.exp=trainer.Experiment(SimpleNamespace(run_dir=str(self.run/'runtime'),resume=None),cfg)
        source=torch.load(args.checkpoint,map_location='cpu',weights_only=False); self.exp.model.load_state_dict(source['model'],strict=True)
        self.source_checkpoint_step=int(source['step']); self.source_checkpoint_sha256=sha256(args.checkpoint); self.base_calibration=source['calibration']
        self.exp.opt=self.exp._build_optimizer(); self.exp.sched=torch.optim.lr_scheduler.CosineAnnealingLR(self.exp.opt,3600,eta_min=args.learning_rate*.05); self.exp.grad_scaler=torch.amp.GradScaler('cuda',enabled=self.exp.amp_dtype==torch.float16)
        self.micro,self.accum=((8,8) if self.kind=='S3-C' else (16,4)); self.exp.prepare_schedule(self.micro,self.accum)
        with np.load(args.bank) as b: self.bank={key:torch.as_tensor(b[key],device=self.exp.device) if b[key].dtype.kind not in 'USO' else b[key].astype(str) for key in b.files}
        self.bank_sha256=sha256(args.bank); self.hlen=int(self.exp.cache['hist_idx'].shape[1]); self.dt=self.exp.cache['dt_next'].clone(); self._repair_terminal_dt()
        self.lambda_new=None; self.top=[]; self.best_physical=None; self.best_contraction=None; self.best_qualified=None; self.swan=None

    def _repair_terminal_dt(self):
        nxt=np.asarray(self.exp.a['next_idx']); dt=np.asarray(self.exp.a['dt_next']); fixed=np.unique(np.concatenate((np.asarray(self.bank['train_fixed_ids'].cpu()),np.asarray(self.bank['validation_fixed_ids'].cpu()))))
        for sid in fixed.tolist():
            if not torch.isfinite(self.dt[sid]):
                pred=np.flatnonzero(nxt==sid); value=float(dt[pred[-1]]) if len(pred) else float(np.nanmedian(dt[self.exp.a['label_id']==self.exp.a['label_id'][sid]])); self.dt[sid]=value
        if not torch.isfinite(self.dt[torch.as_tensor(fixed,device=self.exp.device)]).all(): raise AssertionError('non-finite fixed-point dt')

    def bank_batch(self,step,part):
        ids=self.bank['train_schedule'][step,part*self.micro:(part+1)*self.micro].long()
        return {key:self.bank['train_'+key][ids] for key in ('label_id','fixed_id','scale','du','dp')}

    def free_rollout(self,fixed_ids,a,b,k):
        label=self.exp.tensor('label_id',fixed_ids,torch.long); re=self.exp.tensor('re',fixed_ids); phase=self.exp.tensor('phase',fixed_ids); dt=self.dt[fixed_ids]
        ah=a[:,None].repeat(1,self.hlen,1); bh=b[:,None].repeat(1,self.hlen,1); base0=self.exp.vendor.galerkin_rhs_torch(a,b,label,self.exp.gal); rh=base0[:,None].repeat(1,self.hlen,1)
        pa=[]; pb=[]
        for _ in range(k):
            base=self.exp.vendor.galerkin_rhs_torch(a,b,label,self.exp.gal); x0=self.exp.vendor.make_features_torch(a,b,base,re,phase,self.cfg['model']['phase_harmonics']); x=self.exp.vendor.make_history_features_from_states_torch(x0,a,b,base,ah,bh,rh)
            raw_u,raw_p=self.exp._model_call((x-self.exp.xmt)/self.exp.xst,k); rhs=base+raw_u.float()*self.exp.rst+self.exp.rmt; an=a+dt.unsqueeze(1)*rhs; delta=raw_p.float()*self.exp.pst+self.exp.pmt; bn=self.exp.vendor.pressure_surrogate_torch(an,label,self.exp.sur)+delta
            pa.append(an); pb.append(bn)
            if self.hlen>1: ah=torch.cat([an[:,None],a[:,None],ah[:,1:-1]],1); bh=torch.cat([bn[:,None],b[:,None],bh[:,1:-1]],1); rh=torch.cat([rhs[:,None],base[:,None],rh[:,1:-1]],1)
            a,b=an,bn
        return torch.stack(pa),torch.stack(pb)

    def fixed_inputs(self,batch):
        fixed=batch['fixed_id'].long(); a=self.exp.tensor('a',fixed); b=self.exp.tensor('b',fixed); scale=batch['scale'].float().unsqueeze(1)
        da=scale*self.exp.avt*batch['du'].float(); db=scale*self.exp.bvt*batch['dp'].float(); return fixed,a,b,da,db

    def recovery_loss(self,batch,k):
        fixed,a,b,da,db=self.fixed_inputs(batch); pa,pb=self.free_rollout(fixed,a+da,b+db,k); d2=torch.mean(((pa-a.unsqueeze(0))/self.exp.avt)**2,2)+torch.mean(((pb-b.unsqueeze(0))/self.exp.bvt)**2,2); den=torch.mean((da/self.exp.avt)**2,1)+torch.mean((db/self.exp.bvt)**2,1)+EPS
        recover=torch.mean(d2/den.unsqueeze(0)); d=torch.sqrt(d2+EPS); mono=torch.mean(torch.relu(d[1:]-.99*d[:-1])**2/den.unsqueeze(0)) if k>1 else torch.zeros((),device=a.device); return recover+mono,{'recover':recover,'mono':mono}

    def pair_loss(self,batch,k):
        fixed,a,b,da,db=self.fixed_inputs(batch); ca,cb=self.free_rollout(fixed,a,b,k); pa,pb=self.free_rollout(fixed,a+da,b+db,k); den=torch.sqrt(torch.mean((da/self.exp.avt)**2,1)+torch.mean((db/self.exp.bvt)**2,1)+EPS)
        gain=torch.sqrt(torch.mean(((pa-ca)/self.exp.avt)**2,2)+torch.mean(((pb-cb)/self.exp.bvt)**2,2)+EPS)/den.unsqueeze(0); terminal={4:1.,8:.98,16:.95}[k]; rho=torch.linspace(1.,terminal,k,device=a.device).unsqueeze(1); loss=torch.mean(torch.relu(gain-rho)**2); return loss,{'gain_terminal':gain[-1].mean(),'gain_worst':gain[-1].max()}

    def one_map(self,fixed,z,pressure_only=False):
        if pressure_only:
            a=self.exp.tensor('a',fixed); b=z
        else: a,b=z[:,:32],z[:,32:]
        pa,pb=self.free_rollout(fixed,a,b,1); return pb[0] if pressure_only else torch.cat((pa[0],pb[0]),1)

    def spectral(self,fixed,z,pressure_only):
        z=z.detach().float().requires_grad_(True); v=torch.where(torch.arange(z.shape[1],device=z.device)%2==0,1.,-1.).expand_as(z); v=v/(torch.linalg.vector_norm(v,dim=1,keepdim=True)+EPS)
        def fn(value): return self.one_map(fixed,value,pressure_only)
        _,jv=torch.autograd.functional.jvp(fn,z,v,create_graph=True,strict=False); sigma=torch.linalg.vector_norm(jv,dim=1)/(torch.linalg.vector_norm(v,dim=1)+EPS); return sigma

    def jacobian_loss(self,batch,k):
        count=max(1,self.micro//4); fixed,a,b,_,_=self.fixed_inputs({key:value[:count] for key,value in batch.items()});
        with torch.autocast('cuda',enabled=False):
            pp=self.spectral(fixed,b.float(),True); full=self.spectral(fixed,torch.cat((a,b),1).float(),False); loss=torch.mean(torch.relu(pp-.95)**2+.25*torch.relu(full-.97)**2)
        return loss,{'sigma_pp':pp.mean(),'sigma_full':full.mean(),'sigma_pp_worst':pp.max(),'sigma_full_worst':full.max()}

    def mechanism(self,batch,k):
        if self.kind=='S3-A': return self.recovery_loss(batch,k)
        if self.kind=='S3-B': return self.pair_loss(batch,k)
        return self.jacobian_loss(batch,k)

    def perturb_k(self,step): return 4 if step<=600 else (8 if step<=1600 else 16)

    def grad_norm(self,loss,retain=True):
        grads=torch.autograd.grad(loss,[p for p in self.exp.model.parameters() if p.requires_grad],retain_graph=retain,allow_unused=True); return float(torch.sqrt(sum((g.float()**2).sum() for g in grads if g is not None)+EPS).cpu())

    def calibrate(self):
        clean=self.exp.sample_schedule[0,:self.micro]; batch=self.bank_batch(0,0); k=4
        with torch.autocast('cuda',dtype=self.exp.amp_dtype):
            out=self.exp.rollout(clean,16); base=out['one']+self.base_calibration['lambda_rollout']*out['rollout']+self.base_calibration['lambda_anchor']*out['anchor']; mech,detail=self.mechanism(batch,k)
        bn=self.grad_norm(base); mn=self.grad_norm(mech,False); ratio=.075; self.lambda_new=float(min(ratio*bn/max(mn,EPS),.20*bn/max(mn,EPS))); self.exp.opt.zero_grad(set_to_none=True)
        report={'base_grad_norm':bn,'mechanism_grad_norm':mn,'target_ratio':ratio,'max_ratio':.2,'lambda_new':self.lambda_new,'detail':{k:float(v.detach().cpu()) for k,v in detail.items()}}; atomic_json(self.run/'gradient_calibration.json',report); return report

    def train_step(self,step):
        self.exp.model.train(); self.exp.opt.zero_grad(set_to_none=True); k=self.perturb_k(step); ramp=min(1.,step/300); sums={'loss':0.,'base':0.,'mechanism':0.}; detail_sum={}
        for part in range(self.accum):
            clean=self.exp.sample_schedule[step,part*self.micro:(part+1)*self.micro]; batch=self.bank_batch(step,part)
            with torch.autocast('cuda',dtype=self.exp.amp_dtype):
                out=self.exp.rollout(clean,16); base=out['one']+self.base_calibration['lambda_rollout']*out['rollout']+self.base_calibration['lambda_anchor']*out['anchor']; mech,detail=self.mechanism(batch,k); total=base+ramp*self.lambda_new*mech
            if not torch.isfinite(total):
                raise FloatingPointError('non-finite total loss')
            self.exp.grad_scaler.scale(total/self.accum).backward()
            sums['loss']+=float(total.detach().cpu())/self.accum; sums['base']+=float(base.detach().cpu())/self.accum; sums['mechanism']+=float(mech.detach().cpu())/self.accum
            for key,value in detail.items(): detail_sum[key]=detail_sum.get(key,0.)+float(value.detach().cpu())/self.accum
        self.exp.grad_scaler.unscale_(self.exp.opt); grad=torch.nn.utils.clip_grad_norm_([p for p in self.exp.model.parameters() if p.requires_grad],self.exp.e['grad_clip']);
        if not torch.isfinite(grad):
            raise FloatingPointError('non-finite grad')
        self.exp.grad_scaler.step(self.exp.opt); self.exp.grad_scaler.update(); self.exp.sched.step()
        return sums|detail_sum|{'grad_norm':float(grad.cpu()),'k':k,'ramp':ramp}

    def fixed_validation(self):
        ids=self.bank['validation_fixed_ids'].long(); a=self.exp.tensor('a',ids); b=self.exp.tensor('b',ids); result={'fixed_point':{},'paired_gain':{},'recovery_to_truth':{}}
        with torch.inference_mode(),torch.autocast('cuda',dtype=self.exp.amp_dtype):
            ca,cb=self.free_rollout(ids,a,b,16)
            vu=[]; vp=[]
            for i,label in enumerate(self.bank['validation_fixed_labels'].cpu().tolist()):
                un,ud=self.finalizer.field_sums(ca[-1:,i:i+1],a[None,i:i+1],self.exp.cache['velocity_pod_basis'],self.exp.cache['velocity_mean'],self.velocity_weight); pn,pd=self.finalizer.field_sums(cb[-1:,i:i+1],b[None,i:i+1],self.exp.cache['pressure_pod_basis'],self.exp.cache['pressure_mean'],self.pressure_weight); name=str(self.exp.a['labels'][label]); result['fixed_point'][name]={'physical_velocity':math.sqrt(un/max(ud,EPS)),'physical_pressure':math.sqrt(pn/max(pd,EPS))}; vu.append(result['fixed_point'][name]['physical_velocity']); vp.append(result['fixed_point'][name]['physical_pressure'])
            result['fixed_point']['worst_physical_velocity']=max(vu); result['fixed_point']['worst_physical_pressure']=max(vp)
            for scale in (.001,.005,.01,.02,.05):
                mask=torch.isclose(self.bank['validation_scale'].float(),torch.tensor(scale,device=self.exp.device)); fixed=self.bank['validation_fixed_id'][mask].long(); aa=self.exp.tensor('a',fixed); bb=self.exp.tensor('b',fixed); da=scale*self.exp.avt*self.bank['validation_du'][mask].float(); db=scale*self.exp.bvt*self.bank['validation_dp'][mask].float(); ua,ub=self.free_rollout(fixed,aa,bb,16); pa,pb=self.free_rollout(fixed,aa+da,bb+db,16); den=torch.sqrt(torch.mean((da/self.exp.avt)**2,1)+torch.mean((db/self.exp.bvt)**2,1)+EPS); gain=torch.sqrt(torch.mean(((pa[-1]-ua[-1])/self.exp.avt)**2,1)+torch.mean(((pb[-1]-ub[-1])/self.exp.bvt)**2,1)+EPS)/den; rec=torch.sqrt(torch.mean(((pb[-1]-bb)/self.exp.bvt)**2,1)+torch.mean(((pa[-1]-aa)/self.exp.avt)**2,1)+EPS)/den; result['paired_gain'][str(scale)]={'mean':float(gain.mean().cpu()),'worst':float(gain.max().cpu())}; result['recovery_to_truth'][str(scale)]={'mean':float(rec.mean().cpu()),'worst':float(rec.max().cpu())}
        return result

    def validate(self,step):
        lock=Path(self.args.validation_lock); lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('w') as f:
            fcntl.flock(f,fcntl.LOCK_EX); reports={}
            for k in (1,4,8,16): reports[f'k{k}']=self.finalizer.evaluate_horizon(self.exp,'validation',k)
            if step%400==0: reports['k56']=self.finalizer.evaluate_horizon(self.exp,'validation',56)
            fixed=self.fixed_validation(); fcntl.flock(f,fcntl.LOCK_UN)
        for name,r in reports.items():
            if not r['available']: continue
            for space in ('coefficient_space','physical_reconstruction_area_weighted'):
                vals=[v[space] for v in r['by_re'].values()]; r[space+'_mean']={key:sum(v[key] for v in vals)/len(vals) for key in vals[0]}
        result={'step':step,'horizons':reports,'attractivity':fixed}; (self.run/'validation_history.jsonl').open('a').write(json.dumps(result)+'\n'); emit('validation_complete',experiment=self.kind,step=step); return result

    def selector(self,val):
        k16=val['horizons']['k16']['overall_worst']; p16=k16['physical_reconstruction_area_weighted']; k56=val['horizons'].get('k56'); p56=k56['overall_worst']['physical_reconstruction_area_weighted']['pressure_relative_l2'] if k56 and k56['available'] else float('inf'); gains=[val['attractivity']['paired_gain'][str(s)]['worst'] for s in (.005,.01,.02,.05)]; gain=max(gains); fp=val['attractivity']['fixed_point']['worst_physical_pressure']; finite=k16['finite_fraction']==1 and k16['divergent_windows']==0 and (not k56 or (k56['overall_worst']['finite_fraction']==1 and k56['overall_worst']['divergent_windows']==0)); precision=p16['velocity_relative_l2']<=.0016 and p16['pressure_relative_l2']<=.082 and p56<=.25; qualified=finite and precision and gain<1 and fp<.05
        return {'physical_key':[p16['pressure_relative_l2'],p16['velocity_relative_l2'],p56],'contraction_key':[gain,fp,p16['pressure_relative_l2'],p56],'qualified':qualified,'gain':gain,'fixed_pressure':fp,'precision_gate':precision,'finite_gate':finite}

    def payload(self,step,val):
        return {'schema_version':1,'experiment':self.kind,'step':step,'model':self.exp.model.state_dict(),'optimizer':self.exp.opt.state_dict(),'scheduler':self.exp.sched.state_dict(),'amp_scaler':self.exp.grad_scaler.state_dict(),'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},'validation':val,'config':self.cfg,'source_checkpoint_step':self.source_checkpoint_step,'source_checkpoint_sha256':self.source_checkpoint_sha256,'perturbation_bank_sha256':self.bank_sha256,'base_calibration':self.base_calibration,'lambda_new':self.lambda_new}

    def link(self,source,name):
        target=self.ckpt/name; tmp=self.ckpt/(name+'.tmp'); tmp.unlink(missing_ok=True); os.link(source,tmp); tmp.replace(target)

    def checkpoint(self,step,val):
        path=self.ckpt/f'validation_step_{step:04d}.pt'; tmp=path.with_suffix('.pt.tmp'); torch.save(self.payload(step,val),tmp); tmp.replace(path); self.link(path,'latest.pt'); score=self.selector(val)
        if self.best_physical is None or tuple(score['physical_key'])<tuple(self.best_physical): self.best_physical=score['physical_key']; self.link(path,'best_physical.pt')
        if self.best_contraction is None or tuple(score['contraction_key'])<tuple(self.best_contraction): self.best_contraction=score['contraction_key']; self.link(path,'best_contraction.pt')
        if score['qualified'] and (self.best_qualified is None or tuple(score['contraction_key'])<tuple(self.best_qualified)): self.best_qualified=score['contraction_key']; self.link(path,'best_qualified.pt')
        self.top.append((tuple(score['contraction_key']),path)); self.top.sort(key=lambda x:x[0]); self.top=self.top[:5]; keep={p for _,p in self.top}
        for old in self.ckpt.glob('validation_step_*.pt'):
            if old not in keep: old.unlink()
        atomic_json(self.run/'checkpoint_status.json',{'step':step,'selector':score,'best_physical':self.best_physical,'best_contraction':self.best_contraction,'best_qualified':self.best_qualified,'status':'QUALIFIED_LOCALLY_ATTRACTIVE' if self.best_qualified else 'NO_QUALIFIED_CHECKPOINT','top5':[str(p) for _,p in self.top]}); return score

    def init_eval_assets(self):
        self.finalizer=load_module('s3_finalizer',self.args.finalizer)
        with np.load(self.exp.paths['velocity_pod']) as pod: w=torch.as_tensor(pod['sqrt_point_areas'],device=self.exp.device)
        self.velocity_weight=torch.cat((w,w)); self.pressure_weight=w

    def init_swan(self):
        import swanlab; swanlab.login(api_key=os.environ.get('SWANLAB_API_KEY'),relogin=False,save=False,timeout=20); self.swan=swanlab.init(project='V17indepentMOEV2',name=f'V17-{self.kind}-RTX3090',group='S3-local-attractivity-screen',mode='online',config={'experiment':self.kind,'source_step':8200,'lr':self.args.learning_rate,'micro_batch':self.micro,'grad_accum':self.accum,'bank_sha256':self.bank_sha256,'lambda_new':self.lambda_new},tags=['V17','S3',self.kind,'RTX3090','mechanism-screen'],log_dir=str(self.run/'swanlog'),reinit=True,parallel='shared')

    def run_train(self):
        self.init_eval_assets(); calibration=self.calibrate(); smoke=self.train_step(1); atomic_json(self.run/'smoke.json',{'status':'PASS','calibration':calibration,'step':smoke});
        # Restore the checkpoint model and fresh optimizer after the mutating smoke step.
        source=torch.load(self.args.checkpoint,map_location='cpu',weights_only=False); self.exp.model.load_state_dict(source['model']); self.exp.opt=self.exp._build_optimizer(); self.exp.sched=torch.optim.lr_scheduler.CosineAnnealingLR(self.exp.opt,3600,eta_min=self.args.learning_rate*.05); self.exp.grad_scaler=torch.amp.GradScaler('cuda',enabled=self.exp.amp_dtype==torch.float16)
        if self.args.smoke_only:
            val0=self.validate(0); self.checkpoint(0,val0); before=self.trainer.model_digest(self.exp.model); loaded=torch.load(self.ckpt/'latest.pt',map_location='cpu',weights_only=False); self.exp.model.load_state_dict(loaded['model']); after=self.trainer.model_digest(self.exp.model)
            if before!=after: raise AssertionError('checkpoint save-load model mismatch')
            atomic_json(self.run/'SMOKE_DONE.json',{'status':'PASS','experiment':self.kind,'model_sha256':after,'selector':self.selector(val0)}); emit('smoke_complete',experiment=self.kind); return
        self.init_swan(); import swanlab; atomic_json(self.run/'RUNNING.json',{'status':'RUNNING','pid':os.getpid(),'host':socket.gethostname(),'experiment':self.kind,'started_unix':time.time(),'source_checkpoint_sha256':self.source_checkpoint_sha256,'perturbation_bank_sha256':self.bank_sha256}); emit('training_started',experiment=self.kind,pid=os.getpid())
        val0=self.validate(0); self.checkpoint(0,val0)
        for step in range(1,3601):
            metrics=self.train_step(step)
            if step==1 or step%20==0:
                log={f'train/{k}':v for k,v in metrics.items()}; log['train/lr']=self.exp.opt.param_groups[0]['lr']; swanlab.log(log,step=step); emit('optimizer_step',experiment=self.kind,step=step,**log)
            if step%200==0:
                val=self.validate(step); score=self.checkpoint(step,val); flat={'selector/paired_gain_worst':score['gain'],'selector/fixed_pressure':score['fixed_pressure'],'selector/precision_gate':int(score['precision_gate']),'selector/qualified':int(score['qualified'])}
                for name,r in val['horizons'].items():
                    if r['available']:
                        for field,value in r['overall_worst']['physical_reconstruction_area_weighted'].items(): flat[f'physical/{name}_worst_{field}']=value
                        for field,value in r['physical_reconstruction_area_weighted_mean'].items(): flat[f'physical/{name}_mean_{field}']=value
                        for field,value in r['overall_worst']['coefficient_space'].items(): flat[f'modal/{name}_{field}']=value
                        flat[f'physical/{name}_worst_velocity']=r['overall_worst']['physical_reconstruction_area_weighted']['velocity_relative_l2']; flat[f'physical/{name}_worst_pressure']=r['overall_worst']['physical_reconstruction_area_weighted']['pressure_relative_l2']
                        flat[f'physical/{name}_mean_velocity']=r['physical_reconstruction_area_weighted_mean']['velocity_relative_l2']; flat[f'physical/{name}_mean_pressure']=r['physical_reconstruction_area_weighted_mean']['pressure_relative_l2']
                        flat[f'modal/{name}_velocity']=r['overall_worst']['coefficient_space']['velocity_relative_l2']; flat[f'modal/{name}_pressure']=r['overall_worst']['coefficient_space']['pressure_relative_l2']
                flat['attractivity/k16_fixed_physical_velocity']=val['attractivity']['fixed_point']['worst_physical_velocity']; flat['attractivity/k16_fixed_physical_pressure']=val['attractivity']['fixed_point']['worst_physical_pressure']
                for scale,item in val['attractivity']['paired_gain'].items(): flat[f'attractivity/k16_paired_gain_{scale}']=item['worst']
                for scale,item in val['attractivity']['recovery_to_truth'].items(): flat[f'attractivity/k16_recovery_truth_{scale}']=item['worst']
                swanlab.log(flat,step=step)
        status='QUALIFIED_LOCALLY_ATTRACTIVE' if self.best_qualified else 'NO_QUALIFIED_CHECKPOINT'; atomic_json(self.run/'DONE.json',{'status':status,'step':3600,'finished_unix':time.time()}); swanlab.finish(state='success'); emit('training_complete',experiment=self.kind,status=status)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--experiment',choices=['S3-A','S3-B','S3-C'],required=True); p.add_argument('--trainer',required=True); p.add_argument('--finalizer',required=True); p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--bank',required=True); p.add_argument('--run-dir',required=True); p.add_argument('--validation-lock',required=True); p.add_argument('--learning-rate',type=float,required=True); p.add_argument('--smoke-only',action='store_true'); args=p.parse_args()
    try: S3(args).run_train()
    except Exception as exc: atomic_json(Path(args.run_dir)/'FAILED.json',{'status':'FAILED','error':repr(exc),'time':time.time()}); raise


if __name__=='__main__': main()
