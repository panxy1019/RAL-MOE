"""Recover auditable physical snapshots; never trains or selects by error.

Run with the archived pt_env on the authorized training host.
Output: self-contained physical arrays plus provenance, before any rendering.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
P=R.parent/'Pinball'
SQ=R/'centeredsquare_fusion_runs/E2_T2C_K24_20260730_STRICT_V3'
PB=P/'fluidic_pinball_fusion_v1/runs/E2_T2C_K24_20260806_STRICT_V1'
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--case',choices=['square','circular','pinball'],required=True)
args=parser.parse_args();O=args.output;O.mkdir(parents=True,exist_ok=True)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def record(p):return {'path':str(p),'sha256':sha(p)}
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def gauge(p,w):return p-np.sum(p*w,axis=-1,keepdims=True)/w.sum()
def metrics(fs,w):
 ref=fs['reference'];out={}
 for name,f in fs.items():
  out[name]={}
  for label,sl in [('Eu',slice(0,2)),('Ep',slice(2,3))]:
   out[name][label]=float(np.sqrt(np.sum((f[sl]-ref[sl])**2*w)/np.sum(ref[sl]**2*w)))
 return out
def finalize(key,meta,fields,areas,points):
 assert all(np.isfinite(v).all() for v in fields.values())
 assert (areas>0).all() and len(areas)==fields['reference'].shape[1]
 meta.update(reference_type='POD reconstruction',velocity_definition='u,v in physical coefficient scale; speed=hypot(u,v); pointwise velocity error=norm(u_pred-u_ref)',
  pressure_definition='native kinematic pressure p (pressure divided by density), no density multiplication; weighted zero-mean gauge',
  nondimensional_scales='native POD physical scale retained; no additional u/U or p/U^2 rescaling; independent case color scales, not a common nondimensional comparison',
  gauge_correction='independent quadrature-weighted domain-zero-mean for every pressure field; not fitted to the error',
  interpolation='none for cell fields; piecewise-linear on original surface triangles for Circular point fields',
  fluid_mask='original VTK fluid topology; solid interiors have no triangles',
  integration_weights='native point_areas, full computational domain',missing_points=0,nonfinite_points=0,
  selection_rule='ascending heldout Re, then first archived/common valid window by start index; final output of full saved rollout; no metric sorting',
  snapshot_metrics=metrics(fields,areas),metric_units='fraction (multiply by 100 for percent)',
  training_performed=False,errors_computed_on='full domain, not cropped ROI')
 np.savez_compressed(O/f'{key}.npz',**fields,areas=areas,points=points)
 meta['snapshot_arrays']=record(O/f'{key}.npz');meta['case_id']=key
 dump(O/f'{key}.json',meta)
 print(key,json.dumps({'Re':meta['Re'],'start':meta['window_id'],'time':meta['snapshot_time'],'metrics':meta['snapshot_metrics']}),flush=True)

def square():
 sys.path.insert(0,str(R/'centeredsquare_fusion_v1'))
 from common import ConvexGate,router_probabilities
 for pair in ['sh','hp']:
  cfg=json.loads((R/'paper_consistency_20260920'/f'config_{pair}_boundary_figure.json').read_text())
  z=np.load(cfg['bundle']);i=int(np.lexsort((z['starts'],z['re']))[0]);k=int(z['horizon'])-1
  assert str(z['split'][i])=='heldout'
  bases={};sources=[]
  for name,b in cfg['bases'].items():
   v=np.load(b['velocity']);p=np.load(b['pressure']);w=v['point_areas'].astype(float)
   bases[name]=(v['phi_uv'][:b['r_u']].astype(float),v[b.get('mean_u_key','mean_uv_regime')].astype(float),p['phi_p'][:b['r_p']].astype(float),p[b.get('mean_p_key','mean_p_regime')].astype(float),w,v['points'] if 'points' in v else None)
   sources.extend([record(b['velocity']),record(b['pressure'])])
  def recon(a,b,name):
   pu,mu,pp,mp,w,_=bases[name];uv=np.asarray(a,float)@pu+mu;p=gauge(np.asarray(b,float)@pp+mp,w)
   return np.vstack([uv[0::2],uv[1::2],p])
  m=cfg['methods']['t2c'];c1=m['candidate_1_basis'];c2=m['candidate_2_basis']
  fs={'reference':recon(z['truth_a'][i,k],z['truth_b'][i,k],cfg['truth_basis']),
      'candidate1':recon(z['candidate_1_a'][i,k],z['candidate_1_b'][i,k],c1),
      'candidate2':recon(z['candidate_2_a'][i,k],z['candidate_2_b'][i,k],c2)}
  gatepath=SQ/f'gate_{pair}/best.pt';router=SQ/'e2_router/best.pt';ck=torch.load(gatepath,map_location='cpu',weights_only=False)
  assert ck['router_checkpoint_sha256']==sha(router)
  probs=router_probabilities(router,z['re'][i:i+1]);pairids=np.asarray(ck['pair_indices']);assert np.array_equal(pairids,z['pair_indices'][i])
  e2=int(probs[0].argmax());assert e2 in pairids,'Global E2 choice outside pair'
  probpair=probs[:,pairids];probpair/=probpair.sum(1,keepdims=True);base=np.clip(probpair[:,0],1e-6,1-1e-6)
  f=((z['features'][i:i+1]-ck['feature_mean'])/ck['feature_std']).astype('float32')
  gate=ConvexGate(f.shape[1]);gate.load_state_dict(ck['gate_state'],strict=True);gate.eval()
  with torch.inference_mode():alpha=float(gate(torch.as_tensor(f),torch.as_tensor(np.log(base/(1-base)).astype('float32')))[0])
  fs['prediction']=alpha*fs['candidate1']+(1-alpha)*fs['candidate2'];fs['e2']=fs['candidate1' if e2==pairids[0] else 'candidate2'].copy()
  # Independently compare physical quadrature against frozen modal quadratic cache.
  w=bases[cfg['truth_basis']][4];mm=metrics(fs,w);checks={}
  for method,a in [('candidate1',1.),('candidate2',0.),('prediction',alpha),('e2',float(e2==pairids[0]))]:
   for field,key in [('u','Eu'),('p','Ep')]:
    q=z['quad_'+field][i,k];cached=np.sqrt(max((a*a*q[0]+(1-a)**2*q[1]+2*a*(1-a)*q[2])/q[3],0))
    delta=abs(mm[method][key]-cached);assert delta<1e-7,(method,key,delta);checks[method+'_'+key]=float(delta)
  weights=list(csv.DictReader(open(cfg['weights_csv'])));row=next(v for v in weights if abs(float(v['Re'])-float(z['re'][i]))<1e-5 and int(v['start'])==int(z['starts'][i]))
  assert abs(alpha-float(row[cfg['weight_alpha_column']]))<1e-6
  preflight=json.loads((Path(cfg['bundle']).parent/'PREFLIGHT.json').read_text())
  checkpoints=[record(preflight['source_checkpoint']),record(preflight['target_checkpoint'])]
  assert checkpoints[0]['sha256']==preflight['source_checkpoint_sha256'] and checkpoints[1]['sha256']==preflight['target_checkpoint_sha256']
  ts=z['timestamps'][i].astype(float);dt=np.diff(ts);assert np.max(abs(dt-dt[0]))<1e-5
  seed=[]
  for cp in checkpoints:
   obj=torch.load(cp['path'],map_location='cpu',weights_only=False);a=obj.get('args',{})
   if hasattr(a,'__dict__'):a=vars(a)
   if 'centeredsquare_steady' in cp['path']:
    a=json.loads((R/'centeredsquare_steady_specialist_v1/code/training_centeredsquare_steady_rank999.json').read_text())
   seed.append(a.get('seed',obj.get('seed',1248 if 'Hopf' in cp['path'] else None)))
  assert all(s is not None for s in seed)
  vtk=SQ/'paper_boundary_figures_20260730_V1/inputs/centeredSquare_CN09_graded_Re100_internal_final_reference.vtk'
  meta=dict(case='Centered square',geometry='square, D=1, x=[5,6], y=[1.5,2.5]',regime_or_overlap=pair.upper(),Re=float(z['re'][i]),source_field_path=record(cfg['bundle']),reference_path=record(cfg['bundle']),basis_sources=sources,checkpoint_path=checkpoints,
   model_family='T2-C with fixed sparse-MoE local candidates',original_or_revised_init='archived original (July 2026), no September repair',training_seed=seed,E2_checkpoint=record(router),gate_checkpoint=record(gatepath),gate_seed=int(ck['seed']),
   candidate_r=c1,candidate_s=c2,E2_selection=['Steady','Hopf','Periodic'][e2],E2_probabilities=probs[0].tolist(),alpha=alpha,split='heldout',trajectory_id=f'{pair.upper()}_Re{float(z["re"][i])}',window_id=int(z['starts'][i]),window_index=i,t0=float(ts[0]-dt[0]),dt_output=float(dt[0]),k=k+1,K=k+1,snapshot_time=float(ts[k]),duration=float((k+1)*dt[0]),mesh=record(vtk),data_association='cell',crop_extent=[2,17,0,4],recovery='existing frozen rollout coefficients, freshly decoded; E2/gate CPU inference only',quadratic_metric_max_delta=max(checks.values()),metric_checks=checks)
  finalize('square_'+pair,meta,fs,w,bases[cfg['truth_basis']][5])

def circular():
 base=R/'periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2';vp=base/'global_velocity_pod_area_weighted_l2.npz';pp=base/'global_pressure_pod_area_weighted_l2.npz'
 vpz=np.load(vp);ppz=np.load(pp);w=vpz['point_areas'].astype(float);n=len(w)
 source=R/'experiments/round2_20260917/P_evaluation_v2/proposed/1248/modal_rollouts.pt'
 records=torch.load(source,map_location='cpu',weights_only=False);rec=sorted(records,key=lambda x:(x['Re'],x['start']))[0];vals=rec['values'];assert len(vals)==48
 def recon(a,b):
  uv=np.asarray(a,float)@vpz['phi_uv'].astype(float)+vpz['mean_uv_regime'];p=np.asarray(b,float)@ppz['phi_p'].astype(float)+ppz['mean_p_regime'];return np.vstack([uv[:n],uv[n:],gauge(p,w)])
 pa,pb,ta,tb,t=vals[-1];fs={'reference':recon(ta,tb),'prediction':recon(pa,pb)}
 cp=R/'experiments/round2_20260917/P_proposed/1248/Circular_P_proposed_round2_Re_70p314635_checkpoint.pt'
 proto=json.loads((source.parent/'protocol.json').read_text());assert sha(cp)==proto['checkpoint_sha256']
 # Compare to archived endpoint CSV. Native pressure gauge may differ only by roundoff.
 row=next(v for v in csv.DictReader(open(source.parent/'window_step_errors.csv')) if int(v['start'])==rec['start'] and int(v['step'])==48)
 mm=metrics(fs,w);checks={key:abs(mm['prediction'][key]-float(row[key+'_percent'])/100) for key in ['Eu','Ep']};assert max(checks.values())<1e-5,checks
 times=np.asarray([v[-1] for v in vals]);dt=np.diff(np.r_[0,times]);vtk=R/'steady_specialist_v1/expanded_validation_20260723/visualization/Re_43p500000_last_snapshot_uvp.vtk'
 meta=dict(case='Circular cylinder',geometry='circular cylinder, center=(0,0), D=1',regime_or_overlap='Periodic',Re=float(rec['Re']),source_field_path=record(source),reference_path=record(source),basis_sources=[record(vp),record(pp)],checkpoint_path=[record(cp)],model_family='local sparse-MoE Periodic specialist, original initialization',original_or_revised_init='original, round2 P_proposed',training_seed=1248,E2_checkpoint='N/A',gate_checkpoint='N/A',gate_seed='N/A',candidate_r='N/A',candidate_s='N/A',E2_selection='N/A',alpha='N/A',split='heldout',trajectory_id=f'Periodic_Re{rec["Re"]}',window_id=int(rec['start']),t0=float(rec['initial_time']),dt_output=dt.tolist(),k=48,K=48,snapshot_time=float(rec['initial_time']+t),duration=float(t),mesh=record(vtk),data_association='point',crop_extent=[-2,10,-3,3],recovery='existing frozen rollout cache, no model inference',archived_metric_delta=checks)
 finalize('circular_p',meta,fs,w,vpz['points'])

def pinball():
 sys.path.insert(0,str(P/'fluidic_pinball_fusion_v1/code'))
 from evaluate_full_rollout import load_final_test,select_common_starts,descriptors,method_weights
 from build_boundary_cache import specs,load_runtime,run_rollout,target_data,velocity_field,field_statistics,release
 from common import reconstruct,router_probabilities,chain
 seal=json.loads((PB/'full_rollout_20260807_V1/results/FINAL_TEST_SEAL.json').read_text())
 print('seal keys',list(seal),flush=True)
 ss=specs(P);device=torch.device('cuda');torch.set_num_threads(1)
 for pair in ['SH','HP']:
  tstart=time.time();name='Steady' if pair=='SH' else 'Periodic';s=load_runtime(ss[name],device);s.data,audit=load_final_test(s,P)
  starts,res=select_common_starts(s,pair,56,64);idx=int(np.lexsort((starts,res))[0]);starts=starts[idx:idx+1];res=res[idx:idx+1]
  features=np.concatenate([res[:,None].astype('float32'),descriptors(s,starts)],axis=1);weights,ga=method_weights(PB,pair,features,res)
  pa,pb,ta,tb=run_rollout(s,starts,56);h=load_runtime(ss['Hopf'],device);assert np.allclose(s.data['areas'],h.data['areas'],atol=1e-8,rtol=1e-6)
  old=h.data;h.data=target_data(s,h,starts);ha,hb,_,_=run_rollout(h,starts,56);h.data=old
  qu,qp,_,_=field_statistics(pa,pb,ha,hb,ta,tb,s,h)
  w=np.asarray(s.data['areas'],float)
  def recon(a,b,r):return np.vstack([velocity_field(a,r).T,gauge(reconstruct(b,r.data['phi_p'],r.data['mean_p']),w)])
  fs={'reference':recon(ta[0,-1],tb[0,-1],s),'candidate1':recon(pa[0,-1],pb[0,-1],s),'candidate2':recon(ha[0,-1],hb[0,-1],h)}
  alpha=float(weights['T2-C'][0]);probs=router_probabilities(PB/'e2_router/best.pt',res);e2=int(probs[0].argmax());pairids=ss[name].pair_indices;assert e2 in pairids,'Global E2 outside candidate pair'
  fs['prediction']=alpha*fs['candidate1']+(1-alpha)*fs['candidate2'];fs['e2']=fs['candidate1' if e2==pairids[0] else 'candidate2'].copy()
  mm=metrics(fs,w);checks={}
  for method,a in [('candidate1',1.),('candidate2',0.),('prediction',alpha),('e2',float(e2==pairids[0]))]:
   for key,q in [('Eu',qu[0,-1]),('Ep',qp[0,-1])]:
    d=abs(mm[method][key]-np.sqrt(max((a*a*q[0]+(1-a)**2*q[1]+2*a*(1-a)*q[2])/q[3],0)));assert d<1e-7,(method,key,d);checks[method+'_'+key]=float(d)
  ids=[];cur=int(starts[0])
  for _ in range(56):cur=int(s.data['next'][cur]);ids.append(cur)
  ts=s.data['time'][ids].astype(float);t0=float(s.data['time'][starts[0]]);dt=np.diff(np.r_[t0,ts]);assert np.max(abs(dt-dt[0]))<1e-6
  ref=P/'fluidicPinball_v2/eval_pod_coefficients_v2'/name.lower()/'final_test'/f'Re_{res[0]:07.3f}_pod_coefficients.npz'
  # Bind each restored checkpoint to the historical final-test seal (check content, not filenames).
  sealtext=json.dumps(seal)
  cps=[record(ss[name].checkpoint),record(ss['Hopf'].checkpoint),record(PB/'e2_router/best.pt'),record(PB/f'gate_{pair.lower()}/best.pt')]
  assert all(c['sha256'] in sealtext for c in cps),'checkpoint not in final-test seal'
  vtk=P/'fluidicPinball_v2/rom_assets_v2/reference_vtk/fluidicPinball_production_reference.vtk'
  src=np.load(ss[name].coefficient_view);print('pinball asset spatial keys',[(k,src[k].shape) for k in src.files if any(t in k for t in ['point','coord','center'])],flush=True)
  points=next((src[k] for k in ['points','cell_centers','coordinates'] if k in src),np.empty((0,3)))
  meta=dict(case='Fluidic pinball',geometry='three circular cylinders; original finite-volume topology',regime_or_overlap=pair,Re=float(res[0]),source_field_path=record(ref),reference_path=record(ref),basis_sources=[record(ss[name].coefficient_view),record(ss['Hopf'].coefficient_view)],checkpoint_path=cps[:2],model_family='T2-C with Deep-FNN-H3 candidates',original_or_revised_init='archived B1 Deep-FNN-H3; quadratic-branch repair not applicable',training_seed=[1248,1248],E2_checkpoint=cps[2],gate_checkpoint=cps[3],gate_seed=42001,candidate_r=name,candidate_s='Hopf',E2_selection=['Steady','Hopf','Periodic'][e2],E2_probabilities=probs[0].tolist(),alpha=alpha,split='final_test',trajectory_id=f'{pair}_Re{res[0]}',window_id=int(starts[0]),window_index=idx,t0=t0,dt_output=float(dt[0]),k=56,K=56,snapshot_time=float(ts[-1]),duration=float(ts[-1]-t0),mesh=record(vtk),data_association='cell',crop_extent=[-3,9,-3,3],recovery='new frozen inference of ONE preselected K56 window using original evaluator; no retraining or model selection',elapsed_seconds=time.time()-tstart,final_test_seal=record(PB/'full_rollout_20260807_V1/results/FINAL_TEST_SEAL.json'),metric_checks=checks,quadratic_metric_max_delta=max(checks.values()))
  finalize('pinball_'+pair.lower(),meta,fs,w,points)
  release(s);release(h)

{'square':square,'circular':circular,'pinball':pinball}[args.case]()
