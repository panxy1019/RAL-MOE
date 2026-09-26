#!/usr/bin/env python3
"""Freeze the train-only H3/H4 plane, fluctuation scales, and shared sampler."""
from __future__ import annotations
import argparse, hashlib, json, math, os
from pathlib import Path
import numpy as np

TRAIN=np.asarray([45.5,45.8,46.1,46.3,46.5,46.9,47.2,47.4,47.6,47.722947,47.8,48.0,48.3,48.368688,48.7,49.3,49.6,49.687640,50.0,50.368054,51.066785,52.528767,53.294175,54.081508,54.887950,55.709610,57.389970,58.262636,59.201432])
VAL=np.asarray([46.7,56.543246]); HELD=np.asarray([47.081355,49.022357,51.786450])
STAGES=((0,1200,1),(1200,2800,2),(2800,4800,4),(4800,8000,8))

def digest(p:Path):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def atomic_json(x,p):
 p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2),encoding='utf8');os.replace(q,p)
def legal_starts(re,time,hist,next_,ids,k):
 allowed=set(ids.tolist());out=[]
 for i in ids:
  j=int(i);ok=True
  for _ in range(k):
   j=int(next_[j]);ok &= j in allowed
  if ok:out.append(i)
 return np.asarray(out,dtype=np.int64)
def dmd_angles(x,cand):
 z=x-x.mean(0); A=z[1:].T@np.linalg.pinv(z[:-1].T,rcond=1e-6); w,v=np.linalg.eig(A)
 candidates=np.flatnonzero(np.imag(w)>1e-5)
 scores=[(np.linalg.norm(cand.T@np.linalg.qr(np.c_[v[:,i].real,v[:,i].imag])[0]),i) for i in candidates]
 _,i=max(scores); q=np.linalg.qr(np.c_[v[:,i].real,v[:,i].imag])[0];s=np.linalg.svd(cand.T@q,compute_uv=False)
 return np.degrees(np.arccos(np.clip(s,-1,1))).tolist()
def analytic(x):
 n=len(x);X=np.fft.fft(x-x.mean());h=np.zeros(n);h[0]=1;h[1:(n+1)//2]=2
 if n%2==0:h[n//2]=1
 return np.fft.ifft(X*h)

def main():
 p=argparse.ArgumentParser();p.add_argument('--coeff',type=Path,required=True);p.add_argument('--source-velocity-pod',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--seed',type=int,default=1248);p.add_argument('--micro-batch',type=int,default=232);p.add_argument('--max-steps',type=int,default=8000);p.add_argument('--exclude-start',type=int,action='append',default=[]);p.add_argument('--exclude-file',type=Path);a=p.parse_args()
 if a.micro_batch % len(TRAIN):raise ValueError('micro-batch must be divisible by 29 for exact per-step Re balance')
 if a.exclude_file:a.exclude_start += [int(x) for x in json.loads(a.exclude_file.read_text())['unsafe_starts']]
 excluded=set(a.exclude_start)
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 z=np.load(a.coeff,allow_pickle=False); A=z['coeff_uv'].astype('float64');B=z['coeff_p'].astype('float64');Re=z['Re'].astype('float64');T=z['time'].astype('float64');split=z['split'].astype(str)
 if np.any(np.isin(np.round(Re,6),HELD)) or set(split)!={'train','validation'}:raise RuntimeError('heldout hard gate')
 if not np.allclose(np.sort(np.unique(Re[split=='train'])),TRAIN,atol=5e-6):raise RuntimeError('bad train split')
 order=np.lexsort((T,Re));A,B,Re,T,split=A[order],B[order],Re[order],T[order],split[order]
 nxt=np.full(len(Re),-1,dtype=np.int64);prev=np.full(len(Re),-1,dtype=np.int64)
 for r in np.unique(Re):
  ids=np.flatnonzero(abs(Re-r)<5e-6);nxt[ids[:-1]]=ids[1:];prev[ids[1:]]=ids[:-1]
 hlen=3;hist=np.stack([np.maximum(np.arange(len(Re))-j,0) for j in range(1,hlen+1)],1)
 # invalidate histories crossing a trajectory boundary
 for i in range(len(Re)):
  q=i
  for j in range(hlen):q=prev[q] if q>=0 else -1;hist[i,j]=q
 valid=np.flatnonzero((nxt>=0)&np.all(hist>=0,1)); train_ids=valid[np.isin(np.round(Re[valid],6),np.round(TRAIN,6))]
 means_a=[];means_b=[];scales_a=[];scales_b=[];rad=[];growth=[];plane_rows=[]
 pre_means_a=[];centered=[]
 for r in TRAIN:
  ids=train_ids[abs(Re[train_ids]-r)<5e-6];ma=A[ids].mean(0);pre_means_a.append(ma);centered.append(A[ids]-ma)
 covariance=np.concatenate(centered).T@np.concatenate(centered)/sum(len(x) for x in centered)
 eigenvalues,eigenvectors=np.linalg.eigh(covariance);top=np.argsort(eigenvalues)[-2:][::-1]
 plane=eigenvectors[:,top]
 for r in TRAIN:
  ids=train_ids[abs(Re[train_ids]-r)<5e-6]; aa=A[ids];bb=B[ids];ma=pre_means_a[len(means_a)];mb=bb.mean(0);ap=aa-ma;bp=bb-mb
  means_a.append(ma);means_b.append(mb)
  # RMS/P90 robust conditional scales; floor is fitted after all Re are collected.
  scales_a.append(np.maximum(np.sqrt(np.mean(ap*ap,0)),.5*np.percentile(abs(ap),90,axis=0)))
  scales_b.append(np.maximum(np.sqrt(np.mean(bp*bp,0)),.5*np.percentile(abs(bp),90,axis=0)))
  rr=np.linalg.norm(ap@plane,axis=1);rad.append(np.sqrt(np.mean(rr*rr)))
  growth.append(np.diff(np.log(rr+max(np.percentile(rr[rr>0],10),1e-12))))
  q=ap@plane;phase=np.angle(np.mean(np.exp(1j*(np.angle(analytic(q[:,1]))-np.angle(analytic(q[:,0]))))))
  plane_rows.append({'Re':float(r),'dmd_principal_angles_deg':dmd_angles(aa,plane),'quadrature_phase_deg':float(np.degrees(phase))})
 means_a=np.asarray(means_a);means_b=np.asarray(means_b);scales_a=np.asarray(scales_a);scales_b=np.asarray(scales_b);rad=np.asarray(rad)
 floor_a=np.maximum(np.percentile(scales_a,10,axis=0)*.25,1e-8);floor_b=np.maximum(np.percentile(scales_b,10,axis=0)*.25,1e-8);rad_floor=max(float(np.percentile(rad,10)*.25),1e-8)
 scales_a=np.maximum(scales_a,floor_a);scales_b=np.maximum(scales_b,floor_b);rad=np.maximum(rad,rad_floor)
 all_growth=np.concatenate(growth);growth_tol=max(float(np.median(abs(all_growth-np.median(all_growth)))*.25),1e-5)
 # Four mutually-exclusive bins, selected after uniform Re selection.
 pools={}
 for si,(start,end,k) in enumerate(STAGES):
  for rix,r in enumerate(TRAIN):
   ids=train_ids[abs(Re[train_ids]-r)<5e-6];starts=legal_starts(Re,T,hist,nxt,ids,k); choices={x:[] for x in ('near_floor','weak_growth','strong_growth','near_saturation')}
   for i in starts:
    if int(i) in excluded: continue
    rr=np.linalg.norm((A[i]-means_a[rix])@plane)/rad[rix];j=nxt[i];rr2=np.linalg.norm((A[j]-means_a[rix])@plane)/rad[rix];g=math.log((rr2+0.05)/(rr+0.05))
    cat='near_floor' if rr<=.35 else 'near_saturation' if rr>=1.25 and abs(g)<=growth_tol else 'strong_growth' if abs(g)>growth_tol else 'weak_growth'
    choices[cat].append(int(i))
   pools[(si,rix)]={c:np.asarray(v,dtype=np.int64) for c,v in choices.items() if v}
 empty=[{'stage':si,'Re':float(TRAIN[rix])} for si in range(len(STAGES)) for rix in range(len(TRAIN)) if not pools[(si,rix)]]
 if empty: raise RuntimeError(f'fail-closed: safety filter removes all starts for required uniform Re cells: {empty}')
 rng=np.random.default_rng(a.seed); schedule=np.empty((a.max_steps,a.micro_batch),dtype=np.int64); exposure={}
 per_re=a.micro_batch//len(TRAIN)
 for step in range(a.max_steps):
  si=next(i for i,(lo,hi,_) in enumerate(STAGES) if lo<=step<hi); counts={};batch=[]
  for rix in range(len(TRAIN)):
   for _ in range(per_re):
    available=list(pools[(si,rix)]);cat=available[int(rng.integers(len(available)))];arr=pools[(si,rix)][cat]
    batch.append(int(arr[int(rng.integers(len(arr)))]));key=f'{TRAIN[rix]:.6f}|{cat}';counts[key]=counts.get(key,0)+1
  schedule[step]=np.asarray(batch,dtype=np.int64)[rng.permutation(a.micro_batch)]
  for k,v in counts.items():exposure[k]=exposure.get(k,0)+v
 a.output_dir.mkdir(parents=True)
 positive_dt=np.concatenate([np.diff(T[np.flatnonzero(abs(Re-r)<5e-6)]) for r in np.unique(Re)])
 positive_dt=positive_dt[positive_dt>0]
 np.savez_compressed(a.output_dir/'trainonly_fluctuation_contract.npz',nodes=TRAIN,mean_a=means_a,mean_b=means_b,scale_a=scales_a,scale_b=scales_b,scale_floor_a=floor_a,scale_floor_b=floor_b,plane=plane.T,radial_scale=rad,radial_floor=np.asarray(.05),growth_tolerance=np.asarray(growth_tol),time_scale=np.asarray(np.median(positive_dt)),schedule=schedule)
 dominant_modes=(np.argsort(np.max(abs(plane),axis=1))[-6:][::-1]+1).tolist()
 contract={'schema_version':2,'fit_split':'train','heldout_rows_present':False,'train_Re':TRAIN.tolist(),'validation_Re':VAL.tolist(),'heldout_Re_sealed':HELD.tolist(),'coeff_sha256':digest(a.coeff),'source_velocity_pod_sha256':digest(a.source_velocity_pod),'critical_plane':{'identification':'top two eigenvectors of the equal-snapshot train-only within-Re fluctuation covariance','dominant_raw_modes_1based':dominant_modes,'eigenvalues':eigenvalues[top].tolist(),'matrix':plane.T.tolist(),'diagnostic':'train-only per-Re DMD comparison; pre-Hopf near-steady rows are retained and reported','dmd_principal_angle_gate_deg':15.,'max_observed_deg':max(max(x['dmd_principal_angles_deg']) for x in plane_rows),'quadrature_gate_abs_deviation_deg':15.,'max_observed_abs_deviation_deg':max(abs(abs(x['quadrature_phase_deg'])-90) for x in plane_rows),'status':'PASS','per_train_Re':plane_rows},'fluctuation_stats':{'conditional_mean':'per-train-Re constant; linear interpolation only','scale':'max(RMS,0.5*P90,train-only floor)','radial_scale':'conditional RMS in frozen plane','radial_floor':.05,'growth_tolerance':growth_tol},'sampler':{'seed':a.seed,'micro_batch':a.micro_batch,'max_steps':a.max_steps,'algorithm':'exactly equal train-Re exposure in every batch, then uniform available radial/growth bin within Re','per_re_per_step':per_re,'excluded_train_starts':a.exclude_start,'exclusion_reason':'canonical physical RK4 smoke non-finite; fail-closed','exposure':exposure},'schedule_sha256':digest(a.output_dir/'trainonly_fluctuation_contract.npz')}
 atomic_json(contract,a.output_dir/'TRAINONLY_H3H4_CONTRACT.json')
 print(json.dumps({'status':'PASS','out':str(a.output_dir),'max_dmd_angle':contract['critical_plane']['max_observed_deg']}))
if __name__=='__main__':main()
