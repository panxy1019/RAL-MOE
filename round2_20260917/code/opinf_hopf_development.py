"""Train/validation-only quadratic OpInf pilot; no held-out scores or FOM claims.

Adaptation of polynomial least-squares operator inference (Peherstorfer & Willcox,
2016). Standardized velocity state, affine inverse-Re linear contribution,
compact quadratic monomials, ridge regression, algebraic fitted pressure output.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import solve

R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
OUT=R/'experiments/round2_20260917/OpInf_H_development_v2'
DATA=R/'Hopf/migrated_h4_expanded/assets_r32/expanded_h4_trainval_r32.npz'

def main():
 if OUT.exists():
  trials=json.loads((OUT/'validation_trials.json').read_text())
  if len(trials)!=13:raise RuntimeError('Incomplete directory: inspect before resuming')
  scores=[v['score'] for v in trials if v['score'] is not None]
  (OUT/'status.json').write_text(json.dumps(dict(status='completed_development',selected_available=bool(scores),best_validation_score=min(scores) if scores else None,heldout_evaluated=False),indent=2))
  return
 OUT.mkdir(exist_ok=False)
 z=np.load(DATA,allow_pickle=False)
 a,b,t,re,split=[z[k] for k in ['coeff_uv','coeff_p','time','Re','split']]
 assert set(split)=={'train','validation'}
 tr=split=='train';va=split=='validation'
 assert not set(re[tr])&set(re[va])
 mean=a[tr].mean(0);scale=np.maximum(a[tr].std(0),1e-8)
 pm=b[tr].mean(0);ps=np.maximum(b[tr].std(0),1e-8)
 inv=1/re;rm=inv[tr].mean();rs=inv[tr].std()
 x=(a-mean)/scale;mu=(inv-rm)/rs
 ii,jj=np.triu_indices(a.shape[1])
 def features(x,m):
  x=np.atleast_2d(x);m=np.broadcast_to(np.asarray(m).reshape(-1),(len(x),))[:,None]
  return np.column_stack([np.ones(len(x)),x,x[:,ii]*x[:,jj],m,m*x])
 X=features(x,mu);xs=np.maximum(np.sqrt((X[tr]**2).mean(0)),1e-8)
 # Temporal differentiation strictly within each training trajectory, irregular-time second-order formula.
 derivative=np.full_like(x,np.nan,dtype=float)
 tracks={}
 for rv in np.unique(re):
  ids=np.flatnonzero(re==rv);ids=ids[np.argsort(t[ids])]
  assert np.all(np.diff(t[ids])>0)
  tracks[float(rv)]=ids
  if tr[ids].all(): derivative[ids]=np.gradient(x[ids],t[ids],axis=0,edge_order=2)
 assert np.isfinite(derivative[tr]).all()
 Xt=X[tr]/xs
 gram=Xt.T@Xt/len(Xt)
 rhs=Xt.T@derivative[tr]/len(Xt)
 rhs_p=Xt.T@((b[tr]-pm)/ps)/len(Xt)
 # Eight deterministic, equally spaced validation windows per Re, native observation times.
 windows=[]
 for rv,ids in tracks.items():
  if va[ids].all():
   eligible=np.arange(2,len(ids)-24)
   assert len(eligible)>0
   for start in np.unique(np.linspace(eligible[0],eligible[-1],8,dtype=int)):
    windows.append(ids[start:start+25])
 protocol=dict(status='development_only',data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
   source='https://kiwi.oden.utexas.edu/papers/Non-intrusive-model-reduction-Peherstorfer-Willcox.pdf',
   dynamics_features='1,x,compact_quadratic(x),standardized_inverse_Re,standardized_inverse_Re*x',
   pressure='ridge-fitted algebraic output on the same polynomial features; adaptation, not the neural PPE correction',
   derivative='np.gradient edge_order=2 on each training trajectory only',
   ridge_objective='mean squared residual + lambda * Frobenius norm squared, all feature columns penalized after RMS scaling',
   lambda_grid=np.logspace(-10,2,13).tolist(),validation_windows=len(windows),
   selector='macro-Re mean of all-step velocity and pressure relative POD-field errors over K24; any failed window invalidates candidate',
   truth='POD reconstructed fields, not FOM; final heldout FOM evaluation still required',
   integration='RK4, four equal substeps per output interval; fixed before validation',
   training_Re=np.unique(re[tr]).tolist(),validation_Re=np.unique(re[va]).tolist())
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 # Exact POD-field norm, including means; no full field arrays inside rollout loop.
 w=z['point_areas'];wu=np.concatenate([w,w])
 pu,pp=z['phi_uv'],z['phi_p'];um=z['mean_uv_train'];bp=z['mean_p_train']
 gu=(pu*wu)@pu.T;gp=(pp*w)@pp.T
 vu=(pu*wu)@um;vp=(pp*w)@bp;nu=float((um*um*wu).sum());np0=float((bp*bp*w).sum())
 def err(pred,true,g,v,n):
  d=pred-true
  return np.sqrt(max(float(d@g@d),0)/max(float(true@g@true+2*true@v+n),1e-12))
 trials=[];best=float('inf');start_time=time.time()
 for lam in protocol['lambda_grid']:
  matrix=gram+lam*np.eye(len(gram))
  W=solve(matrix,rhs,assume_a='pos')/xs[:,None]
  V=solve(matrix,rhs_p,assume_a='pos')/xs[:,None]
  scores=[];failed=0
  for ids in windows:
   state=x[ids[0]].copy();m=mu[ids[0]];vals=[]
   def f(s):return (features(s,m)@W)[0]
   with np.errstate(over='ignore',invalid='ignore'):
    for old,new in zip(ids[:-1],ids[1:]):
     dt=(t[new]-t[old])/4
     for _ in range(4):
      k1=f(state);k2=f(state+dt*k1/2);k3=f(state+dt*k2/2);k4=f(state+dt*k3)
      state=state+dt*(k1+2*k2+2*k3+k4)/6
      if not np.isfinite(state).all() or np.max(np.abs(state))>1e6:break
     if not np.isfinite(state).all() or np.max(np.abs(state))>1e6:break
     pressure=(features(state,m)@V)[0]*ps+pm
     vals.append(err(state*scale+mean,a[new],gu,vu,nu)+err(pressure,b[new],gp,vp,np0))
   if len(vals)!=24 or not np.isfinite(vals).all():failed+=1
   else:scores.append((float(re[ids[0]]),float(np.mean(vals))))
  score=float(np.mean([np.mean([s for r,s in scores if r==rv]) for rv in np.unique(re[va])])) if failed==0 else float('inf')
  trial=dict(lambda_value=lam,score=score if np.isfinite(score) else None,failed_windows=failed,elapsed=time.time()-start_time)
  trials.append(trial);print(json.dumps(trial),flush=True)
  (OUT/'validation_trials.json').write_text(json.dumps(trials,indent=2))
  if score<best:
   best=score
   np.savez(OUT/'validation_selected.npz',W=W,V=V,mean=mean,scale=scale,pressure_mean=pm,pressure_scale=ps,inv_Re_mean=rm,inv_Re_scale=rs,lambda_value=lam,ii=ii,jj=jj)
 (OUT/'status.json').write_text(json.dumps(dict(status='completed_development',selected_available=bool(np.isfinite(best)),best_validation_score=best if np.isfinite(best) else None,heldout_evaluated=False),indent=2))

if __name__=='__main__':main()
