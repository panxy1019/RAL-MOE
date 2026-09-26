"""Independent numerical tests for the algebra used in the OpInf adapter."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve
rng=np.random.default_rng(918)
x=rng.normal(size=(1000,4));r=rng.normal(size=(1000,1));i,j=np.triu_indices(4)
F=np.column_stack([np.ones(len(x)),x,x[:,i]*x[:,j],r,r*x])
W=rng.normal(size=(F.shape[1],4));target=F@W
scale=np.sqrt((F*F).mean(0));Z=F/scale
fit=solve(Z.T@Z/len(x)+1e-12*np.eye(Z.shape[1]),Z.T@target/len(x),assume_a='pos')/scale[:,None]
err=float(np.max(np.abs(fit-W)));assert err<1e-8
# RK4's expected refinement behavior on y'=-y.
def integrate(n):
 y=1.;h=1/n
 for _ in range(n):
  k1=-y;k2=-(y+h*k1/2);k3=-(y+h*k2/2);k4=-(y+h*k3);y+=h*(k1+2*k2+2*k3+k4)/6
 return abs(y-np.exp(-1))
ratio=integrate(8)/integrate(16);assert 14<ratio<19
# Blocked [all ux; all uy] weights must reproduce direct two-component integration.
w=rng.uniform(.1,1,7);u=rng.normal(size=(7,2));blocked=np.r_[u[:,0],u[:,1]]
assert np.isclose(np.sum(blocked**2*np.r_[w,w]),np.sum(u*u*w[:,None]))
result=dict(passed=True,regression_max_error=err,RK4_error_refinement_ratio=ratio,blocked_velocity_weighting=True)
(Path(__file__).parents[1]/'opinf_algebra_tests.json').write_text(json.dumps(result,indent=2));print(result)
