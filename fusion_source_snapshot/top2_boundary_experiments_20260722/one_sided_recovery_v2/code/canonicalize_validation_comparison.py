"""Replace only the noncanonical oracle row in the frozen validation comparison."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np

def module(path):
    s=importlib.util.spec_from_file_location('canonical_routes',path); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); return m

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--comparison',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    r=module(Path(__file__).with_name('train_sh_routes.py')); x=json.loads(a.comparison.read_text())
    with np.load(a.cache,allow_pickle=False) as z: d={k:z[k] for k in z.files}
    mask=d['split']=='validation'; qu,qp=d['quad_u'][mask],d['quad_p'][mask]; grid=np.linspace(0,1,10001)[:,None]; alpha=[]
    for i in range(len(qu)):
        eu=(grid*grid*qu[i,:,0]+(1-grid)**2*qu[i,:,1]+2*grid*(1-grid)*qu[i,:,2])/np.maximum(qu[i,:,3],1e-12); ep=(grid*grid*qp[i,:,0]+(1-grid)**2*qp[i,:,1]+2*grid*(1-grid)*qp[i,:,2])/np.maximum(qp[i,:,3],1e-12)
        alpha.append(float(grid[np.argmin(np.mean(np.sqrt(np.maximum(eu,0))+np.sqrt(np.maximum(ep,0)),axis=1)),0]))
    x['baselines']['per_window_convex_oracle']=r.metrics(np.asarray(alpha,np.float32),qu,qp,d['re'][mask]); x['canonical_note']='Original VALIDATION_COMPARISON.json is retained, but its oracle row used a non-metric-matched diagnostic. Route and non-oracle rows were unchanged.'
    a.output.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False))

if __name__=='__main__': main()
