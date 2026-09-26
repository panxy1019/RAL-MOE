import numpy as np,json,sys
from pathlib import Path
ROOT=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
for rel in ['Hopf/migrated_h4_expanded/assets_r32/expanded_h4_trainval_r32.npz','Hopf/migrated_h4_expanded/trainonly_contract/trainonly_fluctuation_contract.npz','Hopf/artifacts/hopf/velocity_pod_hopf.npz','Hopf/artifacts/hopf/pressure_pod_hopf.npz']:
 p=ROOT/rel;z=np.load(p,allow_pickle=True);print(rel,flush=True)
 for k in z.files:
  v=z[k];print(k,v.shape,str(v.dtype),str(v)[:400] if v.size<20 else '',flush=True)
print((ROOT/'Hopf/migrated_h4_expanded/code/evaluate_h4_expanded.py').read_text(),flush=True)
