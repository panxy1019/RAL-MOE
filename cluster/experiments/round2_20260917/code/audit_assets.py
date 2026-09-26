import hashlib
import json
from pathlib import Path
import numpy as np

R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
paths=[R/'periodic_specialist_r32/assets/pressure_poisson_surrogate_periodic.npz',
 R/'Hopf/migrated_h4_expanded/assets_r32/expanded_h4_trainonly_pressure_r32.npz',
 R/'Hopf/migrated_h4_expanded/assets_r32/expanded_h4_trainval_r32.npz']
report=[]
for path in paths:
 z=np.load(path,allow_pickle=False)
 fields={}
 for k in z.files:
  a=z[k];item=dict(shape=list(a.shape),dtype=str(a.dtype))
  if a.size<=10:item['value']=a.tolist()
  fields[k]=item
 rec=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fields=fields)
 report.append(rec)
out=R/'experiments/round2_20260917/asset_audit.json'
out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
