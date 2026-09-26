import json
import os
from pathlib import Path
import subprocess
import sys
import time
R=Path(__file__).resolve().parents[1]
records=[]
for kind in ['proposed','structured','dense']:
 for seed in [1248,1600,2026]:
  tag='dense_fp32' if kind=='dense' else kind
  train=R/f'H_{tag}'/str(seed)
  while kind!='proposed' and not (train/'CONTROL_COMPLETED.json').exists():
   status=R/f'H_{tag}_campaign_status.json'
   if status.exists() and any(x['seed']==seed and not x['smoke'] and x['status']=='failed' for x in json.loads(status.read_text())):break
   time.sleep(30)
  if kind!='proposed' and not (train/'CONTROL_COMPLETED.json').exists():
   records.append(dict(kind=kind,seed=seed,status='training_failed_no_test'));(R/'H_evaluation_status.json').write_text(json.dumps(records,indent=2));continue
  out=R/'H_evaluation'/kind/str(seed)
  if (out/'COMPLETED.json').exists() or (out/'VALIDATION_REJECTED.json').exists():continue
  rec=dict(kind=kind,seed=seed,status='running',start=time.time());records.append(rec)
  with (R/f'H_eval_{kind}_{seed}_resume.log').open('a') as f:
   p=subprocess.Popen([sys.executable,str(R/'code/evaluate_hopf_round2.py'),'--kind',kind,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
   rec['pid']=p.pid;(R/'H_evaluation_status.json').write_text(json.dumps(records,indent=2));rc=p.wait()
  rec.update(status='completed' if rc==0 else 'failed',returncode=rc,end=time.time());(R/'H_evaluation_status.json').write_text(json.dumps(records,indent=2))
  if rc:raise RuntimeError(str(rec))
