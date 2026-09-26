import json
import os
from pathlib import Path
import subprocess
import sys
import time
R=Path(__file__).resolve().parents[1]
records=[]
for kind in ['proposed','dense','structured']:
 for seed in [1248,1600,2026]:
  out=R/'P_evaluation_v2'/kind/str(seed)
  if (out/'COMPLETED.json').exists():continue
  rec=dict(kind=kind,seed=seed,status='running',start=time.time());records.append(rec)
  with (R/f'P_eval_v2_{kind}_{seed}.log').open('x') as f:
   p=subprocess.Popen([sys.executable,str(R/'code/evaluate_periodic_round2.py'),'--kind',kind,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
   rec['pid']=p.pid;(R/'P_evaluation_status.json').write_text(json.dumps(records,indent=2));rc=p.wait()
  rec.update(status='completed' if rc==0 else 'failed',returncode=rc,end=time.time());(R/'P_evaluation_status.json').write_text(json.dumps(records,indent=2))
  if rc:raise RuntimeError(str(rec))
