import json
import os
from pathlib import Path
import subprocess
import sys
import time
R=Path(__file__).resolve().parents[1];done=set();records=[]
while len(done)<12:
 progressed=False
 for reg in ['H','P']:
  status=R/f'{reg}_quadratic_fixed_status.json'
  if not status.exists():continue
  try:runs=json.loads(status.read_text())
  except json.JSONDecodeError:continue
  if any(r['smoke'] and r['status']=='failed' for r in runs):raise RuntimeError(f'{reg} smoke failed; abort fixed evaluation queue')
  for run in runs:
   key=(reg,run['kind'],run['seed'])
   if run['smoke'] or key in done or run['status']=='running':continue
   if run['status']=='failed':
    records.append(dict(regime=reg,kind=key[1],seed=key[2],status='training_failed_no_test'));done.add(key);continue
   rec=dict(regime=reg,kind=key[1],seed=key[2],status='evaluating');records.append(rec)
   script='evaluate_hopf_round2.py' if reg=='H' else 'evaluate_periodic_round2.py'
   with (R/f'fixed_eval_{reg}_{key[1]}_{key[2]}.log').open('x') as log:
    rc=subprocess.run([sys.executable,str(R/'code'/script),'--kind',key[1],'--seed',str(key[2]),'--quadratic-fixed'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')).returncode
   rec.update(status='completed' if rc==0 else 'failed',returncode=rc);done.add(key);progressed=True
   (R/'fixed_evaluation_status.json').write_text(json.dumps(records,indent=2))
   if rc:raise RuntimeError(str(rec))
 if not progressed:time.sleep(30)
subprocess.run([sys.executable,str(R/'code/scorecard.py'),'--fixed'],check=True)
