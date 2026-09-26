"""Evaluate each accepted FP32 H repair checkpoint after its training run finishes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R=Path(__file__).resolve().parents[1]
STATUS=R/'H_quadratic_fixed_fp32_status.json'
OUT_STATUS=R/'H_fp32_evaluation_status.json'
done=set()
records=[]

def save():
 OUT_STATUS.write_text(json.dumps(records,indent=2))

while len(done)<6:
 if not STATUS.exists():
  time.sleep(30);continue
 try:
  runs=json.loads(STATUS.read_text())
 except json.JSONDecodeError:
  time.sleep(5);continue
 if any(r['smoke'] and r['status']=='failed' for r in runs):
  raise RuntimeError('FP32 H smoke failed; evaluation queue stopped')
 progressed=False
 for run in runs:
  key=(run['kind'],run['seed'])
  if run['smoke'] or key in done or run['status']=='running':continue
  if run['status']=='failed':
   records.append(dict(regime='H',kind=key[0],seed=key[1],status='training_failed_no_test'))
   done.add(key);save();progressed=True;continue
  rec=dict(regime='H',kind=key[0],seed=key[1],status='evaluating');records.append(rec);save()
  cmd=[sys.executable,str(R/'code/evaluate_hopf_round2.py'),'--kind',key[0],'--seed',str(key[1]),
       '--quadratic-fixed','--checkpoint-family','quadratic_fixed_fp32',
       '--evaluation-family','H_evaluation_fixed_fp32']
  log=R/f'H_fp32_eval_{key[0]}_{key[1]}.log'
  with log.open('x') as f:
   rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,
      env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')).returncode
  rec.update(status='completed' if rc==0 else 'failed',returncode=rc);done.add(key);save();progressed=True
  if rc:raise RuntimeError(f'Evaluation failed: {key}; see {log}')
 if not progressed:time.sleep(30)

subprocess.run([sys.executable,str(R/'code/scorecard.py'),'--fixed',
 '--h-evaluation-family','H_evaluation_fixed_fp32','--output-suffix','_fixed_fp32'],check=True)
