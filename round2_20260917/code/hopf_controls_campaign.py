import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['dense','structured'],required=True);a=p.parse_args()
 tag=a.kind+'_fp32' if a.kind=='dense' else a.kind
 status=R/f'H_{tag}_campaign_status.json'
 records=json.loads(status.read_text()) if status.exists() else []
 def save():
  status.write_text(json.dumps(records,indent=2))
 for seed,smoke in [(1248,True),(1248,False),(1600,False),(2026,False)]:
  if any(r['seed']==seed and r['smoke']==smoke and r['status'] in ['completed','failed'] for r in records):continue
  out=R/(f'H_{tag}_smoke' if smoke else f'H_{tag}/{seed}')
  if smoke and (out/'CONTROL_COMPLETED.json').exists():continue
  cmd=[sys.executable,str(R/'code/hopf_control.py'),'--kind',a.kind,'--seed',str(seed),'--output',str(out)]
  if smoke:cmd+=['--smoke']
  log=R/f'H_{tag}_{seed}_{"smoke" if smoke else "formal"}.log'
  rec=dict(seed=seed,smoke=smoke,status='running',start=time.time(),output=str(out));records.append(rec)
  env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  with log.open('x') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env);rec['pid']=proc.pid;save();rc=proc.wait()
  rec.update(returncode=rc,status='completed' if rc==0 else 'failed',end=time.time());save()
  if rc and smoke:raise RuntimeError(str(log))
  # Formal numerical failures are experimental outcomes, not grounds to drop other seeds.
if __name__=='__main__':main()
