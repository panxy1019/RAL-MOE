"""User-approved fresh three-seed experiments; immutable legacy runs retained."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--regime',choices=['H','P'],required=True)
 p.add_argument('--family',default='quadratic_fixed')
 p.add_argument('--status-tag',default='quadratic_fixed')
 p.add_argument('--fp32',action='store_true')
 a=p.parse_args()
 if a.fp32 and a.regime!='H':raise ValueError('--fp32 is currently defined only for the H repair campaign')
 status=R/f'{a.regime}_{a.status_tag}_status.json'
 if status.exists():raise FileExistsError(status)
 records=[]
 def save():status.write_text(json.dumps(records,indent=2))
 for smoke,seed,kind in [(True,1248,k) for k in ['proposed','structured']]+[(False,s,k) for s in [1248,1600,2026] for k in ['proposed','structured']]:
  output=R/a.family/a.regime/kind/('smoke' if smoke else str(seed))
  while True:
   gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()
   free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
   if len(gpu)<3 and free>10000:break
   time.sleep(20)
  script='hopf_control.py' if a.regime=='H' else 'periodic_proposed.py'
  key='--kind' if a.regime=='H' else '--architecture'
  cmd=[sys.executable,str(R/'code'/script),key,kind,'--seed',str(seed),'--output',str(output),'--quadratic-fix']
  if smoke:cmd+=['--smoke']
  if a.fp32:cmd+=['--fp32']
  record=dict(regime=a.regime,kind=kind,seed=seed,smoke=smoke,fp32=a.fp32,output=str(output),status='running',start=time.time());records.append(record)
  log=R/f'{a.status_tag}_{a.regime}_{kind}_{seed}_{"smoke" if smoke else "formal"}.log'
  with log.open('x') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
   record['pid']=proc.pid;save();rc=proc.wait()
  record.update(status='completed' if rc==0 else 'failed',returncode=rc,end=time.time());save()
  if rc and smoke:raise RuntimeError(str(log))
if __name__=='__main__':main()
