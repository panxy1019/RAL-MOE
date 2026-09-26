"""Two durable lanes: proposed P and serial P controls, alongside existing H lane."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/experiments/round2_20260917')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--lane',choices=['proposed','controls'],required=True);a=ap.parse_args()
 architectures=['proposed'] if a.lane=='proposed' else ['dense','structured']
 state=dict(lane=a.lane,policy='one process per lane; at least 8 GiB free before each launch; three total training lanes including H',runs=[],status='running')
 path=R/f'P_{a.lane}_status.json'
 if path.exists():raise FileExistsError(path)
 def save():
  temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));os.replace(temp,path)
 def run(kind,seed,smoke=False):
  out=R/(f'P_{kind}_smoke' if smoke else f'P_{kind}')/str(seed)
  record=dict(architecture=kind,seed=seed,smoke=smoke,status='waiting_resources',output=str(out));state['runs'].append(record);save()
  while True:
   free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])
   if free>=8192:break
   time.sleep(20)
  cmd=[sys.executable,str(R/'code/periodic_proposed.py'),'--seed',str(seed),'--architecture',kind,'--output',str(out)]
  if smoke:cmd+=['--smoke']
  logpath=R/f'P_{kind}_{seed}_{"smoke" if smoke else "formal"}.log'
  env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  with logpath.open('x') as log:
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
   record.update(status='running',pid=proc.pid,start_unix=time.time(),log=str(logpath));save()
   rc=proc.wait()
  record.update(status='completed' if rc==0 else 'failed',returncode=rc,end_unix=time.time());save()
  if rc:raise RuntimeError(f'{kind}/{seed} failed, preserving logs and stopping this lane')
  assert (out/'TRAINING_COMPLETED.json').exists()
 try:
  if a.lane=='proposed':assert (R/'P_smoke/TRAINING_COMPLETED.json').exists()
  else:
   for kind in architectures:run(kind,1248,True)
  for seed in [1248,1600,2026]:
   for kind in architectures:run(kind,seed)
  state['status']='completed';save()
 except Exception as exc:
  state.update(status='failed',error=str(exc));save();raise

if __name__=='__main__':main()
