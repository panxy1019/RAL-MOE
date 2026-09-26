import os
from pathlib import Path
import subprocess
import sys
import time
R=Path(__file__).resolve().parents[1]
deadline=time.time()+24*3600
while time.time()<deadline:
 processes=subprocess.check_output(['ps','-eo','args'],text=True)
 # Idle must include Python controllers waiting to launch the next seed/evaluation.
 busy=any(any(tag in line for tag in ['hopf_controls_campaign.py','hopf_evaluation_queue.py','evaluation_queue.py','train_h4_expanded.py','quadratic_fixed_campaign.py','fixed_evaluation_queue.py']) for line in processes.splitlines() if 'python' in line)
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if not busy and not gpu:break
 time.sleep(30)
else:raise TimeoutError('No isolated GPU slot within 24h')
for kind in ['proposed','dense','structured']:
 with (R/f'P_runtime_{kind}.log').open('x') as log:
  subprocess.run([sys.executable,str(R/'code/benchmark_periodic_idle.py'),'--kind',kind],stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
subprocess.run([sys.executable,str(R/'code/scorecard.py')],check=True)
