"""Sequential post-training evaluations; never benchmark concurrently with GPU work."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from dense_periodic import ROOT, write

work=ROOT/'experiments/strengthening_20260915'
state=work/'post_training_status.json'
def process_running(pid):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return False
    return raw.rsplit(')',1)[1].split()[0] not in ('Z','X')

def main():
    job=json.loads((work/'dense_job.json').read_text())
    pid=int(job['pid'])
    marker=work/'dense_periodic_v1/TRAINING_COMPLETED.json'
    while not marker.exists() and process_running(pid):
        write(state,dict(status='WAITING_FOR_TRAINING',pid=pid,updated=time.time()))
        time.sleep(30)
    if not marker.exists() or not json.loads(marker.read_text()).get('completed'):
        raise RuntimeError('Dense training terminated without a completion artifact; no test evaluation launched')
    completed=[]
    for method in ['dense','full','global','galerkin']:
        # Refuse contaminated timing: all inference/other training must have finished.
        while True:
            gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if not gpu:break
            write(state,dict(status='WAITING_FOR_IDLE_GPU',gpu_pids=gpu,completed=completed,updated=time.time()))
            time.sleep(30)
        output=work/f'final_periodic_{method}_v1'
        if output.exists():raise FileExistsError(output)
        command=[sys.executable,'-u',str(work/'code/evaluate_periodic_dense.py'),
                 '--method',method,'--global-windows','--benchmark','--output',str(output)]
        if method=='dense':command.extend(['--checkpoint',str(work/'dense_periodic_v1/best_validation.pt')])
        write(state,dict(status='EVALUATING_AND_BENCHMARKING',method=method,command=command,completed=completed,updated=time.time()))
        with (work/f'final_{method}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        completed.append(method)
    subprocess.run([sys.executable,str(work/'code/summarize_finished.py')],cwd=ROOT,check=True)
    write(state,dict(status='COMPLETED',completed=completed,updated=time.time(),
        scope='Circular Periodic dense/full/global/Galerkin aligned K48 and native runtime; remaining plan items must be audited separately'))

if __name__=='__main__':
    try: main()
    except Exception as e:
        write(state,dict(status='FAILED',error=repr(e),updated=time.time()))
        raise
