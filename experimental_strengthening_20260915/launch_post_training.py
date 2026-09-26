import os
import subprocess
import sys
import json
import signal
from dense_periodic import ROOT, write
work=ROOT/'experiments/strengthening_20260915'
manifest=work/'post_training_job.json'
if manifest.exists():
    if '--replace-waiting' not in sys.argv:raise FileExistsError(manifest)
    status=json.loads((work/'post_training_status.json').read_text())
    assert status['status']=='WAITING_FOR_TRAINING'
    previous=json.loads(manifest.read_text())
    pid=int(previous['pid'])
    from pathlib import Path
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
    assert str(work/'code/finish_periodic_campaign.py').encode() in cmd.split(b'\x00')
    os.kill(pid,signal.SIGTERM)
    manifest.rename(work/f'post_training_job_previous_{pid}.json')
with (work/'post_training.log').open('w') as log:
    p=subprocess.Popen([sys.executable,'-u',str(work/'code/finish_periodic_campaign.py')],cwd=ROOT,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write(manifest,dict(pid=p.pid))
print(p.pid)
