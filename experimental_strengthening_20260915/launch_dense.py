import json
import os
import subprocess
import sys
from pathlib import Path
root = Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
work = root/'experiments/strengthening_20260915'
output = work/'dense_periodic_v1'
if output.exists():
    raise FileExistsError(output)
output.mkdir()
with (work/'dense_train.log').open('w') as log:
    p = subprocess.Popen([sys.executable, '-u', str(work/'code/dense_periodic.py'), '--output', str(output)],
        cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, env={**os.environ, 'OMP_NUM_THREADS': '4'})
(work/'dense_job.json').write_text(json.dumps({'pid':p.pid,'output':str(output),'log':str(work/'dense_train.log')},indent=2))
print(p.pid)
