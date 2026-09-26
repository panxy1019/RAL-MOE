"""Wait for verified inputs, then complete the frozen pipeline; never overwrite old runs."""
from common import *
import subprocess,zipfile,sys
manifest=json.loads((OUT/'split_manifest.json').read_text())['P'];FOUT=OUT/'fno_P';FOUT.mkdir(exist_ok=True)
start=time.time()
while True:
 missing=[]
 for tr in manifest['trajectories']:
  if tr['split']=='heldout':continue
  path=OUT/('raw_training' if tr['split']=='train' else 'raw_validation')/f"{tr['label']}_uvp_pointData.npz"
  if not path.exists() or not zipfile.is_zipfile(path):missing.append(tr['label'])
 if not missing:break
 write(FOUT/'input_wait_status.json',dict(status='waiting_for_raw_training_fields',missing=missing,count=len(missing),updated=time.strftime('%FT%T%z')))
 if time.time()-start>7200:raise RuntimeError('Input wait exceeded2hours; inspect download status, no training launched')
 time.sleep(30)
write(FOUT/'input_wait_status.json',dict(status='all59_development_raw_files_available',updated=time.strftime('%FT%T%z')))
subprocess.run([sys.executable,str(OUT/'code/fno_experiment.py'),'campaign'],check=True)
if (FOUT/'PREDICTIONS_COMPLETED.json').exists():subprocess.run([sys.executable,str(OUT/'code/finalize_fno.py')],check=True)
