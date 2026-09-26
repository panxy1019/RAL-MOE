"""Validate delivered result tables and create a lightweight, checksummed bundle."""
import csv
import hashlib
import json
import math
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parent
status=json.loads((root/'post_training_status.json').read_text())
assert status['status']=='COMPLETED'
assert set(status['completed'])=={'dense','full','global','galerkin'}
checks=[]
for method in ['dense','full','global','galerkin']:
    p=root/f'final_periodic_{method}_v1'
    assert json.loads((p/'COMPLETED.json').read_text())['completed']
    rows=list(csv.DictReader((p/'window_step_errors.csv').open()))
    assert len(rows)==12*48
    assert all(r['finite']=='True' for r in rows)
    assert all(math.isfinite(float(r['Ejoint_percent'])) for r in rows)
    assert all(abs(float(r['Ejoint_percent'])-float(r['Eu_percent'])-float(r['Ep_percent']))<1e-9 for r in rows)
    runtime=json.loads((p/'runtime.json').read_text())
    assert len(runtime['seconds'])==runtime['repeats']==10
    assert runtime['warmups']==3 and all(s>0 for s in runtime['seconds'])
    checks.append(f'{method}: 12 complete K48 windows; joint component identity; 10 timed repeats')
terminal=json.loads((root/'FINAL_PERIODIC_K48.json').read_text())
plotted=json.loads((root/'aligned_rollout_figures/K48_summary.json').read_text())
for r in terminal:
    q=next(t for t in plotted if t['method']==r['method'])
    assert abs(r['Ejoint_percent']-q['Ejoint_percent'])<1e-10
checks.append('Final plotted K48 values match terminal summary')
for name in ['four_method_error_growth','per_Re_physical_time_curves']:
    assert (root/'aligned_rollout_figures'/f'{name}.pdf').stat().st_size>1000
    assert (root/'aligned_rollout_figures'/f'{name}.png').stat().st_size>1000
assert json.loads((root/'GLOBAL_REPLAY_VERIFIED.json').read_text())['passed']
checks.append('Global independent replay passed')
(root/'DELIVERY_VERIFICATION.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))

allowed_dirs={'constant_alpha_v1','global_common_truth_v1','aligned_rollout_figures',
              'final_periodic_dense_v1','final_periodic_full_v1','final_periodic_global_v1','final_periodic_galerkin_v1'}
files=[]
for p in root.iterdir():
    if p.is_file() and p.suffix in {'.py','.md','.tex','.json','.log'} and p.name!='PACKAGE_MANIFEST.json':
        files.append(p)
    elif p.is_dir() and p.name in allowed_dirs:
        files.extend(q for q in p.rglob('*') if q.is_file() and '__pycache__' not in q.parts)
files=sorted(files)
manifest={str(p.relative_to(root)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
mp=root/'PACKAGE_MANIFEST.json'
mp.write_text(json.dumps(manifest,indent=2))
target=root/'RAL_MoE_ROM_experimental_strengthening_completed_20260915.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for p in [*files,mp]:z.write(p,p.relative_to(root).as_posix())
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    for name,meta in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==meta['sha256']
print(json.dumps({'zip':str(target),'files':len(files)+1,'bytes':target.stat().st_size,'integrity':'passed'},indent=2))
