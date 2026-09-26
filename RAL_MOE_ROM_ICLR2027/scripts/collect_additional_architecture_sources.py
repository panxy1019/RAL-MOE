"""Fetch read-only selected Pinball candidate metadata and exact source snapshots."""
import json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parents[1];O=P/'build/network_architecture_audit_20260912';O.mkdir(parents=True,exist_ok=True)
code=r'''
import pathlib,json,torch,hashlib
B=pathlib.Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy')
scopes={'pinball_steady':B/'Pinball/steady_b1_rank999_20260806','pinball_hopf':B/'Pinball/hopf_b1_rank999_20260806','pinball_periodic':B/'particalMOE/fluidic_pinball_periodic_v2_b1'}
scopes['pinball_outer']=B/'Pinball/fluidic_pinball_fusion_v1'
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,pathlib.Path):return str(v)
 if hasattr(v,'shape'):return {'shape':list(v.shape),'dtype':str(v.dtype)}
 if v is None or isinstance(v,(str,int,float,bool)):return v
 return str(v)
out={}
for name,root in scopes.items():
 row={'status':'CANDIDATE: publication mapping requires result comparison','root':str(root),'checkpoints':[],'sources':{},'records':{}}
 for p in root.glob('runs/**/'+('best.pt' if name=='pinball_outer' else 'best_validation.pt')):
  if 'smoke' in str(p):continue
  ck=torch.load(p,map_location='cpu',weights_only=False)
  entry={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'metadata':{},'state_dicts':{}}
  for k,v in ck.items():
   if isinstance(v,dict) and v and all(isinstance(t,torch.Tensor) for t in v.values()):entry['state_dicts'][k]={'elements':sum(t.numel() for t in v.values()),'tensors':clean(v)}
   elif k in ('args','config','step','best_step','epoch','best_epoch','norm_stats','validation','best_score','seed','feature_mean','feature_std','pair_indices','router_checkpoint_sha256','cache_sha256'):entry['metadata'][k]=clean(v)
  row['checkpoints'].append(entry);del ck
  conf=p.parent/'config.json'
  if conf.exists():row['records'][str(conf)]=json.loads(conf.read_text())
 for p in (root/'code').glob('*.py'):row['sources'][str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}
 for p in root.glob('*/ROLLOUT_SUMMARY.json'):row['records'][str(p)]=json.loads(p.read_text())
 if name=='pinball_outer':
  for p in root.glob('runs/*/FINAL_FUSION_SUMMARY.json'):row['records'][str(p)]=json.loads(p.read_text())
 out[name]=row
print(json.dumps(out))
'''
c=subprocess.run(['ssh','-o','BatchMode=yes','-p','20381','root@10.10.164.243','/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/.runtime/pt_env/bin/python','-'],input=code,text=True,capture_output=True,timeout=120)
if c.returncode:raise RuntimeError(c.stderr)
d=json.loads(c.stdout)
for n,r in d.items():
 target=O/'sources'/n;target.mkdir(parents=True,exist_ok=True)
 for path,s in r['sources'].items():
  (target/Path(path).name).write_text(s.pop('text'),encoding='utf-8')
 print(n,'checkpoints',len(r['checkpoints']),'sources',len(r['sources']),'records',len(r['records']))
(O/'pinball_candidates.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
