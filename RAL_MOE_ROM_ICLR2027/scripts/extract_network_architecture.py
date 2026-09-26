"""Read-only checkpoint/source audit; writes only versioned local audit artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
OUT = PAPER / 'build/network_architecture_audit_20260912'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PAPER / 'build/parameter_count_runtime'))
sys.path.insert(0, str(ROOT / 'centeredsquare_fusion_v1'))
import torch
from common import ConvexGate, Router

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def clean(v):
    if isinstance(v, dict): return {str(k): clean(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [clean(x) for x in v]
    if isinstance(v, Path): return str(v)
    if isinstance(v, torch.Tensor):
        return {'shape':list(v.shape),'dtype':str(v.dtype)}
    if hasattr(v,'tolist'): return v.tolist()
    if v is None or isinstance(v,(str,int,float,bool)): return v
    return str(v)

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

base=ROOT/'centeredsquare_fusion_v1/results/E2_T2C_K24_20260730_STRICT_V3'
result={'scope':'Checkpoint-extracted facts; unresolved publication identities are not promoted to confirmed paper configurations.','local':{}}
paths={'square_e2':base/'e2_router/best.pt','square_sh_original':base/'gate_sh/best.pt','square_hp_original':base/'gate_hp/best.pt'}
for p in (PAPER/'experiments/iclr_seed_study_20260911/results_v1/centered_square').glob('*/*/seed_*/best.pt'):
    paths['seed_study/'+p.parent.relative_to(PAPER/'experiments/iclr_seed_study_20260911/results_v1/centered_square').as_posix()]=p
for name,path in paths.items():
    ck=torch.load(path,map_location='cpu',weights_only=False)
    key='model_state' if name=='square_e2' else 'gate_state'
    state=ck[key]
    dim=state['net.0.weight' if key=='model_state' else 'correction.net.0.weight'].shape[1]
    model=Router() if key=='model_state' else ConvexGate(dim)
    model.load_state_dict(state,strict=True)
    item={'path':str(path),'sha256':sha(path),'input_dim':dim,'total_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'model_repr':str(model),'state_dict':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in state.items()},'metadata':clean({k:v for k,v in ck.items() if k!=key})}
    config=path.parent/'config.json'
    if config.exists(): item['resolved_config']=json.loads(config.read_text())
    history=path.parent/'history.json'
    if history.exists():
        hist=json.loads(history.read_text()); item['recorded_last_step']=hist[-1].get('step'); item['history_records']=len(hist)
    result['local'][name]=item
    print(name,'first input',dim,'trainable',item['total_trainable'],flush=True)
dump('local_checkpoints.json',result)

inventory=json.loads((ROOT/'iclr_expert_routing_analysis/logs/checkpoint_inventory.json').read_text())
remote_paths={k:v['path'] for k,v in inventory['checkpoints'].items() if k in ('hopf','periodic')}
for boundary in ('sh','hp'):
    pre=json.loads((base/f'cache_{boundary}/PREFLIGHT.json').read_text())
    remote_paths['square_'+('steady' if boundary=='sh' else 'periodic')]=pre['source_checkpoint']
    remote_paths['square_hopf']=pre['target_checkpoint']
remote_code=r'''
import json,hashlib,pathlib,torch,numpy as np
def clean(v):
 if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)): return [clean(x) for x in v]
 if isinstance(v,pathlib.Path): return str(v)
 if isinstance(v,(torch.Tensor,np.ndarray)): return {'shape':list(v.shape),'dtype':str(v.dtype)}
 if isinstance(v,np.generic): return v.item()
 if v is None or isinstance(v,(str,int,float,bool)): return v
 return str(v)
output={}
for name,p in PATHS.items():
 try:
  path=pathlib.Path(p); h=hashlib.sha256()
  with path.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
  ck=torch.load(path,map_location='cpu',weights_only=False)
  row={'path':p,'sha256':h.hexdigest(),'keys':list(ck),'metadata':{},'state_dicts':{}}
  for k,v in ck.items():
   if isinstance(v,dict) and v and all(isinstance(x,torch.Tensor) for x in v.values()):
    row['state_dicts'][k]={'elements':sum(x.numel() for x in v.values()),'tensors':clean(v)}
   elif k not in ('optimizer','optimizer_state','rng','history','gradient_audits','validation','protection','baseline'):
    row['metadata'][k]=clean(v)
   if k in ('optimizer','optimizer_state') and isinstance(v,dict):
    row['optimizer_groups']=[{a:clean(b) for a,b in g.items() if a!='params'} for g in v.get('param_groups',[])]
  output[name]=row
  del ck
 except Exception as e: output[name]={'path':p,'status':'UNRESOLVED','error':str(e)}
print(json.dumps(output))
'''
remote_code='PATHS='+repr(remote_paths)+'\n'+remote_code
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','20381','root@10.10.164.243','/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/.runtime/pt_env/bin/python','-']
completed=subprocess.run(cmd,input=remote_code,text=True,capture_output=True,timeout=150)
if completed.returncode: raise RuntimeError(completed.stderr)
remote=json.loads(completed.stdout)
dump('remote_checkpoints.json',remote)
for name in ('hopf','periodic'):
    assert remote[name]['sha256']==inventory['checkpoints'][name]['sha256']
    for field in ('elements','tensors'):
        assert remote[name]['state_dicts']['model_state'][field]==inventory['checkpoints'][name]['state_dicts']['model_state'][field]
for boundary in ('sh','hp'):
    pre=json.loads((base/f'cache_{boundary}/PREFLIGHT.json').read_text())
    assert remote['square_'+('steady' if boundary=='sh' else 'periodic')]['sha256']==pre['source_checkpoint_sha256']
    assert remote['square_hopf']['sha256']==pre['target_checkpoint_sha256']
dump('remote_checkpoints.json',remote)
print('PASS: five remote checkpoint hashes reverified against evaluation provenance',flush=True)
