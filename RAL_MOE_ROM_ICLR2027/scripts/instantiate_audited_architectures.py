"""Instantiate source classes without training/data loading; verify all frozen shapes."""
import importlib.util, inspect, json, sys
from pathlib import Path
from types import SimpleNamespace, ModuleType
P=Path(__file__).resolve().parents[1]; R=P.parent; O=P/'build/network_architecture_audit_20260912'
sys.path.insert(0,str(P/'build/parameter_count_runtime'))
import torch
torch.set_num_threads(2)
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
d=json.loads((O/'remote_checkpoints.json').read_text())
periodic=load('audit_periodic',R/'iclr_expert_routing_analysis/source/periodic_specialist_r32/code/train_periodic_moe.py')
package=ModuleType('periodic_moe_3090');package.train_periodic_moe=periodic;sys.modules['periodic_moe_3090']=package
result={}
for name,ck in d.items():
 key='model' if name=='square_steady' else 'model_state'; frozen=ck['state_dicts'][key]
 inp=frozen['tensors']['encoder.net.0.weight']['shape'][1]
 args=ck['metadata'].get('args',ck['metadata'].get('config',{}))
 if name in ('hopf','square_hopf'):
  source=R/('iclr_expert_routing_analysis/source/Hopf/migrated_h4_expanded/code/train_hopf_moe_expanded.py' if name=='hopf' else 'fusion_source_snapshot/CenteredSquare_Hopf_H4_20260728/code/train_centeredsquare_hopf_base.py')
  mod=load('audit_'+name,source); rank=32 if name=='hopf' else 11
  cfg=dict(hidden_dim=256,expert_hidden=1024,num_blocks=3,experts=6,top_k=2,expert_blocks=4,quadratic_rank=4,dropout=.04,temperature=.8,adaptive_gate_initial_logit=6.)
  stats={'rhs_mean':torch.zeros(rank),'rhs_scale':torch.ones(rank),'pressure_mean':torch.zeros(rank),'pressure_scale':torch.ones(rank)}
  model=mod.build_model(inp,SimpleNamespace(**cfg),stats,torch.device('cpu'))
 elif name=='square_steady':
  source=R/'fusion_source_snapshot/centeredsquare_steady_specialist_v1/code/train_v16_4_v2_r32_compat.py';mod=load('audit_square_steady',source)
  cfg=dict(args['model']);cfg.update(in_dim=inp,out_dim=args['r_u'],pressure_dim=args['r_p'],num_experts=cfg['experts_per_group'],num_operator_spaces=cfg['shared_experts_per_group'],closure_mode='baseline',pressure_base_mode='static',film_base_hidden=64,film_base_scale=.2,attractor_conditioned=False)
  model=mod.OperatorSpaceMoEROM(**{k:cfg[k] for k in inspect.signature(mod.OperatorSpaceMoEROM).parameters if k in cfg})
  for n,p in model.named_parameters():
   if n.startswith(('velocity_expert_groups','velocity_shared_experts','closure_confidence_head')): p.requires_grad_(False)
 else:
  source=R/('iclr_expert_routing_analysis/source/periodic_specialist_r32/code/train_periodic_moe.py' if name=='periodic' else 'fusion_source_snapshot/centered_square_periodic_v1/code/train_square_periodic_moe_optimized.py')
  mod=periodic if name=='periodic' else load('audit_square_periodic',source)
  cfg=dict(args);cfg.update(in_dim=inp,out_dim=args['r_u'],pressure_dim=args['r_p'],num_operator_spaces=args['num_shared_experts'])
  model=mod.OperatorSpaceMoEROM(**{k:cfg[k] for k in inspect.signature(mod.OperatorSpaceMoEROM).parameters if k in cfg})
 shapes={n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in model.state_dict().items()}
 assert shapes==frozen['tensors'],name
 row={'source':str(source),'shape_check':'PASS','checkpoint_sha256':ck['sha256'],'resolved_constructor_values':cfg,'input_dim':inp,'total_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'nontrainable_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad),'state_dict_elements':sum(v.numel() for v in model.state_dict().values()),'buffers':sum(v.numel() for v in model.buffers()),'modules':{n:str(m) for n,m in model.named_children()},'parameter_shapes':{n:{'shape':list(p.shape),'numel':p.numel(),'requires_grad':p.requires_grad} for n,p in model.named_parameters()}}
 (O/(name+'_model.txt')).write_text(str(model)+'\n',encoding='utf-8')
 result[name]=row
 print(name,inp,row['total_trainable'],row['nontrainable_parameters'],flush=True)
 del model
(O/'instantiated_models.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
pinball=json.loads((O/'pinball_candidates.json').read_text()); extra={}
for name,row in pinball.items():
 if name=='pinball_outer': continue
 source=next((O/'sources'/name).glob('train_*.py'))
 mod=load('audit_'+name,source);ck=row['checkpoints'][0];frozen=ck['state_dicts']['model_state']
 mod.H3_DIM=frozen['tensors']['trunk.net.1.weight']['shape'][1]
 mod.R_U=frozen['tensors']['heads.velocity.2.weight']['shape'][0]
 mod.R_P=frozen['tensors']['heads.pressure.2.weight']['shape'][0]
 model=mod.DeepFNNH3();shapes={n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in model.state_dict().items()}
 assert shapes==frozen['tensors'],name
 extra[name]={'source':str(source),'shape_check':'PASS','input_dim':mod.H3_DIM,'r_u':mod.R_U,'r_p':mod.R_P,'total_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'publication_identity':'CANDIDATE; see numerical provenance discussion','model_repr':str(model),'curriculum_in_source':mod.STAGES}
 (O/(name+'_model.txt')).write_text(str(model)+'\n',encoding='utf-8')
 print(name,extra[name]['total_trainable'],flush=True)
(O/'pinball_instantiated.json').write_text(json.dumps(extra,indent=2)+'\n',encoding='utf-8')
