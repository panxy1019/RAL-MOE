"""Active-capacity controls for the verified Circular Periodic configuration."""
import inspect
import json
from pathlib import Path
import types
import torch
from torch import nn

ROOT=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE')
TARGET=8308959  # Verified per-sample inference-path count of frozen Periodic model.

def patch(trainer,kind,output):
 source=ROOT/'experiments/strengthening_20260915/code/dense_periodic.py'
 text=source.read_text()
 old='target = sum(p.numel() for p in self.parameters() if p.requires_grad)'
 assert text.count(old)==1
 text=text.replace(old,f'target = {TARGET}')
 text=text.replace('reference_total_trainable=target','reference_active_trainable=target')
 text=text.replace('total allocated trainable parameters, NOT active parameters',
                   'reference per-sample active trainable parameters; not equal FLOPs or latency')
 module=types.ModuleType('round2_dense_control');module.__file__=str(source)
 exec(compile(text,str(source),'exec'),module.__dict__)
 original=trainer.OperatorSpaceMoEROM
 Dense=module.patch(trainer,output/'architecture.json')
 if kind=='dense':return
 assert kind=='structured'
 class Structured(Dense):
  def __init__(self,*args,**kwargs):
   cfg=inspect.signature(original.__init__).bind(None,*args,**kwargs);cfg.apply_defaults();cfg=cfg.arguments
   super().__init__(*args,**kwargs)
   del self.dense_correction
   common=sum(p.numel() for p in self.parameters())
   def pair(width):
    options=(width,cfg['expert_blocks'],cfg['quadratic_rank'],cfg['quadratic_scale'],cfg['dropout'])
    return nn.ModuleList([trainer.PhysicsAwareExpert(cfg['hidden_dim'],cfg['out_dim'],cfg['out_dim'],*options),
      trainer.PhysicsAwareExpert(cfg['hidden_dim'],cfg['out_dim']+cfg['pressure_dim'],cfg['pressure_dim'],*options)])
   # Parameter count is affine in expansion width; count without allocating a search grid.
   c16=sum(p.numel() for p in pair(16).parameters());c17=sum(p.numel() for p in pair(17).parameters())
   width=max(16,round(16+(TARGET-common-c16)/(c17-c16)))
   self.structured_correction=pair(width)
   actual=sum(p.numel() for p in self.parameters())
   assert abs(actual-TARGET)/TARGET<.005
   (output/'architecture.json').write_text(json.dumps(dict(kind='single nonrouted structured correction per channel',
      target_active=TARGET,total_trainable=actual,expert_hidden=width,common_parameters=common,
      matching='reference per-sample active parameters, not FLOPs or runtime',
      removed='all learned routers, shared/routed expert banks',
      initialization_note='Native zero initialization sets both quadratic factors to zero; this branch requires a separate effective-gradient audit before mechanistic claims.',
      retained='encoder, linear+low-rank-quadratic+nonlinear map, native projected operators and pressure confidence'),indent=2))
  def forward(self,x,return_expert_stack=True,pressure_state_override=None,return_closure_params=False):
   h,attr=self._condition_attractor(self._encode(x));a,state=self._state_slices(x)
   if pressure_state_override is not None:state=pressure_state_override
   u=self.structured_correction[0](h,a);p=self.structured_correction[1](h,state)
   params=self._closure_params(h);params.update(attr)
   gate=self.rom_regime_gate(h)
   if gate is not None:params['rom_gate']=gate
   ones=x.new_ones((len(x),1))
   result=(u,p,[ones,ones],x.new_empty((0,1,self.out_dim)))
   return (*result,params) if return_closure_params else result
 def initialize(model,scalers,logit):
  with torch.no_grad():
   for expert,bias in zip(model.structured_correction,[-scalers['rhs_op_mean']/scalers['rhs_op_scale'],-scalers['pressure_mean']/scalers['pressure_scale']]):
    expert.linear.weight.zero_();expert.mlp_head[-1].weight.zero_();expert.mlp_head[-1].bias.copy_(bias)
    # Preserve native physical-zero initialization; do not silently change quadratic-factor policy.
    if expert.quad_left is not None:expert.quad_left.zero_();expert.quad_right.zero_()
   model.closure_confidence_head[-1].weight.zero_();model.closure_confidence_head[-1].bias.fill_(float(logit))
 trainer.OperatorSpaceMoEROM=Structured
 trainer.initialize_physical_zero_residual=initialize
