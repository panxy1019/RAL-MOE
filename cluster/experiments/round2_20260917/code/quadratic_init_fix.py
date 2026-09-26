"""Keep initial correction zero without killing gradients of both factors."""
import json
from pathlib import Path
import torch

def install(trainer,output):
 original=trainer.initialize_physical_zero_residual
 def initialize(model,scalers,logit):
  # Capture native random initialization before zero-residual initialization.
  right={n:p.detach().clone() for n,p in model.named_parameters() if n.endswith('quad_right')}
  assert right and all(torch.count_nonzero(v)>0 for v in right.values())
  original(model,scalers,logit)
  with torch.no_grad():
   for n,p in model.named_parameters():
    if n in right:p.copy_(right[n])
  tests=[]
  for name,module in model.named_modules():
   if not hasattr(module,'quad_left') or module.quad_left is None:continue
   left=module.quad_left;rr=module.quad_right
   assert torch.count_nonzero(left)==0 and torch.count_nonzero(rr)>0
   # Local diagnostic does not advance the training RNG or populate parameter .grad.
   state=torch.arange(1,module.state_dim+1,device=left.device,dtype=left.dtype)[None,:]/module.state_dim
   q=(torch.einsum('bs,ors->bor',state,left)*torch.einsum('bs,ors->bor',state,rr)).sum()
   gl,gr=torch.autograd.grad(q,(left,rr))
   assert float(q)==0 and torch.count_nonzero(gl)>0 and torch.count_nonzero(gr)==0
   tests.append(dict(module=name,initial_output=0,left_gradient_norm=float(gl.norm()),right_gradient_norm=float(gr.norm())))
  Path(output,'quadratic_initialization_test.json').write_text(json.dumps(dict(passed=True,
    change='restore native random right factor; left stays zero; no extra RNG draws',modules=tests),indent=2))
 trainer.initialize_physical_zero_residual=initialize
