from common import *
from fno_model import *
torch.set_num_threads(2);torch.manual_seed(1248)
conv=SpectralConv2d(2,4)
with torch.no_grad():
 conv.positive.zero_();conv.negative.zero_()
 for i in range(2):conv.positive[i,i]=1;conv.negative[i,i]=1
y=torch.arange(32)*2*torch.pi/32;x=torch.arange(48)*2*torch.pi/48
signal=(torch.sin(y[:,None])+torch.cos(2*x[None,:]))[None,None].repeat(2,2,1,1)
err=float((conv(signal)-signal).detach().abs().max());assert err<2e-6
model=FNO2d(32,12,True);history=torch.randn(1,3,3,32,48);coords=torch.randn(1,2,32,48);mask=torch.ones(1,1,32,48);mask[:,:,14:18,22:26]=0
out=model(history,coords,mask,torch.zeros(1),torch.ones(1));assert out.shape==(1,3,32,48)
assert float(out[:,:,14:18,22:26].detach().abs().max())==0
loss=out.square().mean();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
param=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
write(OUT/'fno_unit_tests.json',dict(spectral_low_frequency_identity_max_abs=err,output_shape=list(out.shape),solid_mask_exact_zero=True,finite_forward_backward=True,
  real_scalar_parameter_count=param,scope='synthetic CPU unit tests only, not a trained FNO or CFD evaluation',fourier_layers=4,padding='8 cells on high-x/high-y sides',history=3,dt_feature='known output interval'))
print('FNO_UNIT_PASS',flush=True)
