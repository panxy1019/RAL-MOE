"""Independent autoregressive 2D FNO. Four Fourier blocks; no ROM/PPE calls."""
import torch
from torch import nn
import torch.nn.functional as F
class SpectralConv2d(nn.Module):
 def __init__(self,width,modes):
  super().__init__();self.modes=modes;scale=1/width
  self.positive=nn.Parameter(scale*torch.randn(width,width,modes,modes,dtype=torch.cfloat))
  self.negative=nn.Parameter(scale*torch.randn(width,width,modes,modes,dtype=torch.cfloat))
 def forward(self,x):
  z=torch.fft.rfft2(x);out=torch.zeros_like(z);m=self.modes
  if x.shape[-2]<2*m or z.shape[-1]<m:raise ValueError('Modes exceed grid Nyquist limit')
  out[:,:,:m,:m]=torch.einsum('bixy,ioxy->boxy',z[:,:,:m,:m],self.positive)
  out[:,:,-m:,:m]=torch.einsum('bixy,ioxy->boxy',z[:,:,-m:,:m],self.negative)
  return torch.fft.irfft2(out,s=x.shape[-2:])
class FNO2d(nn.Module):
 def __init__(self,width=32,modes=12,increment=True):
  super().__init__();self.increment=increment
  # history 3x(u,v,p), coordinates x/y, fluid mask, inverse Re, output dt
  self.lift=nn.Conv2d(14,width,1);self.spectral=nn.ModuleList([SpectralConv2d(width,modes) for _ in range(4)])
  self.local=nn.ModuleList([nn.Conv2d(width,width,1) for _ in range(4)])
  self.head=nn.Sequential(nn.Conv2d(width,128,1),nn.GELU(),nn.Conv2d(128,3,1))
 def forward(self,history,coords,mask,param,dt):
  b,_,_,ny,nx=history.shape
  def broadcast(v):return v[:,None,None,None].expand(b,1,ny,nx)
  x=torch.cat([history.reshape(b,9,ny,nx),coords.expand(b,-1,-1,-1),mask.expand(b,-1,-1,-1),broadcast(param),broadcast(dt)],1)
  x=F.pad(self.lift(x),(0,8,0,8))
  for i,(spectral,local) in enumerate(zip(self.spectral,self.local)):
   x=spectral(x)+local(x)
   if i<3:x=F.gelu(x)
  y=self.head(x[:,:,:ny,:nx]);y=y+history[:,-1] if self.increment else y
  return y*mask
