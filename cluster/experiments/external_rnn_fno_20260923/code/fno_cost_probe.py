from common import *
from fno_model import *
torch.set_num_threads(2)
def main():
 out=OUT/'fno_gate/P';z=np.load(out/'visual_1536x1024.npz');data=torch.tensor(z['grid'].transpose(2,0,1),device='cuda',dtype=torch.float32)[None]
 data/=data.square().mean((0,2,3),keepdim=True).sqrt();h=data[:,None].repeat(1,3,1,1,1)
 coord=torch.tensor(np.stack(np.meshgrid(z['x']/15,z['y']/10)),device='cuda',dtype=torch.float32)[None];mask=torch.tensor(z['mask'],device='cuda',dtype=torch.float32)[None,None];mu=torch.zeros(1,device='cuda')
 m=FNO2d().cuda();opt=torch.optim.AdamW(m.parameters(),lr=1e-3);forward=[];back=[]
 for i in range(4):
  opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter();p=m(h,coord,mask,mu,mu);loss=((p-data)**2*mask).mean();torch.cuda.synchronize();mid=time.perf_counter();loss.backward();opt.step();torch.cuda.synchronize();end=time.perf_counter()
  if i:forward.append(mid-t);back.append(end-t)
 record=dict(grid=[1536,1024],width=32,modes=12,FP32=True,training_step_seconds=back,forward_seconds=forward,
    scope='warm resource probe only on validation field, weights discarded; not formal model training',
    full_validation_48windows_K48_serial_forward_estimate_seconds=float(np.median(forward))*48*48,
    training_4000_optimizer_updates_estimate_hours=float(np.median(back))*4000/3600)
 write(out/'cost_probe.json',record);print(json.dumps(record),flush=True)
if __name__=='__main__':main()
