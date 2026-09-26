"""Memory gate on the smallest registered FNO at the selected physical grid."""
from common import *
from fno_model import *
from scipy.sparse import load_npz
import gc
torch.set_num_threads(2)
def main():
 out=OUT/'fno_gate/P';decision=json.loads((out/'decision.json').read_text())['decisions']['P'];assert decision['status']=='interpolation_passed_memory_visual_pending'
 nx,ny=decision['nx'],decision['ny'];z=np.load(out/f'visual_{nx}x{ny}.npz');field=z['grid'].transpose(2,0,1).astype(np.float32)
 mask=torch.tensor(z['mask'],device='cuda',dtype=torch.float32)[None,None]
 coords=np.stack(np.meshgrid(z['x'],z['y'])).astype(np.float32);coords=(coords-np.array([5,0],dtype=np.float32)[:,None,None])/np.array([15,10],dtype=np.float32)[:,None,None]
 coords=torch.tensor(coords,device='cuda')[None];data=torch.tensor(field,device='cuda')[None];scale=data.square().mean((0,2,3),keepdim=True).sqrt().clamp_min(1e-8);data=data/scale
 history=data[:,None].repeat(1,3,1,1,1);records=[]
 for checkpointed in [False,True]:
  try:
   torch.cuda.empty_cache();m=FNO2d(width=32,modes=12,increment=True).cuda();opt=torch.optim.AdamW(m.parameters(),lr=1e-3);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
   if checkpointed:
    from torch.utils.checkpoint import checkpoint
    pred=checkpoint(m,history,coords,mask,torch.zeros(1,device='cuda'),torch.zeros(1,device='cuda'),use_reentrant=False)
   else:pred=m(history,coords,mask,torch.zeros(1,device='cuda'),torch.zeros(1,device='cuda'))
   loss=((pred-data)**2*mask).sum()/(3*mask.sum());loss.backward();opt.step();torch.cuda.synchronize()
   peak=torch.cuda.max_memory_allocated();records.append(dict(checkpointed=checkpointed,status='finite' if torch.isfinite(loss) else 'nonfinite',loss=float(loss.detach()),seconds=time.perf_counter()-t,peak_gpu_bytes=peak,within_22GiB=peak<22*2**30))
   del pred,loss,m,opt;gc.collect();torch.cuda.empty_cache()
   if records[-1]['within_22GiB']:break
  except torch.cuda.OutOfMemoryError as e:
   records.append(dict(checkpointed=checkpointed,status='OOM',message=str(e)))
   m=opt=pred=loss=None
   gc.collect();torch.cuda.empty_cache()
 write(out/'memory_smoke.json',dict(grid=[nx,ny],width=32,modes=12,layer_count=4,batch=1,dtype='FP32/complex64',TBPTT=1,records=records,
  scope='Single training step on a validation field for resource validation only; parameters discarded, not a fitted or evaluated baseline'))
 print(json.dumps(records),flush=True)
if __name__=='__main__':main()
