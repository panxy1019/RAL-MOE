"""Final full-field FNO aggregation and serial native/physical latency."""
from fno_experiment import *
import shutil

def latency(m,data):
 s=int(data.d['test_starts'][0]);hist0=data.initial(s);back=load_npz(OUT/'fno_gate/P/maps/grid_to_fv_1536x1024.npz');w=data.w.cpu().numpy()
 def run(K,mapped):
  history=data.initial(s) if mapped else hist0.clone();last=None
  for k in range(K):
   mu,dt=data.condition(s,k);pred=data.normalize_output(m(history,data.coords,data.mask,mu,dt));physical=data.physical(pred)[0].reshape(3,-1).T.cpu().numpy()
   if mapped:
    field=back@physical;field[:,2]-=field[:,2]@w/w.sum();last=field
   else:last=physical
   history=torch.cat([history[:,1:],pred[:,None]],1)
  return last
 results=[]
 with torch.inference_mode():
  for K in [24,48]:
   for mapped in [False,True]:
    for _ in range(3):run(K,mapped)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();samples=[]
    for _ in range(10):
     torch.cuda.synchronize();t=time.perf_counter();value=run(K,mapped);torch.cuda.synchronize();samples.append(time.perf_counter()-t)
    results.append(dict(K=K,include_mapping=mapped,seconds=samples,median=float(np.median(samples)),iqr=float(np.percentile(samples,75)-np.percentile(samples,25)),
      mean=float(np.mean(samples)),sample_sd=float(np.std(samples,ddof=1)),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),warmups=3,repeats=10,batch=1,
      scope='in-memory physical FV history -> normalized grid -> all output fields transferred CPU -> inverse FV mapping/gauge' if mapped else 'pre-mapped normalized grid history on GPU -> all native physical output fields transferred to CPU; no FV mapping'))
 return dict(results=results,device=torch.cuda.get_device_name(),dtype='FP32/complex64 neural; sparse interpolation stored FP64',cpu_threads=2,
   initial_global_id=int(data.d['global_ids'][s]),Re=float(data.d['Re'][s]),IO_excluded=True,synchronization='before/after each repetition')
def main():
 if not (FOUT/'PREDICTIONS_COMPLETED.json').exists():raise RuntimeError('Predictions not completed')
 seeds=[];data=Data(test=True)
 for seed in SEEDS:
  out=FOUT/'formal'/str(seed);run=json.loads((out/'summary.json').read_text())
  if not run.get('heldout_evaluated'):continue
  rr=list(csv.DictReader((out/'per_step_metrics.csv').open()));wr=[];br=[]
  for K in [24,48]:
   for window in sorted(set(r['window'] for r in rr),key=int):
    part=[r for r in rr if r['window']==window and int(r['step'])<=K];okay=len(part)==K and all(r['finite']=='True' for r in part)
    row=dict(seed=seed,K=K,window=int(window),Re=float(part[0]['Re']),complete=okay)
    for key in ['Eu','Ep','native_Eu','native_Ep']:row[key]=float(np.mean([float(r[key]) for r in part])) if okay else float('nan')
    row['joint']=row['Eu']+row['Ep'];row['native_joint']=row['native_Eu']+row['native_Ep'];wr.append(row)
   for rv in sorted(set(r['Re'] for r in wr)):
    part=[r for r in wr if r['Re']==rv and r['K']==K];row=dict(seed=seed,K=K,Re=rv,complete=sum(r['complete'] for r in part),total=len(part))
    for key in ['Eu','Ep','joint','native_Eu','native_Ep','native_joint']:row[key]=float(np.mean([r[key] for r in part])) if all(r['complete'] for r in part) else float('nan')
    br.append(row)
   part=[r for r in br if r['K']==K];row=dict(seed=seed,K=K,complete=sum(r['complete'] for r in part),total=32)
   for key in ['Eu','Ep','joint','native_Eu','native_Ep','native_joint']:row[key]=float(np.mean([r[key] for r in part])) if row['complete']==32 else float('nan')
   seeds.append(row)
  csvwrite(out/'metrics_by_window.csv',wr);csvwrite(out/'metrics_by_re.csv',br);shutil.copy2(OUT/'split_manifest.json',out/'split_manifest.json');shutil.copy2(FOUT/'data_prepared.json',out/'data_audit.json')
  ck=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);cfg=ck['config']['architecture'];m=FNO2d(cfg['width'],cfg['modes'],cfg['increment']).cuda();m.load_state_dict(ck['model']);m.eval()
  write(out/'latency.json',latency(m,data));(out/'environment.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True)+'\n'+torch.__config__.show());del m;torch.cuda.empty_cache()
  run['horizons']=[r for r in seeds if r['seed']==seed];run['checkpoint_sha256']=sha(out/'best_checkpoint.pt');write(out/'summary.json',run)
 summary=[]
 for K in [24,48]:
  rr=[r for r in seeds if r['K']==K];okay=len(rr)==3 and all(r['complete']==32 for r in rr)
  row=dict(K=K,complete_seeds=sum(r['complete']==32 for r in rr),expected_seeds=3,complete_windows=sum(r['complete'] for r in rr),total_windows=96)
  for key in ['Eu','Ep','joint','native_Eu','native_Ep','native_joint']:
   row[key+'_mean']=float(np.mean([r[key] for r in rr])) if okay else None;row[key+'_sample_sd']=float(np.std([r[key] for r in rr],ddof=1)) if okay else None
  summary.append(row)
 csvwrite(FOUT/'seed_metrics.csv',seeds);csvwrite(FOUT/'summary.csv',summary);write(FOUT/'COMPLETED.json',dict(time=time.strftime('%FT%T%z'),summary=summary))
 print('FNO_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
