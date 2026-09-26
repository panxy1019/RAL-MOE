"""One-time sealed tests followed by serial end-to-end timing."""
from rnn_experiment import *
import shutil,subprocess,platform

def timing(m,d,reg,seed):
 g=np.load(OUT/f'{reg}_geometry.npz');u=g['phi_u'].astype(float);p=g['phi_p'].astype(float);w=g['areas'];wu=np.r_[w,w];mu=g['mean_u'].astype(float);mp=g['mean_p'].astype(float)
 p-=(p@w/w.sum())[:,None];mp-=mp@w/w.sum()
 encu=(u*wu).T@np.linalg.inv((u*wu)@u.T);encp=(p*w).T@np.linalg.inv((p*w)@p.T)
 s=int(d['test_starts'][0]);state=d['state'][s-2:s+1];input_u=state[:,:32]@u+mu;input_p=state[:,32:]@p+mp;tt=tensors(d)
 def run(K):
  aa=(input_u-mu)@encu;pp=input_p-(input_p@w/w.sum())[:,None];bb=(pp-mp)@encp
  history=torch.tensor((np.c_[aa,bb]-d['mean'])/d['scale'],device='cuda',dtype=torch.float32)[None]
  values=predict(m,d,[s],K,tt,initial=history)*tt['scale']+tt['mean'];coeff=values[0].cpu().numpy()
  uu=coeff[:,:32]@u+mu;pp=coeff[:,32:]@p+mp;pp-=(pp@w/w.sum())[:,None]
  return uu,pp,coeff
 results=[]
 with torch.inference_mode():
  for K in [24,48]:
   for _ in range(3):run(K)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();samples=[]
   for _ in range(10):
    torch.cuda.synchronize();t=time.perf_counter();uu,pp,c=run(K);torch.cuda.synchronize();samples.append(time.perf_counter()-t)
   cached=np.load(OUT/'formal'/reg/m.cell_identity/str(seed)/'test_predictions.npz')['coeff'][0,:K]
   delta=float(np.max(abs(c-cached)))
   results.append(dict(K=K,seconds=samples,median=float(np.median(samples)),mean=float(np.mean(samples)),sample_sd=float(np.std(samples,ddof=1)),
     iqr=float(np.percentile(samples,75)-np.percentile(samples,25)),warmups=3,repeats=10,batch=1,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
     max_modal_difference_vs_cached=delta,start_global=int(d['global_ids'][s]),Re=float(d['Re'][s])))
 return dict(results=results,scope='common in-memory POD-reconstructed physical 3-frame history -> normalization/encoding + GPU rollout + all decoded velocity/pressure fields + gauge; includes transfers; excludes load/IO/metrics',
   dtype='FP32 network, FP64 encode/decode',device=torch.cuda.get_device_name(),cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),synchronization='CUDA before and after each repetition',
   peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)

def main():
 assert (OUT/'TRAINING_COMPLETED.json').exists()
 selected=OUT/'selected_configs.json';assert selected.exists()
 marker=OUT/'TEST_STARTED.json'
 if marker.exists():raise FileExistsError('Sealed test already started: inspect and explicitly resume, do not silently overwrite')
 write(marker,dict(time=time.strftime('%FT%T%z'),selected_sha256=sha(selected)))
 summaryrows=[];allsteps=[];runs=[]
 environment=subprocess.check_output(['nvidia-smi'],text=True)+'\n'+subprocess.check_output(['lscpu'],text=True)+'\n'+torch.__config__.show()
 for reg in ['H','P']:
  d=load(reg,'sealed_test');tt=tensors(d);starts=d['test_starts'];true=d['state'][starts[:,None]+np.arange(1,49)]
  for cell in ['LSTM','GRU']:
   for seed in SEEDS:
    out=OUT/'formal'/reg/cell/str(seed);s=json.loads((out/'summary.json').read_text())
    shutil.copy2(OUT/'split_manifest.json',out/'split_manifest.json');shutil.copy2(OUT/'data_audit.json',out/'data_audit.json');(out/'environment.txt').write_text(environment)
    if not s['accepted']:
     s.update(heldout_evaluated=False,test_error='validation_or_training_rejected');write(out/'summary.json',s);runs.append(s);continue
    checkpoint=out/'best_checkpoint.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False);m=PODRNN(cell,ck['config']['config']).cuda();m.load_state_dict(ck['model']);m.eval();m.cell_identity=cell
    with torch.inference_mode():pred=(predict(m,d,starts,48,tt)*tt['scale']+tt['mean']).double().cpu().numpy()
    score,eu,ep=metrics(pred,d,starts);finite=np.isfinite(pred).all(2)
    bad=(np.linalg.norm(pred[:,:,:32],axis=2)>10*np.max(np.linalg.norm(true[:,:,:32],axis=2),axis=1,keepdims=True))|(np.linalg.norm(pred[:,:,32:],axis=2)>10*np.max(np.linalg.norm(true[:,:,32:],axis=2),axis=1,keepdims=True))|~finite
    path=out/'test_predictions.npz';np.savez_compressed(path,coeff=pred,global_starts=d['global_ids'][starts],times=d['time'][starts[:,None]+np.arange(1,49)],Re=d['Re'][starts])
    write(out/'test_predictions_or_hashes.json',dict(path=str(path),sha256=sha(path),checkpoint_sha256=sha(checkpoint),reference='common train-only regime POD reconstruction'))
    win=[];re_rows=[];horizons={}
    for K in [24,48]:
     for i,start in enumerate(starts):
      okay=bool(finite[i,:K].all());failed=np.flatnonzero(~finite[i,:K]);div=np.flatnonzero(bad[i,:K])
      win.append(dict(regime=reg,model=cell,seed=seed,K=K,window=int(d['global_ids'][start]),Re=float(d['Re'][start]),complete=okay,
       Eu=float(eu[i,:K].mean()) if okay else float('nan'),Ep=float(ep[i,:K].mean()) if okay else float('nan'),joint=float((eu[i,:K]+ep[i,:K]).mean()) if okay else float('nan'),
       divergent=bool(len(div)),first_nonfinite_step=int(failed[0]+1) if len(failed) else None,first_divergence_step=int(div[0]+1) if len(div) else None))
     selected_rows=[x for x in win if x['K']==K];by=[]
     for rv in sorted(set(x['Re'] for x in selected_rows)):
      rr=[x for x in selected_rows if x['Re']==rv];valid=all(x['complete'] for x in rr)
      row=dict(regime=reg,model=cell,seed=seed,K=K,Re=rv,complete=sum(x['complete'] for x in rr),total=len(rr),divergent=sum(x['divergent'] for x in rr))
      for k in ['Eu','Ep','joint']:row[k]=float(np.mean([x[k] for x in rr])) if valid else float('nan')
      re_rows.append(row);by.append(row)
     allokay=all(x['complete'] for x in selected_rows)
     row=dict(regime=reg,model=cell,seed=seed,K=K,complete=sum(x['complete'] for x in selected_rows),total=len(selected_rows),divergent=sum(x['divergent'] for x in selected_rows),parameter_count=s['parameter_count'])
     for k in ['Eu','Ep','joint']:row[k]=float(np.mean([x[k] for x in by])) if allokay else float('nan')
     summaryrows.append(row);horizons[str(K)]=row
    for i,start in enumerate(starts):
     for k in range(48):allsteps.append(dict(regime=reg,model=cell,seed=seed,window=int(d['global_ids'][start]),Re=float(d['Re'][start]),step=k+1,time=float(d['time'][start+k+1]),Eu=float(eu[i,k]),Ep=float(ep[i,k]),finite=bool(finite[i,k]),divergent=bool(bad[i,k])))
    csvwrite(out/'metrics_by_window.csv',win);csvwrite(out/'metrics_by_re.csv',re_rows)
    s.update(heldout_evaluated=True,horizons=horizons,checkpoint_sha256=sha(checkpoint));write(out/'summary.json',s);runs.append(s)
    write(out/'latency.json',timing(m,d,reg,seed));del m;torch.cuda.empty_cache()
    print('TEST_COMPLETE',reg,cell,seed,horizons,flush=True)
 aggregate=[]
 for reg in ['H','P']:
  for cell in ['LSTM','GRU']:
   for K in [24,48]:
    rr=[r for r in summaryrows if r['regime']==reg and r['model']==cell and r['K']==K]
    good=len(rr)==3 and all(r['complete']==r['total'] for r in rr)
    row=dict(regime=reg,model=cell,K=K,accepted_seeds=len(rr),complete_seeds=sum(r['complete']==r['total'] for r in rr),expected_seeds=3,
     complete_windows=sum(r['complete'] for r in rr),total_windows=3*(42 if reg=='H' else 32),divergent_windows=sum(r['divergent'] for r in rr),parameter_count=rr[0]['parameter_count'] if rr else None)
    for key in ['Eu','Ep','joint']:
     row[key+'_mean']=float(np.mean([r[key] for r in rr])) if good else None;row[key+'_sample_sd']=float(np.std([r[key] for r in rr],ddof=1)) if good else None
    aggregate.append(row)
 csvwrite(OUT/'rnn_seed_metrics.csv',summaryrows);csvwrite(OUT/'rnn_summary.csv',aggregate);csvwrite(OUT/'rnn_per_step_metrics.csv',allsteps)
 write(OUT/'TEST_COMPLETED.json',dict(time=time.strftime('%FT%T%z'),runs=len(runs),summary=aggregate));print('ALL_RNN_TESTS_COMPLETE',flush=True)
if __name__=='__main__':main()
