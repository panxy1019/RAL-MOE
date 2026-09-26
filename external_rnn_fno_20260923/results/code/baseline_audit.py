from common import *
import runpy,sys
def main():
 results=[];score=json.loads((OLD/'scorecard_current.json').read_text())
 for reg in ['H','P']:
  d=np.load(OUT/f'{reg}_sealed_test.npz');windows=[]
  if reg=='H':
   for p in sorted((OLD/'H_evaluation/proposed/1248').glob('*_modal.npz')):
    z=np.load(p);re=float(p.name.split('_')[1])
    for a,b,ta,tb in zip(z['pred_a'],z['pred_b'],z['true_a'],z['true_b']):windows.append((re,a,b,ta,tb))
  else:
   for r in torch.load(OLD/'P_evaluation_v2/proposed/1248/modal_rollouts.pt',weights_only=False):
    v=r['values'];windows.append((r['Re'],np.array([x[0] for x in v]),np.array([x[1] for x in v]),np.array([x[2] for x in v]),np.array([x[3] for x in v])))
  for K in [24,48]:
   by={}
   for re,a,b,ta,tb in windows:
    eu=error(a[:K],ta[:K],d['gu'],d['cu'],float(d['nu']));ep=error(b[:K],tb[:K],d['gp'],d['cp'],float(d['np0']))
    by.setdefault(re,[]).append([eu.mean(),ep.mean()])
   result=np.mean([np.mean(v,0) for v in by.values()],0)
   old=next(x for x in score if x['regime']==reg and x['method']=='proposed' and x['seed']==1248 and x['K']==K)
   expected=np.array([old['Eu_percent_window_mean'],old['Ep_percent_window_mean']]);delta=abs(result-expected)
   assert np.max(delta)<5e-5,(reg,K,result,expected)
   results.append(dict(regime=reg,K=K,Eu=float(result[0]),Ep=float(result[1]),old_Eu=float(expected[0]),old_Ep=float(expected[1]),max_abs_difference=float(delta.max()),rounding_tolerance_pass=True))
 write(OUT/'baseline_metric_audit.json',results)
 # One actual frozen P checkpoint replay; do not regard concurrent runtime as a latency benchmark.
 source=ROOT/'experiments/qlrom_20260923/code/timing_neural.py';text=source.read_text()
 text=text.replace("ROOT/'experiments/qlrom_20260923/timing'/kind","OUT_REPLAY/kind")
 text=text.replace('range(3): end_to_end()','range(0): end_to_end()').replace('range(10):','range(1):')
 namespace={'__name__':'__main__','__file__':str(source),'OUT_REPLAY':OUT/'baseline_checkpoint_replay'}
 sys.argv=['replay','--kind','proposed'];exec(compile(text,str(source),'exec'),namespace)
 replay=json.loads((OUT/'baseline_checkpoint_replay/proposed/timing.json').read_text());assert replay['max_modal_difference_from_cached']<1e-6
 write(OUT/'baseline_checkpoint_replay/verification.json',dict(checkpoint_sha256=replay['checkpoint_sha256'],max_modal_difference=replay['max_modal_difference_from_cached'],passed=True,timing_discarded='Concurrent RNN screening; this is an identity/numerical replay only'))
 print('BASELINE_REPLAY_PASS',flush=True)
if __name__=='__main__':main()
