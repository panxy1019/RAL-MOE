import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/experiments/round3_20260921/field_diagnostics')
def read(name):
 rows=list(csv.DictReader((D/name).open()))
 for r in rows:
  for k,v in r.items():
   if v in ['True','False']:r[k]=v=='True'
   else:
    try:r[k]=float(v)
    except ValueError:pass
 return rows
rows=read('field_curves.csv');pr=read('phase_curves.csv')
pal={'reference':'#333333','candidate1':'#3366AA','candidate2':'#BB7722','fusion':'#9A4488'};styles=['--',':','-.','-'];plt.rcParams.update({'font.size':9})
for pair in ['sh','hp']:
 fig,axs=plt.subplots(2,2,figsize=(9,5.8),layout='constrained')
 for m,ls in zip(pal,styles):
  rr=[r for r in rows if r['neighbors']==24 and r['pair']==pair and r['method']==m]
  axs[0,0].plot(np.arange(1,25)*4,[np.mean([r['energy'] for r in rr if r['step']==k]) for k in range(1,25)],label=m,color=pal[m],ls=ls)
 for ax,key,label in zip([axs[0,1],axs[1,0],axs[1,1]],['deficit_over_reference','momentum_excess_rms','ppe_excess_rms'],['Energy deficit / reference energy','Momentum fusion excess RMS','PPE fusion excess RMS']):
  rr=[r for r in rows if r['neighbors']==24 and r['pair']==pair and r['method']=='fusion'];ks=range(1,25) if 'deficit' in key else range(2,24)
  ax.plot(np.array(list(ks))*4,[np.mean([r[key] for r in rr if r['step']==k]) for k in ks],color=pal['fusion']);ax.set_ylabel(label)
 axs[0,0].set_ylabel('Total kinetic energy');axs[0,0].legend(fontsize=8)
 for ax in axs.flat:ax.set_xlabel('Prediction time');ax.grid(alpha=.2)
 fig.suptitle(pair.upper()+' | all 16 windows; energy and additional fusion residual')
 for ext in ['png','pdf']:fig.savefig(D/f'{pair}_energy_excess.{ext}',dpi=180)
 plt.close(fig)
 fig,axs=plt.subplots(4,4,figsize=(12,8),layout='constrained')
 for w,ax in enumerate(axs.flat):
  window=[r for r in pr if r['pair']==pair and r['window']==w]
  for m,ls in zip(pal,styles):
   if m=='reference':continue
   rr=[r for r in window if r['method']==m]
   ax.plot([r['lead_time'] for r in rr],[r['phase_difference_rad'] for r in rr],color=pal[m],ls=ls,label=m,lw=1)
  ax.set_title(f"w{w}, Re={window[0]['Re']:g}, "+('resolved' if window[0]['phase_resolved'] else 'unresolved'),fontsize=9)
  ax.set_ylim(-np.pi,np.pi);ax.grid(alpha=.2)
  if w>=12:ax.set_xlabel('Prediction time')
  if w%4==0:ax.set_ylabel('Phase difference (rad)')
 axs[0,0].legend(fontsize=7)
 fig.suptitle(pair.upper()+' | all windows, detrended Hilbert diagnostic; unresolved curves are not accuracy evidence')
 for ext in ['png','pdf']:fig.savefig(D/f'{pair}_phase_all_windows.{ext}',dpi=180)
 plt.close(fig)
print('additional figures complete')
