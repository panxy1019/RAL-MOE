"""Replot existing diagnostic measurements; no simulation or data alteration."""
from pathlib import Path
import csv
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'field_redraw_20260922/.runtime'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'round3_20260921/field_diagnostics/field_curves.csv'
rows = list(csv.DictReader(SOURCE.open()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                     'pdf.fonttype':42,'axes.linewidth':0.6})
fig, axes = plt.subplots(2,3,figsize=(7.2,3.6),layout='constrained')
metrics = [('deficit_over_reference',100,'Energy deficit (%)'),
           ('momentum_excess_rms',1,'Momentum defect RMS'),
           ('ppe_excess_rms',1,'Pressure defect RMS')]
for i,pair in enumerate(['sh','hp']):
    base=[r for r in rows if r['pair']==pair and int(r['neighbors'])==24
          and r['method']=='fusion']
    assert len({r['window'] for r in base})==16
    for j,(key,scale,title) in enumerate(metrics):
        steps=range(1,25) if j==0 else range(2,24)
        vals=[]
        for k in steps:
            selected=[float(r[key])*scale for r in base if int(r['step'])==k]
            assert len(selected)==16 and np.isfinite(selected).all()
            vals.append(np.mean(selected))
        ax=axes[i,j]
        ax.plot(np.array(list(steps))*4,vals,color=['#3366AA','#A05A24'][i],lw=1.5)
        ax.set_title(f'({chr(97+i*3+j)}) '+title,fontsize=8)
        ax.set_xlabel('Prediction lead time')
        ax.set_xlim(0,96); ax.set_xticks([0,24,48,72,96])
        ax.set_ylim(bottom=0)
        if j==0:ax.set_ylabel(['S–H','H–P'][i],fontweight='bold')
        else:ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
        ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
fig.savefig(ROOT/'figures/fusion_defects_compact.pdf')
fig.savefig(ROOT/'figures/fusion_defects_compact.png',dpi=220)
print('Saved compact figure: 16 windows per interface, original 24-neighbor diagnostics.')
