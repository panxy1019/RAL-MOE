"""Render the frozen S/H K56 capability curves from the scan CSV."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import math

def main():
    p=argparse.ArgumentParser(); p.add_argument('--curves',type=Path,required=True); p.add_argument('--summary',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    rows=list(csv.DictReader(a.curves.open()))
    re=[float(r['Re']) for r in rows]; s=[float(r['S_joint_error_K56_mean']) for r in rows]; h=[float(r['H_joint_error_K56_mean']) for r in rows]
    summary=json.loads(a.summary.read_text()); crossings=summary['capacity_crossings']
    try:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7.2,4.2),layout='constrained')
        ax.plot(re,s,'o-',label='Steady native (S-only)',color='#1f77b4')
        ax.plot(re,h,'s-',label='Hopf phase-free (H-only on S history)',color='#d62728')
        for item in crossings:
            if item['linear_interpolated_Re_cap'] is not None:
                ax.axvline(item['linear_interpolated_Re_cap'],color='#333333',linestyle='--',linewidth=1,label=f"Re_cap={item['linear_interpolated_Re_cap']:.3f}")
        ax.set_xlabel('Re'); ax.set_ylabel('K56 joint field error (velocity + pressure)'); ax.set_yscale('log'); ax.grid(True,which='both',alpha=.25); ax.legend(fontsize=8)
        fig.savefig(a.output,format='svg'); return
    except ModuleNotFoundError:
        pass
    width,height,left,right,top,bottom=800,440,78,25,25,60
    xmin,xmax=min(re),max(re); logs=[math.log10(v) for v in s+h]; ymin,ymax=min(logs),max(logs)
    def px(x): return left+(x-xmin)/(xmax-xmin)*(width-left-right)
    def py(y): return top+(ymax-math.log10(y))/(ymax-ymin)*(height-top-bottom)
    def path(values): return ' '.join(f'{px(x):.2f},{py(y):.2f}' for x,y in zip(re,values))
    text=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<style>text{font-family:Arial,sans-serif;font-size:12px}.axis{stroke:#222}.grid{stroke:#ddd}.s{fill:none;stroke:#1f77b4;stroke-width:2}.h{fill:none;stroke:#d62728;stroke-width:2}</style>', f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/><line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>', f'<polyline class="s" points="{path(s)}"/><polyline class="h" points="{path(h)}"/>']
    for x,y in zip(re,s): text.append(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="3" fill="#1f77b4"/>')
    for x,y in zip(re,h): text.append(f'<rect x="{px(x)-3:.2f}" y="{py(y)-3:.2f}" width="6" height="6" fill="#d62728"/>')
    for item in crossings:
        x=item['linear_interpolated_Re_cap']
        if x is not None: text.append(f'<line x1="{px(x):.2f}" y1="{top}" x2="{px(x):.2f}" y2="{height-bottom}" stroke="#333" stroke-dasharray="5,4"/><text x="{px(x)+4:.2f}" y="{top+14}">Re_cap={x:.3f}</text>')
    text += [f'<text x="{width/2-100}" y="{height-15}">Re</text>', f'<text x="8" y="18">K56 joint field error (log scale)</text>', f'<text x="{width-235}" y="{top+18}" fill="#1f77b4">● Steady native (S-only)</text>', f'<text x="{width-235}" y="{top+35}" fill="#d62728">■ Hopf phase-free (H-only)</text>', '</svg>']
    a.output.write_text(''.join(text),encoding='utf-8')

if __name__=='__main__': main()
