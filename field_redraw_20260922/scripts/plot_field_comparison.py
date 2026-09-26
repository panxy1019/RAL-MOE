"""Standalone renderer: only NumPy/Matplotlib, saved fields and original triangles.

python scripts/plot_field_comparison.py --config config/figure_config.json
No checkpoint, GPU, VTK, or SSH is needed to change the appearance.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.runtime').is_dir():sys.path.insert(0,str(ROOT/'.runtime'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize,TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'config/figure_config.json');parser.add_argument('--minimal',action='store_true',help='Six PNGs with case names and numeric colorbars only');parser.add_argument('--annotated',action='store_true',help='Two requested minimal figures with labels inside panels');a=parser.parse_args()
if a.annotated:a.minimal=True
C=json.loads(a.config.read_text());D=ROOT/'data';O=ROOT/('output_annotated_below' if a.annotated else 'output_minimal' if a.minimal else 'output');O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':C['font_family'],'font.size':C['font_size'],'axes.titlesize':8,'xtick.labelsize':C['tick_size'],'ytick.labelsize':C['tick_size'],'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.4,'savefig.facecolor':'white','figure.facecolor':'white'})
ids=list(dict.fromkeys(C['overview']+sum(C['fusion'].values(),[])))
DATA={};manifest=[];qa=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for key in ids:
 meta=json.loads((D/f'{key}.json').read_text());arr=np.load(D/f'{key}.npz');geo=np.load(D/meta['plot_geometry'])
 assert sha(D/f'{key}.npz')==meta['snapshot_arrays']['sha256']
 assert meta['nonfinite_points']==0
 xy=geo['xy'];tri=mtri.Triangulation(xy[:,0],xy[:,1],geo['triangles'])
 DATA[key]=(meta,arr,geo,tri);meta['plot_scale_records']=[];meta['local_array']=f'data/{key}.npz';meta['local_geometry']=f'data/{key}_geometry.npz';meta['local_geometry_sha256']=sha(D/f'{key}_geometry.npz');manifest.append(meta)
 # Recalculate metrics without the extraction helper.
 for method in ['prediction','e2','candidate1','candidate2']:
  if method not in arr:continue
  diff=arr[method]-arr['reference'];ref=arr['reference'];w=arr['areas']
  got=[float(np.sqrt(np.einsum('ij,j->',d*d,w)/np.einsum('ij,j->',r*r,w))) for d,r in [(diff[:2],ref[:2]),(diff[2:],ref[2:])]]
  assert max(abs(got[j]-meta['snapshot_metrics'][method][name]) for j,name in enumerate(['Eu','Ep']))<1e-12

def scalar(f,field):return np.hypot(f[0],f[1]) if field=='velocity' else f[2]
def error(f,r,field):return np.hypot(*(f[:2]-r[:2])) if field=='velocity' else abs(f[2]-r[2])
def norms(key,field,methods,stem):
 meta,z,_,_=DATA[key];values=[scalar(z[m],field) for m in ['reference']+methods];errors=[error(z[m],z['reference'],field) for m in methods]
 lo=min(float(v.min()) for v in values);hi=max(float(v.max()) for v in values);em=max(float(e.max()) for e in errors)
 if field=='pressure':hi=max(abs(lo),abs(hi));lo=-hi
 else:lo=min(0.,lo)
 norm=TwoSlopeNorm(vcenter=0,vmin=lo,vmax=hi) if field=='pressure' else Normalize(lo,hi)
 en=Normalize(0,max(em,1e-15))
 meta['plot_scale_records'].append(dict(figure=stem,field=field,methods=methods,field_vmin=lo,field_vmax=hi,error_vmin=0,error_vmax=em,clipped=False,saturation_fraction={m:0.0 for m in methods},crop_extent=C['crop_extent'][key]))
 return norm,en

def panel(ax,key,values,norm,cmap,show_y=False):
 meta,z,g,tri=DATA[key];v=values[g['value_ids']]
 if meta['data_association']=='cell':art=ax.tripcolor(tri,facecolors=v,shading='flat',cmap=cmap,norm=norm,rasterized=True)
 else:art=ax.tripcolor(tri,v,shading='gouraud',cmap=cmap,norm=norm,rasterized=True)
 xmin,xmax,ymin,ymax=C['crop_extent'][key];ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),aspect='equal');ax.set_anchor('C')
 ax.set_xticks(np.linspace(xmin,xmax,3));ax.set_yticks([ymin,(ymin+ymax)/2,ymax]);ax.tick_params(length=2,pad=1.5,width=.4,labelleft=show_y)
 ax.get_xticklabels()[0].set_ha('left');ax.get_xticklabels()[-1].set_ha('right')
 if a.minimal:ax.set_xticks([]);ax.set_yticks([])
 for spine in ax.spines.values():spine.set_color('#888888');spine.set_linewidth(.35)
 return art

def cbar(fig,slot,norm,cmap,label):
 ax=fig.add_subplot(slot);bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=ax,orientation='horizontal')
 ticks=[norm.vmin,(norm.vmin+norm.vmax)/2,norm.vmax];bar.set_ticks(ticks)
 vmax=max(abs(norm.vmin),abs(norm.vmax));exponent=int(np.floor(np.log10(vmax))) if 0<vmax<.01 else 0
 bar.formatter=FuncFormatter(lambda v,p:f'{v/10**exponent:.3g}');bar.update_ticks();bar.ax.tick_params(length=2,pad=1,labelsize=7);bar.outline.set_linewidth(.35)
 bar.ax.get_xticklabels()[0].set_ha('left');bar.ax.get_xticklabels()[-1].set_ha('right')
 bar.ax.set_title(label+(rf' $\times10^{{{exponent}}}$' if exponent else ''),fontsize=7.5,pad=3)
 if a.minimal:
  bar.ax.set_title('')
  bar.formatter=FuncFormatter(lambda v,p: '0' if v==0 else f'{v:.2g}')
  bar.update_ticks()

def title(meta,index,overview=False):
 reg={'HP':'H–P','SH':'S–H','Periodic':'P'}[meta['regime_or_overlap']]
 return f"({chr(97+index)}) {meta['case']}  |  Re={meta['Re']:.4f}  |  {reg}  |  k={meta['k']}/{meta['K']}"

def rowheight(key):
 x0,x1,y0,y1=C['crop_extent'][key];return 1.47*(y1-y0)/(x1-x0)+.32
def save(fig,stem):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();W,H=fig.get_size_inches()*fig.dpi
 outside=[]
 for txt in fig.findobj(matplotlib.text.Text):
  if not txt.get_visible() or not txt.get_text():continue
  bb=txt.get_window_extent(renderer)
  if bb.width and (bb.x0 < -1 or bb.x1>W+1 or bb.y0 < -1 or bb.y1>H+1):outside.append(txt.get_text())
 assert not outside,(stem,outside)
 for ax in fig.axes:
  if ax.get_aspect()==1.0:
   # Data unit x/y lengths in display coordinates must match exactly.
   p=ax.transData.transform([[0,0],[1,0],[0,1]]);ratio=np.linalg.norm(p[1]-p[0])/np.linalg.norm(p[2]-p[0]);assert abs(ratio-1)<1e-10
 if a.annotated:
  widths=[ax.get_window_extent(renderer).width for ax in fig.axes if ax.get_aspect()==1.0]
  assert max(widths)-min(widths)<1e-8,('unequal panel widths',widths)
 if a.minimal:
  allowed={'(a) Circular cylinder','(b) Centered square','(c) Fluidic pinball','(a) Centered square','(b) Fluidic pinball'}
  if a.annotated:allowed.update({'Reference','Periodic specialist','T2-C','E2 Top-1','Velocity error','E2 pressure error','T2-C pressure error'})
  for ax in fig.axes:
   assert not ax.get_title()
   if ax.get_aspect()==1.0:assert len(ax.get_xticks())==len(ax.get_yticks())==0
   for txt in ax.texts:assert txt.get_text() in allowed
  assert len(fig.texts)==0
 for ext in (['png'] if a.minimal else ['pdf','png']):fig.savefig(O/f'{stem}.{ext}',dpi=C['dpi'])
 qa.append({'figure':stem,'width_inches':float(fig.get_size_inches()[0]),'height_inches':float(fig.get_size_inches()[1]),'equal_aspect_check':'PASS','text_canvas_bounds':'PASS','font_min_pt':7,'dpi':C['dpi'],'minimal':a.minimal,'png_sha256':sha(O/f'{stem}.png'),**({} if a.minimal else {'pdf_sha256':sha(O/f'{stem}.pdf')})})
 plt.close(fig);print(stem,flush=True)

def overview(field):
 keys=C['overview'];stem=f'field_overview_three_cases_{field}';heights=[]
 for key in keys:heights.extend([.42,rowheight(key),.30,.23])
 fig=plt.figure(figsize=(C['width_inches'],sum(heights)+.36));gs=fig.add_gridspec(len(heights),3,height_ratios=heights,left=.072,right=.987,bottom=.042,top=.94,wspace=.16,hspace=0)
 cmap=C[field+'_cmap'];lab=r'$\|\mathbf{u}\|_2$' if field=='velocity' else r'$p^\circ$';elab=r'$\|\mathbf{u}_{pred}-\mathbf{u}_{ref}\|_2$' if field=='velocity' else r'$|p^\circ_{pred}-p^\circ_{ref}|$'
 for i,key in enumerate(keys):
  meta,z,_,_=DATA[key];norm,en=norms(key,field,['prediction'],stem)
  ax=fig.add_subplot(gs[4*i,:]);ax.axis('off');ax.text(0,.96,title(meta,i),va='top',fontsize=8.3,weight='bold')
  model='Local specialist; original initialization' if key.startswith('circular') else 'T2-C; Deep-FNN-H3 candidates' if key.startswith('pinball') else 'T2-C; fixed sparse-MoE candidates'
  ax.text(0,.40,f"{model}   ·   t={meta['snapshot_time']:.4f}",va='top',fontsize=7.7)
  for col,m in enumerate(['reference','prediction','error']):
   pa=fig.add_subplot(gs[4*i+1,col]);v=error(z['prediction'],z['reference'],field) if m=='error' else scalar(z[m],field)
   panel(pa,key,v,en if m=='error' else norm,C['error_cmap'] if m=='error' else cmap,col==0)
  sub=gs[4*i+2,:].subgridspec(2,3,height_ratios=[.62,.38],wspace=.32)
  cbar(fig,sub[1,:2],norm,cmap,lab+' (shared)');cbar(fig,sub[1,2],en,C['error_cmap'],elab)
 for j,label in enumerate(['POD reference','Prediction','Absolute error']):
  bb=gs[1,j].get_position(fig);fig.text((bb.x0+bb.x1)/2,.982,label,ha='center',va='top',fontsize=8.5,weight='bold')
 save(fig,stem)

def fusion(field,boundary):
 keys=C['fusion'][boundary];stem=f'fusion_comparison_{boundary}_square_pinball_{field}';heights=[]
 for key in keys:heights.extend([.43,rowheight(key),.28,.16,rowheight(key),.28,.18])
 fig=plt.figure(figsize=(C['width_inches'],sum(heights)+.38));gs=fig.add_gridspec(len(heights),3,height_ratios=heights,left=.072,right=.987,bottom=.04,top=.945,wspace=.16,hspace=0)
 cmap=C[field+'_cmap'];lab=r'$\|\mathbf{u}\|_2$' if field=='velocity' else r'$p^\circ$';elab=r'$\|\mathbf{u}_{pred}-\mathbf{u}_{ref}\|_2$' if field=='velocity' else r'$|p^\circ_{pred}-p^\circ_{ref}|$'
 for i,key in enumerate(keys):
  meta,z,_,_=DATA[key];norm,en=norms(key,field,['e2','prediction'],stem);offset=7*i
  ax=fig.add_subplot(gs[offset,:]);ax.axis('off');ax.text(0,.96,title(meta,i),va='top',fontsize=8.3,weight='bold')
  family='Deep-FNN-H3' if key.startswith('pinball') else 'sparse-MoE'
  ax.text(0,.42,f"{family} candidates: {meta['candidate_r'].title()} / Hopf   ·   t={meta['snapshot_time']:.4f}",va='top',fontsize=7.7)
  for j,method in enumerate(['reference','e2','prediction']):panel(fig.add_subplot(gs[offset+1,j]),key,scalar(z[method],field),norm,cmap,j==0)
  sub=gs[offset+2,:].subgridspec(2,3,height_ratios=[.62,.38]);cbar(fig,sub[1,1:],norm,cmap,lab+' (shared across three fields)')
  note=fig.add_subplot(gs[offset+4,0]);note.axis('off');note.text(0,.75,f"Frozen E2 selects\n{meta['E2_selection']}\n\nT2-C weight\nα({meta['candidate_r'].title()}) = {meta['alpha']:.4f}",va='top',fontsize=8,linespacing=1.25)
  for j,method in enumerate(['e2','prediction'],1):
   pa=fig.add_subplot(gs[offset+4,j]);panel(pa,key,error(z[method],z['reference'],field),en,C['error_cmap']);pa.set_title('E2 error' if method=='e2' else 'T2-C error',pad=3,fontsize=8)
  sub=gs[offset+5,:].subgridspec(2,3,height_ratios=[.62,.38]);cbar(fig,sub[1,1:],en,C['error_cmap'],elab+' (shared)')
 for j,label in enumerate(['POD reference','E2 Top-1','T2-C']):
  bb=gs[1,j].get_position(fig);fig.text((bb.x0+bb.x1)/2,.98,label,ha='center',va='top',fontsize=8.5,weight='bold')
 save(fig,stem)

def minimal_figure(field,boundary=None):
 keys=C['overview'] if boundary is None else C['fusion'][boundary]
 stem=f'field_overview_three_cases_{field}' if boundary is None else f'fusion_comparison_{boundary}_square_pinball_{field}'
 heights=[]
 for key in keys:
  x0,x1,y0,y1=C['crop_extent'][key];h=1.63*(y1-y0)/(x1-x0)
  heights.extend([.25,h,.08,.075,.25] if boundary is None else [.25,h,.08,.075,.24,h,.08,.075,.25])
 fig=plt.figure(figsize=(C['width_inches'],sum(heights)+.12))
 gs=fig.add_gridspec(len(heights),3,height_ratios=heights,left=.02,right=.98,bottom=.012,top=.988,wspace=.09,hspace=0)
 labels={'circular':'(a) Circular cylinder','square':'(b) Centered square','pinball':'(c) Fluidic pinball'}
 if boundary is not None:labels={'square':'(a) Centered square','pinball':'(b) Fluidic pinball'}
 cmap=C[field+'_cmap'];step=5 if boundary is None else 9
 for i,key in enumerate(keys):
  meta,z,_,_=DATA[key];offset=i*step
  methods=['prediction'] if boundary is None else ['e2','prediction'];norm,en=norms(key,field,methods,stem)
  name=fig.add_subplot(gs[offset,:]);name.axis('off');name.text(0,.88,labels[key.split('_')[0]],va='top',fontsize=8.5,weight='bold')
  order=['reference','prediction','error'] if boundary is None else ['reference','e2','prediction']
  for j,m in enumerate(order):
   iserr=m=='error';v=error(z['prediction'],z['reference'],field) if iserr else scalar(z[m],field)
   ax=fig.add_subplot(gs[offset+1,j]);panel(ax,key,v,en if iserr else norm,C['error_cmap'] if iserr else cmap)
   if a.annotated:
    label={'reference':'Reference','e2':'E2 Top-1','prediction':'Periodic specialist' if key.startswith('circular') else 'T2-C','error':'Velocity error'}[m]
    inside_label(ax,label)
  if boundary is None:
   cbar(fig,gs[offset+3,:2],norm,cmap,'');cbar(fig,gs[offset+3,2],en,C['error_cmap'],'')
  else:
   cbar(fig,gs[offset+3,:],norm,cmap,'')
   for j,m in enumerate(['e2','prediction'],1):
    ax=fig.add_subplot(gs[offset+5,j]);panel(ax,key,error(z[m],z['reference'],field),en,C['error_cmap'])
    if a.annotated:inside_label(ax,'E2 pressure error' if m=='e2' else 'T2-C pressure error')
   cbar(fig,gs[offset+7,1:],en,C['error_cmap'],'')
 save(fig,stem)

def inside_label(ax,label):
 ax.text(.025,.965,label,transform=ax.transAxes,ha='left',va='top',fontsize=7,
         color='#151515',bbox={'facecolor':'white','edgecolor':'none','alpha':.86,'pad':1.4},zorder=10)

def annotated_below(field,boundary=None):
 keys=C['overview'] if boundary is None else C['fusion'][boundary]
 stem=f'field_overview_three_cases_{field}' if boundary is None else f'fusion_comparison_{boundary}_square_pinball_{field}'
 # Same three-column width everywhere. Extra slot height makes width the
 # limiting dimension; equal data aspect preserves both circle and square geometry.
 panel_width=C['width_inches']*.96/(3+2*.09)
 heights=[]
 for key in keys:
  x0,x1,y0,y1=C['crop_extent'][key];h=panel_width*(y1-y0)/(x1-x0)+.06
  heights.extend([.25,h,.23,.06,.075,.26] if boundary is None else [.25,h,.23,.06,.075,.25,h,.23,.06,.075,.26])
 fig=plt.figure(figsize=(C['width_inches'],sum(heights)/.976))
 gs=fig.add_gridspec(len(heights),3,height_ratios=heights,left=.02,right=.98,bottom=.012,top=.988,wspace=.09,hspace=0)
 labels={'circular':'(a) Circular cylinder','square':'(b) Centered square','pinball':'(c) Fluidic pinball'} if boundary is None else {'square':'(a) Centered square','pinball':'(b) Fluidic pinball'}
 def label_at(slot,label):
  ax=fig.add_subplot(slot);ax.axis('off');ax.text(.5,.55,label,ha='center',va='center',fontsize=7.5)
 for i,key in enumerate(keys):
  meta,z,_,_=DATA[key];offset=i*(6 if boundary is None else 11)
  methods=['prediction'] if boundary is None else ['e2','prediction'];norm,en=norms(key,field,methods,stem);cmap=C[field+'_cmap']
  name=fig.add_subplot(gs[offset,:]);name.axis('off');name.text(0,.88,labels[key.split('_')[0]],va='top',fontsize=8.5,weight='bold')
  order=['reference','prediction','error'] if boundary is None else ['reference','e2','prediction']
  for j,m in enumerate(order):
   iserr=m=='error';v=error(z['prediction'],z['reference'],field) if iserr else scalar(z[m],field)
   ax=fig.add_subplot(gs[offset+1,j]);panel(ax,key,v,en if iserr else norm,C['error_cmap'] if iserr else cmap)
   ax.set_anchor('S')
   label_at(gs[offset+2,j],{'reference':'Reference','e2':'E2 Top-1','prediction':'Periodic specialist' if key.startswith('circular') else 'T2-C','error':'Velocity error'}[m])
  if boundary is None:
   cbar(fig,gs[offset+4,:2],norm,cmap,'');cbar(fig,gs[offset+4,2],en,C['error_cmap'],'')
  else:
   cbar(fig,gs[offset+4,:],norm,cmap,'')
   for j,m in enumerate(['e2','prediction'],1):
    ax=fig.add_subplot(gs[offset+6,j]);panel(ax,key,error(z[m],z['reference'],field),en,C['error_cmap']);ax.set_anchor('S')
    label_at(gs[offset+7,j],'E2 pressure error' if m=='e2' else 'T2-C pressure error')
   cbar(fig,gs[offset+9,1:],en,C['error_cmap'],'')
 save(fig,stem)

if a.annotated:
 annotated_below('pressure','hp')
 annotated_below('velocity')
else:
 for f in ['pressure','velocity']:
  (minimal_figure if a.minimal else overview)(f)
  for b in ['hp','sh']:(minimal_figure if a.minimal else fusion)(f,b)
MD=O if a.minimal else D
(MD/'panel_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
rows=[]
for m in manifest:
 for method,metrics in m['snapshot_metrics'].items():
  if method!='reference':rows.append(dict(case=m['case_id'],method=method,Re=m['Re'],window_id=m['window_id'],k=m['k'],K=m['K'],time=m['snapshot_time'],Eu_percent=100*metrics['Eu'],Ep_percent=100*metrics['Ep'],joint_percent=100*(metrics['Eu']+metrics['Ep'])))
with (MD/'snapshot_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(O/'render_qa.json').write_text(json.dumps(qa,indent=2));print('Metrics, manifest and render QA saved.')
