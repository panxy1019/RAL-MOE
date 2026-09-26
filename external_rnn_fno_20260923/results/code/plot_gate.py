from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
for reg in ['H','P']:
 out=OUT/'fno_gate'/reg;z=np.load(out/'visual_1536x1024.npz');xy=z['points'][:,:2];unique,ind=np.unique(xy,axis=0,return_index=True)
 tri=mtri.Triangulation(unique[:,0],unique[:,1]);cent=unique[tri.triangles].mean(1);tri.set_mask((cent**2).sum(1)<.5**2)
 fig,ax=plt.subplots(2,3,figsize=(13,5),layout='constrained')
 for row,(component,label) in enumerate([(1,'Transverse velocity'),(2,'Pressure')]):
  orig=z['original'][ind,component];rest=z['restored'][ind,component];lo,hi=np.quantile(orig,[.005,.995])
  for col,(v,title) in enumerate([(orig,'CFD'),(rest,'FV-grid-FV'),(rest-orig,'Interpolation error')]):
   lim=max(abs(v.min()),abs(v.max()));im=ax[row,col].tripcolor(tri,v,shading='gouraud',cmap='RdBu_r',vmin=lo if col<2 else -lim,vmax=hi if col<2 else lim)
   ax[row,col].set(xlim=(-2,10),ylim=(-3,3),aspect='equal',title=f'{label}: {title}');ax[row,col].add_patch(plt.Circle((0,0),.5,color='white',ec='black',lw=.5));fig.colorbar(im,ax=ax[row,col],shrink=.8)
 fig.suptitle(f'{reg}: validation field; selected largest audit grid 1536 x 1024')
 fig.savefig(out/'roundtrip_visual.png',dpi=180);plt.close(fig)
