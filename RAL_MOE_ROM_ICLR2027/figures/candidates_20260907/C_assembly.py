"""Candidate C: independent local charts meet only in physical space.

An admissible H-P pair is shown schematically; Top-1 is the identity case.
No source-paper edits. All trajectories/meshes are explanatory vector glyphs.
"""
from pathlib import Path
import json
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

OUT = Path(__file__).with_suffix("")
INK, MUTED = "#233344", "#657585"
BLUE, PURPLE, WARM, GREEN = "#487FA8", "#8066AB", "#CE8849", "#4E8F7E"
BG_BLUE, BG_PURPLE, BG_WARM, BG_GREEN = "#EDF5FB", "#F1EDF8", "#FFF4E8", "#EEF7F3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 19,
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none",
    "svg.hashsalt": "ral-c-compact-20260907", "savefig.facecolor": "white"})
fig = plt.figure(figsize=(12, 5.28))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0,100), ylim=(44,0), aspect="equal")
ax.axis("off")
texts = []


def t(x,y,s,size=19,color=INK,bold=False,ha="center",**kw):
    a = ax.text(x,y,s,fontsize=size,color=color,weight="bold" if bold else "normal",
        ha=ha,va="center",linespacing=1.1,zorder=8,**kw)
    texts.append(a)


def box(x,y,w,h,fill="white",edge="#BFCAD2",dash=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=.65",
        fc=fill,ec=edge,lw=1.25,ls=(0,(4,3)) if dash else "-",zorder=1))


def line(points,color=INK,arrow=True,dash=False,lw=1.5,z=3):
    x,y=zip(*points)
    ax.plot(x,y,color=color,lw=lw,ls=(0,(4,3)) if dash else "-",
        solid_joinstyle="round",solid_capstyle="round",zorder=z)
    if arrow:
        end,start=np.array(points[-1]),np.array(points[-2])
        d=end-start
        start=end-d/np.linalg.norm(d)*min(.8,np.linalg.norm(d))
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=14,
            shrinkA=0,shrinkB=0,color=color,lw=lw,zorder=z+.1))


def dot(x,y,color=INK):
    ax.add_patch(Circle((x,y),.14,color=color,zorder=5))


def grid(x,y,w=4.5,h=3.9,color=BLUE):
    ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],
        fc=BG_BLUE if color==BLUE else BG_GREEN,ec=color,lw=1.1,zorder=3))
    for i in range(1,5):
        line([(x+w*i/5,y),(x+w*i/5,y+h)],color,False,lw=.6)
    for j in range(1,4):
        line([(x,y+h*j/4),(x+w,y+h*j/4)],color,False,lw=.6)


def plane(x,y,letter,kind):
    # Different projected axes represent noncommensurate reduced charts.
    poly = [(x,y+2),(x+11,y),(x+16,y+4.8),(x+4,y+6.2)]
    ax.add_patch(Polygon(poly,fc=BG_WARM,ec="#DDB58E",lw=1.05,zorder=2))
    line([(x+3,y+4.6),(x+11.8,y+3.1)],WARM,True,lw=1.1)
    line([(x+3,y+4.6),(x+1,y+2.6)],WARM,True,lw=1.1)
    tt=np.linspace(0,2*np.pi,160)
    if kind=="H":
        rr=np.linspace(.18,1.25,len(tt))
        xx,yy=x+8+rr*np.cos(2*tt)*1.9, y+3+rr*np.sin(2*tt)*.8
    else:
        xx,yy=x+8+2.6*np.cos(tt),y+3+1.0*np.sin(tt)
    ax.plot(xx,yy,color=WARM,lw=1.9,zorder=4)
    t(x-1.6,y+3.2,"$"+letter+"$",25,WARM,True)


def decoder(x,y):
    ax.add_patch(Polygon([(x,y+.6),(x+4,y),(x+4,y+4.8),(x,y+4.2)],
        fc=BG_BLUE,ec=BLUE,lw=1.2,zorder=2))
    t(x+2.2,y+2.4,"$D$",25,BLUE)


# Outer routing. The initial history does not enter E2.
t(40.25,1.8,"Regime routing",20,PURPLE,True)
t(4.5,7.1,r"$\mu$",30,PURPLE)
line([(7,7.1),(11,7.1)],PURPLE)
box(11,4.1,10,6,BG_PURPLE,PURPLE)
t(16,7.1,"E2",22,PURPLE,True)
line([(21,7.1),(28,7.1)],PURPLE)
t(24.5,5.2,r"$\pi$",24,PURPLE)
box(28,3.8,24.5,7.8,"#FCFBFE","#B9ADCB",True)
line([(35.4,7),(37.85,7)],"#AAA4B7",False,lw=1.1)
line([(42.65,7),(45.1,7)],"#AAA4B7",False,lw=1.1)
for x,name,active in [(33,"S",False),(40.25,"H",True),(47.5,"P",True)]:
    box(x-2.4,4.9,4.8,4.2,BG_WARM if active else "#F2F3F5",
        WARM if active else "#C7CDD3")
    t(x,7,"$"+name+"$",23,WARM if active else MUTED,True)
t(14.5,15.8,"Top-1 or\nadjacent Top-2",18.5,MUTED)
t(14.5,19.4,"H-P shown",18.5,MUTED)

# Independent local charts. One representative H-P pair is depicted.
box(28,14.1,24.5,23.3,"#FFFCF8","#E8D0B9",True)
t(40.2,16,"Independent charts",18.5,WARM,True)
plane(33.2,17.5,"H","H")
plane(33.2,27.7,"P","P")
t(40.25,35.9,"ROM + sparse MoE",19,WARM,True)
line([(40.25,11.6),(40.25,14.1)],PURPLE)

# Input history and initialization are orthogonal to chart selection.
for dx,dy in [(.5,-.5),(.25,-.25),(0,0)]:
    grid(3.1+dx,22.5+dy,3.4,3.1)
t(10.5,24,r"$\mathcal{H}_n$",27,BLUE)
t(8,29,"History",19,MUTED)
line([(13.2,24),(24.2,24),(24.2,21),(28,21)],BLUE)
line([(24.2,24),(24.2,31),(28,31)],BLUE)
dot(24.2,24,BLUE)

# Each chart is reconstructed before any common-space assembly.
t(58.3,16,"Decode",19,BLUE,True)
for yy in [20.7,30.9]:
    line([(49.7,yy),(55.4,yy)])
    decoder(55.4,yy-2.4)
    line([(59.4,yy),(67.3,yy)])

# Shared physical coordinates allow a convex mix. Top-1 is an identity.
box(64.5,3.8,34,32.6,"#F7FBF9","#B5D2C8")
t(81.5,6.5,"Common physical space",19.5,GREEN,True)
t(81.5,10.3,"Top-1: pass-through",18.5,MUTED)
for yy,rr in [(20.7,"H"),(30.9,"P")]:
    grid(67.3,yy-1.95,4.6,3.9,GREEN)
    t(69.6,yy+3.4,r"$\widehat{q}^{("+rr+")}$",22,GREEN)
# Separate, visibly parallel inputs to T2-C.
line([(71.9,20.7),(77,20.7),(77,23.6),(80.1,23.6)])
line([(71.9,30.9),(77,30.9),(77,26.6),(80.1,26.6)])
box(80.1,21.7,9,6.8,BG_GREEN,GREEN)
t(84.6,24,"T2-C",21,GREEN,True)
t(84.6,27,"Top-2",18.5,MUTED)
line([(89.1,25.1),(93,25.1)])
grid(93,23.4,3.2,3.4,GREEN)
t(94.6,30.1,r"$\widehat{q}$",26,GREEN)
# Top-1 bypass carries only one reconstructed field.
dot(73.5,20.7)
line([(73.5,20.7),(73.5,14.1),(94.6,14.1),(94.6,23.4)],GREEN,lw=1.25)

# Chart-independent history descriptor for the assembly gate.
dot(19,24,BLUE)
line([(19,24),(19,40),(31.5,40)],BLUE)
box(31.5,38,17,4,BG_BLUE,BLUE)
t(40,40,r"Descriptor $d_n$",19,BLUE)
line([(48.5,40),(69.5,40)],BLUE)
box(69.5,38,21.5,4,BG_PURPLE,PURPLE)
t(80.25,40,r"$\alpha(\mu,\bar\pi,d_n)$",24,PURPLE)
line([(91,40),(97.3,40),(97.3,35.3),(84.6,35.3),(84.6,28.5)],PURPLE)

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
outside=[]
overlaps=[]
for i,a in enumerate(texts):
    b=a.get_window_extent(renderer)
    if b.x0<0 or b.y0<0 or b.x1>fig.bbox.x1 or b.y1>fig.bbox.y1:
        outside.append(a.get_text())
    for other in texts[i+1:]:
        if b.overlaps(other.get_window_extent(renderer)):
            overlaps.append([a.get_text(),other.get_text()])
if outside:
    raise ValueError(f"Text outside canvas: {outside}")
for ext in ("pdf","svg","png"):
    metadata={"Title":"RAL-MoE-ROM / C: independent charts, common physical space"}
    if ext=="pdf": metadata.update({"CreationDate":None,"ModDate":None,"Author":""})
    if ext=="svg": metadata["Date"]=None
    fig.savefig(OUT.with_suffix("."+ext),dpi=220,metadata=metadata)
visible=" ".join(re.sub(r"\$.*?\$","",a.get_text()) for a in texts)
stats={"ordinary_english_words":len(re.findall(r"[A-Za-z][A-Za-z0-9-]*",visible)),
       "min_source_font_pt":min(a.get_fontsize() for a in texts),
       "min_font_at_397pt_width":min(a.get_fontsize() for a in texts)*397/864,
       "width_pt":864,"height_pt":380.16,"variant":"C"}
stats.update({"text_overlap_candidates":overlaps,"text_outside_canvas":outside})
OUT.with_suffix(".json").write_text(json.dumps(stats,indent=2)+"\n")
print(json.dumps(stats,indent=2))
