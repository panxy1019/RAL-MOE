"""Candidate A: compact horizontal overview with one closure zoom.

Editable vector shapes only; schematic trajectories are not CFD results.
Run with C:/ProgramData/anaconda3/python.exe. See A_notes.md for semantics.
"""
from pathlib import Path
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np

OUT = Path(__file__).with_suffix("")
INK, MUTED, LINE = "#263644", "#72808C", "#C9D2D9"
PURPLE, PURPLE_BG = "#8066AB", "#F1EDF8"
WARM, WARM_BG = "#CE8849", "#FFF4E8"
BLUE, BLUE_BG = "#487FA8", "#EDF5FB"
TEAL, TEAL_BG = "#4E8F7E", "#EEF7F3"
W, H = 120, 52
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 18.5,
    "mathtext.fontset": "stix", "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none",
    "svg.hashsalt": "ral-compact-A-20260907",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})
fig = plt.figure(figsize=(12, 5.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(H, 0), aspect="equal")
ax.axis("off")
TEXT = []


def label(x, y, s, size=18.5, color=INK, weight="normal", ha="center", **kwargs):
    t = ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                ha=ha, va="center", linespacing=1.08, zorder=8, **kwargs)
    TEXT.append(t)
    return t


def box(x, y, w, h, fill="white", edge=LINE, radius=.65, lw=1.2, dashed=False):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        fc=fill, ec=edge, lw=lw, zorder=2,
        linestyle=(0, (3.5, 3.5)) if dashed else "solid")
    ax.add_patch(p)
    return p


def route(points, color=INK, lw=1.25, arrow=True, dashed=False, zorder=3):
    xx, yy = zip(*points)
    ax.plot(xx, yy, color=color, lw=lw, solid_capstyle="round",
            solid_joinstyle="round", ls=(0, (3, 3)) if dashed else "-",
            zorder=zorder)
    if arrow:
        p, q = np.asarray(points[-2]), np.asarray(points[-1])
        d = q - p
        start = q - d / np.linalg.norm(d) * min(.8, np.linalg.norm(d))
        ax.add_patch(FancyArrowPatch(start, q, arrowstyle="-|>",
            mutation_scale=12, color=color, lw=lw, shrinkA=0, shrinkB=0,
            zorder=zorder+.1))


def dot(x, y, color=INK, radius=.15):
    ax.add_patch(Circle((x, y), radius, fc=color, ec="none", zorder=5))


def sum_node(x, y, color=INK, radius=.85, symbol="+"):
    ax.add_patch(Circle((x, y), radius, fc="white", ec=color, lw=1.2, zorder=5))
    label(x, y-.08, symbol, 20, color)


def grid(x, y, color=BLUE, tint=BLUE_BG, w=3.7, h=3.2):
    """Neutral grid glyph: no invented scalar values or flow field."""
    ax.add_patch(Polygon([(x,y), (x+w,y), (x+w,y+h), (x,y+h)],
                        fc=tint, ec=color, lw=1, zorder=4))
    for k in range(1, 4):
        route([(x+w*k/4,y), (x+w*k/4,y+h)], color, .45, False, zorder=5)
    for k in range(1, 3):
        route([(x,y+h*k/3), (x+w,y+h*k/3)], color, .45, False, zorder=5)


def encoder(x, y, regime):
    ax.add_patch(Polygon([(x,y-2), (x+4,y-1.3), (x+4,y+1.3), (x,y+2)],
                        fc=BLUE_BG, ec=BLUE, lw=1.15, zorder=4))
    label(x+1.9, y, "$E_"+regime+"$", 20, BLUE)


def decoder(x, y, regime):
    ax.add_patch(Polygon([(x,y-1.3), (x+4,y-2), (x+4,y+2), (x,y+1.3)],
                        fc=BLUE_BG, ec=BLUE, lw=1.15, zorder=4))
    label(x+2.1, y, "$D_"+regime+"$", 20, BLUE)


def native_path(x, y, width, phase=0):
    tt = np.linspace(0, 1, 100)
    yy = y + .63*np.sin(2*np.pi*tt+phase)
    ax.plot(x+width*tt, yy, color=WARM, lw=1.6, zorder=4)
    for t in (.04, .46, .91):
        ax.add_patch(Circle((x+width*t, y+.63*np.sin(2*np.pi*t+phase)),
                           .22, fc="white", ec=WARM, lw=1.1, zorder=5))
    route([(x+width-.5,yy[-1]), (x+width+.2, yy[-1])], WARM, 1.4)


# The only incoming edge to E2 is the physical parameter.
label(4.0, 8.0, r"$\mu$", 27, PURPLE)
route([(6.2,8), (9,8)], PURPLE)
box(9, 5.45, 10, 5.1, PURPLE_BG, PURPLE)
label(14, 8, "E2", 20, PURPLE, "bold")
route([(19,8), (28.4,8)], PURPLE)
label(23.7, 4.9, r"$\pi(\mu)$", 22, PURPLE)
label(23.7, 11.0, "fixed", 18.5, MUTED)

# Adjacency is drawn as a chain, not an all-to-all regime selector.
label(36.9, 4.9, "Top-1 / Top-2", 18.5, PURPLE)
box(28.4, 7, 17, 11, "#FBFAFD", "#D8CEE7", lw=.9)
route([(31.6,11.6), (42.2,11.6)], PURPLE, 1.1, False)
for x, ch in [(31.6,"S"), (36.9,"H"), (42.2,"P")]:
    ax.add_patch(Circle((x,11.6), 1.7, fc="white", ec=PURPLE, lw=1.2, zorder=5))
    label(x, 11.6, "$"+ch+"$", 21, PURPLE)
label(36.9, 15.7, r"$S$-$H$ / $H$-$P$", 18.5, MUTED)

# Selection activates chart(s); it is not a cross-chart state transformation.
label(65.0, 3.5, "Local rollouts", 19, INK, "bold")
box(51, 7.6, 28.5, 17.9, "#FFFCF8", "#E1BE9C", lw=1.15)
route([(45.4,11.6), (47.2,11.6), (47.2,6.1), (65,6.1), (65,7.6)], PURPLE)
route([(52.4,17.5), (78.1,17.5)], "#E5D6C5", .75, False, True)
for y, regime, phase in [(13,"r",0), (21,"s",.8)]:
    encoder(52.3, y, regime)
    route([(56.3,y), (59,y)], BLUE)
    native_path(59.6, y, 14.8, phase)
    label(66.8, y-2.35, r"$a^{("+regime+r")}$", 21, WARM)
    route([(75.2,y), (82,y)], WARM)
    decoder(82, y, regime)
    route([(86,y), (89,y)], BLUE)
    grid(89, y-1.6)
    label(90.9, y+3.0, r"$\widehat q^{("+regime+r")}$", 21, BLUE)
    route([(92.7,y), (97.4,y)], TEAL)
label(65.2, 24.0, "Top-2 only", 18.5, MUTED)
label(86.5, 3.5, "Decode", 18.5, INK, "bold")

# Both incoming quantities are physical fields. Top-1 is the identity case.
label(106.2, 3.5, "Physical fusion", 19, TEAL, "bold")
box(97.4, 8.6, 17.3, 16.0, TEAL_BG, TEAL, lw=1.5)
label(106.1, 10.8, "T2-C", 20, TEAL, "bold")
route([(97.4,13), (101,13), (106,17)], TEAL)
route([(97.4,21), (101,21), (106,17)], TEAL)
label(100.8, 15.6, r"$\alpha$", 21, TEAL)
label(101.0, 23.0, r"$1-\alpha$", 19, TEAL)
sum_node(106.6,17,TEAL)
route([(107.5,17), (116.1,17)], TEAL, 1.5)
label(118,17,r"$\widehat q$",24,TEAL)
label(106.2,27.0,"Top-1: identity",18.5,MUTED)

# History branches into selected encoders and a chart-independent descriptor.
label(6.4,24.9,"History",18.5,BLUE)
label(6.4,28.5,r"$\mathcal{H}$",26,BLUE)
route([(10.0,28.5), (49.0,28.5), (49.0,13), (52.3,13)], BLUE)
dot(49,21,BLUE)
route([(49,21),(52.3,21)],BLUE)
label(37.5,26.3,"initialize",18.5,BLUE)
dot(14,28.5,BLUE)
route([(14,28.5), (14,31.5), (92.6,31.5)], BLUE)
label(57.5,31.5,r"chart-independent $d$",18.5,BLUE,
      bbox=dict(facecolor="white",edgecolor="none",pad=2.7))
box(92.6,29,22.1,5.0,PURPLE_BG,PURPLE,lw=1.15)
label(103.65,31.5,r"$\alpha(\mu,\bar\pi,d)$",22,PURPLE)
route([(114.7,31.5),(116.6,31.5),(116.6,23),(114.7,23)],PURPLE)

# One shallow zoom retains the sparse closure as a visual mechanism.
route([(2.4,35), (117.6,35)],LINE,.75,False)
label(7.6,37.9,"One chart",18.5,MUTED)
label(6.8,44.6,r"$(a,b)_n$",22,BLUE)
route([(12.1,44.6),(14,44.6)],BLUE,arrow=False)
dot(14,44.6,BLUE)
route([(14,44.6),(14,38.8),(18,38.8)],BLUE)
box(18,36.5,28,4.6,BLUE_BG,BLUE,lw=1.15)
label(32,38.8,r"Galerkin $\mathcal{F}_r^G$",18.5,BLUE)

# Group Top-1, then shared expert in parallel with Top-2 routed experts.
box(18,43.1,49,8.0,WARM_BG,WARM,lw=1.35)
label(55.5,42.0,"Sparse closure",18.5,WARM,"bold",
      bbox=dict(facecolor="white",edgecolor="none",pad=1.4))
route([(14,44.6),(14,47.6),(20,47.6)],BLUE)
box(20,45.2,9.8,4.8,PURPLE_BG,PURPLE,radius=.4,lw=1)
label(24.9,47.6,r"$g^\star$",23,PURPLE)
label(24.9,42.5,"Top-1",18.5,PURPLE)
route([(29.8,47.6),(33,47.6),(33,45.1),(36.3,45.1)],PURPLE)
route([(33,47.6),(33,49),(36.3,49)],PURPLE)
dot(33,47.6,PURPLE)
box(36.3,43.6,25.0,3.0,"white",WARM,radius=.3,lw=1)
label(48.8,45.1,r"Shared $E_0$",18.5,WARM)
box(35.7,47.0,25.6,4.0,"#FFFCF8","#DDB993",radius=.3,lw=.8,dashed=True)
for x, e in [(36.3,"1"),(44.9,"2")]:
    box(x,47.4,7.6,3.3,"#FFE4C6",WARM,radius=.3,lw=1)
    label(x+3.8,49.0,r"$E_{"+e+r"}$",20,WARM)
label(58.3,49.0,"Top-2",18.5,WARM)
route([(61.3,45.1),(64.6,45.1),(64.6,46.7)],WARM,arrow=False)
route([(61.3,49),(64.6,49),(64.6,48.3)],WARM,arrow=False)
sum_node(64.6,47.5,WARM,.85,r"$\Sigma$")
route([(65.4,47.5),(69.3,47.5)],WARM)
label(72.1,47.5,r"$\rho_r^u$",23,WARM)
route([(46,38.8),(78,38.8)],BLUE)
route([(74.5,47.5),(78.8,47.5),(78.8,39.7)],WARM)
sum_node(78.8,38.8)
route([(79.7,38.8),(88,38.8)])
label(83.8,36.8,r"$\dot a$",22,INK)
box(88,36.3,14,5.0,BLUE_BG,BLUE)
label(95,38.8,"RK",20,BLUE,"bold")
label(95,43.5,"Velocity",18.5,BLUE)
route([(102,38.8),(108,38.8)],BLUE)
label(113,38.8,r"$a_{n+1}^{(r)}$",25,BLUE)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
outside = []
overlaps = []
for i, t in enumerate(TEXT):
    b = t.get_window_extent(renderer)
    if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.x1 or b.y1 > fig.bbox.y1:
        outside.append(t.get_text())
    for u in TEXT[i+1:]:
        c = u.get_window_extent(renderer)
        if b.overlaps(c):
            overlaps.append([t.get_text(),u.get_text()])
if outside:
    raise ValueError(f"Text outside canvas: {outside}")
english_words = []
for t in TEXT:
    plain = re.sub(r"\$[^$]*\$", "", t.get_text())
    english_words.extend(re.findall(r"[A-Za-z]+(?:-[A-Za-z0-9]+)*", plain))
for suffix in ("pdf","svg","png"):
    meta = {"Title":"RAL-MoE-ROM candidate A: compact horizontal overview"}
    if suffix == "pdf":
        meta.update({"CreationDate":None,"ModDate":None})
    if suffix == "svg":
        meta["Date"] = None
    fig.savefig(OUT.with_suffix("."+suffix), dpi=240, metadata=meta)
report = {
    "outputs":[str(OUT.with_suffix("."+s)) for s in ("pdf","svg","png")],
    "english_word_count":len(english_words), "english_words":english_words,
    "min_source_font_pt":min(t.get_fontsize() for t in TEXT),
    "min_font_at_397pt_width":18.5*397/(12*72),
    "text_overlap_candidates":overlaps,"text_outside_canvas":outside,
    "vector_only":True}
OUT.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
