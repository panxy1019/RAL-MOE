"""Editable vector Figure 1: publication-scale hierarchy, not a box transcript.

Python + Matplotlib + NumPy; exports PDF, text-editable SVG and 220 dpi PNG.
All trajectory and mesh glyphs are schematic, not experimental flow results.
Design sources and semantic checks: ../DIAGRAM_DESIGN_NOTES_20260907.md.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, FancyArrowPatch
import numpy as np

OUT = Path(__file__).with_suffix("")
W, H = 100, 67.5
INK, MUTED = "#233344", "#617487"
BLUE, BLUE_BG = "#487FA8", "#EDF5FB"
PURPLE, PURPLE_BG = "#8066AB", "#F1EDF8"
WARM, WARM_BG = "#CE8849", "#FFF4E8"
GREEN, GREEN_BG = "#4E8F7E", "#EEF7F3"
GRAY, GRAY_BG = "#B6C1CA", "#F5F7F9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17,
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "svg.hashsalt": "ral-20260907",
    "savefig.facecolor": "white", "figure.facecolor": "white"})
fig = plt.figure(figsize=(13, 13*H/W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(H, 0), aspect="equal")
ax.axis("off")
TEXT = []


def text(x, y, s, size=17, color=INK, weight="normal", ha="center", **kw):
    t = ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha,
                va="center", linespacing=1.16, zorder=6, **kw)
    TEXT.append(t)
    return t


def box(x, y, w, h, fill="white", edge=GRAY, radius=.55, lw=1.2, dashed=False):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}", fc=fill, ec=edge,
        lw=lw, zorder=2, linestyle=(0, (4, 3)) if dashed else "solid")
    ax.add_patch(p)
    return p


def route(points, color=INK, lw=1.5, arrow=True, dashed=False, z=3):
    xx, yy = zip(*points)
    ax.plot(xx, yy, color=color, lw=lw, solid_joinstyle="round",
            solid_capstyle="round", ls=(0, (4, 3)) if dashed else "-", zorder=z)
    if arrow:
        start, end = np.array(points[-2]), np.array(points[-1])
        vector = end - start
        length = np.linalg.norm(vector)
        tipstart = end-vector/length*min(.9, length)
        ax.add_patch(FancyArrowPatch(tipstart, end, arrowstyle="-|>",
            mutation_scale=13, color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=z+.1))


def junction(x, y, color=INK):
    ax.add_patch(Circle((x, y), .14, fc=color, ec="none", zorder=5))


def merge(x, y, symbol="+", radius=.9):
    ax.add_patch(Circle((x, y), radius, fc="white", ec=INK, lw=1.5, zorder=5))
    text(x, y-.04, symbol, 21)


def field_tile(x, y, w=3.7, h=3, stack=True, color=BLUE):
    """Neutral vector mesh cells; deliberately not a simulated field."""
    if stack:
        for dx, dy in [(.55, -.55), (.28, -.28)]:
            ax.add_patch(Polygon([(x+dx, y+dy), (x+w+dx, y+dy),
                (x+w+dx, y+h+dy), (x+dx, y+h+dy)],
                fc="white", ec=color, lw=.7, zorder=2))
    ax.add_patch(Polygon([(x,y), (x+w,y), (x+w,y+h), (x,y+h)],
        fc=BLUE_BG if color == BLUE else GREEN_BG, ec=color, lw=1, zorder=3))
    for i in range(1, 5):
        route([(x+w*i/5,y), (x+w*i/5,y+h)], color, .45, False)
    for j in range(1, 4):
        route([(x,y+h*j/4), (x+w,y+h*j/4)], color, .45, False)


def chart_glyph(x, y, kind, color):
    t = np.linspace(0, 1, 100)
    if kind == "S":
        xx, yy = x+2.5*t, y-.65*np.exp(-3*t)*np.cos(6*np.pi*t)
    elif kind == "H":
        phase, r = 4.7*np.pi*t, .1+.7*t
        xx, yy = x+1.3+r*np.cos(phase), y+r*np.sin(phase)
    else:
        xx, yy = x+2.5*t, y-.65*np.sin(3*np.pi*t)
    ax.plot(xx, yy, color=color, lw=1.45, zorder=4)


def trajectory(x, y, w, phase=0):
    """Schematic native-coordinate path, with successive states."""
    t = np.linspace(0, 1, 120)
    yy = y+.52*np.sin(2*np.pi*t+phase)
    ax.plot(x+w*t, yy, color=WARM, lw=1.5, zorder=4)
    for tt in [.05, .48, .92]:
        ax.add_patch(Circle((x+w*tt, y+.52*np.sin(2*np.pi*tt+phase)),
                            .21, fc="white", ec=WARM, lw=1.1, zorder=5))
    route([(x+w-.45,yy[-1]), (x+w+.15,yy[-1])], WARM, 1.3)


def decoder(x, y, label):
    ax.add_patch(Polygon([(x,y+.6), (x+3.8,y), (x+3.8,y+4.4), (x,y+3.8)],
                         fc=BLUE_BG, ec=BLUE, lw=1.15, zorder=2))
    text(x+2.1, y+2.2, label, 21, BLUE)


# (a) The parameter/control lane is distinct from the physical-history lane.
text(1.5, 2, "(a) Across regime-local charts", 22, weight="bold", ha="left")
text(98.5, 2, "LEVEL 2  /  REGIME ROUTING", 16.5, MUTED, ha="right")
route([(1.5,4), (98.5,4)], GRAY, .65, False)
text(5, 10.6, "Parameter", 17)
text(4.1, 14, r"$\mu$", 29, PURPLE)
route([(6.8,14), (10,14)], PURPLE)
box(10, 10.5, 12, 7, PURPLE_BG, PURPLE)
text(16, 12.7, "E2 router", 18, weight="bold")
for xx, ys in [(13.1,[15,16.2]), (16,[14.6,15.6,16.6]), (18.9,[15,16.2])]:
    if xx > 13.1:
        px, pys = (13.1,[15,16.2]) if xx == 16 else (16,[14.6,15.6,16.6])
        for py in pys:
            for cy in ys:
                route([(px,py), (xx,cy)], PURPLE, .6, False)
    for yy in ys:
        ax.add_patch(Circle((xx,yy), .2, fc="white", ec=PURPLE, lw=.9, zorder=4))
text(16, 19, "parameter only", 16.5, MUTED)
route([(22,14), (29,14)], PURPLE)
text(25.5, 12.25, r"$\boldsymbol{\pi}(\mu)$", 22, PURPLE)
text(25.5, 16.3, "fixed", 16.5, MUTED)
box(29, 6.3, 17, 17, "#FAFAFC", GRAY, dashed=True)
text(37.5, 8.1, "Top-1 / Top-2", 18, weight="bold")
for yy, name, letter, edge, fill in [
    (10,"Steady","S",BLUE,BLUE_BG), (14.3,"Hopf","H",WARM,WARM_BG),
    (18.6,"Periodic","P",GREEN,GREEN_BG)]:
    box(30.1, yy, 14.8, 3.3, fill, edge)
    text(31.2, yy+1.65, name, 16.5, ha="left")
    text(39.3, yy+1.65, "$"+letter+"$", 20)
    chart_glyph(41, yy+1.65, letter, edge)
text(37.5, 24.7, "Pairs: S-H or H-P", 16.5, MUTED)

text(57.5, 6.3, "Independent\nrollouts", 16.5, weight="bold")
box(49.5, 8.6, 16, 15.9, "#FFFCF8", WARM, radius=.8)
route([(50.6,16.2), (64.4,16.2)], "#E7CFB7", .9, False, True)
text(57.5, 10.4, r"Specialist $r$", 18, weight="bold")
trajectory(52, 13.4, 10.6)
text(57.5, 18.4, r"Specialist $s$", 18, weight="bold")
trajectory(52, 21.4, 10.6, .7)
text(63.6, 23.3, "Top-2 only", 16.5, MUTED, ha="right")
route([(46,14), (47.7,14), (47.7,12), (49.5,12)], PURPLE)
route([(47.7,14), (47.7,20), (49.5,20)], PURPLE)
junction(47.7, 14, PURPLE)
text(73.7, 6.3, "Reconstruct", 16.5, weight="bold")
for yy, rr in [(12,"r"), (20,"s")]:
    route([(65.5,yy), (68,yy)])
    decoder(68, yy-2.2, "$D_"+rr+"$")
    route([(71.8,yy), (74.2,yy)])
    field_tile(74.2, yy-1.5, 3.9, 3, False)
    text(76.2, yy+2.65, r"$\widehat{\boldsymbol{q}}^{("+rr+")}$", 19, BLUE)
    route([(78.1,yy), (81.6,yy)])
# The first reconstructed field participates in BOTH alternatives. The second
# exists only for Top-2. This explicit branch prevents a false s-only fusion.
junction(79.6, 12)
route([(79.6,12), (79.6,18.2), (81.6,18.2)])
text(87.9, 6.3, "Physical space", 16.5, weight="bold")
box(81.6, 9.5, 12.6, 13.5, GREEN_BG, GREEN, radius=.8, lw=1.4)
text(87.9, 12, "Top-1", 17, weight="bold")
text(87.9, 14.3, "Identity", 17, GREEN)
route([(82.8,16), (93,16)], "#C3DBD2", .9, False)
text(87.9, 18.4, "Top-2", 17, weight="bold")
text(87.9, 20.9, "T2-C", 19, GREEN, weight="bold")
route([(94.2,16.25), (96,16.25)])
field_tile(96, 14.9, 2.6, 2.7, True, GREEN)
text(97.3, 20.1, r"$\widehat{\boldsymbol{q}}_{n+1}$", 21, GREEN)

# History initializes only selected charts; descriptor feeds T2-C, never E2.
field_tile(2.3, 25.1, 3.5, 2.8)
text(9, 26.4, r"$\mathcal{H}_n$", 25, BLUE)
text(7.5, 30.1, "Physical history", 16.5, MUTED)
route([(11.6,26.4), (57.5,26.4), (57.5,24.5)], BLUE, 1.35)
text(50.3, 27.7, "initialize", 16.5, BLUE)
junction(14.5, 26.4, BLUE)
route([(14.5,26.4), (14.5,30.2), (20,30.2)], BLUE, 1.35)
box(20, 28, 22, 4.5, BLUE_BG, BLUE)
text(31, 29.2, "Chart-independent", 16.5)
text(31, 31.2, r"descriptor $\boldsymbol{d}_n$", 16.5)
route([(42,30.2), (80.8,30.2)], BLUE, 1.35)
box(80.8, 28.3, 14.2, 3.8, PURPLE_BG, PURPLE)
text(87.9, 30.2, r"T2-C weight $\alpha$", 17)
text(87.9, 25.6, r"$\mu,\;\bar\pi_r$", 21, PURPLE)
route([(87.9,26.8), (87.9,28.3)], PURPLE, 1.35)
route([(95,30.2), (98.3,30.2), (98.3,24.7), (92.5,24.7), (92.5,23)], PURPLE, 1.35)

# (b) The sparse velocity branch is expanded; pressure is an algebraic branch.
box(1.1, 34.3, 97.8, 32, "#FBFCFD", "#CDD6DE", radius=1, lw=.9)
text(2.4, 36.4, "(b) Inside one specialist", 22, weight="bold", ha="left")
text(97.6, 36.4, "LEVEL 1  /  SPARSE EXPERT ROUTING", 16.5, MUTED, ha="right")
field_tile(2.5, 48, 3.2, 2.7)
text(8.4, 49.2, r"$\mathcal{H}_n$", 25, BLUE)
route([(10.8,49.2), (13,49.2)])
ax.add_patch(Polygon([(13,45.9), (21,47), (21,51.4), (13,52.5)],
                     fc=WARM_BG, ec=WARM, lw=1.2, zorder=2))
text(16.7, 48.2, "Encoder", 17)
text(17, 50.6, r"$\phi_r$", 23, WARM)
route([(21,49.2), (22.3,49.2)])
text(26.1, 49.2, r"$(\boldsymbol{a}_n,\boldsymbol{b}_n)$", 21)
text(25.9, 52.1, "local history", 16.5, MUTED)
route([(29.9,49.2), (32.1,49.2)], arrow=False)
junction(32.1, 49.2)
route([(32.1,49.2), (32.1,42.5), (36,42.5)])
box(36, 40.2, 32, 4.6, BLUE_BG, BLUE)
text(52, 42.5, r"Projected dynamics $\mathcal{F}_r^G(a,b;\mu)$", 17.5)
box(35.2, 47.2, 33.8, 15.2, WARM_BG, WARM, radius=.9, lw=1.3)
text(52.1, 49, "Sparse MoE velocity closure", 18, weight="bold")
route([(32.1,49.2), (32.1,55.8), (37.1,55.8)])
text(33.5, 54.1, r"$\boldsymbol{\xi}_n$", 22)
box(37.1, 53, 7.9, 5.6, PURPLE_BG, PURPLE)
text(41.05, 55.1, "Group", 16.5, weight="bold")
text(41.05, 57, "router", 16.5)
text(41.05, 60.1, r"$g^\star$ (Top-1)", 16.5, MUTED)
box(49, 51.2, 12, 3.4, "white", WARM)
text(55, 52.9, r"Shared $E_0$", 17.5)
route([(45,55.8), (46.7,55.8), (46.7,52.9), (49,52.9)], PURPLE, 1.25)
route([(46.7,55.8), (46.7,58), (48.3,58)], PURPLE, 1.25)
junction(46.7, 55.8, PURPLE)
box(48.3, 55.75, 13.4, 4.5, "white", "#DFB68E", radius=.4, lw=.9, dashed=True)
for xx, label, active in [(49,r"$E_{e_1}$",True), (53.3,r"$E_{e_2}$",True),
                          (57.6,r"$\cdots$",False)]:
    box(xx, 56.4, 3.4, 3.2, "#FFE4C6" if active else "#F4F3F1",
        WARM if active else GRAY, radius=.35)
    text(xx+1.7, 58, label, 19, INK if active else MUTED)
text(55, 61, "Top-2 routed", 16.5)
route([(61,52.9), (63.3,52.9), (63.3,55.8), (64.5,55.8)])
route([(61.7,58), (63.3,58), (63.3,55.8)], arrow=False)
merge(65.5, 55.8, r"$\Sigma$", .85)
text(65.5, 59.2, r"$\omega$", 21, MUTED)
route([(66.35,55.8), (73.2,55.8), (73.2,43.4)])
text(70.8, 54, r"$\boldsymbol{\rho}^u$", 22, WARM)
route([(68,42.5), (72.3,42.5)])
merge(73.2, 42.5)
route([(74.1,42.5), (78,42.5)])
text(75.9, 40.3, r"$\dot{\boldsymbol{a}}$", 23)
box(78, 40, 13, 5, GRAY_BG, GRAY)
text(84.5, 41.4, "Velocity", 17, weight="bold")
text(84.5, 43.6, "RK step", 17, MUTED)
route([(91,42.5), (93.1,42.5)])
text(96.1, 42.5, r"$\boldsymbol{a}_{n+1}$", 22, BLUE)
# New velocity plus current local features enter the pressure map (not an ODE).
route([(96.1,44.3), (96.1,49.8), (84.5,49.8), (84.5,54)], BLUE, 1.35)
box(78, 54, 13, 6, BLUE_BG, BLUE, dashed=True)
text(84.5, 56.1, "Algebraic", 17.5, weight="bold")
text(84.5, 58.1, "pressure map", 17)
route([(91,57), (93.1,57)])
text(96.1, 57, r"$\boldsymbol{b}_{n+1}$", 22, BLUE)
text(84.5, 62, "No pressure integration", 16.5, MUTED)
route([(32.1,55.8), (32.1,64.9), (73.7,64.9), (73.7,58.5), (78,58.5)], BLUE, 1.2, dashed=True)
text(54, 64.9, "local features", 16.5, BLUE,
     bbox=dict(facecolor="#FBFCFD", edgecolor="none", pad=2.8))
# The structured expert is illustrated once; every shared/routed expert uses it.
text(16.5, 56.8, "Each structured expert", 16.5, MUTED)
for xx, label in [(7,"Nonlinear"), (16.5,"Affine")]:
    text(xx, 62.2, label, 16.5, MUTED)
t = np.linspace(0, 1, 60)
ax.plot(4.5+5*t, 59.6-.65*np.tanh(5*(t-.5)), color=WARM, lw=1.5, zorder=4)
route([(14,60.2), (19,58.9)], WARM, 1.5, False)
for xx in (23.5, 27.4):
    box(xx, 58.7, 1.2, 1.8, WARM_BG, WARM, radius=.12, lw=.9)
    for yy in (59.3, 59.9):
        route([(xx,yy), (xx+1.2,yy)], WARM, .7, False)
text(26.0, 59.6, r"$\odot$", 21, WARM)
text(26, 63.0, "Low-rank\nbilinear", 16.5, MUTED)
text(11.8, 59.6, "+", 18, MUTED)
text(21.1, 59.6, "+", 18, MUTED)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
outside = []
for t in TEXT:
    b = t.get_window_extent(renderer)
    if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.x1 or b.y1 > fig.bbox.y1:
        outside.append(t.get_text())
if outside:
    raise ValueError(f"Text outside canvas: {outside}")
for suffix in ("pdf", "svg", "png"):
    meta = {"Title": "RAL-MoE-ROM: hierarchical routing and local dynamics"}
    if suffix == "pdf":
        meta.update({"Author": "", "CreationDate": None, "ModDate": None})
    if suffix == "svg":
        meta["Date"] = None
    fig.savefig(OUT.with_suffix("."+suffix), dpi=220, metadata=meta)
print(json.dumps({"outputs": [str(OUT.with_suffix("."+s)) for s in ("pdf","svg","png")],
    "width_pt": 936, "height_pt": 936*H/W,
    "min_source_font_pt": min(t.get_fontsize() for t in TEXT),
    "min_font_at_397pt_width": min(t.get_fontsize() for t in TEXT)*397/936,
    "schematic_icons_only": True}, indent=2))
