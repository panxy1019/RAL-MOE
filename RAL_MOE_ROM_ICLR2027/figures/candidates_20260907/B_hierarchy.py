"""Candidate B: a compact hierarchy of regime-local sparse specialists.

This script owns B_hierarchy.{pdf,svg,png} only. Geometry is drawn as vectors.
"""

from pathlib import Path
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon


OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 20,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})

PURPLE = "#8066AB"
WARM = "#CE8849"
BLUE = "#487FA8"
TEAL = "#4E8F7E"
INK = "#24313C"
GRAY = "#9DA7AF"
LIGHT = "#D4DADF"
TINT_PURPLE = "#F5F2F9"
TINT_WARM = "#FCF5EE"
TINT_BLUE = "#F0F6FA"
TINT_TEAL = "#F1F7F5"

fig = plt.figure(figsize=(12, 5.2), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 12), ylim=(0, 5.2))
ax.axis("off")
LABELS = []
ARTISTS = []


def text(x, y, s, size=20, color=INK, weight="normal", ha="center", va="center"):
    LABELS.append((s, size))
    a = ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, zorder=10)
    ARTISTS.append(a)
    return a


def box(x, y, w, h, edge=LIGHT, face="white", lw=1.6, radius=.08, dash="solid"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=face, edgecolor=edge, linewidth=lw,
                       linestyle=dash, zorder=2)
    ax.add_patch(p)
    return p


def line(points, color=GRAY, lw=1.5, dash="solid", z=3):
    ax.plot([p[0] for p in points], [p[1] for p in points],
            color=color, linewidth=lw, linestyle=dash, zorder=z)


def arrow(a, b, color=GRAY, lw=1.6, scale=11, dash="solid", rad=0):
    p = FancyArrowPatch(a, b, arrowstyle="-|>", color=color,
                       linewidth=lw, mutation_scale=scale,
                       linestyle=dash, connectionstyle=f"arc3,rad={rad}",
                       shrinkA=0, shrinkB=0, zorder=4)
    ax.add_patch(p)
    return p


def dot(x, y, r=.18, color=WARM, face="white", lw=1.6):
    p = Circle((x, y), r, facecolor=face, edgecolor=color, lw=lw, zorder=5)
    ax.add_patch(p)


# Outer regime routing: mu alone enters E2; pi is frozen for the trajectory.
text(.26, 4.76, r"$\mu$", size=25)
arrow((.46, 4.76), (.88, 4.76), PURPLE, lw=1.8)
box(.90, 4.40, 3.0, .69, PURPLE, TINT_PURPLE, lw=1.8)
text(2.40, 4.87, "Regime router", size=22, weight="medium")
text(2.40, 4.56, r"E2  ·  $\pi(\mu)$ fixed", size=18.5, color=PURPLE)
text(6.47, 4.79, "Top-1 / adjacent Top-2", size=21, color=PURPLE)
text(5.74, 4.46, r"$S$-$H$ / $H$-$P$", size=18.5, color=PURPLE)
line([(2.40, 4.40), (2.40, 4.18)], PURPLE, 1.7)
line([(1.14, 4.18), (8.49, 4.18)], PURPLE, 1.7)
arrow((1.14, 4.18), (1.14, 3.98), GRAY, dash=(0, (3, 3)))
arrow((4.70, 4.18), (4.70, 3.98), PURPLE, lw=2)
arrow((8.49, 4.18), (8.49, 3.98), PURPLE, lw=2)

# Initial physical history: a separate dashed path only initializes the locals.
box(10.01, 4.40, 1.49, .66, edge=GRAY, face="white", lw=1.5)
text(10.75, 4.73, r"$\mathcal{H}_0$", size=26)
line([(10.01, 4.55), (9.62, 4.55), (9.62, 4.30), (6.97, 4.30)],
     GRAY, 1.35, dash=(0, (4, 3)))
arrow((6.97, 4.30), (6.97, 3.68), GRAY, 1.35, dash=(0, (4, 3)))
arrow((9.17, 4.30), (9.17, 3.68), GRAY, 1.35, dash=(0, (4, 3)))
text(8.23, 4.48, "init.", size=18.5, color=GRAY)

# Inactive steady specialist and the two admissible, independently evolving
# local charts. There are no latent-space arrows between these cards.
box(.48, 1.24, 1.34, 2.74, edge=LIGHT, face="#FAFBFC", dash=(0, (4, 3)))
text(1.15, 3.66, r"$S$", size=25, color=GRAY)
text(1.15, 2.81, "ROM", size=20, color=GRAY)
text(1.15, 2.45, "+", size=22, color=GRAY)
text(1.15, 2.10, "MoE", size=20, color=GRAY)
text(1.15, 1.54, r"$\mathbf{a}_S(t)$", size=21, color=GRAY)

box(2.20, 1.24, 5.10, 2.74, edge=PURPLE, face="white", lw=1.9)
text(2.54, 3.68, r"$H$", size=25, color=PURPLE)
text(4.80, 3.68, "Local chart", size=21, weight="medium")
box(2.48, 2.29, 1.08, .65, edge=BLUE, face=TINT_BLUE, lw=1.5)
text(3.02, 2.615, "ROM", size=19.5, color=BLUE)

# The inner closure is a genuine parallel shared + sparse weighted sum.
box(3.87, 1.87, 3.16, 1.52, edge=WARM, face=TINT_WARM, lw=1.5)
text(5.45, 3.19, "Sparse MoE", size=20.5, color=WARM, weight="medium")
text(4.33, 2.83, "shared", size=18.5, color=WARM)
diamond = Polygon([(5.63, 2.99), (5.79, 2.83), (5.63, 2.67), (5.47, 2.83)],
                  closed=True, facecolor="white", edgecolor=WARM, lw=1.4, zorder=5)
ax.add_patch(diamond)
text(6.31, 2.83, "Top-2", size=18.5, color=WARM)
arrow((5.53, 2.70), (5.24, 2.47), WARM, 1.4, scale=9)
arrow((5.72, 2.70), (6.02, 2.47), WARM, 1.4, scale=9)
for x, label in [(4.33, r"$E_0$"), (5.20, r"$E_1$"), (6.06, r"$E_2$")]:
    dot(x, 2.29, .195, color=WARM, face="white")
    text(x, 2.29, label, size=20, color=WARM)
text(6.69, 2.29, r"$\cdots$", size=21, color=LIGHT)
line([(4.33, 2.09), (4.33, 1.68), (5.20, 1.68)], WARM, 1.45)
line([(6.06, 2.09), (6.06, 1.68), (5.20, 1.68)], WARM, 1.45)
arrow((5.20, 2.09), (5.20, 1.86), WARM, 1.45, scale=8)
dot(5.20, 1.68, .18, color=WARM, face="white")
text(5.20, 1.68, r"$\Sigma_{\omega}$", size=20, color=WARM)

# Physics plus learned correction supplies the local velocity rollout.
line([(3.02, 2.29), (3.02, 1.68)], BLUE, 1.55)
arrow((5.00, 1.68), (3.22, 1.68), WARM, 1.55, scale=10)
dot(3.02, 1.68, .18, color=BLUE, face="white")
text(3.02, 1.68, "+", size=23, color=BLUE)
line([(3.21,1.60),(3.61,1.50)], BLUE, 1.5)
dot(3.77,1.46,.145,color=BLUE,face="white",lw=1.1)
text(3.77,1.46,r"$\int$",size=18.5,color=BLUE)
arrow((3.92,1.44),(4.13,1.44),BLUE,1.5,scale=9)
text(4.50, 1.42, r"$\mathbf{a}_H(t)$", size=22, color=BLUE)

box(7.69, 1.24, 1.62, 2.74, edge=PURPLE, face="white", lw=1.9)
text(8.50, 3.68, r"$P$", size=25, color=PURPLE)
text(8.50, 2.81, "ROM", size=21, color=BLUE)
text(8.50, 2.45, "+", size=22, color=WARM)
text(8.50, 2.10, "MoE", size=21, color=WARM)
text(8.50, 1.53, r"$\mathbf{a}_P(t)$", size=22, color=BLUE)

# Chart-specific physical reconstruction precedes the common-space fusion.
arrow((4.80, 1.22), (4.80, 1.02), BLUE, 1.6, scale=10)
arrow((8.50, 1.22), (8.50, 1.02), BLUE, 1.6, scale=10)
box(3.59, .51, 2.43, .49, edge=TEAL, face=TINT_TEAL, lw=1.5)
box(7.35, .51, 2.32, .49, edge=TEAL, face=TINT_TEAL, lw=1.5)
text(4.80, .76, "Reconstruct", size=20, color=TEAL)
text(8.51, .76, "Reconstruct", size=20, color=TEAL)

# Two independently reconstructed fields join only in the physical output.
line([(6.02, .76), (6.47, .76), (6.47, .27), (10.70, .27)], TEAL, 1.65)
arrow((10.70, .27), (10.70, .58), TEAL, 1.65, scale=10)
arrow((9.67, .76), (10.51, .76), TEAL, 1.65, scale=10)
dot(10.70, .76, .18, color=TEAL, face=TINT_TEAL)
text(10.70, .76, r"$\Sigma$", size=22, color=TEAL)
arrow((10.90, .76), (11.34, .76), TEAL, 1.7, scale=10)
text(11.63, .76, r"$\widehat{\mathbf{q}}$", size=23, color=TEAL)
text(10.74, 1.15, "Physical fusion", size=19, color=TEAL)

# T2-C is conditioned by mu, the fixed pair probabilities, and an initial
# physical-history descriptor. Its output weights physical fields only.
arrow((10.75, 4.40), (10.75, 3.03), GRAY, 1.5, scale=10)
text(10.99, 3.63, r"$\mathbf{d}$", size=23, color=GRAY)
box(9.72, 2.18, 2.10, .83, edge=PURPLE, face=TINT_PURPLE, lw=1.6)
text(10.77, 2.79, "T2-C", size=22, color=PURPLE, weight="medium")
text(10.77, 2.43, r"$\alpha(\mu,\bar\pi,\mathbf{d})$", size=20, color=PURPLE)
line([(11.82,2.59),(11.93,2.59),(11.93,.97),(10.86,.97)],PURPLE,1.4)
arrow((10.86,.97),(10.77,.92),PURPLE,1.4,scale=10)

# The depicted pair is an example; the muted S card is not advanced here.
text(2.35, .22, "H-P example", size=18.5, color=PURPLE)

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
outside=[]
overlaps=[]
for i,a in enumerate(ARTISTS):
    b=a.get_window_extent(renderer)
    if b.x0<0 or b.y0<0 or b.x1>fig.bbox.x1 or b.y1>fig.bbox.y1:
        outside.append(a.get_text())
    for other in ARTISTS[i+1:]:
        if b.overlaps(other.get_window_extent(renderer)):
            overlaps.append([a.get_text(),other.get_text()])
if outside:
    raise ValueError(f"Text outside: {outside}")
for extension in ("pdf", "svg", "png"):
    metadata={"Title":"RAL-MoE-ROM / B: two-level hierarchy"}
    if extension=="pdf": metadata.update({"CreationDate":None,"ModDate":None})
    if extension=="svg": metadata["Date"]=None
    fig.savefig(OUT / f"B_hierarchy.{extension}", dpi=220,
                facecolor="white", edgecolor="none", metadata=metadata)

ordinary = []
for label, _ in LABELS:
    stripped = re.sub(r"\$.*?\$", "", label)
    ordinary.extend(re.findall(r"[A-Za-z]+(?:-[A-Za-z0-9]+)*", stripped))
report = {
    "variant": "B_hierarchy",
    "figure_inches": [12, 5.2],
    "min_font_pt": min(size for _, size in LABELS),
    "min_font_at_397pt_width": min(size for _, size in LABELS) * 397 / 864,
    "ordinary_word_count": len(ordinary),
    "ordinary_words": ordinary,
    "text_overlap_candidates":overlaps,
    "text_outside_canvas":outside,
    "output_files": [str(OUT / f"B_hierarchy.{ext}") for ext in ("pdf", "svg", "png")],
}
print(json.dumps(report, indent=2))
(OUT / "B_hierarchy.json").write_text(json.dumps(report,indent=2)+"\n")
