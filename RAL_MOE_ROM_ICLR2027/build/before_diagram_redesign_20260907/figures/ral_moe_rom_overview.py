"""Editable native-vector Figure 1; no raster assets.
Design reference: user-provided September 6 hierarchical pipeline.
"""
from pathlib import Path
import math
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

W, H = 1024, 726
OUT = Path(__file__).with_suffix('.pdf')
FONT_DIR = Path('C:/Windows/Fonts')
for suffix, name in [('', 'times.ttf'), ('-Bold', 'timesbd.ttf'),
                     ('-Italic', 'timesi.ttf'), ('-BoldItalic', 'timesbi.ttf')]:
    pdfmetrics.registerFont(TTFont('FigureSerif' + suffix, str(FONT_DIR / name)))
pdfmetrics.registerFontFamily('FigureSerif', normal='FigureSerif', bold='FigureSerif-Bold',
                            italic='FigureSerif-Italic', boldItalic='FigureSerif-BoldItalic')
C = canvas.Canvas(str(OUT), pagesize=(W, H), pageCompression=1, invariant=1)
C.setTitle('RAL-MoE-ROM: global hierarchy and regime-local specialist')
C.setAuthor('')
INK = HexColor('#152130')
PURPLE = (HexColor('#f0eafa'), HexColor('#9a78bf'))
BLUE = (HexColor('#eaf3ff'), HexColor('#7199ce'))
ORANGE = (HexColor('#fff0df'), HexColor('#dc9960'))
GREEN = (HexColor('#ecf6e7'), HexColor('#88b375'))
GREY = (HexColor('#f5f6f8'), HexColor('#8793a0'))


def paragraph(txt, x, y, w, h, size=18, bold=False, align=1):
    txt = txt.replace('<sub>', f'<sub size="{size * .68}" rise="{size * .18}">')
    txt = txt.replace('<sup>', f'<sup size="{size * .68}" rise="{size * .38}">')
    style = ParagraphStyle('diagram', fontName='FigureSerif-Bold' if bold else 'FigureSerif',
                           fontSize=size, leading=size * 1.12, textColor=INK,
                           alignment=align, splitLongWords=False)
    p = Paragraph(txt, style)
    if p.minWidth() > w + .01:
        raise ValueError(f'Text too wide: {txt!r}: {p.minWidth():.1f} > {w}')
    _, ph = p.wrap(w, H)
    if ph > h + .01:
        raise ValueError(f'Text exceeds box height: {txt!r}: {ph:.1f} > {h}')
    p.drawOn(C, x, H - y - (h + ph) / 2)


def box(x, y, w, h, txt='', palette=GREY, size=18, bold=False):
    C.setFillColor(palette[0]); C.setStrokeColor(palette[1]); C.setLineWidth(1.15)
    C.roundRect(x, H-y-h, w, h, 6, stroke=1, fill=1)
    if txt:
        paragraph(txt, x+5, y+3, w-10, h-6, size, bold)


def line(points, arrow=True, width=1.6):
    C.saveState(); C.setStrokeColor(INK); C.setFillColor(INK); C.setLineWidth(width)
    p = C.beginPath(); p.moveTo(points[0][0], H-points[0][1])
    for x, y in points[1:]: p.lineTo(x, H-y)
    C.drawPath(p)
    if arrow:
        x, y = points[-1]; px, py = points[-2]
        a = math.atan2(y-py, x-px); d, s = 7, 3.5
        tri = C.beginPath(); tri.moveTo(x, H-y)
        for side in (1, -1):
            tri.lineTo(x-d*math.cos(a)+side*s*math.sin(a), H-(y-d*math.sin(a)-side*s*math.cos(a)))
        tri.close(); C.drawPath(tri, fill=1, stroke=0)
    C.restoreState()


def plus(x, y, radius=11):
    C.setFillColor(white); C.setStrokeColor(INK); C.setLineWidth(1.5)
    C.circle(x, H-y, radius, fill=1, stroke=1)
    paragraph('+', x-radius, y-radius-1, 2*radius, 2*radius+3, 22)


# (a): separate preference and physical-history lanes.
paragraph('(a) Global hierarchical pipeline of RAL-MoE-ROM', 14, 5, 995, 32, 23, True, 0)
box(16, 54, 81, 51, 'Parameter<br/><i>μ</i>', PURPLE, 17.5)
box(115, 54, 143, 51, '<b>E2 router</b><br/>(parameter-only)', PURPLE)
box(276, 54, 130, 51, 'Preference <i>π</i><br/>[<i>π</i><sub>S</sub>, <i>π</i><sub>H</sub>, <i>π</i><sub>P</sub>]', BLUE)
box(424, 54, 116, 51, '<b>Top-1 or<br/>Top-2</b>', GREY, 18)
for a,b in [(97,115),(258,276),(406,424)]: line([(a,79.5),(b,79.5)])
paragraph('S-H and H-P pairs only', 351, 110, 205, 24)

# Each orange rollout node names its selected independent autonomous specialist.
box(584, 50, 165, 60, '<b>Top-1: S, H, or P</b><br/>Autonomous rollout', ORANGE)
box(773, 50, 130, 60, 'Physical<br/>reconstruction', ORANGE)
box(925, 50, 85, 60, '<b>Predicted<br/>fields</b>', GREEN)
line([(540,79.5),(584,79.5)])
line([(749,80),(773,80)]); line([(903,80),(925,80)])
line([(540,79.5),(562,79.5),(562,258)], False)
paragraph('Top-2', 584, 141, 68, 22, 17.5)
for yy, title in [(166, 'Specialist A'), (230, 'Specialist B')]:
    box(584, yy, 165, 54, '<b>' + title + '</b><br/>Independent rollout', ORANGE)
    box(773, yy, 130, 54, 'Physical<br/>reconstruction', ORANGE)
    line([(562,yy+27),(584,yy+27)]); line([(749,yy+27),(773,yy+27)])
box(925, 176, 85, 94, '<b>Physical<br/>space<br/>T2-C<br/>fusion</b>', GREEN)
line([(903,193),(925,193)]); line([(903,257),(925,257)])
box(925, 301, 85, 54, '<b>Predicted<br/>fields</b>', GREEN)
line([(967.5,270),(967.5,301)])
paragraph('Fusion after reconstruction in common physical space', 578, 117, 435, 25, 17.5)

C.saveState(); C.setFillColor(HexColor('#faf9f7')); C.setStrokeColor(HexColor('#9aa0a9')); C.setDash(5,4); C.setLineWidth(1)
C.roundRect(145,H-253,377,103,7,fill=1,stroke=1); C.restoreState()
paragraph('Regime-local specialists', 150, 152, 367, 26, 19, True)
for xx, txt in [(158, '<b>Steady</b> <i>S</i>'), (278, '<b>Hopf</b> <i>H</i>'), (398, '<b>Periodic</b> <i>P</i>')]:
    box(xx, 184, 111, 37, txt, ORANGE)
paragraph('Independent local reduced charts', 150, 226, 367, 23)
box(16, 267, 103, 55, 'Physical<br/>history <i>H</i><sub>n</sub>', BLUE, bold=True)
line([(119,294.5),(135,294.5),(135,264),(453.5,264)], False)
for xx in [213.5,333.5,453.5]: line([(xx,264),(xx,253)])
paragraph('Initialize selected native chart(s)', 151, 270, 371, 26)
box(315, 313, 194, 49, 'Chart-independent<br/>descriptor <i>d</i><sub>n</sub>', BLUE)
line([(67.5,322),(67.5,337.5),(315,337.5)])
paragraph('From physical history', 122, 307, 184, 24)
box(584, 313, 217, 49, '<b>T2-C weight <i>α</i></b><br/>correction Δ<sub>ψ</sub>(<i>μ</i>, <i>d</i><sub>n</sub>)', PURPLE)
line([(509,337.5),(584,337.5)])
paragraph('<i>μ</i>, pair preference <i>π̄</i>', 584, 286, 217, 22)
line([(692.5,308),(692.5,313)])
line([(801,337.5),(913,337.5),(913,223),(925,223)])

# (b): shared / routed experts act in parallel; only velocity is integrated.
C.saveState(); C.translate(0, -32)
C.setFillColor(HexColor('#fffaf5')); C.setStrokeColor(HexColor('#bea88e')); C.setLineWidth(1)
C.roundRect(8,H-686,1008,338,12,fill=1,stroke=1)
paragraph('(b) Zoom-in: one regime-local specialist', 20, 358, 700, 31, 23, True, 0)
paragraph('Local reduced chart', 772, 363, 232, 26, 18, align=2)
box(20, 440, 91, 51, 'Physical<br/>history <i>H</i><sub>n</sub>', BLUE, 17.5, True)
box(130, 440, 98, 51, 'Local<br/>encoder', ORANGE, 18, True)
box(247, 440, 107, 51, 'Local state /<br/>history', BLUE)
line([(111,465.5),(130,465.5)]); line([(228,465.5),(247,465.5)])
line([(354,465.5),(389,465.5),(389,459),(422,459)])
paragraph('Features<br/><i>ξ</i><sub>n</sub>', 345, 414, 76, 41, 18)
C.setFillColor(HexColor('#fff0df')); C.setStrokeColor(HexColor('#dc9960')); C.setLineWidth(1.2)
C.roundRect(411,H-525,367,126,9,fill=1,stroke=1)
paragraph('Sparse MoE velocity correction', 418, 400, 351, 26, 20, True)
box(422, 436, 87, 49, 'Group<br/>router', PURPLE, bold=True)
box(537, 429, 118, 34, 'Shared expert', ORANGE, 17.5, True)
box(537, 477, 118, 34, 'Top-2 experts', ORANGE, 17.5, True)
line([(509,460.5),(523,460.5),(523,446),(537,446)])
line([(509,460.5),(523,460.5),(523,494),(537,494)])
line([(655,446),(685,446),(685,459),(700,459)])
line([(655,494),(685,494),(685,483),(700,483)])
plus(710,471,13)
paragraph('Weighted sum', 657, 500, 116, 23, 17)
paragraph('Nonlinear + affine + low-rank bilinear', 416, 530, 365, 27)
paragraph('Shared and routed experts<br/>within the selected group', 416, 554, 365, 39, 17)
box(411, 595, 367, 43, 'Projected velocity backbone <i>F</i><sub>r</sub><sup>G</sup>', BLUE, 19, True)
line([(354,465.5),(378,465.5),(378,616.5),(411,616.5)])
line([(723,471),(800,471)])
line([(778,616.5),(812,616.5),(812,483)])
plus(812,471,12)
line([(824,471),(844,471)])
box(844, 444, 161, 54, '<b>Velocity time<br/>integration</b> (RK)', GREY)
paragraph('Velocity derivative', 783, 414, 225, 24, 18)
box(844, 518, 161, 35, 'Velocity <i>a</i><sub>n+1</sub>', BLUE, 19, True)
line([(924.5,498),(924.5,518)])
box(844, 567, 161, 67, 'Algebraic pressure<br/>reconstruction<br/>(local features)', GREY, 17.8)
line([(924.5,553),(924.5,567)])
box(844, 641, 161, 35, 'Pressure <i>b</i><sub>n+1</sub>', BLUE, 19, True)
line([(924.5,634),(924.5,641)])
box(431, 641, 347, 35, 'Updated local prediction (<i>a</i><sub>n+1</sub>, <i>b</i><sub>n+1</sub>)', BLUE)
line([(844,535.5),(834,535.5),(834,658.5)], False)
line([(844,658.5),(778,658.5)])
C.restoreState()
C.showPage(); C.save()
print(OUT)
