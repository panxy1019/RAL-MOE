"""Read PDF structure, audit no manuscript edits, compose a selection sheet."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
from pypdf import PdfReader
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
checks=[]


def check(v,s):
    if not v:
        raise AssertionError(s)
    checks.append(s)


stats={}
for stem in ("A_flow","B_hierarchy","C_assembly"):
    pdf=PdfReader(HERE/(stem+".pdf"))
    check(len(pdf.pages)==1,stem+": one PDF page")
    check(len(pdf.pages[0].images)==0,stem+": no raster image objects")
    check(float(pdf.pages[0].mediabox.width)==864,stem+": 12-inch vector source")
    svg=(HERE/(stem+".svg")).read_text("utf-8")
    check("<text" in svg and "<image" not in svg,stem+": editable vector SVG")
    data=json.loads((HERE/(stem+".json")).read_text())
    check(not data["text_overlap_candidates"],stem+": no text/text bounding-box overlap")
    check(not data["text_outside_canvas"],stem+": all text inside canvas")
    check(data["min_font_at_397pt_width"]>=8.5,stem+": minimum 8.5 pt at paper width")
    stats[stem]={"font_at_paper_width":data["min_font_at_397pt_width"],
        "ordinary_english_words":data.get("english_word_count",
            data.get("ordinary_word_count",data.get("ordinary_english_words"))),
        "pdf_sha256":hashlib.sha256((HERE/(stem+".pdf")).read_bytes()).hexdigest(),
        "page_size_pt":[float(pdf.pages[0].mediabox.width),float(pdf.pages[0].mediabox.height)]}
manifest=json.loads((ROOT/"build/diagram_redesign_20260907/upload_manifest.json").read_text())
for rel,digest in manifest.items():
    check(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,
          "Current manuscript preserved: "+rel)
previous=json.loads((ROOT/"build/diagram_redesign_20260907/verification.json").read_text())
for rel,key in [("build/pdf/main_diagram_redesign_20260907.pdf","pdf_sha256"),
                ("build/overleaf_upload_diagram_redesign_20260907.zip","zip_sha256")]:
    check(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==previous[key],
          "Previous deliverable preserved: "+rel)

# Comparison sheet made from the PDF renders, never from experimental data.
font=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",31)
small=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",23)
sheet=Image.new("RGB",(1640,2420),"#edf1f5")
draw=ImageDraw.Draw(sheet)
labels=[("A_flow","A  /  COMPLETE FLOW","Overview + one compact velocity-closure inset"),
        ("B_hierarchy","B  /  TWO-LEVEL ROUTING","Regime selection outside; sparse expert selection inside"),
        ("C_assembly","C  /  INDEPENDENT CHARTS","Different local coordinates; fusion only after reconstruction")]
for i,(stem,title,subtitle) in enumerate(labels):
    y=15+i*800
    draw.text((28,y+8),title,fill="#233344",font=font)
    draw.text((28,y+50),subtitle,fill="#617487",font=small)
    im=Image.open(HERE/"qa"/(stem+".png")).convert("RGB")
    im.thumbnail((1600,700))
    sheet.paste(im,(20,y+82))
sheet.save(HERE/"comparison.png")
result={"status":"PASS","checks_passed":len(checks),"manuscript_unchanged":True,
        "variants":stats,"checks":checks}
(HERE/"verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
with zipfile.ZipFile(HERE/"RAL_MoE_ROM_three_candidates.zip","w",zipfile.ZIP_DEFLATED) as archive:
    for stem in ("A_flow","B_hierarchy","C_assembly"):
        for ext in ("pdf","svg","png","py","json"):
            archive.write(HERE/(stem+"."+ext),stem+"."+ext)
    for name in ("README.md","A_notes.md","B_notes.md","C_notes.md","comparison.png",
                 "verification.json","verify_candidates.py"):
        archive.write(HERE/name,name)
print(json.dumps({k:v for k,v in result.items() if k!="checks"},indent=2))
