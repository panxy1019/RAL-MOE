"""Verify unchanged user PNG, focused source changes and standalone upload ZIP."""
import hashlib
import json
from pathlib import Path
import re
import zipfile
from pypdf import PdfReader
from PIL import Image, ImageDraw, ImageChops

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"build/user_figure_20260908"
BASE=ROOT/"build/before_user_figure_20260908"
SOURCE=Path("C:/Users/panxy1019/Downloads/ChatGPT Image Sep 8, 2026, 03_11_19 PM.png")
FIGURE=ROOT/"figures/ral_moe_rom_overview_user_20260908.png"
PDF=ROOT/"build/pdf/main_user_figure_20260908.pdf"
ZIP=ROOT/"build/overleaf_upload_user_figure_20260908.zip"
checks=[]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(v,s):
    if not v: raise AssertionError(s)
    checks.append(s)


check(sha(SOURCE)==sha(FIGURE),"User PNG copied byte-for-byte without editing or recompression")
changed=[]
for old in sorted(BASE.rglob("*")):
    if not old.is_file(): continue
    rel=old.relative_to(BASE)
    check((ROOT/rel).is_file(),"Original file retained: "+rel.as_posix())
    if sha(old)!=sha(ROOT/rel):
        changed.append(rel.as_posix())
    else:
        check(True,"Byte-identical original: "+rel.as_posix())
check(changed==["sections/introduction.tex"],"Only existing introduction file modified")
stripfig=lambda s:re.sub(r"\\begin\{figure\}.*?\\end\{figure\}","",s,flags=re.S)
intro=(ROOT/"sections/introduction.tex").read_text("utf-8-sig")
check(stripfig(intro)==stripfig((BASE/"sections/introduction.tex").read_text("utf-8-sig")),
      "Introduction outside Figure 1 unchanged")
check("figures/ral_moe_rom_overview_user_20260908.png" in intro,"Figure 1 uses latest user image")
check("Purple arrows denote" not in intro and "Each $D_r$" not in intro,
      "Obsolete diagram-specific caption descriptions removed")
aux=(RUN/"compile/main.aux").read_text("utf-8-sig")
labels={k:(v,int(p)) for k,v,p in re.findall(
    r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{(\d+)\}",aux)}
for key,value in {"fig:ral_overview":"1","fig:periodic_internal_routing":"3",
                  "app:pinball":"H","app:internal_routing":"I","app:parameter_counts":"J"}.items():
    check(labels[key][0]==value,"Label retained: "+key)
conclusion=re.search(r"\\numberline \{6\}Conclusion\}\{(\d+)\}",aux)
check(conclusion and int(conclusion.group(1))<=9,"Conclusion within nine pages")
log=(RUN/"compile/main.log").read_text("utf-8-sig")
check(not re.search(r"Overfull \\[hv]box|(?:Reference|Citation).*undefined|"
                    r"There were undefined|multiply defined|Missing character|Font shape .* undefined",log),
      "No overfull, unresolved reference/citation or missing character")
pdf=PdfReader(PDF)
check(len(pdf.pages)==23,"Compiled paper has 23 pages")
original=Image.open(SOURCE).convert("RGB")
matches=[im.image.convert("RGB") for im in pdf.pages[labels["fig:ral_overview"][1]-1].images
         if im.image.size==original.size]
check(len(matches)==1,"Latest full-resolution image embedded on Figure 1 page")
check(ImageChops.difference(original,matches[0]).getbbox() is None,
      "Embedded figure pixels exactly match original PNG")
with zipfile.ZipFile(ZIP) as archive:
    check(len(archive.namelist())==31 and "main.tex" in archive.namelist(),
          "Upload ZIP contains 31 referenced files with main.tex at root")
    check("figures/ral_moe_rom_overview_user_20260908.png" in archive.namelist(),
          "Upload ZIP includes latest PNG")
    check("figures/ral_moe_rom_overview.pdf" not in archive.namelist(),
          "Unused prior overview PDF not included in upload ZIP")
    for rel in archive.namelist():
        check(archive.read(rel)==(ROOT/rel).read_bytes(),"Packaged file matches: "+rel)
packaged=PdfReader(RUN/"package_compile/main.pdf")
check(len(packaged.pages)==len(pdf.pages),"Independent package compile has same page count")
for n,(a,b) in enumerate(zip(pdf.pages,packaged.pages),1):
    check(a.extract_text()==b.extract_text(),f"Package page {n}: identical text")
    check(a.get_contents().get_data()==b.get_contents().get_data(),
          f"Package page {n}: identical page content stream")
    check((RUN/f"render/final-{n:02}.png").read_bytes()==(RUN/f"render/package-{n:02}.png").read_bytes(),
          f"Package page {n}: identical rendered appearance")
for batch,start in enumerate(range(1,24,12),1):
    sheet=Image.new("RGB",(990,1780),"#e8edf1")
    draw=ImageDraw.Draw(sheet)
    for idx,n in enumerate(range(start,min(start+12,24))):
        im=Image.open(RUN/f"render/final-{n:02}.png").convert("RGB")
        im.thumbnail((310,408))
        x,y=10+(idx%3)*330,24+(idx//3)*440
        sheet.paste(im,(x,y))
        draw.text((x,y-18),f"Page {n}",fill="#233344")
    sheet.save(RUN/f"contact-{batch}.png")
result={"status":"PASS","checks_passed":len(checks),"pages":len(pdf.pages),
    "figure_1_page":labels["fig:ral_overview"][1],"conclusion_page":int(conclusion.group(1)),
    "user_image_pixels":original.size,"image_dpi_at_397pt_width":original.width*72/397,
    "user_image_sha256":sha(SOURCE),"pdf_sha256":sha(PDF),"zip_sha256":sha(ZIP),
    "changed_existing_files":changed,"experimental_data_unchanged":True,
    "overleaf_upload_confirmed":False,"checks":checks}
(RUN/"verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k!="checks"},indent=2))
