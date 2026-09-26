"""Narrow regression audit for the September 7 Figure 1 redraw."""
import hashlib
import json
from pathlib import Path
import re
import zipfile
from pypdf import PdfReader
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "build/diagram_redesign_20260907"
BASE = ROOT / "build/before_diagram_redesign_20260907"
PDF = ROOT / "build/pdf/main_diagram_redesign_20260907.pdf"
ZIP = ROOT / "build/overleaf_upload_diagram_redesign_20260907.zip"
checks = []


def check(value, label):
    if not value:
        raise AssertionError(label)
    checks.append(label)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


changed = []
for previous in sorted(BASE.rglob("*")):
    if not previous.is_file():
        continue
    relative = previous.relative_to(BASE)
    check((ROOT / relative).is_file(), f"Preserved file: {relative.as_posix()}")
    if sha(previous) != sha(ROOT / relative):
        changed.append(relative.as_posix())
    else:
        check(True, f"Byte-identical baseline file: {relative.as_posix()}")
check(set(changed) == {"figures/ral_moe_rom_overview.py",
                       "figures/ral_moe_rom_overview.pdf",
                       "sections/introduction.tex"},
      "Only diagram generator, diagram PDF and introduction caption changed")
without_figure = lambda t: re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", t, flags=re.S)
check(without_figure((ROOT / "sections/introduction.tex").read_text("utf-8-sig")) ==
      without_figure((BASE / "sections/introduction.tex").read_text("utf-8-sig")),
      "Introduction outside Figure 1 exactly unchanged")
diagram = PdfReader(ROOT / "figures/ral_moe_rom_overview.pdf")
check(len(diagram.pages) == 1, "Figure 1 is a single vector page")
check(len(diagram.pages[0].images) == 0, "Figure 1 contains no raster image objects")
svg = (ROOT / "figures/ral_moe_rom_overview.svg").read_text("utf-8")
check("<image" not in svg and "<text" in svg, "SVG contains editable text and no raster images")
aux = (RUN / "compile/main.aux").read_text("utf-8-sig")
labels = {key: (number, int(page)) for key, number, page in re.findall(
    r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{(\d+)\}", aux)}
for key, value in {"fig:ral_overview": "1", "fig:periodic_internal_routing": "3",
                   "app:pinball": "H", "app:internal_routing": "I",
                   "app:parameter_counts": "J"}.items():
    check(labels[key][0] == value, f"Label retained: {key} = {value}")
conclusion = re.search(r"\\numberline \{6\}Conclusion\}\{(\d+)\}", aux)
check(conclusion and int(conclusion.group(1)) <= 9, "Scientific main text ends within nine pages")
log = (RUN / "compile/main.log").read_text("utf-8-sig")
check(not re.search(r"Overfull \\[hv]box|(?:Reference|Citation).*undefined|"
                    r"There were undefined|multiply defined|Missing character|Font shape .* undefined", log),
      "No overfull boxes, unresolved citations/references or missing characters")
with zipfile.ZipFile(ZIP) as archive:
    check("main.tex" in archive.namelist(), "Upload main.tex at ZIP root")
    check(len(archive.namelist()) == 31, "Upload contains exactly 31 referenced paper files")
    for relative in archive.namelist():
        check(archive.read(relative) == (ROOT / relative).read_bytes(),
              f"Packaged file matches workspace: {relative}")
pdf = PdfReader(PDF)
packaged = PdfReader(RUN / "package_compile/main.pdf")
check(len(pdf.pages) == len(packaged.pages) == 23, "Both independent compilations have 23 pages")
for n, (a, b) in enumerate(zip(pdf.pages, packaged.pages), 1):
    check(a.extract_text() == b.extract_text(), f"Package page {n} text matches")
    check(a.get_contents().get_data() == b.get_contents().get_data(),
          f"Package page {n} page stream matches")
    check((RUN / f"render/final-{n:02}.png").read_bytes() ==
          (RUN / f"render/package-{n:02}.png").read_bytes(),
          f"Package page {n} rasterized appearance matches")
result = {"status": "PASS", "checks_passed": len(checks), "pages": len(pdf.pages),
          "changed_baseline_files": changed, "scientific_content_unchanged": True,
          "conclusion_page": int(conclusion.group(1)), "figure_1_page": labels["fig:ral_overview"][1],
          "pdf_sha256": sha(PDF), "zip_sha256": sha(ZIP),
          "figure_pdf_sha256": sha(ROOT / "figures/ral_moe_rom_overview.pdf"),
          "checks": checks}
(RUN / "verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))

# Contact sheets are QA intermediates, not altered publication assets.
for batch, start in enumerate(range(1, 24, 12), 1):
    sheet = Image.new("RGB", (990, 1780), "#e8edf1")
    draw = ImageDraw.Draw(sheet)
    for idx, page_number in enumerate(range(start, min(start+12, 24))):
        thumbnail = Image.open(RUN / f"render/final-{page_number:02}.png").convert("RGB")
        thumbnail.thumbnail((310, 408))
        x, y = 10+(idx % 3)*330, 24+(idx // 3)*440
        sheet.paste(thumbnail, (x, y))
        draw.text((x, y-18), f"Page {page_number}", fill="#233344")
    sheet.save(RUN / f"contact-{batch}.png")
