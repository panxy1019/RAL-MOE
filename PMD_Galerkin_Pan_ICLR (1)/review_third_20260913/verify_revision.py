"""Read-only manuscript checks; writes generated QA artifacts, not manuscript source."""
import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfReader
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "review_third_20260913"
BEFORE = QA / "before"
paths = [Path("main.tex")]
paths += [p.relative_to(ROOT) for folder in ("sections", "appendix") for p in sorted((ROOT / folder).glob("*.tex"))]
texts = {str(p).replace("\\", "/"): (ROOT / p).read_text(encoding="utf-8") for p in paths}
strip_comments = lambda s: re.sub(r"(?<!\\)%[^\n]*", "", s)
combined = "\n".join(map(strip_comments, texts.values()))
labels = re.findall(r"\\label\{([^}]+)\}", combined)
refs = re.findall(r"\\(?:ref|eqref|plaineqref|pageref)\{([^}]+)\}", combined)
headings = re.findall(r"\\paragraph\{([^}]+)\}", combined)
numbers = lambda s: re.findall(r"\d+(?:\.\d+)?", s)
table_checks = {}
for name, new in texts.items():
    old = (BEFORE / name).read_text(encoding="utf-8")
    old_tables = re.findall(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", old, re.S)
    new_tables = re.findall(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", new, re.S)
    table_checks[name] = len(old_tables) == len(new_tables) and all(numbers(a.split(r"\midrule", 1)[-1]) == numbers(b.split(r"\midrule", 1)[-1]) for a, b in zip(old_tables, new_tables))
result_checks = {name: re.findall(r"\d+\.\d+", texts[name]) == re.findall(r"\d+\.\d+", (BEFORE / name).read_text(encoding="utf-8")) for name in ["sections/experiments.tex", "appendix/circular_cylinder.tex", "appendix/fluidic_pinball.tex", "appendix/internal_routing.tex", "appendix/parameter_counts.tex"]}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
bib_checks = {name: sha(ROOT/name) == sha(BEFORE/name) for name in ["references.bib", "legacy_references.bib"]}
log = (ROOT / "output/third_round_pdf/main.log").read_text(encoding="utf-8", errors="replace")
reader = PdfReader(ROOT / "output/third_round_pdf/main.pdf")
pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
aux = (ROOT / "output/third_round_pdf/main.aux").read_text(encoding="utf-8")
label_map = dict(re.findall(r"\\newlabel\{((?:app|tab|fig):[^}]+)\}\{\{([^}]+)\}", aux))
stale = re.findall(r"bilinear\s+responses|Reynolds-number trajectories|regime cover\b|data partitions|fixed-run|Periodic core|convex correction|E2 prob\.|equal weighting|E2-probability weighting|E2-probability blending", combined, re.I)
checks = {
    "undefined_source_refs": sorted(set(refs)-set(labels)),
    "duplicate_labels": [x for x,n in collections.Counter(labels).items() if n>1],
    "duplicate_paragraph_headings": [x for x,n in collections.Counter(headings).items() if n>1],
    "table_numeric_sequences_unchanged": table_checks,
    "result_file_numeric_sequences_unchanged": result_checks,
    "bibliography_hashes_unchanged": bib_checks,
    "all_displayed_equations_unchanged": all(re.findall(r"\\begin\{(equation\*?|align\*?)\}(.*?)\\end\{\1\}", texts[name], re.S) == re.findall(r"\\begin\{(equation\*?|align\*?)\}(.*?)\\end\{\1\}", (BEFORE/name).read_text(encoding="utf-8"), re.S) for name in texts),
    "descriptor_definition_unchanged": re.search(r"\\begin\{equation\}\s*\\bm s_n.*?\\end\{equation\}", texts["appendix/routing_details.tex"], re.S).group() == re.search(r"\\begin\{equation\}\s*\\bm s_n.*?\\end\{equation\}", (BEFORE/"appendix/routing_details.tex").read_text(encoding="utf-8"), re.S).group(),
    "stale_terms_in_tex": stale,
    "unresolved_markers_in_pdf": pdf_text.count("??"),
    "undefined_reference_warnings": len(re.findall(r"(?:Reference|Citation).*?undefined|There were undefined", log, re.S)) if "undefined" in log else 0,
    "multiply_defined_warnings": len(re.findall(r"multiply defined", log)),
    "overfull_boxes": re.findall(r"Overfull[^\n]+", log),
    "underfull_box_count": len(re.findall(r"Underfull", log)),
    "pdf_pages": len(reader.pages),
    "label_map": label_map,
}
(QA / "verification.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
diffs = []
stats = []
for p in paths:
    if sha(ROOT/p) != sha(BEFORE/p):
        diff = subprocess.run(["git", "diff", "--no-index", "--", str(BEFORE/p), str(ROOT/p)], capture_output=True, text=True, encoding="utf-8")
        diffs.append(diff.stdout)
        stat = subprocess.run(["git", "diff", "--no-index", "--numstat", "--", str(BEFORE/p), str(ROOT/p)], capture_output=True, text=True, encoding="utf-8")
        stats.append(stat.stdout.strip())
(QA / "revision.diff").write_text("\n".join(diffs), encoding="utf-8")
(QA / "diff_summary.txt").write_text("\n".join(stats), encoding="utf-8")
(QA / "compiled_text.txt").write_text(pdf_text, encoding="utf-8")
render = QA / "rendered"
render.mkdir(exist_ok=True)
doc = pdfium.PdfDocument(str(ROOT / "output/third_round_pdf/main.pdf"))
thumbs = []
for i in range(len(doc)):
    page = doc[i]
    img = page.render(scale=1.25).to_pil().convert("RGB")
    img.save(render/f"page-{i+1:02d}.png")
    img.thumbnail((280, 385))
    tile = Image.new("RGB", (300, 415), "#e5e7eb")
    tile.paste(img, ((300-img.width)//2, 23))
    ImageDraw.Draw(tile).text((10, 5), f"Page {i+1}", fill="black")
    thumbs.append(tile)
for start in range(0, len(thumbs), 9):
    group = thumbs[start:start+9]
    sheet = Image.new("RGB", (900, 415*((len(group)+2)//3)), "white")
    for i,tile in enumerate(group):
        sheet.paste(tile, ((i%3)*300, (i//3)*415))
    sheet.save(render/f"contact-{start//9+1}.png")
print(json.dumps(checks, ensure_ascii=False, indent=2))
assert not checks["undefined_source_refs"] and not checks["duplicate_labels"]
assert not checks["duplicate_paragraph_headings"] and not stale
assert all(table_checks.values()) and all(result_checks.values()) and all(bib_checks.values())
assert checks["descriptor_definition_unchanged"] and checks["all_displayed_equations_unchanged"]
assert not checks["overfull_boxes"] and not checks["unresolved_markers_in_pdf"]
