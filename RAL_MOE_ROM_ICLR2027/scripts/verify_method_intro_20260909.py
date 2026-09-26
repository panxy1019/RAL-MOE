"""Audit the focused method/intro revision and independently compiled upload package."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

import pdfplumber
from PIL import Image, ImageChops, ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "build/method_intro_20260909"
BASE = ROOT / "build/before_method_intro_20260909"
PDF = ROOT / "build/pdf/main_method_intro_revision_20260909.pdf"
ZIP = ROOT / "build/overleaf_upload_method_intro_20260909.zip"
checks = []


def read(path):
    return path.read_text(encoding="utf-8-sig")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(ok, message):
    if not ok:
        raise AssertionError(message)
    checks.append(message)


def envs(text, kind):
    return re.findall(r"\\begin\{" + kind + r"\}.*?\\end\{" + kind + r"\}", text, re.S)


def equations(text):
    return re.findall(r"\\begin\{(equation\*?|align\*?)\}(.*?)\\end\{\1\}", text, re.S)


def compact(text):
    return re.sub(r"\s+", "", text)


old_manifest = json.loads(read(ROOT / "build/user_figure_20260908/upload_manifest.json"))
manifest = json.loads(read(RUN / "upload_manifest.json"))
check(set(old_manifest) == set(manifest), "Same 31 referenced source/asset paths")
changed = []
for rel in sorted(old_manifest):
    check(sha(BASE / rel) == old_manifest[rel], "Verified baseline snapshot: " + rel)
    check(sha(ROOT / rel) == manifest[rel], "Current source matches manifest: " + rel)
    if sha(ROOT / rel) != sha(BASE / rel):
        changed.append(rel)
check(changed == ["appendix/routing_details.tex", "appendix/specialist_details.tex",
                  "sections/experiments.tex", "sections/introduction.tex", "sections/method.tex"],
      "Exactly five scoped manuscript files changed")

old_exp = read(BASE / "sections/experiments.tex")
new_exp = read(ROOT / "sections/experiments.tex")
for kind, count in [("tabular", 4), ("tablenotes", 2)]:
    check(len(envs(new_exp, kind)) == count and envs(old_exp, kind) == envs(new_exp, kind),
          f"All {count} experimental {kind} blocks unchanged")
numbers = lambda s: set(re.findall(r"\d+(?:\.\d+)?", s))
check(numbers(old_exp) == numbers(new_exp), "Experimental numeric value set unchanged")
for rel in sorted(k for k in manifest if k.startswith("appendix/") and k.endswith(".tex")):
    before, after = read(BASE / rel), read(ROOT / rel)
    normalized = after.replace(r"\bm\gamma_n^{(r)}", r"\bm\alpha_n^{(r)}")
    normalized = normalized.replace(r"\Delta_\psi", r"\delta").replace(r"c_\psi", r"c_\vartheta")
    check(equations(before) == equations(normalized), "All appendix display equations retained: " + rel)

intro = read(ROOT / "sections/introduction.tex")
method = read(ROOT / "sections/method.tex")
specialist = read(ROOT / "appendix/specialist_details.tex")
check("Learning reduced-order dynamical models" in intro, "ROM identity explicit in opening")
check("ROM couples a low-dimensional state representation" in intro, "Representation/dynamics bridge present")
check(len(re.findall(r"\\subsection\{", method)) == 4, "Method retains four subsections")
check(len(equations(method)) == 6, "Six main method displays, including two added equations")
check("two-level routing hierarchy" in method and "three operations" in method,
      "Three operations remain within two routing levels")
check(r"\bm\alpha_n" not in method + specialist, "No pressure/T2-C alpha collision")
check(r"\mathcal Q_r^{PP}(\bm a_{n+1}^{(r)};\mu)" in method and
      r"\odot\bm\rho^p_{\theta_r}(\bm\xi_n^{(r)})" in method and
      r"\bm\eta^p_{\theta_r}(\bm\xi_n^{(r)})" in method, "Pressure branch time indices retained")
loss = next(s for _, s in equations(method) if "eq:main_rollout_loss" in s)
app_loss = next(s for _, s in equations(specialist) if "eq:app_true_rollout_loss" in s)
# Compare the two sums, allowing only harmless TeX spacing and label conventions.
def loss_formula(text):
    text = re.sub(r"\\label\{[^}]+\}", "", text)
    for command in (r"\!", r"\,", r"\left", r"\right"):
        text = text.replace(command, "")
    text = compact(text).rstrip(".,")
    # ||error||^2/d and (1/d)||error||^2 are the same normalization.
    return re.sub(r"\\frac\{(\\\|.*?\\\|_2\^2)\}\{(d_u|d_p)\}",
                  lambda m: r"\frac{1}{" + m[2] + "}" + m[1], text)
check(loss_formula(loss) == loss_formula(app_loss), "Main rollout loss equals appendix loss exactly")

aux = read(RUN / "compile/main.aux")
labels = {k: (v, int(p)) for k, v, p in re.findall(
    r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{(\d+)\}", aux)}
sources = "\n".join(read(ROOT / rel) for rel in manifest if rel.endswith(".tex"))
references = set(re.findall(r"\\(?:ref|eqref|plaineqref|autoref)\{([^}]+)\}", sources))
references = {key for key in references if "#" not in key}  # Ignore macro parameters.
check(not references - labels.keys(), "Every explicit source cross-reference resolves")
for label, expected in {"eq:main_pressure_closure": "4", "eq:main_rollout_loss": "5",
                        "eq:main_t2c": "6", "fig:ral_overview": "1", "app:routing": "D",
                        "app:internal_routing": "I", "app:parameter_counts": "J"}.items():
    check(labels[label][0] == expected, "Expected equation/section label: " + label)
log = read(RUN / "compile/main.log")
check(not re.search(r"Overfull \\[hv]box|(?:Reference|Citation).*undefined|There were undefined|"
                    r"multiply defined|Missing character|Font shape .* undefined", log),
      "No overfull, unresolved reference/citation, duplicate label or missing glyph warning")

paper = PdfReader(PDF)
packaged = PdfReader(RUN / "package_compile/main.pdf")
check(len(paper.pages) == len(packaged.pages) == 23, "Both builds produce 23 pages")
source_png = ROOT / "figures/ral_moe_rom_overview_user_20260908.png"
user_png = Path("C:/Users/panxy1019/Downloads/ChatGPT Image Sep 8, 2026, 03_11_19 PM.png")
check(sha(source_png) == sha(user_png), "Latest user PNG remains byte-identical")
original = Image.open(user_png).convert("RGB")
embedded = [im.image.convert("RGB") for im in paper.pages[labels["fig:ral_overview"][1]-1].images
            if im.image.size == original.size]
check(len(embedded) == 1 and ImageChops.difference(original, embedded[0]).getbbox() is None,
      "Full-resolution Figure 1 embedded with identical pixels")
with zipfile.ZipFile(ZIP) as archive:
    check(set(archive.namelist()) == set(manifest), "ZIP contains only referenced sources/assets")
    for rel in manifest:
        check(archive.read(rel) == (ROOT / rel).read_bytes(), "Exact packaged source: " + rel)
for n, (local_page, package_page) in enumerate(zip(paper.pages, packaged.pages), 1):
    check(local_page.extract_text() == package_page.extract_text(), f"Package page {n}: identical text")
    check(local_page.get_contents().get_data() == package_page.get_contents().get_data(),
          f"Package page {n}: identical page content stream")
    check((RUN / f"render/final-{n:02}.png").read_bytes() ==
          (RUN / f"render/package-{n:02}.png").read_bytes(),
          f"Package page {n}: identical rendered appearance")


def layout_metrics(path, baseline=False):
    with pdfplumber.open(path) as doc:
        def find(pat, pages):
            hits = [(n, m) for n in pages for m in doc.pages[n-1].search(pat, case=False)]
            assert len(hits) == 1, (pat, hits)
            return hits[0]
        n3, s3 = find(r"3\s+RAL-MOE-ROM", [4])
        n4, s4 = find(r"4\s+EXPERIMENTS", [5])
        _, refs = find(r"REFERENCES", [9])
        # Section span in 648 pt text-height equivalents; subtract the intervening
        # Table 1 float in the baseline using its continuation text position.
        span = 648 + s4["top"] - s3["top"]
        float_height = 0
        if baseline:
            _, continuation = find(r"pressure-coordinate", [5])
            body = [c for c in doc.pages[4].chars if 108 <= c["x0"] < 504 and 70 < c["top"] < 730]
            float_height = continuation["top"] - min(c["top"] for c in body)
            span -= float_height
        return {"method_start_page": n3, "method_end_page": n4,
                "method_text_height_equivalents": round(span / 648, 3),
                "baseline_intervening_table_and_spacing_pt": round(float_height, 2),
                "references_start_page": 9, "references_start_y_pt": round(refs["top"], 2)}


old_layout = layout_metrics(ROOT / "build/pdf/main_user_figure_20260908.pdf", True)
new_layout = layout_metrics(PDF)
for batch, start in enumerate(range(1, 24, 12), 1):
    sheet = Image.new("RGB", (990, 1780), "#e8edf1")
    draw = ImageDraw.Draw(sheet)
    for idx, n in enumerate(range(start, min(start+12, 24))):
        im = Image.open(RUN / f"render/final-{n:02}.png").convert("RGB")
        im.thumbnail((310, 408))
        x, y = 10 + idx % 3 * 330, 24 + idx // 3 * 440
        sheet.paste(im, (x, y))
        draw.text((x, y-18), f"Page {n}", fill="#233344")
    sheet.save(RUN / f"contact-{batch}.png")

result = {"status": "PASS", "checks_passed": len(checks), "pages": len(paper.pages),
          "main_paper_pages_including_statements": 9,
          "changed_manuscript_files": changed, "method_definition_changed": False,
          "experimental_values_changed": False, "experimental_unique_numeric_literals": len(numbers(new_exp)),
          "baseline_method_source_lines": len(read(BASE / "sections/method.tex").splitlines()),
          "final_method_source_lines": len(method.splitlines()),
          "baseline_layout": old_layout, "final_layout": new_layout,
          "method_page_equivalent_increase": round(new_layout["method_text_height_equivalents"] -
                                                    old_layout["method_text_height_equivalents"], 3),
          "pdf_sha256": sha(PDF), "zip_sha256": sha(ZIP), "figure_sha256": sha(source_png),
          "underfull_box_messages": len(re.findall(r"Underfull \\[hv]box", log)),
          "latex_warnings": re.findall(r"LaTeX Warning:[^\n]+", log),
          "overleaf_synced_this_revision": False, "checks": checks}
(RUN / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
