"""Verify this editorial round against its pre-edit snapshot and source tables.

Run from the project root with the bundled Python runtime. This is an internal
check, not part of the anonymous Overleaf package. No metrics are recomputed.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from pypdf import PdfReader


def read(path):
    return path.read_text(encoding="utf-8-sig")


def table(text, label):
    for match in re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", text, re.S):
        if r"\label{" + label + "}" in match.group():
            return match.group()
    raise AssertionError(f"Missing table {label}")


def body(text):
    return text.split(r"\toprule", 1)[1].split(r"\bottomrule", 1)[0]


def rows(text):
    result = []
    for row in body(text).split(r"\\"):
        row = re.sub(r"\\(?:midrule|addlinespace)", "", row).strip()
        if "&" in row:
            result.append([cell.strip() for cell in row.split("&")])
    return result[1:]  # Column headings are not data.


def numbers(text):
    return re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", text)


def decimal(text):
    values = re.findall(r"\d+\.\d+", text)
    assert len(values) == 1, text
    return values[0]


def triple_rows(text):
    return [[decimal(cell) for cell in row[-3:]] for row in rows(text)]


def math_blocks(text):
    pattern = r"\\begin\{(equation\*?|align\*?|gather\*?)\}.*?\\end\{\1\}"
    return [re.sub(r"\s+", "", match.group()) for match in re.finditer(pattern, text, re.S)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, default=Path("build/pdf/main.pdf"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve()
    pdf_path = (root / args.pdf).resolve()
    checks = []

    def check(condition, message):
        assert condition, message
        checks.append(message)

    experiments = read(root / "sections/experiments.tex")
    previous = read(baseline / "sections/experiments.tex")
    t1_old = table(previous, "tab:benchmark_summary")
    splits = table(read(root / "appendix/implementation.tex"), "tab:chart_splits")
    check(numbers(body(t1_old)) == numbers(body(splits)),
          "All 9 chart ranges, 27 split counts, and 9 horizons migrated exactly")
    t1 = table(experiments, "tab:benchmark_summary")
    check(len(rows(t1)) == 3 and "Primary role" in t1 and "Validation" not in body(t1),
          "Main Table 1 contains benchmark scope and roles, not split counts")

    t2_old = rows(table(previous, "tab:pinball_autonomous"))
    t2 = table(experiments, "tab:pinball_autonomous")
    t2_new = rows(t2)
    check(len(t2_old) == len(t2_new) == 8, "Table 2 retains all 8 predictor rows")
    for index, (old, new) in enumerate(zip(t2_old, t2_new)):
        check(numbers(old[0]) == numbers(new[0]), f"Table 2 row {index + 1}: same horizon")
        if index == 0:
            check(all("N/E" in cell for cell in new[2:]), "Steady POD-Galerkin uses N/E")
        else:
            check([decimal(cell) for cell in old[2:5]] == [decimal(cell) for cell in new[2:5]],
                  f"Table 2 row {index + 1}: field-error values unchanged")
    check("All 448" in t2 and "141 of 320" in t2 and "including threshold-violating" in t2,
          "Divergence counts and finite-window qualification retained in Table 2 notes")

    square = read(root / "appendix/square_cylinder.tex")
    full = table(square, "tab:square_fusion_full")
    square_source = table(read(root / "source_snapshot/SECOND_NUMERICAL_EXAMPLE_CENTEREDSQUARE.tex"),
                          "tab:centered_square_fusion_aggregate")
    full_values = triple_rows(full)
    check(len(full_values) == 14 and full_values == triple_rows(square_source),
          "All 42 field-error values in Appendix G match the authoritative source table")
    migrated_indices = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13]
    check([full_values[i] for i in migrated_indices] ==
          triple_rows(table(previous, "tab:routing_fusion")),
          "All 30 former main-table field-error values migrated exactly to Appendix G")
    joint = table(experiments, "tab:routing_fusion")
    expected_joint = [[full_values[i][-1] for i in indices]
                      for indices in ([1, 2, 3, 4, 5, 6], [7, 9, 10, 11, 12, 13])]
    check([[decimal(cell) for cell in row[1:]] for row in rows(joint)] == expected_joint,
          "Main Table 3 matches joint errors and best fixed specialists in Appendix G")
    check([re.findall(r"\\textbf\{([^}]+)\}", " ".join(row)) for row in rows(joint)] ==
          [["1.1958"], ["0.8465"]], "Only deployable T2-C is bold in main Table 3")
    check(all(r"\textbf" not in " ".join(row) for row in rows(full) if "oracle" in row[1]),
          "Oracle rows are not bold in Appendix G")

    circular_old = table(read(baseline / "appendix/circular_cylinder.tex"), "tab:circular_full")
    circular = table(read(root / "appendix/circular_cylinder.tex"), "tab:circular_full")
    check(numbers(body(circular)) == numbers(body(circular_old)),
          "All 44 circular-cylinder range endpoints unchanged in Appendix F")
    t4 = table(experiments, "tab:compact_ablation")
    pattern = r"\d+\.\d+--\d+\.\d+"
    check(Counter(re.findall(pattern, body(t4))) == Counter(re.findall(pattern, body(circular_old))),
          "Main Table 4 includes exactly the 22 original circular field-error ranges")
    for model, old_name in [("RAL-MoE-ROM", "RAL-MoE-ROM"), ("Vanilla-FNN-MoE", "Vanilla"),
                            ("DataOnly-MoE", "Data only"), ("Global MoE", "Global")]:
        main_model_row = body(t4).split(model, 1)[1].split(r"\addlinespace", 1)[0]
        actual = re.findall(pattern, main_model_row)
        expected = [value for row in rows(circular_old) if row[1] == old_name
                    for cell in row[2:] for value in re.findall(pattern, cell)]
        check(actual == expected, f"Table 4 model/regime mapping correct: {model}")
    check("E2 Top-1" not in body(t4) and "Overlap" not in body(t4) and r"\resizebox" not in t4,
          "Table 4 does not mix square overlaps, outer routing, or resized table content")

    for filename, label in [("square_cylinder.tex", "tab:square_parameter"),
                            ("fluidic_pinball.tex", "tab:pinball_overlap")]:
        check([numbers(" ".join(row)) for row in rows(table(read(baseline / "appendix" / filename), label))]
              == [numbers(" ".join(row)) for row in rows(table(read(root / "appendix" / filename), label))],
              f"Existing appendix table data unchanged: {label}")

    for filename in ["sections/method.tex", "appendix/rom_derivation.tex",
                     "appendix/specialist_details.tex", "appendix/routing_details.tex"]:
        check(math_blocks(read(root / filename)) == math_blocks(read(baseline / filename)),
              f"Display-equation blocks unchanged: {filename}")
    figures = list((baseline / "figures").rglob("*"))
    check(all(hashlib.sha256(path.read_bytes()).digest() ==
              hashlib.sha256((root / path.relative_to(baseline)).read_bytes()).digest()
              for path in figures if path.is_file() and path.suffix != ".tex"),
          "All pre-round experimental image assets unchanged")
    diagram_path = Path("figures/ral_moe_rom_framework_overview.tex")
    diagram = read(root / diagram_path)
    diagram_old = read(baseline / diagram_path)
    node_pattern = r"\\node\[.*?\].*?\{(.*?)\};"
    check(re.findall(node_pattern, diagram, re.S) == re.findall(node_pattern, diagram_old, re.S),
          "Framework diagram node labels unchanged; only spacing, wrapping, and wire path adjusted")
    wire_pattern = r"\\draw\[(?:arrow|signal)\][^;]+;"
    def endpoints(text):
        pairs = []
        for wire in re.findall(wire_pattern, text):
            names = re.findall(r"\(([A-Za-z]+)(?:\.[^)]*)?\)", wire)
            pairs.append([names[0], names[-1]])
        return pairs
    check(endpoints(diagram) == endpoints(diagram_old), "Framework diagram connections unchanged")

    sources = [root / "main.tex", *sorted((root / "sections").glob("*.tex")),
               *sorted((root / "appendix").glob("*.tex"))]
    combined = "\n".join(read(path) for path in sources)
    banned = ["journal experiments", "retained experiment", "retained source", "retained runs",
              "retained training", "current evidence boundary", "data contracts", "auditability",
              "unverified", "we do not append", "conversion directory", "author-side verification"]
    normalized = re.sub(r"\s+", " ", combined).lower()
    check(not any(phrase in normalized for phrase in banned), "Specified production/meta phrases removed")
    labels = re.findall(r"\\label\{([^}]+)\}", combined)
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", combined)
    # The framework figure is a TeX input with its label in the main text.
    check(len(labels) == len(set(labels)), "No duplicate manuscript labels")
    check(set(refs) <= set(labels), "All manuscript cross-reference labels exist")
    log = read(pdf_path.with_suffix(".log"))
    check(not re.search(r"Overfull|undefined references|Citation .* undefined|Reference .* undefined|Font shape .* undefined|^!", log, re.M),
          "Compile log has no overfull boxes, unresolved refs/cites, font-shape fallback, or fatal errors")
    pdf = PdfReader(pdf_path)
    check(len(pdf.pages) == 19, "Final PDF has 19 pages")
    aux = read(pdf_path.with_suffix(".aux"))
    check(r"{6}Conclusion}{8}" in aux and r"{A}Full problem setup and weighted POD construction}{10}" in aux,
          "Conclusion remains on page 8; appendices start on page 10")
    for label, page in [("tab:benchmark_summary", 5), ("tab:pinball_autonomous", 5),
                        ("tab:routing_fusion", 6), ("tab:compact_ablation", 6),
                        ("tab:chart_splits", 14), ("tab:square_fusion_full", 16),
                        ("tab:square_parameter", 16), ("tab:pinball_overlap", 19)]:
        check(bool(re.search(r"\\newlabel\{" + re.escape(label) + r"\}\{\{[^}]+\}\{" + str(page) + r"\}", aux)),
              f"Verified table placement: {label}, page {page}")

    report = {"status": "PASS", "checks_passed": len(checks), "checks": checks,
              "pdf_pages": len(pdf.pages), "main_text_last_page": 8,
              "baseline": str(baseline), "pdf_file": str(pdf_path),
              "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["status", "checks_passed", "pdf_pages", "main_text_last_page"]}))


if __name__ == "__main__":
    main()
