from __future__ import annotations

import copy
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


INPUT = Path(r"C:\Users\panxy1019\Desktop\开题报告提交修改版8.28.docx")
OUTPUT = Path(r"C:\Users\panxy1019\Desktop\开题报告提交修改版8.28-参考文献超链接已核对.docx")

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W}


def qn(local: str) -> str:
    return f"{{{W}}}{local}"


def paragraph_text(p: etree._Element) -> str:
    return "".join(p.xpath(".//w:t/text()", namespaces=NS))


def run_text(r: etree._Element) -> str:
    return "".join(r.xpath(".//w:t/text()", namespaces=NS))


def make_text_run(text: str, rpr: etree._Element | None = None) -> etree._Element:
    r = etree.Element(qn("r"))
    if rpr is not None:
        r.append(copy.deepcopy(rpr))
    t = etree.SubElement(r, qn("t"))
    if text[:1].isspace() or text[-1:].isspace() or "  " in text:
        t.set(f"{{{XML}}}space", "preserve")
    t.text = text
    return r


def replace_run_with_text_parts(p: etree._Element, r: etree._Element, left: str, right: str) -> None:
    parent = r.getparent()
    idx = parent.index(r)
    rpr = r.find(qn("rPr"))
    parts = []
    if left:
        parts.append(make_text_run(left, rpr))
    if right:
        parts.append(make_text_run(right, rpr))
    parent.remove(r)
    for part in reversed(parts):
        parent.insert(idx, part)


def split_direct_run_at(p: etree._Element, offset: int) -> None:
    if offset <= 0 or offset >= len(paragraph_text(p)):
        return
    pos = 0
    for child in p.xpath(".//w:r[.//w:t]", namespaces=NS):
        text = run_text(child)
        nxt = pos + len(text)
        if pos < offset < nxt:
            cut = offset - pos
            replace_run_with_text_parts(p, child, text[:cut], text[cut:])
            return
        pos = nxt


def make_field_run(kind: str, value: str | None = None) -> etree._Element:
    r = etree.Element(qn("r"))
    if kind == "instr":
        instr = etree.SubElement(r, qn("instrText"))
        instr.set(f"{{{XML}}}space", "preserve")
        instr.text = value or ""
    else:
        fld = etree.SubElement(r, qn("fldChar"))
        fld.set(qn("fldCharType"), kind)
    return r


def flatten_hyperlink_fields(p: etree._Element) -> int:
    """Remove complex-field plumbing while retaining its displayed result runs."""
    children = list(p)
    active = False
    is_hyperlink = False
    phase = None
    remove: list[etree._Element] = []
    removed_fields = 0
    for child in children:
        fld = child.find(".//w:fldChar", namespaces=NS)
        instr = child.find(".//w:instrText", namespaces=NS)
        if fld is not None:
            typ = fld.get(qn("fldCharType"))
            if typ == "begin":
                active = True
                is_hyperlink = False
                phase = "instr"
                remove.append(child)
            elif active and typ == "separate":
                phase = "result"
                remove.append(child)
            elif active and typ == "end":
                remove.append(child)
                if is_hyperlink:
                    removed_fields += 1
                active = False
                is_hyperlink = False
                phase = None
        elif active and phase == "instr":
            if instr is not None and "HYPERLINK" in (instr.text or ""):
                is_hyperlink = True
            remove.append(child)
    # This document uses only hyperlink fields in the body area being processed.
    for child in remove:
        if child.getparent() is p:
            p.remove(child)
    return removed_fields


def insert_hyperlink_field(p: etree._Element, start: int, end: int, bookmark: str) -> None:
    split_direct_run_at(p, end)
    split_direct_run_at(p, start)

    pos = 0
    first = None
    last = None
    for child in p.xpath(".//w:r[.//w:t]", namespaces=NS):
        text = run_text(child)
        nxt = pos + len(text)
        if text and pos >= start and nxt <= end:
            if first is None:
                first = child
            last = child
        pos = nxt
    if first is None or last is None:
        raise RuntimeError(f"Could not isolate hyperlink span {start}:{end} in {paragraph_text(p)!r}")

    parent = first.getparent()
    if last.getparent() is not parent:
        raise RuntimeError(
            f"Hyperlink span crosses revision containers {start}:{end} in {paragraph_text(p)!r}"
        )
    first_idx = parent.index(first)
    parent.insert(first_idx, make_field_run("begin"))
    parent.insert(first_idx + 1, make_field_run("instr", f' HYPERLINK \\l "{bookmark}" '))
    parent.insert(first_idx + 2, make_field_run("separate"))
    last_idx = parent.index(last)
    parent.insert(last_idx + 1, make_field_run("end"))


def add_zero_length_bookmark(p: etree._Element, name: str, bookmark_id: int, offset: int = 0) -> None:
    split_direct_run_at(p, offset)
    pos = 0
    insert_idx = 0
    children = list(p)
    for idx, child in enumerate(children):
        if child.tag == qn("pPr"):
            insert_idx = idx + 1
            continue
        if child.tag == qn("r"):
            if pos >= offset:
                insert_idx = idx
                break
            pos += len(run_text(child))
            insert_idx = idx + 1
        else:
            insert_idx = idx + 1
    start = etree.Element(qn("bookmarkStart"))
    start.set(qn("id"), str(bookmark_id))
    start.set(qn("name"), name)
    end = etree.Element(qn("bookmarkEnd"))
    end.set(qn("id"), str(bookmark_id))
    p.insert(insert_idx, start)
    p.insert(insert_idx + 1, end)


BOOKMARK_TARGETS: dict[str, tuple[int, int]] = {
    "codex_ref_ahrens_2007": (193, 0),
    "codex_ref_alsalman_2002": (195, 0),
    "codex_ref_bartlomiejczyk_2006": (196, 0),
    "codex_ref_chang_2007": (198, 0),
    "codex_ref_chen_2022": (200, 0),
    "codex_ref_chernov_2004": (201, 0),
    "codex_ref_chou_2021": (202, 0),
    "codex_ref_debot_2011": (203, 0),
    "codex_ref_diaz_2015": (205, 0),
    "codex_ref_fantinuoli_2017": (206, 0),
    "codex_ref_gile_1995": (208, 0),
    "codex_ref_gile_2009": (210, 0),
    "codex_ref_herbert_1952": (213, 0),
    "codex_ref_housen_2012": (214, 0),
    "codex_ref_kahng_2024": (215, 0),
    "codex_ref_kurz_1995": (216, 0),
    "codex_ref_lambert_2014": (218, 0),
    "codex_ref_mcallister_2000": (220, 0),
    "codex_ref_mead_2002": (222, 0),
    "codex_ref_rinne_2000": (225, 0),
    "codex_ref_robinson_2007": (226, 0),
    "codex_ref_schweda_1992": (227, 0),
    "codex_ref_seeber_2011": (229, 0),
    "codex_ref_seleskovitch_1999": (230, 0),
    "codex_ref_skehan_1989": (231, 0),
    "codex_ref_skehan_1998": (233, 0),
    "codex_ref_skehan_2009": (234, 0),
    "codex_ref_sun_2025": (235, 0),
    "codex_ref_tavakoli_2005": (236, 0),
    "codex_ref_tommola_heleva_1998": (237, 0),
    "codex_ref_tommola_etal_2000": (239, 0),
    "codex_ref_williams_1994": (240, 0),
    "codex_ref_xu_2018": (242, 0),
    "codex_ref_zhao_2022": (245, 0),
    "codex_ref_zhou_2024": (247, 0),
    "codex_ref_zou_2023": (248, 0),
    "codex_ref_bao_1996": (249, 0),
    "codex_ref_bao_2005": (251, 0),
    "codex_ref_chai_2016": (252, 0),
    "codex_ref_fu_2013": (252, 48),
    "codex_ref_he_2020": (253, 0),
    "codex_ref_liu_heping_2005": (254, 0),
    "codex_ref_liu_heping_2007": (255, 0),
    "codex_ref_liu_jin_2011": (257, 0),
    "codex_ref_song_2021": (258, 0),
    "codex_ref_wang_2019": (259, 0),
    "codex_ref_xu_ran_2018": (261, 0),
    "codex_ref_xu_ran_2020": (263, 0),
    "codex_ref_yang_2005": (264, 0),
    "codex_ref_yu_2009": (265, 0),
    "codex_ref_yuan_2019": (266, 0),
    "codex_ref_zhang_jiliang_2003": (267, 0),
    "codex_ref_zhang_wenzhong_1999": (268, 0),
    "codex_ref_zhang_wu_2001": (270, 0),
}


# Regexes intentionally preserve the visible citation text and only add field plumbing.
CITATION_PATTERNS: list[tuple[str, str]] = [
    (r"Bartłomiejczyk\s*\(2006\)", "codex_ref_bartlomiejczyk_2006"),
    (r"Chou et al\.\s*\(2021\)", "codex_ref_chou_2021"),
    (r"Herbert\s*\(1952\)", "codex_ref_herbert_1952"),
    (r"Seleskovitch\s*\(1999\)", "codex_ref_seleskovitch_1999"),
    (r"Schweda Nicholson\s*\(1992\)", "codex_ref_schweda_1992"),
    (r"Chernov\s*\(2004\)", "codex_ref_chernov_2004"),
    (r"McAllister\s*\(2000\)", "codex_ref_mcallister_2000"),
    (r"Williams\s*\(1994\)", "codex_ref_williams_1994"),
    (r"Tommola\s*&\s*Helevä\s*\(1998\)", "codex_ref_tommola_heleva_1998"),
    (r"Al-Salman\s*&\s*Al-Khanji\s*\(2002\)", "codex_ref_alsalman_2002"),
    (r"Chang\s*&\s*Schallert\s*\(2007\)", "codex_ref_chang_2007"),
    (r"俞小娟\s*[（(]2009[）)]", "codex_ref_yu_2009"),
    (r"Chen[（(]2022[）)]", "codex_ref_chen_2022"),
    (r"Zou\s*与\s*Wang[（(]2023[）)]", "codex_ref_zou_2023"),
    (r"Gile\s*\(2009\)", "codex_ref_gile_2009"),
    (r"Gile[（(]1995[）)]", "codex_ref_gile_1995"),
    (r"Gile[（(]2009[）)]", "codex_ref_gile_2009"),
    (r"Díaz-Galaz et al\.\s*\(2015\)", "codex_ref_diaz_2015"),
    (r"Díaz-Galaz\s+等[（(]2015[）)]", "codex_ref_diaz_2015"),
    (r"Fantinuoli\s*\(2017\)", "codex_ref_fantinuoli_2017"),
    (r"Xu\s*\(2018\)", "codex_ref_xu_2018"),
    (r"鲍刚[（(]1996[）)]", "codex_ref_bao_1996"),
    (r"鲍刚[（(]2005[）)]", "codex_ref_bao_2005"),
    (r"刘和平[（(]2005[）)]", "codex_ref_liu_heping_2005"),
    (r"刘和平[（(]2007[）)]", "codex_ref_liu_heping_2007"),
    (r"杨承淑[（(]2005[）)]", "codex_ref_yang_2005"),
    (r"张吉良[（(]2003[）)]", "codex_ref_zhang_jiliang_2003"),
    (r"柴明颎[（(]2016[）)]", "codex_ref_chai_2016"),
    (r"王斌华[（(]2019[）)]", "codex_ref_wang_2019"),
    (r"刘进[（(]2011[）)]", "codex_ref_liu_jin_2011"),
    (r"徐然[（(]2018[）)]", "codex_ref_xu_ran_2018"),
    (r"徐然[（(]2020[）)]", "codex_ref_xu_ran_2020"),
    (r"Sun[（(]2025[）)]", "codex_ref_sun_2025"),
    (r"Zhou\s*与\s*Dong[（(]2024[）)]", "codex_ref_zhou_2024"),
    (r"Skehan,\s*1998", "codex_ref_skehan_1998"),
    (r"Lambert\s*&\s*Kormos,\s*2014", "codex_ref_lambert_2014"),
    (r"Ahrens\s*\(2007\)", "codex_ref_ahrens_2007"),
    (r"Mead\s*\(2002\)", "codex_ref_mead_2002"),
    (r"Seeber\s*\(2011\)", "codex_ref_seeber_2011"),
    (r"Housen et al\.,\s*2012", "codex_ref_housen_2012"),
    (r"Robinson,\s*2007", "codex_ref_robinson_2007"),
    (r"Skehan[（(]1989[）)]", "codex_ref_skehan_1989"),
    (r"Tavakoli\s*&\s*Skehan[（(]2005[）)]", "codex_ref_tavakoli_2005"),
    (r"Skehan[（(]2009[）)]", "codex_ref_skehan_2009"),
    (r"de Bot\s*&\s*Larsen-Freeman,\s*2011", "codex_ref_debot_2011"),
    (r"Kahng,\s*2024", "codex_ref_kahng_2024"),
    (r"张文忠[（(]1999[）)]", "codex_ref_zhang_wenzhong_1999"),
    (r"张文忠、吴旭东[（(]2001[）)]", "codex_ref_zhang_wu_2001"),
    (r"宋姝娴、李德超、查建设[（(]2021[）)]", "codex_ref_song_2021"),
    (r"Zhao[（(]2022[）)]", "codex_ref_zhao_2022"),
    (r"袁帅、万宏瑜\s*[（(]2019[）)]", "codex_ref_yuan_2019"),
    (r"符荣波\s*[（(]2013[）)]", "codex_ref_fu_2013"),
    (r"何妍、李德凤、李丽青\s*[（(]2020[）)]", "codex_ref_he_2020"),
    (r"Kurz\s*\(1995\)", "codex_ref_kurz_1995"),
    (r"Rinne\s+等\s*\(2000\)", "codex_ref_rinne_2000"),
    (r"Tommola\s+等\s*\(2000\)", "codex_ref_tommola_etal_2000"),
]


def collect_matches(text: str) -> list[tuple[int, int, str, str]]:
    matches: list[tuple[int, int, str, str]] = []
    for pattern, bookmark in CITATION_PATTERNS:
        for m in re.finditer(pattern, text):
            if any(not (m.end() <= a or m.start() >= b) for a, b, _, _ in matches):
                continue
            matches.append((m.start(), m.end(), bookmark, m.group()))
    matches.sort()
    return matches


def count_hyperlink_fields(root: etree._Element) -> int:
    return sum(1 for t in root.xpath(".//w:instrText/text()", namespaces=NS) if "HYPERLINK" in t)


def main() -> None:
    with ZipFile(INPUT, "r") as zin:
        document_xml = zin.read("word/document.xml")
        root = etree.fromstring(document_xml)
        paras = root.xpath(".//w:body/w:p", namespaces=NS)
        before = count_hyperlink_fields(root)

        removed = 0
        for p in paras[31:190]:
            removed += flatten_hyperlink_fields(p)

        max_id = max(
            [int(x) for x in root.xpath(".//w:bookmarkStart/@w:id", namespaces=NS) if str(x).isdigit()]
            or [0]
        )
        for seq, (name, (p_idx, offset)) in enumerate(BOOKMARK_TARGETS.items(), start=max_id + 1):
            add_zero_length_bookmark(paras[p_idx], name, seq, offset)

        inserted: list[tuple[int, str, str]] = []
        for p_idx in range(31, 190):
            p = paras[p_idx]
            text = paragraph_text(p)
            matches = collect_matches(text)
            for start, end, bookmark, visible in reversed(matches):
                insert_hyperlink_field(p, start, end, bookmark)
                inserted.append((p_idx, visible, bookmark))

        # Structural validation before packaging.
        all_names = set(root.xpath(".//w:bookmarkStart/@w:name", namespaces=NS))
        missing_targets = []
        for instr in root.xpath(".//w:instrText/text()", namespaces=NS):
            m = re.search(r'HYPERLINK\s+\\l\s+"([^"]+)"', instr)
            if m and m.group(1) not in all_names:
                missing_targets.append(m.group(1))
        if missing_targets:
            raise RuntimeError(f"Missing bookmark targets: {sorted(set(missing_targets))}")

        after = count_hyperlink_fields(root)
        out_xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
        with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = out_xml if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, data)

    print(f"output={OUTPUT}")
    print(f"hyperlink_fields_before={before}")
    print(f"flattened_existing_fields={removed}")
    print(f"inserted_fields={len(inserted)}")
    print(f"hyperlink_fields_after={after}")
    counts: dict[str, int] = {}
    for _, _, bookmark in inserted:
        counts[bookmark] = counts.get(bookmark, 0) + 1
    for bookmark in sorted(counts):
        print(f"{bookmark}={counts[bookmark]}")


if __name__ == "__main__":
    main()
