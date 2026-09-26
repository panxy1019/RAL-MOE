#!/usr/bin/env python3
"""Static LaTeX audit used when no TeX engine is installed."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "cdm_grom_theory_main.tex"
SOURCES = [
    MAIN,
    ROOT / "sections" / "continuous_delta_memory_rom.tex",
    ROOT / "appendices" / "continuous_delta_memory_proofs.tex",
]

REQUIRED_RESULT_LABELS = {
    "lem:expansions",
    "thm:local-consistency",
    "cor:global-first-order",
    "prop:gradient-flow",
    "cor:fixed-equilibrium",
    "thm:local-wellposedness",
    "thm:memory-boundedness",
    "prop:memory-passivity",
    "prop:fixed-key",
    "thm:volterra",
    "thm:linear-elimination",
    "cor:exact-realization",
    "thm:reciprocal-energy",
    "thm:kernel-error",
}


def remove_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def check_balanced_braces(text: str, source: Path) -> None:
    depth = 0
    for index, character in enumerate(text):
        if character == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif character == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth < 0:
                raise AssertionError(f"extra closing brace in {source} at {index}")
    if depth:
        raise AssertionError(f"unbalanced braces in {source}: depth={depth}")


def check_environments(text: str, source: Path) -> None:
    stack: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([^{}]+)\}", text):
        action, environment = match.groups()
        if action == "begin":
            stack.append(environment)
        else:
            if not stack or stack[-1] != environment:
                raise AssertionError(
                    f"environment mismatch in {source}: closing {environment}, "
                    f"stack={stack[-3:]}"
                )
            stack.pop()
    if stack:
        raise AssertionError(f"unclosed environments in {source}: {stack}")


def main() -> None:
    for source in SOURCES:
        assert source.is_file(), f"missing source: {source}"

    texts = {source: remove_comments(source.read_text(encoding="utf-8")) for source in SOURCES}
    for source, text in texts.items():
        check_balanced_braces(text, source)
        check_environments(text, source)

    combined = "\n".join(texts.values())
    labels = re.findall(r"\\label\{([^{}]+)\}", combined)
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    assert not duplicate_labels, f"duplicate labels: {duplicate_labels}"
    label_set = set(labels)

    brace_references = set(
        re.findall(r"\\(?:ref|eqref)\{([^{}]+)\}", combined)
    )
    hyper_references = set(re.findall(r"\\hyperref\[([^\]]+)\]", combined))
    undefined = sorted((brace_references | hyper_references) - label_set)
    assert not undefined, f"undefined references: {undefined}"

    missing_results = sorted(REQUIRED_RESULT_LABELS - label_set)
    assert not missing_results, f"missing result labels: {missing_results}"

    appendix = texts[SOURCES[2]]
    proof_targets = set(
        re.findall(
            r"\\subsection\{Proof of \\hyperref\[([^\]]+)\]",
            appendix,
        )
    )
    missing_proofs = sorted(REQUIRED_RESULT_LABELS - proof_targets)
    extra_proofs = sorted(proof_targets - REQUIRED_RESULT_LABELS)
    assert not missing_proofs, f"results without appendix proof: {missing_proofs}"
    assert not extra_proofs, f"proofs without result statement: {extra_proofs}"

    main_text = texts[MAIN]
    input_paths = re.findall(r"\\input\{([^{}]+)\}", main_text)
    missing_inputs = [
        item for item in input_paths if not (ROOT / f"{item}.tex").is_file()
    ]
    assert not missing_inputs, f"missing input files: {missing_inputs}"

    assert "\\cite{" not in combined, "unexpected unresolved citation command"
    assert "TODO citation" in combined, "required citation marker is absent"

    print("PASS verify_latex_structure")
    print(f"sources={len(SOURCES)}")
    print(f"labels={len(labels)}")
    print(f"references={len(brace_references | hyper_references)}")
    print(f"result_statements={len(REQUIRED_RESULT_LABELS)}")
    print(f"appendix_proofs={len(proof_targets)}")
    print(f"inputs={input_paths}")


if __name__ == "__main__":
    main()

