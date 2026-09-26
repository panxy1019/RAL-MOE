# CDM-GROM Theory Package

This directory is an independent theory package. It does not modify the
existing HPRS-MoE-ROM or neural CTDM work.

## Primary deliverables

- `CONTINUOUS_DELTA_MEMORY_ROM_THEORY_CN.md` — complete Chinese method note.
- `sections/continuous_delta_memory_rom.tex` — English Sections 2–5.
- `appendices/continuous_delta_memory_proofs.tex` — complete Appendices A–D.
- `cdm_grom_theory_main.tex` — independent LaTeX entry.
- `NOTATION_AND_ASSUMPTIONS.md` — notation table and A1–A9.
- `THEOREM_DEPENDENCY_MAP.md` — result/assumption/proof dependency graph.
- `PROOF_REVIEW_CHECKLIST.md` — strict derivation review and open items.
- `THEORY_SANITY_CHECK_REPORT.md` — numerical algebra-check evidence.
- `LATEX_COMPILATION_REPORT.md` — compile/static-validation status.

## Verification scripts

The five requested mathematical scripts are:

- `scripts/verify_continuous_limit.py`
- `scripts/verify_gradient_flow.py`
- `scripts/verify_memory_energy_identity.py`
- `scripts/verify_fixed_key_reduction.py`
- `scripts/verify_linear_memory_kernel.py`

An additional source-only checker is provided:

- `scripts/verify_latex_structure.py`

No script performs training or uses held-out flow data.

