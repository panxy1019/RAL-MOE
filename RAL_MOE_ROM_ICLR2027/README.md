# RAL-MoE-ROM ICLR 2027 manuscript workspace

This directory is an independent, anonymous ICLR 2027 manuscript derived from the downloaded journal project. The journal source is not modified.

## Build entry point

- `main.tex`
- bibliography: `references.bib` and `legacy_references.bib`
- official style: `iclr2027_conference.sty` and `iclr2027_conference.bst`

Compile with a standard PDFLaTeX/BibTeX sequence or upload `build/overleaf_upload_final_p0_p1_20260906.zip` to Overleaf.
The verified current PDF is `build/pdf/main_final_p0_p1_20260906.pdf`: 23 pages total, scientific main text and conclusion through page 9, references starting on page 9, and appendices starting on page 11.

## Current revision: final P0/P1 (2026-09-06)

- Figure 1 is a native vector schematic redrawn from the user's newest September 6 reference, with parameter-only E2, separate history branches, independent reconstruction before fusion, velocity-only integration, and algebraic pressure.
- Table 2 defines Periodic core and numerical finiteness; Table 3 labels Best single as a posteriori and nondeployable. All original experimental performance values are unchanged.
- The AI disclosure is filled; authors must confirm its accuracy and complete the separate submission-form disclosure.
- Section 4.1 now reports the existing POD--DeepONet one-step/recursive-finiteness contrast, explicitly not a cadence-matched accuracy comparison.
- Four recent Scientific-ML/PDE MoE works use verified official BibTeX exports.
- Appendix I rounds routing diagnostics. New Appendix J gives exact Hopf/Periodic trainable and active counts from actual model classes and recorded checkpoint configurations, distinguishing native diagnostic execution from prediction-only execution. No parameter-matched claim or missing-model estimate is made.
- All 23 pages were visually checked; 246 regression/package checks pass. The 31-file ZIP was compiled from its independent extraction directory and all page text, content streams, and 90 dpi renders match.

See `FINAL_P0_P1_REVISION_REPORT.md`, `build/parameter_counts.json`, and `build/final_p0_p1_verification.json`. Editable Figure 1 source: `figures/ral_moe_rom_overview.py`. Previous sources are preserved in `build/before_final_p0_p1_20260906/`. Online Overleaf and historical E-drive mirrors were not modified.

## Previous revision: internal routing and user framework image

- Figure 1 uses the user's September 5 21:33:58 PNG byte-for-byte, without the obsolete crop intended for the preceding image. The caption preserves the parameter-only E2 and algebraic-pressure definitions.
- Figure 3 is a Periodic-only, three-panel group/velocity/pressure routing diagnostic generated from existing `macro_stage1` statistics; it appears on page 8 after Table 4 and its interpretation.
- Appendix I (pages 20-21) discloses the fixed Hopf support, compact Table 10, and Periodic rollout-position Figure 6. It does not include the independent S4 routing panel.
- Limitations and the reproducibility statement reflect the descriptive, post-hoc nature of the analysis; no original performance numbers or method definitions changed.
- The regression check passes 112 checks. The 30-file anonymous ZIP was independently extracted and compiled; all 21 page texts, content streams, and 85 dpi page renders match the working build.

See `ROUTING_INTEGRATION_REPORT.md`, `build/routing_integration_verification.json`, and `build/routing_package_verification.json` for the complete record. The preserved pre-round snapshot is `build/before_routing_reexecution_20260905_215237/`. The AI-use disclosure TODO remains for the authors to complete accurately.

Use the dated PDF and ZIP above as the authoritative delivery. The older table-cleanup files and any E-drive mirror not explicitly refreshed in this round are historical copies.

## Previous revision: table logic and narrative cleanup (2026-09-05)

- Table 1 summarizes benchmark coverage, ranks, horizons, and scientific roles; all nine chart-wise splits are in Appendix E.
- Table 2 isolates pinball autonomous prediction, retaining the horizons, pressure errors, and finite-window/divergence qualifications.
- Table 3 reports square-overlap joint errors, adds the best fixed specialist, and reserves bold for deployable T2-C. All component errors are in Appendix G.
- Table 4 contains only circular-cylinder local architecture ablations. Original min--max ranges are retained; no means were estimated from ranges.
- Formal manuscript text no longer contains conversion/audit notes or missing-file inventories. The reproducibility statement correctly points to Appendices A--E.
- Title line breaks, Times bold/italic font encoding, framework-label spacing, and appendix float placement were checked. Method equations, framework labels/connections, and experimental image data are unchanged.

The internal check `scripts/verify_table_cleanup.py` passes 55 checks against `build/before_table_cleanup_20260905_105002/` and the source tables; its output is `build/table_cleanup_verification.json`. The 27-file upload was independently extracted and compiled with Tectonic 0.17.0, producing identical PDF page content to the working build. All 19 pages were rendered and inspected. No overfull boxes, unresolved citations/references, or undefined font shapes remain. Some underfull-box notices and an environment-level Fontconfig configuration warning remain; the expected normal, bold, and italic fonts are embedded.

To recheck the dated delivery, run Python with `scripts/verify_table_cleanup.py --baseline build/before_table_cleanup_20260905_105002 --pdf build/pdf/main_table_cleanup_20260905.pdf --report build/table_cleanup_verification.json` from this directory.

## Directory policy

- `sections/`: concise ICLR main text
- `appendix/`: derivations, implementation details, and full numerical tables
- `figures/`: the supplied framework PNG, vector routing diagnostics, compact main-text comparison, and full appendix panels
- `source_snapshot/`: author-side audit copy only; **do not upload this directory with the anonymous submission**
- `build/`: upload and compilation artifacts

The upload excludes this README, the internal report/checks, `source_snapshot/`, backups, local tools, and rendered QA images. The pre-round backup is retained under `build/before_table_cleanup_20260905_105002/`.

The compact centered-square figure is generated by `scripts/build_compact_figure.ps1`. For each overlap it retains the CFD reference, E2-selected specialist, and T2-C velocity fields plus the selected-specialist and T2-C velocity-error maps. It draws source bands at native resolution and does not synthesize experimental fields. Complete velocity, pressure, and error panels are included in the appendix.

## Submission reminders

- Keep `\iclrfinalcopy` disabled for review.
- Replace the AI-use TODO with the authors' accurate disclosure before submission.
- Verify that the main text is within the official nine-page limit after final compilation.
- Keep all author names, affiliations, acknowledgements, repository identities, and machine paths out of the anonymous upload.
