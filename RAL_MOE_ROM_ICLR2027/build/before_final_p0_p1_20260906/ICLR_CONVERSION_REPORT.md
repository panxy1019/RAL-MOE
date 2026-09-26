# ICLR 2027 conversion and review-response report

## Current round: table logic and meta-language cleanup (2026-09-05)

This round implements `ICLR_table_logic_and_meta_cleanup_codex.md`. It preserves the paper's section structure, method, experimental results, and train/validation/test partitions.

### Implemented changes

- **Table 1 / experimental scope:** three benchmark rows now show full Reynolds-number coverage, local velocity/pressure ranks, main horizons, and primary evaluation roles. All nine chart ranges, 27 split counts, and nine horizons were transferred exactly to Appendix E, Table 5.
- **Table 2 / autonomous dynamics:** the first column is `Evaluation setting`, all four horizons remain, and the divergence column was replaced by explicit notes. Steady POD--Galerkin is N/E with 448/448 criterion violations. Hopf retains 0.4378/48.50/48.94 with the 141/320 qualification; finite threshold-violating windows are not incorrectly described as excluded survivors. Periodic core is identified as another periodic evaluation, not a fourth regime.
- **Table 3 / cross-regime assembly:** the main text now contains only joint field errors for the two square overlaps. Best single means the best fixed specialist per overlap: Hopf (1.7298) for S--H and Periodic (1.2079) for H--P. Only T2-C (1.1958 and 0.8465) is bold; the oracle remains a posteriori. Appendix G, Table 7 reproduces all 42 component/joint error values from the authoritative square source, including all 30 values migrated from the former main table.
- **Table 4 / local architecture:** removed the square-overlap column and outer-routing rows. All 22 circular-cylinder velocity/pressure ranges (44 endpoints) use the full precision already present in Appendix F. No matching per-trajectory raw values for the complete four-model, three-regime ablation were located, so no mean was invented. Vanilla-FNN-MoE's Hopf N/E is correctly explained as failure of the validation stability gate, not an observed held-out failure.
- **Narrative:** benchmark roles and transitions are explicit; metric/rounding conventions and table-specific N/E reasons are stated. Removed internal production/audit language and missing-record inventories from the formal manuscript. Limitations now uses the requested fixed-run scientific qualification, Appendix E headings are scientific, Appendix C keeps the actual training objective, and the reproducibility statement references Appendices A--E for method details.
- **Typesetting:** all four main tables use booktabs and small text, without resizebox. Enabled T1 encoding to restore the expected Times normal/bold/italic faces; added semantic title line breaks and constrained framework-node text widths to remove label collisions. Framework labels and connections are unchanged. Float barriers keep the complete square tables on page 16 before their figure pages (17--18), and the pinball appendix is continuous on page 19.

### Verification and deliverables

- `scripts/verify_table_cleanup.py` passes **55 checks**; details and the final PDF hash are in `build/table_cleanup_verification.json`.
- Display-equation blocks in the method and derivation/specialist/routing appendices are unchanged. Experimental image assets are byte-identical to the pre-round backup. Only framework layout, not its node labels or connections, changed.
- The final PDF has **19 US-Letter pages**. Conclusion and references start on page 8; appendices start on page 10. Tables 1--2 are on page 5, Tables 3--4 on page 6, Table 5 on page 14, and square Tables 7--8 on page 16.
- All 19 pages were rendered for visual inspection. The tables, title, framework, equations, and field panels are unclipped. No overfull boxes, unresolved references/citations, duplicate labels, undefined font shapes, or fatal build errors remain. Some underfull-box warnings and the local Fontconfig default-configuration warning remain; the PDF embeds normal, bold, and italic Times-compatible fonts correctly.
- `build/overleaf_upload_table_cleanup_20260905.zip` contains exactly **27 required source/style/bibliography/figure files** and was independently extracted and compiled. All 19 resulting page content streams and extracted page texts match the working build. The archive excludes internal reports, scripts, snapshots, backups, QA renders, local tools, and machine-specific paths.
- Current outputs: `build/pdf/main_table_cleanup_20260905.pdf`, `build/overleaf_upload_table_cleanup_20260905.zip`, and the identical canonical `build/overleaf_upload.zip`.
- The E-drive mirror's old `build/pdf/main.pdf` was locked by another application, so it was preserved rather than forcibly closed or overwritten. The dated PDF and matching dated log/auxiliary files are the verified current delivery. Source files and both upload ZIP names are current.
- Pre-round recovery snapshot: `build/before_table_cleanup_20260905_105002/`. The downloaded journal project was not modified. No online Overleaf project was changed in this round.

The AI-use disclosure remains an author-owned TODO and must be completed truthfully before submission. The internal evidence-recovery items below still apply, but are no longer rendered as missing-file lists in the paper.

## Earlier review round (historical record)

The sections below describe the preceding 18-page version. Its table layout, formal-paper limitation wording, and page-count verification are superseded by the current-round record above.

## Review corrections implemented

- Corrected the outer E2 router to a parameter-only, fixed trajectory-level router. Physical-history descriptors now enter only the T2-C correction gate.
- Redrew Figure 1 as a vector three-panel schematic with separate parameter/history streams and explicit group Top-1, channel Top-2, shared, nonlinear, affine, and low-rank bilinear components.
- Removed the unsupported single `lambda_route R_route` term and restored the regime-wise standard-deviation-normalized, dimension-normalized, closed-loop multi-step objective.
- Defined divergence as failure of the original predefined criterion, including non-finite states and threshold violations; no unavailable numerical threshold was invented.
- Moved Amsallem and Farhat (2012) to projection-ROM stabilization and strengthened the local-ROM and learned-dynamics references using bibliography entries already retained in the project.
- Eliminated the `Eq. equation N` failure caused by the local `eqref` redefinition.
- Rebuilt Figure 2 to retain only the CFD reference, E2-selected specialist, T2-C prediction, and the two corresponding velocity-error rows. The scientific caption contains no production notes.
- Added the complete centered-square velocity, gauge-corrected pressure, and pointwise-error panels to the appendix.
- Added a compact ablation table without generating new metrics and softened claims about global representations and finite-horizon stability.
- Compressed the Introduction while clarifying the two routing levels, expanded Related Work, and added cross-geometry, capacity-matching, checkpoint, seed, and divergence-threshold limitations.

## Appendix restored

- Full problem statement, pressure gauge, finite-volume inner products, weighted POD optimization and SVD construction.
- Projected momentum operators and gauge-consistent Pressure--Poisson map.
- Semi-explicit velocity/pressure specialist, three-state standardized feature construction, group and channel routing, shared plus routed experts, structured expert map, frozen-context RK4, and progressive rollout objective.
- E2 adjacency and admissibility, all six physical-history descriptors, the 7-dimensional T2-C input, gate architecture, quadratic field-error objective, and validation selection rule.
- Held-out-by-Re data contract, chart ranks, known mesh/cadence values, and evidence boundaries for missing reproducibility fields.

## Evidence deliberately not fabricated

The retained final-manuscript package does not contain matching final checkpoints or complete configs. Consequently, this revision does not report internal expert utilization/non-collapse plots, final total or active parameter counts, a parameter-matched fairness claim, multi-seed uncertainty, optimizer/training schedule values, or the numerical divergence threshold. An older V16 archive was inspected but rejected as evidence for the final three-local-specialist RAL model because its architecture and training contract do not match.

## Verification

- Tectonic completes the LaTeX/BibTeX build with no undefined citations, undefined references, duplicate labels, missing inputs, overfull boxes, or fatal errors.
- The generated PDF is US Letter and has 18 pages. The conclusion and references begin on page 8; the appendices begin on page 10, so the main paper is within the nine-page text limit.
- All 18 pages were rendered for visual inspection. The vector framework, compact main-text comparison, tables, equations, and two full appendix figure pages are legible and unclipped.
- Submission-source scans find none of the known incorrect E2-input wording, fictitious routing term, `Eq. equation` string, raster-production caption text, or unresolved-reference markers.
- The anonymous upload excludes `source_snapshot/`, local build tools, rendered QA images, and machine-specific paths.

## Author actions still required

- Complete the AI-use statement truthfully.
- Recover the final checkpoint/configuration/evaluator records before claiming exact reproducibility, utilization analysis, parameter fairness, numerical divergence thresholds, or multi-seed statistics.
- Run one final Overleaf PDFLaTeX pass and recheck page/float placement after any author edits.
- Restore authors, affiliations, acknowledgements, and non-anonymous artifact links only for the camera-ready version.

## Source preservation

The downloaded journal source remains at its original location and was not edited. `source_snapshot/` is an author-side audit copy and is excluded from the anonymous Overleaf archive.
