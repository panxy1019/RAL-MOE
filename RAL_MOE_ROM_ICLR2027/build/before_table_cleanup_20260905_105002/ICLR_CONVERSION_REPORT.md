# ICLR 2027 conversion and review-response report

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
