# Circular aggregation and Pinball identity audit — 2026-09-20

Read-only cluster inspection; no training or model evaluation. CPU checkpoint deserialization and strict architecture loading only. After the audit, parent agent authorized the narrow paper edits listed below. Remote base R = `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE` on `root@10.10.164.243:20381`.

## 1. Circular: no common aggregation supports the current main table

Confirmed original raw Hopf source: `R/Hopf/migrated_h4_expanded/native_attractor_audit_20260723_v1/results/HOPF_NATIVE_ATTRACTOR_AUDIT.json`. Its held-out rows are transcribed without rounding in `R/paper_experiments/reports/figure_data/HOPF_STRICT_ATTRACTOR_PER_RE.csv` (last three rows). The report generator `R/paper_experiments/code/generate_final_reports.py:160–170` explicitly filters `split == heldout`, selects horizon 56, and uses `statistics.mean` separately for velocity and pressure.

| Re | K56 velocity (%) | K56 pressure (%) |
|---|---:|---:|
| 47.081356048583984 | 0.0032075548550635166 | 0.03551875900586877 |
| 49.02235412597656 | 0.028787521520837964 | 0.4321343565488563 |
| 51.78644943237305 | 0.0060956492707790254 | 0.041411763594408735 |
| Equal-Re mean | **0.012696908548893502** | **0.16968829304971128** |

The same means appear in `R/paper_experiments/reports/tables/TABLE_A1_KLONG_PERCENT.csv`, Hopf/Proposed Specialist MoE row, and revision4 `reports/tables/TABLE_4_2_1_GLOBAL_INCLUDED.csv`. Thus an explicitly source-backed mean row is **0.0127 / 0.1697**, replacing both main-table Hopf proposed entries if that row is defined as equal-Re mean. Appendix range **0.0032–0.0288 / 0.0355–0.4321** is supported. Main pressure **0.1338** is neither this mean nor range midpoint (**0.2338** from rounded bounds); main velocity **0.0160** is the midpoint of rounded bounds. A caption-only rename to “midrange” is invalid.

The existing Circular table has further provenance errors, so the remaining entries must not simply be re-labelled mean:

- `R/paper_experiments/revisions/revision5_supplemental_evaluation_20260725/manuscript/NUMERICAL_EXPERIMENTS_DRAFT_REVISION5_FULL.md:46–53` labels Steady **0.0330–0.1723 velocity as Vanilla**, not Proposed, and **0.0384–0.2995 as Proposed S3-B**, not Vanilla. These are the opposite assignments in the current paper appendix.
- `R/paper_experiments/revisions/revision5_supplemental_evaluation_20260725/reports/REVISION5_SUPPLEMENTAL_EVALUATION_ZH.md`, §2.2, assigns Steady pressure **0.2500–1.3249 to Global MoE**, while the paper assigns it to Proposed. The paper's Global pressure bounds **1.5226–7.4363** combine Proposed's minimum and Vanilla's maximum from the prior source. These are substantive mapping concerns, not merely mean-vs-midrange ambiguity.
- That same revision5 report gives equal-Re Global K56 means Steady **0.1809 / 0.6639**, Hopf **0.0187 / 0.1880**, Periodic **1.2147 / 4.9544**. Do not put its Periodic K56 values in a K48 row.
- Earlier `TABLE_A1_KLONG_PERCENT.csv` supports mean Periodic Proposed **0.8132 / 3.4445**, Vanilla **2.2109 / 9.5010**, DataOnly **1.0107 / 4.2341**; these are archived evaluations, not established identities for every present table row. Do not silently replace all present rows with an earlier evaluation vintage.

Recommended paper treatment: use the verified Hopf row with an explicit mean label and raw-source citation; suppress or explicitly mark the remaining Circular scalar entries as provenance/aggregation unresolved until a consistent run-to-row manifest is reconstructed. Retaining their numeric values while saying “mean” or “midrange” is unsupported. Restoring the current appendix ranges wholesale is also unsafe because of the demonstrated Steady label mixing. No complete corrected Circular comparison table was established in this bounded audit.

## 2. Pinball Periodic: identity closed by corrected protocol hash

Let P = `R/fluidic_pinball_periodic_v2_b1`.

The exact published numbers occur in:

- `P/evaluation_final_test_corrected_dt1_all_offsets_k1_k32/ROLLOUT_SUMMARY.json`, `aggregate.32`: velocity `0.002896951825176883`, pressure `0.005518001349021991`, joint `0.008414953235640295` (multiply by 100 for paper percentages).
- `P/evaluation_final_test_corrected_dt1_all_offsets_k56_core/ROLLOUT_SUMMARY.json`, `aggregate.56`: velocity `0.006221541295027626`, pressure `0.010496483450489385`, joint `0.016718024254909584`.

Both name `P/runs/FluidicPinballV2_B1_Deep_FNN_H3_seed1248/best_validation.pt`, best/optimizer step 8000. `P/CORRECTED_FINAL_TEST_PROTOCOL_MANIFEST.json` records checkpoint SHA-256 **c89a0cbf6cc0920664875f00234afaac49bc19b1c5c5f985cbba0b09e2dbdbda**, identical to the live file hash measured this turn. The original unseal manifest `P/evaluation_final_test/FINAL_TEST_UNSEAL_MANIFEST.json` carries the same hash. This closes the prior missing-hash gap at the archived-manifest level.

Live CPU verification: loaded that checkpoint, observed `variant=b1`, `optimizer_step=8000`, instantiated `DeepFNNH3` from `P/code/train_b1_fluidic_pinball.py`, and strict-loaded `model_state`: **all keys matched**. Trainable count **1,792,959**, first weight `[512,263]`. Source lines 159–199 define six 512-wide dense layers and velocity/pressure/pressure-gate heads, no sparse MoE experts/router. Evaluator `P/code/evaluate_b1_validation.py:151–155` explicitly checks `variant == b1`, instantiates `K.DeepFNNH3`, and strict-loads it. The pressure gate is not a sparse expert router.

### Correct Table 2 identity, metric, and scope

Change **only the two verified Periodic learned-model row labels** from RAL-MoE-ROM to **Deep-FNN-H3 specialist (B1)**; retain their numeric values. Do not attribute those rows to sparse MoE evidence or claim all four settings establish RAL-MoE-ROM performance. Update adjacent prose accordingly. Steady/Hopf identities and the three POD–Galerkin baseline rows were not closed by this audit; do not relabel them to B1 by analogy.

The matching columns are **field_mean**, not endpoint error: evaluator lines 206–209 take `.mean()` over windows and time, and lines 218–225 take an equal mean over the selected per-Re rows. Use a table-specific definition such as `100 mean_Re mean_windows mean_{j=1..K} e_field(j)`. The sum is the reported joint field mean. `joint_terminal` is a distinct statistic: **1.4387025342633327%** for K32 and **3.077718655445746%** for K56. Do not relabel the published mean numbers as endpoint results.

Corrected manifest protocol:

- effective dt **1.0**, temporal stride **4** from original dt 0.25 snapshots, all four temporal offsets;
- K32 test Re **22.25, 24, 27.5, 31, 34.5, 38, 43, 50, 57**;
- K56 periodic-core Re **27.5, 31, 34.5, 38, 43, 50, 57**;
- unchanged checkpoint, no hyperparameter/model selection after unseal;
- original `evaluation_final_test` dt0.25 metrics are explicitly invalid for B1 comparison due to cadence mismatch. The corrected directories above are the relevant sources.

Baseline comparability remains unresolved here: no audit of POD–Galerkin row cadence, test sets, mean aggregation or final hashes was performed. Retain them only as separately reported results with that limitation; do not assert a matched quantitative superiority comparison from this identity audit alone.

## 3. Authorized narrow paper changes

- `sections/experiments.tex`: Hopf proposed scalar pair corrected to 0.0127/0.1697; caption defines this as the verified equal-Re mean and flags other retained Circular numbers as unreconciled, including Steady assignment inconsistencies. No blanket mean/midrange relabel.
- Same file: only the two verified Periodic learned-model labels changed to Deep-FNN-H3 (B1); original numeric values retained. Caption and narrative specify window means and remove claims that these rows demonstrate sparse MoE performance or a fully matched baseline improvement.
- `appendix/fluidic_pinball.tex` (actual filename; no `appendix/pinball.tex` exists): added verified architecture, fixed-checkpoint corrected cadence, exact K32/K56 Re sets, all offsets, averaging order, and separate endpoint values. Other Pinball/overlap identities remain explicitly outside this closure.
- No figure block edited. Applied with `apply_patch`; searched resulting files to verify intended identity, caveat, and number replacements. No LaTeX compile performed by this worker.
