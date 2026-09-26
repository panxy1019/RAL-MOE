# Descriptor correction verification

The current appendix and the supplied `paper_before` backup already contained the eight-channel correction when this task began. No six-channel formula remained in `appendix/routing_details.tex`. This pass verified every channel against the centered-square source, explicitly defined its cell-area inner products and weighted pressure gauge, and made the raw descriptor dimension `R^8` explicit. The equation label `eq:app_t2c_descriptor` and all existing labels are preserved.

## Source mapping

- `centeredsquare_fusion_v1/common.py:16`: EPS is `1e-12`.
- `common.py:78-81`: pressure uses the area-weighted zero-mean gauge.
- `common.py:109-145`: eight channels are, in order, log velocity RMS, log pressure RMS, log1p normalized latest velocity rate, log1p normalized previous velocity rate, tanh latest log-energy growth, tanh growth difference, log1p normalized latest pressure rate, and log1p normalized difference of velocity rates.
- Both time intervals are floored at EPS. Each amplitude denominator is `sqrt(max(energy, EPS)) + EPS`; the previous velocity-rate channel uses the previous velocity energy. The two RMS log channels floor `energy / total_area` at EPS before taking half its logarithm. Growth floors energy before taking its logarithm.
- The eighth channel has no extra division by a time interval. Calling it a discretized second derivative would be incorrect.
- `build_boundary_cache.py:509`: the parameter precedes the eight descriptors, yielding nine gate inputs.
- `train_t2c_gate.py:87-93`: means and population standard deviations use training windows only; standard deviations below `1e-8` are replaced by one.
- `common.py:180-203`: the gate has widths 9--64--64--1 with SiLU activations and a zero-initialized final correction layer. The parameter count is `(9*64+64)+(64*64+64)+(64+1)=4865`.

## Verification and limits

Read-only source inspection checked the eight formulas, feature order, numerical floors, standardization and architecture. Static paper search found the descriptor used in `sections/method.tex` with consistent notation; no equation label was removed. No executable source, checkpoint, or experimental number was changed. No experiments or LaTeX compilation were run by this subtask.

These equations document the verified centered-square implementation only. They do not establish an identical descriptor or normalization for every benchmark. No external reference scales have been introduced. Existing claims outside the descriptor subsection, including the mu-only ablation, are outside this subtask's independent verification scope.
