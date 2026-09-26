from pathlib import Path
import re

p=Path(r'C:\Users\panxy1019\Desktop\STABLEMOE\PMD_Galerkin_Pan_ICLR (2)')
def edit(name, old, new):
    f=p/name
    s=f.read_text(encoding='utf-8')
    if old not in s: raise ValueError((name,old[:100]))
    f.write_text(s.replace(old,new),encoding='utf-8')

edit('main.tex','In this paper, We introduce','We introduce')
edit('main.tex','joint velocity–pressure error','joint velocity--pressure error ($E_u+E_p$)')
edit('main.tex','used solely to assist with language polishing and LaTeX formatting.','used to assist with language editing, manuscript organization, literature lookup, consistency checks of reported claims, and LaTeX formatting. No new experimental results were generated in this manuscript-revision workflow.')
edit('main.tex','Appendices~A--E document',r'Appendices~\ref{app:problem_pod}--\ref{app:implementation} document')
edit('main.tex','Appendices~F--H provide',r'Appendices~\ref{app:circular}--\ref{app:pinball} provide')

f=p/'sections/introduction.tex'; s=f.read_text(encoding='utf-8')
start=s.index('Localization, however,'); end=s.index('Our contributions are threefold:')
s=s[:start]+r'''Local ROMs reduce the burden on a single representation, but their coefficients refer to different physical modes and cannot be averaged directly as shared features. Change-of-basis maps and combinations of reconstructed outputs can connect such models. Our question is how to select or combine local predictions during autonomous forecasting without imposing shared coordinates or repeatedly transferring the evolving state between models. Mixture-of-experts (MoE) architectures provide a framework for conditional specialization \citep{jacobs1991,jordan1994,shazeer2017,fedus2022}; here we separate correction within a local ROM from the assembly of complete local predictions.

We introduce the Regime-Adaptive Localized Mixture-of-Experts Reduced-Order Model (RAL-MoE-ROM). Overlapping regions of parameter space have independently constructed local reduced coordinate systems, referred to as charts. Each local specialist augments a projected ROM with structured sparse corrections trained through autonomous multi-step rollouts. An outer parameter-conditioned router provides model preferences within an applicability-constrained candidate set. For an applicable adjacent pair, a convex weight inferred from the parameter and initial physical history remains fixed throughout the prediction window. Both specialists evolve independently in their own coordinates; only their reconstructed outputs are combined, with no feedback into either trajectory.

'''+s[end:]
s=s.replace('through a history-conditioned convex weight','through a convex weight conditioned on the initial history and fixed over the prediction window')
s=s.replace('(a) A parameter-only outer router selects one regime-local specialist or an adjacent pair. Selected specialists encode the same physical history in their native coordinates, evolve independently, and are combined only after reconstruction; recent physical history conditions the Top-2 fusion but not the outer router. (b) Each specialist augments the projected Galerkin backbone with a structured sparse MoE correction, while pressure is reconstructed algebraically. Flow thumbnails are schematic.', '(a) The parameter-conditioned outer router supplies preferences; applicability rules restrict selection to one local specialist or an adjacent pair. Selected specialists encode the initial physical history in their own coordinates, evolve independently, and reconstruct their fields before fusion. The fusion weight uses the initial history and remains fixed over the window; the fused output is not fed back. (b) Each specialist augments projected Galerkin dynamics with a structured sparse correction and an algebraic pressure update.')
f.write_text(s,encoding='utf-8')

edit('sections/method.tex','These components are organized by a two-level routing hierarchy. Overlapping regime-local parameter regions are equipped with independent reduced charts.','The two functional levels are correction within a local ROM and selection or fusion between complete local ROMs. Overlapping regions of parameter space are equipped with independent reduced charts.')
edit('sections/method.tex','Routing is hierarchical within each specialist.','Each specialist uses group-and-expert routing.')
edit('sections/method.tex','Thus velocity is advanced dynamically, whereas pressure is reconstructed algebraically rather than evolved as a separate recurrent state.','Pressure is updated algebraically rather than through a separate differential equation. The updated pressure enters the state and history used for the next velocity step.')
edit('sections/method.tex','Chart-independent descriptors extracted from the initial history $\\mathcal H_n$ enter only the T2-C gate described below.','The T2-C gate uses chart-independent descriptors extracted from the initial history $\\mathcal H_n$; the applicability mask separately checks the initialization and horizon.')
edit('sections/method.tex','T2-C predicts a history-conditioned logit correction','T2-C predicts an initial-history-conditioned logit correction')

edit('sections/related_work.tex',r'\citep{colanera2025}. These approaches motivate local reduced representations over restricted regions of parameter or state space, but they also introduce a coordination problem because independently constructed local models generally differ in'+ '\n' +'their mean fields, bases, scalings, and reduced operators. RAL-MoE-ROM addresses this issue differently: each reduced chart is paired with a complete autonomous specialist, the specialists retain independent coordinates during rollout, and cross-regime coupling is performed only after reconstruction in the common physical space. The resulting problem is therefore not merely one of selecting a local basis or switching between local coordinates, but of assembling independently evolved local dynamical systems.',r'''\citep{colanera2025}. The recent fl-ROM formulation advances local models in parallel, blends reconstructed states using fuzzy memberships, and projects the blended state back into local coordinates for the next step \citep{colanera2026fuzzy}. RAL-MoE-ROM instead keeps the selected trajectories independent over the entire forecast window, with a weight conditioned on the initial history and no feedback from fusion. Its distinguishing design is this forecast organization together with learned local dynamical corrections, rather than physical-space blending alone.''')
edit('sections/related_work.tex','These approaches demonstrate the value of conditional specialization, but they do not address the same cross-chart assembly problem considered here.','These approaches motivate conditional specialization. Here the focus is the organization of autonomous local ROMs with independent coordinate systems.')

edit('sections/experiments.tex','All are two-dimensional systems in which varying the Reynolds number traverses steady, Hopf-onset, and periodic regimes.','All are two-dimensional systems covering steady, Hopf-onset, and periodic regimes across Reynolds numbers. The parameter is fixed within each trajectory, and models are trained separately for each geometry.')
edit('sections/experiments.tex','A rollout window is labeled ``Divergent\'\' if it violates the predefined rollout-divergence criterion, either through a non-finite state or a\nthreshold violation. N/E denotes a setting that is not evaluable under this criterion; table-specific notes identify the cause.', 'We distinguish validation-stage exclusion (V/F), test-rollout failure, and an unreported result (N/R); an unreported value alone is not evidence of divergence.')
edit('sections/experiments.tex','The validation-selected T2-C errors are also close to the diagnostic convex oracle in both overlaps.','The oracle selects its weight using a full-window objective, whereas this table reports terminal-step error. Its displayed values are diagnostic references, not lower bounds on terminal error; numerical proximity does not establish terminal-optimal fusion.')
edit('sections/experiments.tex','Validation-selected cross-regime routing and fusion on the centered-square benchmark at $K=24$.','Test errors for centered-square routing and fusion models selected on validation trajectories ($K=24$).')
edit('sections/experiments.tex','The oracle is an a posteriori convex reference and is not available at inference.','The oracle fits a full-window objective using future reference fields; its terminal error is not a terminal-optimal lower bound and it is unavailable at inference.')
edit('sections/experiments.tex',r'\shortstack{Convex\\oracle}',r'\shortstack{Diagnostic\\oracle}')
edit('sections/experiments.tex','A dash indicates that no baseline error is reported.','N/R denotes an unreported baseline error.')
edit('sections/experiments.tex','& -- & -- & --', '& N/R & N/R & N/R')
edit('sections/experiments.tex','Circular-cylinder autonomous-rollout errors (\\%). A dash indicates that the rollout diverged.','Circular-cylinder autonomous-rollout errors (\\%). V/F denotes exclusion at validation, with no held-out evaluation; N/R denotes an unreported pressure error. Neither label is interpreted as test-rollout divergence.')
edit('sections/experiments.tex','& Vanilla-FNN-MoE & - & -','& Vanilla-FNN-MoE & V/F & V/F')
edit('sections/experiments.tex','& DataOnly-MoE & 0.3380 & -','& DataOnly-MoE & 0.3380 & N/R')
edit('sections/experiments.tex','& DataOnly-MoE & 0.2570 & -','& DataOnly-MoE & 0.2570 & N/R')
edit('sections/experiments.tex','the Vanilla-FNN-MoE Hopf model is not evaluable under the prescribed rollout criterion. Removing the projected ROM backbone produces the largest degradation, with particularly large pressure errors for DataOnly-MoE in the Steady and Hopf regimes.','the Vanilla-FNN-MoE Hopf model is excluded at validation and has no reported test error. DataOnly-MoE increases the reported velocity errors in all regimes and the Periodic pressure error; its Steady and Hopf pressure errors are not reported.')

edit('appendix/circular_cylinder.tex',r'\caption{Circular-cylinder autonomous-rollout error ranges (\%). }',r'\caption{Circular-cylinder autonomous-rollout error ranges (\%). V/F denotes validation-stage exclusion; N/R denotes an unreported pressure error.}')
edit('appendix/circular_cylinder.tex','& Vanilla-FNN-MoE & - & -','& Vanilla-FNN-MoE & V/F & V/F')
edit('appendix/circular_cylinder.tex','& DataOnly-MoE & 0.1637--0.5122 & -','& DataOnly-MoE & 0.1637--0.5122 & N/R')
edit('appendix/circular_cylinder.tex','& DataOnly-MoE & 0.1260--0.3880 & -','& DataOnly-MoE & 0.1260--0.3880 & N/R')

edit('appendix/square_cylinder.tex','This terminal metric differs from the full-window objective used by the diagnostic convex oracle.','The diagnostic convex oracle chooses one weight per window by minimizing a full-window objective using the future reference fields. The table then evaluates that weight at the terminal step. It therefore does not report the minimum attainable terminal error over convex weights. Proximity to this reference cannot be interpreted as terminal optimality.')
edit('appendix/square_cylinder.tex','The convex oracle is an a posteriori reference unavailable during prediction.','The diagnostic oracle fits a full-window objective using future reference fields; its reported terminal error is not a terminal-optimal lower bound.')
edit('appendix/square_cylinder.tex',' & Convex oracle &',' & Diagnostic oracle &')
edit('appendix/fluidic_pinball.tex','T2-C also remains close to the corresponding window-wise convex oracle, indicating that the learned fusion captures most of the available benefit from convex physical-space assembly in these overlap evaluations.','The diagnostic oracle uses future reference fields to choose a window-wise convex weight and is unavailable during prediction. These values do not establish that the learned weight is optimal for the reported metric.')
f=p/'appendix/fluidic_pinball.tex'; s=f.read_text(encoding='utf-8'); s=re.sub(r'\\textbf\{(0\.(?:5889|6763|0928|0955))\}',r'\1',s); f.write_text(s,encoding='utf-8')
edit('appendix/routing_details.tex','descriptor computed from the current physical state and the two preceding\nstates.','descriptor computed from the physical state at initialization and the two preceding states. The descriptor is evaluated once per prediction window, not updated from subsequent predicted states.')

edit('sections/conclusion.tex','cross-regime coupling is performed only after reconstruction in physical space.','output-level fusion is performed only after reconstruction in physical space, with no feedback into the local trajectories.')
edit('sections/conclusion.tex','The present study assumes a validation-supported regime coverage and evaluates pairwise fusion only for adjacent specialists on two-dimensional incompressible flows.','The evidence is limited to held-out trajectories within the covered Reynolds-number ranges, with fixed parameters and separately trained models for each geometry. The study assumes validation-supported coverage and evaluates adjacent-pair fusion; it does not establish cross-geometry transfer, online switching under time-varying parameters, or a computational speedup. Convex output fusion alone does not ensure satisfaction of the full nonlinear governing equations.')

b=r'''
@article{vlachas2018,
  author = {Vlachas, Pantelis R. and Byeon, Wonmin and Wan, Zhong Y. and Sapsis, Themistoklis P. and Koumoutsakos, Petros},
  title = {Data-driven forecasting of high-dimensional chaotic systems with long short-term memory networks},
  journal = {Proceedings of the Royal Society A},
  volume = {474}, pages = {20170844}, year = {2018}, doi = {10.1098/rspa.2017.0844}
}
@article{kramer2024,
  author = {Kramer, Boris and Peherstorfer, Benjamin and Willcox, Karen E.},
  title = {Learning Nonlinear Reduced Models from Data with Operator Inference},
  journal = {Annual Review of Fluid Mechanics}, volume = {56}, pages = {521--548}, year = {2024},
  doi = {10.1146/annurev-fluid-121021-025220}
}
@article{bhat2025,
  author = {Bhat, Sourabh P. and Barral, Nicolas and Ricchiuto, Mario},
  title = {Error-based efficient parameter space partitioning for mesh adaptation and local reduced order models},
  journal = {Computer Methods in Applied Mechanics and Engineering}, volume = {435}, pages = {117649}, year = {2025},
  doi = {10.1016/j.cma.2024.117649}
}
@misc{manti2025,
  author = {Manti, Simone and Tsai, Ping-Hsuan and Lucantonio, Alessandro and Iliescu, Traian},
  title = {Symbolic Regression of Data-Driven Reduced Order Model Closures for Under-Resolved, Convection-Dominated Flows},
  year = {2025}, howpublished = {arXiv preprint arXiv:2502.04703}, url = {https://arxiv.org/abs/2502.04703}
}
@misc{colanera2025,
  author = {Colanera, Antonio and Magri, Luca},
  title = {Quantized local reduced-order modeling in time ({ql-ROM})},
  year = {2025}, howpublished = {arXiv preprint arXiv:2506.13738}, url = {https://arxiv.org/abs/2506.13738}
}
@misc{colanera2026fuzzy,
  author = {Colanera, Antonio and Magri, Luca},
  title = {Fuzzy local reduced order models ({fl-ROMs})},
  year = {2026}, howpublished = {arXiv preprint arXiv:2609.00031v2}, url = {https://arxiv.org/abs/2609.00031v2}
}
'''
with (p/'references.bib').open('a',encoding='utf-8') as f:f.write(b)
print('Manuscript edits applied.')
