# Candidate B: nested hierarchy

This candidate makes the two routing levels visible by containment: E2 selects regime-local cards, and the enlarged H card contains the sparse closure. The H-P pair is explicitly an example; the muted S card does not contribute to this example's output. The visual uses 12 by 5.2 inches, native vector geometry, embedded TrueType PDF fonts, editable SVG text, and labels of at least 18.5 pt (8.50 pt at a 397 pt paper width).

## Semantics and deliberate omissions

- E2 receives only the physical parameter. Its trajectory-level probabilities are marked fixed. Top-1 and adjacent Top-2 are the allowed outer modes; an admissible Top-2 pair is S-H or H-P, never S-P or all three.
- The dashed initial-history branch initializes the selected native charts. Its separate descriptor d enters T2-C. Initial history is not an E2 input, and the fusion output has no feedback edge.
- H and P have independent coordinates and independent rollouts. Each reconstructs in physical space before the weighted fusion. There is no latent-space blend.
- The expanded H closure depicts one shared expert E0 plus two active experts E1 and E2 in parallel, with muted remaining experts. They enter a weighted sum; the shared expert does not precede the selected experts. Expert indices are illustrative. The sparse router's group-selection stage and channel-wise replication are omitted for space.
- The projected ROM term and learned weighted closure add before the velocity rollout; the integral glyph represents velocity advancement. The displayed a_H(t) and a_P(t) denote velocity coordinates. Pressure is supplied by the algebraic physics-data map, not integrated as a combined velocity-pressure recurrent state. RK internals, pressure equations, and expert sub-branches are intentionally omitted.
- T2-C's alpha(mu, pi_bar, d) label makes its three conditioning quantities explicit. In an overlap, the gate combines fixed pair-normalized E2 probabilities with the initial physical-history descriptor and parameter. Its output acts only on reconstructed physical fields.

## Caption suggestion

RAL-MoE-ROM nests sparse closure experts inside regime-local reduced models. The parameter-only E2 router selects one specialist or an admissible adjacent pair; the H-P case is illustrated. Selected specialists initialize from the same physical history, evolve independently, and reconstruct before T2-C combines their physical fields. The shared and selected closure experts contribute in parallel. Pressure remains algebraic; only velocity coordinates are advanced.

## Verification

The builder prints the ordinary English word count and smallest font size. The PDF is rendered with Poppler and visually reviewed after generation. Figure source, PDF, SVG, and PNG share the B_hierarchy basename. No current manuscript or existing figure is modified.
