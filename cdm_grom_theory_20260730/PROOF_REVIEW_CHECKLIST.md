# CDM-GROM Strict Proof Review

## 1. Overall assessment

After two algebraic passes and deterministic sanity checks, no known sign,
dimension, or inequality error remains in the stated CDM-GROM results.
However, the manuscript is a self-contained draft rather than a
peer-reviewed proof. The claims are deliberately limited to what the stated
assumptions support.

## 2. Problems found and corrected during review

### R1 — Local consistency target

Initial draft issue: the theorem compared the recurrence with the Euler
tangent \(S(t_n)+h_n\dot S(t_n)\), which demonstrates consistency but is not
the complete one-step truncation statement requested.

Correction: Theorem 1 now states and proves

\[
\|S_{n+1}^{rec}-S(t_{n+1})\|_F\le C_{\rm loc}h_n^2.
\]

The proof has separate recurrence and exact-solution Taylor remainders. The
sanity script now verifies the true one-step defect as well as the expansion
residual.

### R2 — Infinite-time memory bound

Initial draft issue: the memory estimate was followed by an unconditional
\(\limsup_{t\to\infty}\), even though Theorem 2 proves only local existence
for the coupled system.

Correction: the finite-time bound is stated on the maximal existence
interval. The limsup conclusion is now explicitly conditional on a global
coupled solution.

### R3 — Reciprocal absorbing bound

Initial draft issue: the energy inequality was correct, but global
continuation was not tied explicitly to local Lipschitz regularity.

Correction: the global absorbing conclusion now requires A2 in addition to
A9 and the Galerkin absorbing estimate. Bounded augmented energy plus the ODE
continuation criterion then excludes finite-time escape.

### R4 — Gradient-flow regularizer

Initial draft issue: symmetry of \(\Gamma\) is sufficient for the gradient
identity, but calling the second term a regularizer is physically meaningful
only when it is nonnegative.

Correction: Proposition 1 now assumes symmetric positive-semidefinite
\(\Gamma\); Corollary 2 strengthens this to positive definite \(\Gamma\).

### R5 — Passivity interpretation

Initial draft issue: the dissipation inequality was present without an
explicit supply-rate interpretation.

Correction: the manuscript now identifies storage \(\|S\|_F^2/2\), input
\(v\), read output \(y=S^\top k\), supply
\(\eta\|v\|^2/2\), and output penalty \(\eta\|y\|^2/2\).

## 3. Result-by-result review

| Result | Review status | Important qualification |
|---|---|---|
| Lemma 1 | Complete | Uniform remainder requires bounded coefficient sets |
| Theorem 1 | Complete | Requires smooth sampled coefficients/exact trajectory |
| Corollary 1 | Complete | Requires common compact trajectory set and one-step Lipschitz stability |
| Proposition 1 | Complete | Fixed coefficients, \(\eta>0\), symmetric PSD \(\Gamma\) |
| Corollary 2 | Complete | Fixed input and SPD \(\Gamma\) |
| Theorem 2 | Complete | Local existence only |
| Theorem 3 | Complete | A7 must hold along the solution; limsup needs global existence |
| Proposition 2 | Complete | Pointwise dissipativity; does not bound \(a\) |
| Proposition 3 | Complete | “Minimal” is limited to the observable fixed-address matrix component |
| Theorem 4 | Complete | Linear time-invariant auxiliary blocks |
| Theorem 5 | Complete | Exact only for linear/linearized block dynamics |
| Corollary 3 | Complete | Stability uses A8; reduced realization is approximate |
| Theorem 6 | Complete | Reciprocal subclass may suppress Hopf growth |
| Theorem 7 | Complete | Finite-time, conservative bound; assumes bounded approximate solution |

## 4. Remaining items requiring human or future project review

These are not hidden proof gaps; they are external validation tasks.

1. **Reference audit.** No references were fabricated. The manuscript retains
   a TODO marker for finite-memory, Mori--Zwanzig, balanced truncation, and
   related ROM literature. A domain expert must supply and verify citations.
2. **Actual TeX compile.** No TeX engine is installed locally. Static checks
   pass, but the PDF and compiler log still require a TeX-enabled machine.
3. **Operator realization.** The current CenteredSquare bundle retains only
   rank-11 modes. A pure operator realization requires a new train-only
   higher-rank POD/Galerkin construction.
4. **A8 verification.** The unresolved block must be checked at every
   parameter node. If \(A_{uu}\) is not Hurwitz, the exact unresolved
   realization is not a stable fading-memory closure.
5. **A7 verification.** For \(v=V[1,a,a\otimes a,\mu]^\top\), boundedness of
   \(v\) follows only after a bound on \(a\), or after using a bounded analytic
   feature. It must not be assumed from compactness of \(\mu\) alone.
6. **Nonnormal reduced blocks.** Positive spectral real parts of
   \(\Lambda_j\) imply exponential stability but do not give monotone
   Euclidean energy. A Lyapunov metric is needed if such an energy statement
   is desired.
7. **Hopf expressivity.** Real decaying poles and the reciprocal subclass may
   be too restrictive near onset. Rotational \(2\times2\) blocks are a future
   extension, not part of the present theorems.
8. **Nonlinear memory.** State-dependent addresses create a nonlinear
   fading-memory operator. The linear Volterra and Galerkin-elimination
   theorems must not be cited as exact statements for that model.
9. **Time discretization.** A future CenteredSquare realization must verify
   joint state/memory time-step convergence because the previously tested
   direct continuous Galerkin backbone was unstable at the native snapshot
   interval.

## 5. Claim-safety conclusion

The following claims are supported:

- the stated recurrence is first-order consistent with the continuous
  delta-memory ODE;
- fixed-input delta memory is a regularized gradient flow;
- the memory subsystem has an explicit conditional bound;
- fixed-address memory reduces to auxiliary states;
- finite auxiliary states realize finite exponential Volterra kernels;
- a linear resolved--unresolved Galerkin block yields the stated exact kernel;
- reciprocal coupling gives the stated energy identity;
- the stated finite-time kernel-error estimate follows from Gronwall.

The following claims are not supported and are not made:

- finite-step equivalence to discrete KDA;
- unconditional global stability of CDM-GROM;
- nonlinear Mori--Zwanzig exactness;
- universal representation of Hopf growth;
- empirical superiority on the CenteredSquare dataset.

