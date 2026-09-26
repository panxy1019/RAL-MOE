# CDM-GROM Notation and Assumptions

## 1. Scope

This document fixes the notation and the minimal assumptions used by the
Continuous Delta-Memory Galerkin Reduced-Order Model (CDM-GROM). CDM-GROM is a
deterministic, operator-based reduced-order model. It contains no neural
network, learned router, attention layer, external phase variable, or explicit
delay-window concatenation.

The pressure variable is algebraic. The dynamical state is the pair
\((a,S)\), or an equivalent minimal auxiliary realization \((a,y_1,\ldots,y_m)\).

## 2. Core notation

| Symbol | Dimension/domain | Meaning |
|---|---:|---|
| \(\mu\) | \(\mathcal P\subset\mathbb R^{d_\mu}\) | Physical parameter; for the motivating flow, Reynolds number or a smooth reparameterization of it |
| \(a(t;\mu)\) | \(\mathbb R^r\) | Resolved velocity POD coefficients |
| \(b(t;\mu)\) | \(\mathbb R^{r_p}\) | Algebraic pressure POD coefficients |
| \(\mathcal P_p(a;\mu)\) | \(\mathbb R^{r_p}\) | Pressure–Poisson reconstruction map |
| \(f_G(a,b;\mu)\) | \(\mathbb R^r\) | Semi-explicit velocity POD–Galerkin vector field |
| \(F_G(a;\mu)\) | \(\mathbb R^r\) | Pressure-eliminated field \(f_G(a,\mathcal P_p(a;\mu);\mu)\) |
| \(\phi(a;\mu)\) | \(\mathbb R^{d_\phi}\) | Fixed analytic feature vector, e.g. \([1,a,a\otimes a,\mu]^\top\) |
| \(S(t)\) | \(\mathbb R^{d_k\times d_v}\) | Matrix delta-memory state |
| \(K,V,Q\) | compatible fixed matrices | Operator-derived feature maps; never neural-network weights |
| \(k(a;\mu)\) | \(\mathbb R^{d_k}\) | Normalized write address \(K\phi/(\|K\phi\|+\varepsilon)\) |
| \(v(a;\mu)\) | \(\mathbb R^{d_v}\) | Write target \(V\phi\) |
| \(q(a;\mu)\) | \(\mathbb R^{d_k}\) | Normalized read address \(Q\phi/(\|Q\phi\|+\varepsilon)\) |
| \(\Gamma(a,\mu)\) | \(\mathbb R^{d_k\times d_k}\) | Diagonal channel-wise forgetting-rate matrix |
| \(\eta(a,\mu)\) | \(\mathbb R_{\ge 0}\) | Delta write rate |
| \(y=S^\top q\) | \(\mathbb R^{d_v}\) | Memory read |
| \(B\) | \(\mathbb R^{r\times d_v}\) | Fixed closure-output matrix |
| \(r_{\rm mem}=By\) | \(\mathbb R^r\) | Memory closure |
| \(h_n\) | \(\mathbb R_{>0}\) | Physical step \(t_{n+1}-t_n\) |
| \(D_n\) | \(\mathbb R^{d_k\times d_k}\) | \(e^{-h_n\Gamma_n}\) |
| \(\beta_n\) | \([0,1)\) | \(1-e^{-h_n\eta_n}\) |
| \(y_j\) | \(\mathbb R^{d_j}\) | Auxiliary memory state in a finite realization |
| \(B_j\) | \(\mathbb R^{r\times d_j}\) | Auxiliary-to-resolved coupling |
| \(C_j\) | \(\mathbb R^{d_j\times r}\) | Resolved-to-auxiliary coupling |
| \(\Lambda_j\) | \(\mathbb R^{d_j\times d_j}\) | Stable auxiliary decay matrix |
| \(K_m(\tau)\) | \(\mathbb R^{r\times r}\) | Finite exponential memory kernel |
| \(g(t)\) | \(\mathbb R^r\) | Initial-slip/orthogonal-dynamics forcing |
| \(\|\cdot\|\) | — | Euclidean vector norm or induced spectral matrix norm |
| \(\|\cdot\|_F\) | — | Frobenius norm |
| \(\langle X,Y\rangle_F\) | — | \(\operatorname{tr}(X^\top Y)\) |

The normalization denominator uses a fixed \(\varepsilon>0\). Consequently,
\(\|k\|\le 1\) and \(\|q\|\le 1\), including when the unnormalized feature map
vanishes.

## 3. Numbered assumptions

The assumptions are intentionally modular. No result may cite “all
assumptions” when only a subset is required.

### A1 — Compact parameter domain

\(\mathcal P\subset\mathbb R^{d_\mu}\) is compact.

### A2 — Local regularity of the pressure-eliminated Galerkin field

For every \(\mu\in\mathcal P\), \(F_G(\cdot;\mu)\) is locally Lipschitz in
\(a\). On every compact \(A\subset\mathbb R^r\), its local Lipschitz constant
may be chosen uniformly in \(\mu\in\mathcal P\).

### A3 — Local regularity of memory coefficients

The maps \(k,v,q,\Gamma,\eta\) are continuous in \((a,\mu)\) and locally
Lipschitz in \(a\), uniformly in \(\mu\) on compact subsets. The matrix
\(\Gamma(a,\mu)\) is real, diagonal, and therefore symmetric.

### A4 — Address normalization

\[
\|k(a;\mu)\|\le 1,\qquad \|q(a;\mu)\|\le 1
\]
for all admissible \((a,\mu)\).

### A5 — Strict channel-wise forgetting

There is a constant \(\gamma_{\min}>0\) such that
\[
\Gamma(a,\mu)\succeq\gamma_{\min}I_{d_k}
\]
for all admissible \((a,\mu)\).

### A6 — Bounded nonnegative write rate

There is an \(\eta_{\max}<\infty\) such that
\[
0\le\eta(a,\mu)\le\eta_{\max}
\]
for all admissible \((a,\mu)\).

### A7 — Bounded write target for the memory bound

For the memory-state boundedness theorem only, there is a
\(v_{\max}<\infty\) such that
\[
\|v(a;\mu)\|\le v_{\max}
\]
along the considered solution. A7 is not silently inferred from A1–A6; it
must be checked from the feature map or from a separately established bound on
\(a(t)\).

### A8 — Stable unresolved linear block

In the linear resolved–unresolved correspondence, \(A_{uu}\) is Hurwitz:
every eigenvalue has strictly negative real part. This assumption is sufficient
for exponential decay of \(e^{A_{uu}t}\). It is not asserted for an arbitrary
nonlinear Navier–Stokes orthogonal dynamics.

### A9 — Reciprocal energy coupling

For the energy-stable subclass,
\[
\dot a=F_G(a)-\sum_{j=1}^m C_j^\top y_j,\qquad
\dot y_j=\eta_jC_ja-\lambda_jy_j,
\]
where \(\eta_j>0\) and \(\lambda_j>0\) are constants. The weighted memory
energy is \(\|y_j\|^2/(2\eta_j)\).

## 4. Theorem-specific regularity clauses

These clauses are not global assumptions and are invoked only where stated.

| Clause | Use |
|---|---|
| C1 | The coefficient functions sampled by the discrete recurrence and the exact memory solution are sufficiently smooth on \([0,T]\) for uniform second-order Taylor remainders. |
| C2 | For first-order global convergence, the exact trajectory remains in a compact set, the continuous vector field is Lipschitz there, and \(h=\max_n h_n\to0\). |
| C3 | For the kernel-error theorem, exact and approximate resolved solutions remain in a common ball of radius \(M\), and the two kernels belong to \(L^1(0,T)\). |
| C4 | For the absorbing-ball extension, \(a^\top F_G(a)\le c_0-c_1\|a\|^2\) with \(c_0\ge0\) and \(c_1>0\). |

## 5. Claims deliberately excluded

1. The continuous ODE is not claimed to equal finite-step discrete KDA.
2. Memory boundedness alone is not a global-stability proof for \((a,S)\).
3. The linear resolved–unresolved elimination is not an exact nonlinear
   Mori–Zwanzig theorem.
4. The reciprocal subclass is not claimed to reproduce every Hopf growth
   mechanism.
5. A full matrix state is not claimed to be minimal when the addresses are
   fixed.
6. No novelty claim about the first use of memory in ROMs is made here.

