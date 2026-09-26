# CDM-GROM Theory Sanity-Check Report

## 1. Scope

This report records deterministic algebraic checks for the CDM-GROM theory.
The checks do not train a model and do not constitute a flow simulation or a
substitute for proof. They verify dimensions, signs, asymptotic orders, and
equivalences that are easy to invalidate through implementation mistakes.

Execution date: 2026-07-31 (Asia/Shanghai)

Runtime:

```text
Python 3.12.13
NumPy 2.3.5
```

All Python files also passed `python -m py_compile`.

## 2. Required mathematical checks

### 2.1 Discrete-to-continuous limit

Script: `scripts/verify_continuous_limit.py`

Status: **PASS**

Observed behavior:

| Quantity | Finest observed orders |
|---|---:|
| Expansion residual \(S_{n+1}^{rec}-S_n-hF(S_n)\) | 1.9912, 1.9956, 1.9978 |
| True one-step defect \(S_{n+1}^{rec}-S(t_{n+1})\) | 1.9846, 1.9923, 1.9961 |
| Finite-time global recurrence error | 1.0006, 1.0003, 1.0002 |
| Deliberately wrong write-sign residual | 0.9975, 0.9987, 0.9994 |

Interpretation:

- The recurrence has a second-order local defect relative to the continuous
  memory ODE.
- Its fixed-time global error is first order.
- Reversing the delta-write sign leaves an \(O(h)\) mismatch, so the sign in
  the derived ODE is identifiable by the convergence test.

### 2.2 Gradient-flow identity

Script: `scripts/verify_gradient_flow.py`

Status: **PASS**

```text
relative_gradient_error = 3.084832e-10
flow_identity_error     = 1.922963e-16
equilibrium_residual    = 1.022752e-16
```

The central finite-difference gradient agrees with
\[
\nabla_S\mathcal J
=-k(v-S^\top k)^\top+\eta^{-1}\Gamma S,
\]
and the ODE agrees with \(-\eta\nabla_S\mathcal J\).

### 2.3 Memory and reciprocal-energy identities

Script: `scripts/verify_memory_energy_identity.py`

Status: **PASS**

```text
memory identity error         = 0.000000e+00
completing-square margin      = 1.156143e+00
passivity inequality margin   = 1.393057e+00
reciprocal cross-term error   = 4.440892e-16
```

The exact Frobenius energy identity, both stated upper bounds, and the
reciprocal-coupling cancellation were verified independently.

### 2.4 Fixed-key reduction

Script: `scripts/verify_fixed_key_reduction.py`

Status: **PASS**

```text
read derivative error          = 9.155134e-16
orthogonal derivative error    = 5.057163e-16
trajectory read error          = 1.262192e-15
orthogonal trajectory error    = 4.416239e-15
```

The observable vector \(y=S^\top e\) follows the reduced auxiliary equation,
while \(P_\perp S\) is unobservable and decays exactly as
\(e^{-\gamma t}P_\perp S(0)\).

### 2.5 Linear finite-state/memory-kernel equivalence

Script: `scripts/verify_linear_memory_kernel.py`

Status: **PASS**

```text
block/auxiliary trajectory error = 2.775558e-17
max convolution reconstruction    = 2.111291e-10
transfer-function error           = 0.000000e+00
kernel identity error             = 0.000000e+00
```

The block Galerkin system, its auxiliary-state realization, the Volterra
convolution reconstruction, and the Laplace-domain transfer function agree
within numerical precision and quadrature error.

## 3. Additional LaTeX structural check

Script: `scripts/verify_latex_structure.py`

Status: **PASS**

```text
LaTeX sources                 = 3
labels                        = 161
references                    = 61
formal result statements      = 14
matching appendix proofs      = 14
missing/duplicate labels      = 0
undefined references          = 0
missing input files           = 0
```

This is a source-structure audit, not a PDF compilation. The absence of a
local TeX engine is documented in `LATEX_COMPILATION_REPORT.md`.

## 4. Reproduction command

Use the bundled workspace Python:

```powershell
$py = 'C:\Users\panxy1019\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
Get-ChildItem .\scripts\verify_*.py |
  Sort-Object Name |
  ForEach-Object { & $py $_.FullName }
```

Every script exits nonzero on a failed assertion.

## 5. Interpretation boundary

The passing checks establish internal algebraic consistency for the tested
finite-dimensional identities. They do not establish:

- stability of a particular CenteredSquare CDM-GROM realization;
- validity of A7 for an unbounded polynomial feature map;
- the Hurwitz property of a future unresolved Galerkin block;
- nonlinear Mori--Zwanzig exactness;
- adequacy of real decaying poles for Hopf memory;
- correctness of references or novelty claims.

