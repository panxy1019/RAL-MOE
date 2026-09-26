# Continuous limit of the matched discrete KDA

Start from

\[
S_{n+1}=(I-\beta_n k_nk_n^\top)D_nS_n+\beta_nk_nv_n^\top,
\]

with

\[
D_n=\exp(-\Delta t_n\Gamma_n),\qquad
\beta_n=\eta_n\Delta t_n+O(\Delta t_n^2).
\]

Since

\[
D_n=I-\Delta t_n\Gamma_n+O(\Delta t_n^2),
\]

retaining first-order terms gives

\[
\frac{S_{n+1}-S_n}{\Delta t_n}
=-\Gamma_nS_n-\eta_nk_nk_n^\top S_n+\eta_nk_nv_n^\top+O(\Delta t_n).
\]

Using \(k k^\top S=k(S^\top k)^\top\), the limit is

\[
\dot S=-\Gamma S+\eta k(v-S^\top k)^\top.
\]

The implementation uses

\[
\Gamma=\operatorname{Diag}(\gamma_{\min}+\operatorname{softplus}(\cdot)),
\qquad
\eta=\eta_{\max}\operatorname{sigmoid}(\cdot).
\]

These plus signs are not optional:

1. `gamma_min + softplus` guarantees
   \(\Gamma\succeq\gamma_{\min}I\). `gamma_min - softplus` can create negative
   damping.
2. The write term is \(+\eta k(v-S^\top k)^\top\). A negative write term is not
   the continuous limit of the stated discrete KDA and reverses the online
   reconstruction-error correction.

The independent convergence test compares the stated discrete update against both
sign choices. It must show first-order convergence only to the plus-write equation
before any Hopf training is allowed.

For a single head, setting \(y=S^\top k\) yields

\[
\frac12\frac{d}{dt}\|S\|_F^2
=-\langle S,\Gamma S\rangle_F+\eta y^\top(v-y).
\]

If \(\Gamma\succeq\gamma_{\min}I\), \(0\le\eta\le\eta_{\max}\), and \(v\) is
bounded, then

\[
\frac12\frac{d}{dt}\|S\|_F^2
\le-\gamma_{\min}\|S\|_F^2+\frac{\eta_{\max}}4\|v\|_2^2.
\]

This proves an ultimate bound for the memory subsystem. It does not by itself
prove global stability of the coupled velocity-pressure ROM.
