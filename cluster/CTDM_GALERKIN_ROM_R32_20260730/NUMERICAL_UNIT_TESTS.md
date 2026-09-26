# CTDM pre-training numerical unit tests

All mathematical tests ran with `/root/miniconda3/envs/pt_env/bin/python` on
CPU in double precision (PyTorch `2.13.0+cu126`). No Hopf model training was
started before these tests passed.

## DISCRETE_TO_CONTINUOUS_CONVERGENCE

The matched discrete update used

\[
D_n=\exp(-\Delta t_n\Gamma_n),\qquad
\beta_n=1-\exp(-\eta_n\Delta t_n).
\]

| dt | Relative terminal error |
|---:|---:|
| 0.03125 | 1.1512245e-2 |
| 0.015625 | 5.7479335e-3 |
| 0.0078125 | 2.8719213e-3 |
| 0.00390625 | 1.4354495e-3 |

Observed orders were `1.00205`, `1.00103`, and `1.00051`; the expected
first-order continuous-limit convergence passed.

For the sign-reversed write equation, the errors remained
`1.7078`--`1.7090` and did not converge to zero. This independently rejects the
negative-write equation as the limit of the stated discrete KDA.

## RK4_ORDER_TEST

| dt | Relative terminal error |
|---:|---:|
| 0.125 | 7.9938955e-8 |
| 0.0625 | 4.9567907e-9 |
| 0.03125 | 3.0875656e-10 |
| 0.015625 | 1.9267467e-11 |

Observed orders were `4.01142`, `4.00486`, and `4.00223`. The continuous
memory RK4 implementation achieved the expected fourth-order convergence on
the smooth manufactured problem.

## MEMORY_BOUNDEDNESS_TEST

- Simulated physical time: `200`
- Integration step: `0.05`
- Maximum observed memory Frobenius norm: `0.993638`
- Final memory Frobenius norm: `0.879372`
- Dissipation-bound scale: `1.299038`

The memory remained finite and below the derived ultimate-bound scale. Its tail
did not exhibit monotonic growth.

## Runtime-contract smoke tests

After the frozen B0 checkpoint established the actual
`493 = 109 + 2 x 192` input contract, all B1--B4 smoke tests were rerun from
scratch with B1 input 493 and B2--B4 current input 109.

- Horizons: K4, K8, K16, K32, and K56.
- CPU path: bfloat16 autocast in `pt_env`, with matrix memory forced to float32.
- CUDA path: bfloat16 forward, backward, finite loss, and checkpoint roundtrip.
- B3: discrete update count exactly equaled the rollout horizon.
- B4: zero discrete updates and nonzero within-RK4-stage memory change.
- B3/B4 K56 memory-parameter gradient norms were nonzero (`0.14881` and
  `0.14884`).
- B3/B4 parameter tensors were name/shape identical, with `2,238,718`
  trainable parameters each.
- The B4 K56 benchmark passed micro-batches 4, 8, and 16. The deterministic
  selector chose micro-batch 16 and gradient accumulation 1.

## Gate decision

All mathematical, CPU, CUDA, shape, gradient, checkpoint, and K56 benchmark
gates passed before formal Hopf training began.
