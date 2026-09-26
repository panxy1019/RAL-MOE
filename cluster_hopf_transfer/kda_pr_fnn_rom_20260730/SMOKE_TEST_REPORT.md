# KDA-PR-FNN-ROM smoke-test report

Date: 2026-07-30

All B1, B2 and B3 smoke tests passed on the RTX 4090 with the `pt_env`
environment. The final rerun followed the numerically stable multi-timescale
tau initialization.

Verified:

1. runtime feature dimensions are B1=178 and B2/B3 current-only=46;
2. KDA memory resets for each trajectory and is not shared across Reynolds cases;
3. memory is unchanged through all four RK stages;
4. memory is written once per accepted macro state;
5. KDA matrix updates and reads remain float32 under BF16 AMP;
6. forward and backward values are finite;
7. all initialized parameters are finite;
8. state-dict checkpoint reconstruction is strict and complete;
9. autonomous rollout features do not read future truth;
10. identical zero-memory resets reproduce identical rollout results.

The first K56 benchmark exposed overflow in a naive inverse-softplus
initialization for long tau channels. It was corrected with the stable
large-argument identity branch and the entire smoke suite was rerun.

Parameter counts:

| Model | Trainable parameters | Dynamic memory state |
|---|---:|---:|
| B1 Deep-FNN-H3 | 1,745,797 | 0 |
| B2 KDA radial-off | 1,830,621 | 1,024 |
| B3 KDA radial-on | 1,830,621 | 1,024 |

B2 has 4.86% more trainable parameters than B1, below the specified 10%
parameter-matching threshold.
