# KDA-PR-FNN-ROM method contract

B1 replaces the frozen MoE with a six-layer width-512 FNN while retaining the
178-dimensional three-state input. B2 uses the same FNN and output heads, removes
explicit history, and adds four 16-by-16 KDA memories driven by the
46-dimensional current feature. B3 differs from B2 only by enabling the
train-only radial regularizer.

KDA implements channel-wise physical-time forgetting, a true rank-one
delta-error write, and fixed-size matrix memory. Memory is reset for every
trajectory, updated exactly once per accepted macro state, held fixed through
all four RK stages, retained numerically across truncated-BPTT boundaries, and
kept in float32 under AMP.

Every training batch is made from continuous windows of one Reynolds trajectory.
The 8000-step matched budget uses K4/K8/K16/K32/K56 for
20%/20%/25%/20%/15% of optimizer steps. Autonomous rollout never uses future
truth as a model input.

The program cannot load a held-out file. It creates validation-only checkpoints;
held-out evaluation requires a separate later workflow and is allowed only after
the B2-versus-B1 validation gate is evaluated exactly as frozen.
