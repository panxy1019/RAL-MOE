# Model and asset inventory

Generated from this run's strict-load, file-hash, and prediction-invariance evidence.

All three runtime classes are the original `OperatorSpaceMoEROM`, with original `PhysicsAwareExpert` implementations.
Router attributes: `group_router`, `velocity_group_routers`, `pressure_group_routers`.
Expert attributes: `velocity_expert_groups`, `pressure_expert_groups`, `velocity_shared_experts`, `pressure_shared_experts`.
Group selection is shared across the two channels. Expert IDs are zero-based and group-local.

Steady S4 is an independent diagnosis, not the current manuscript Steady reference.

## steady

- Checkpoint: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/checkpoint/frozen_s4_validation_step_1200.pt`
- Checkpoint SHA256: `bcab5661af39e3262103ed887ce9846415467452b453f3dcc14e133f0f8bd80d`
- Original model/expert definitions: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/code/train_v16_4_v2_r32_compat.py`
- Original rollout/evaluation: `train_s2b_3090.Experiment.rollout; finalize_s2b_3090.evaluate_horizon`
- Integrator: Euler; native additive algebraic pressure closure
- Precision: torch.bfloat16
- Window policy: All legal heldout K56 windows; six per Re; native batching
- Strict load: no missing/unexpected keys; checkpoint and model state unchanged.
- Hook-on/off prediction equality: bitwise equality for every evaluated window.

### Held-out windows

| Label | Actual stored Re | Count | Start indices |
|---|---:|---:|---|
| Re_24p630436 | 24.630435943603516 | 6 | [130, 131, 132, 133, 134, 135] |
| Re_32p740068 | 32.74006652832031 | 6 | [386, 387, 388, 389, 390, 391] |
| Re_39p685479 | 39.68547821044922 | 6 | [641, 642, 643, 644, 645, 646] |
| Re_45p142703 | 45.142704010009766 | 6 | [1087, 1088, 1089, 1090, 1091, 1092] |

### Routing configuration

```json
{
  "num_regime_groups": 3,
  "experts_per_group": 6,
  "shared_per_group": 1,
  "top_k": 2,
  "group_top_k": 1,
  "temperature": 0.95,
  "group_temperature": 0.9,
  "gate_floor": 0.0,
  "group_gate_floor": 0.0,
  "shared_scale": 1.0,
  "routed_scale": 0.85
}
```

### Original source and asset fingerprints

| Absolute path | SHA256 |
|---|---|
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/code/train_s2b_3090.py` | `b7f1ffaf8a341501a504f08663035f4e8863a8f28eca1946b57bb871961a08e2` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/code/finalize_s2b_3090.py` | `05ab315e359817c0a03aa66c3f1b79e573e4efcb21fa7a24a2dccd6cf1cec2ce` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/code/train_v16_4_v2_r32_compat.py` | `11099078fe195c1f571acc2ea35e30338ae46c0cc8c9911e9d9b1a5ce65ddb2c` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/source_artifacts/steady/velocity_pod_steady.npz` | `abaab63d11e172a5b40e86b48715f247311eaf324a06094757f2f98870f40948` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/source_artifacts/steady/pressure_pod_steady.npz` | `5056f7470f2439bccc2cfb68448e9a4dd61d89145aa7108ab8672ba6dd2be997` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/source_artifacts/steady/normalization_steady.npz` | `87a5a2326d70adb3e57901168f0ac1b555f23c742ae64a09132031452b51f6bb` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/source_artifacts/steady/velocity_rom_steady.npz` | `780eb5be79311290a2f36837651ae2a97a107c48e1df4cf683b2d344805b325c` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/source_artifacts/steady/pressure_poisson_surrogate_steady.npz` | `8e6c5b689b96acf8f651fc64d8d0f82526e6a5bf5098722e5e1ce51469cc2b1c` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/code/training_s2b_portable.json` | `c5f44462e1a68ad6470461479615a72d21a9c025b1d21fef784b49c903f67455` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/source_artifacts/steady/Global_POD_AreaWeighted_L2/global_pressure_pod_area_weighted_l2.npz` | `5056f7470f2439bccc2cfb68448e9a4dd61d89145aa7108ab8672ba6dd2be997` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/source_artifacts/steady/Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz` | `abaab63d11e172a5b40e86b48715f247311eaf324a06094757f2f98870f40948` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/source_artifacts/steady/Global_POD_AreaWeighted_L2/mesh_l2_point_area_weights.npz` | `d9e7169b622ee02362e7364c41137a5652fc9924f824ccaf78fb89f353a3e1ef` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/source_artifacts/steady/Global_POD_AreaWeighted_L2/pod_area_weighted_l2_metadata.json` | `4674956542882e33c710bde5e19efd0ce74c72d6c68d3118ee2e6103fc94d4e8` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/steady_specialist_v1/source_artifacts/steady/Global_POD_AreaWeighted_L2/pod_snapshot_index.csv` | `b1584b874576161757e827d090702791c0ce4e14c8dc0d02df5ec15f09ad98d5` |

## hopf

- Checkpoint: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/migrated_h4_expanded/final_evaluation/20260722_h4_expanded_final/HopfExpanded34_H4_NormalFormRadial_r32/final.pt`
- Checkpoint SHA256: `02148741ed8bc9fbec88709f69b10492e263514b2edcee76652485b399962235`
- Original model/expert definitions: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/migrated_h4_expanded/periodic_moe_3090/train_periodic_moe.py`
- Original rollout/evaluation: `train_hopf_moe_expanded.rollout_batch/autonomous_step; audit_hopf_native_attractor.rollout_arrays/horizon_metrics`
- Integrator: Native RK4; pressure from stage 1 closure with predicted next velocity
- Precision: float32; no autocast in original native audit
- Window policy: Original native audit: five evenly spaced legal K56 windows per heldout Re
- Strict load: no missing/unexpected keys; checkpoint and model state unchanged.
- Hook-on/off prediction equality: bitwise equality for every evaluated window.

### Held-out windows

| Label | Actual stored Re | Count | Start indices |
|---|---:|---:|---|
| Re_47.081356 | 47.081356048583984 | 5 | [449, 474, 499, 524, 550] |
| Re_49.022354 | 49.02235412597656 | 5 | [2061, 2086, 2111, 2136, 2162] |
| Re_51.786449 | 51.78644943237305 | 5 | [3190, 3215, 3240, 3265, 3291] |

### Routing configuration

```json
{
  "num_regime_groups": 1,
  "experts_per_group": 6,
  "shared_per_group": 1,
  "top_k": 2,
  "group_top_k": 1,
  "temperature": 0.8,
  "group_temperature": 1.0,
  "gate_floor": 0.0,
  "group_gate_floor": 0.0,
  "shared_scale": 1.0,
  "routed_scale": 0.75
}
```

### Original source and asset fingerprints

| Absolute path | SHA256 |
|---|---|
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/migrated_h4_expanded/code/train_hopf_moe_expanded.py` | `f7facb3b6070290e9d6725f09fdd0810f0cf7e1775d35c20563a1b691d2c6975` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/migrated_h4_expanded/code/evaluate_h4_expanded.py` | `3cc9562ae07175f0272fd6ede561cd9df60d4d8b52c20fda7b100e49816f2b34` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/migrated_h4_expanded/native_attractor_audit_20260723_v1/code/audit_hopf_native_attractor.py` | `2a26492fdd40f0c97f7f973aaea336bb430a795f898d3c63db6265b614415d56` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/migrated_h4_expanded/periodic_moe_3090/train_periodic_moe.py` | `500c2179734e6c3a6db6469629d4727637ae4552843b4d3154ca7b281a102314` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/migrated_h4_expanded/native_attractor_audit_20260723_v1/results/HOPF_NATIVE_ATTRACTOR_AUDIT.json` | `de6211930abd0eff9815cb45eba6b39b396bb64c2e45d7c3024b8820645499d9` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/migrated_h4_expanded/assets_r32/expanded_h4_trainval_r32.npz` | `53d05065d518de82432e55b191dcee6c3501e09822dfedbf6162e061151f4d45` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/migrated_h4_expanded/assets_r32/expanded_h4_trainonly_galerkin_r32.npz` | `ebfb05f23c7fe909cb9b92fdea9098d849927bc732652a3881a4f32bbc6e6e26` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/migrated_h4_expanded/assets_r32/expanded_h4_trainonly_pressure_r32.npz` | `9df6f17413ff884a04b5bbba9e2b791563bde412b5d17649f9badda4e151e44c` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/migrated_h4_expanded/trainonly_contract/trainonly_fluctuation_contract.npz` | `2e3aa04095ab0ccceed2a98e3582a7096d1a80993ab486a7896f2dff7afc1e19` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/artifacts/hopf/velocity_pod_hopf.npz` | `2f951192f6eeaa69e7d911aa21552454d1a99a49ab796166b4a13df16016a192` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/artifacts/hopf/pressure_pod_hopf.npz` | `cc128a4361ab2a239165ac22a48407290de7d085b857bbff6726988457cfddf3` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/Hopf/artifacts/hopf/projection_snapshots_velocity_hopf.csv` | `9f54039d14e084bb8da8a4291edc561e9228219626e94f51745743ef7baa6c5a` |

## periodic

- Checkpoint: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/checkpoint/FINAL_PERIODIC_SPECIALIST.pt`
- Checkpoint SHA256: `b2052a746fdaab60f83e65d1b91d038a9dded8255ce74f41e3f6fca4c9e717c5`
- Original model/expert definitions: `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/code/train_periodic_moe.py`
- Original rollout/evaluation: `train_periodic_moe.integrate_autonomous_step_np; evaluate_periodic_r32_portable window policy and weighted geometry`
- Integrator: Native RK4; pressure from stage 1 closure with predicted next velocity
- Precision: float32; original portable evaluator without autocast
- Window policy: Original portable evaluator: K48-capable starts at stride 8; 13 per Re
- Strict load: no missing/unexpected keys; checkpoint and model state unchanged.
- Hook-on/off prediction equality: bitwise equality for every evaluated window.

### Held-out windows

| Label | Actual stored Re | Count | Start indices |
|---|---:|---:|---|
| Re_70p314635 | 70.31463623046875 | 13 | [976, 984, 992, 1000, 1008, 1016, 1024, 1032, 1040, 1048, 1056, 1064, 1072] |
| Re_100p352251 | 100.35224914550781 | 13 | [3874, 3882, 3890, 3898, 3906, 3914, 3922, 3930, 3938, 3946, 3954, 3962, 3970] |
| Re_149p059229 | 149.05923461914062 | 13 | [7258, 7266, 7274, 7282, 7290, 7298, 7306, 7314, 7322, 7330, 7338, 7346, 7354] |
| Re_189p862278 | 189.86227416992188 | 13 | [9513, 9521, 9529, 9537, 9545, 9553, 9561, 9569, 9577, 9585, 9593, 9601, 9609] |

### Routing configuration

```json
{
  "num_regime_groups": 3,
  "experts_per_group": 6,
  "shared_per_group": 1,
  "top_k": 2,
  "group_top_k": 1,
  "temperature": 0.8,
  "group_temperature": 0.9,
  "gate_floor": 0.0,
  "group_gate_floor": 0.0,
  "shared_scale": 1.0,
  "routed_scale": 0.75
}
```

### Original source and asset fingerprints

| Absolute path | SHA256 |
|---|---|
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/code/train_periodic_moe.py` | `500c2179734e6c3a6db6469629d4727637ae4552843b4d3154ca7b281a102314` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/code/evaluate_periodic_r32_portable.py` | `7edd5a61b48391af350d6725f5c3192db2b5bdc50fb0da21aa14252136c3c67f` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/reproduction/best_epoch85_heldout_eval_v2/periodic_r32_multihorizon_evaluation.json` | `43d4dcb69d607ea5805275bf3cd5fad0238d867742a8473901fe12a5cdadd5a1` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/global_velocity_pod_area_weighted_l2.npz` | `a2c12dc3ea25772e7bd4216b37683df9dbcce804063f9900bdaebac3e25dc7a2` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/global_pressure_pod_area_weighted_l2.npz` | `f6e0a23432581c78e9f649d60bb0fceb9087a6a1d0631778bde444338c16239f` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/assets/velocity_rom_periodic.npz` | `8af11eaaab376d7c146fa45884f5d56d8bc4a6e48780cbc57cd8cb3f8f4ce87b` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/assets/pressure_poisson_surrogate_periodic.npz` | `bf64858428733e90bda185b504ab98ef7fc96f5e0f3bb3d6d788d758e2b15a09` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/pod_snapshot_index.csv` | `94525b6d5b4318aa1c6f11762ef2b1021b14d6da3e12f882a48e97c6cc0dd358` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/mesh_l2_point_area_weights.npz` | `d9e7169b622ee02362e7364c41137a5652fc9924f824ccaf78fb89f353a3e1ef` |
| `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE/periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2/pod_area_weighted_l2_metadata.json` | `2ddaf5f4533225f3717b7e805bfee52ec000c23d6183ea404323a3efc59f06f0` |

## Normalization and compatible data views

- Steady: original relocated `training_s2b_portable.json` points to `steady_specialist_v1/source_artifacts/steady`; `Experiment` uses its `Global_POD_AreaWeighted_L2` compatibility view to construct arrays and fits the original training-only normalization. POD/ROM physical assets are listed above.
- Hopf: the original trainer reconstructs normalization from the train view; all ten normalization fields match the checkpoint exactly (maximum absolute difference 0). The full split coefficient/index assets are read only for evaluation, with held-out IDs explicitly selected.
- Periodic: `scalers` are loaded directly from the frozen checkpoint; local POD data view is `periodic_specialist_r32/assets/Global_POD_AreaWeighted_L2`, with native velocity ROM and pressure surrogate paths listed above.
- Initial history is observed; later a/b/history are predicted. Native Re/time metadata are retained. Steady and Periodic phase features are retained as provided by the original data-index contract; their upstream construction is not re-audited here.

## Source authority for historical result comparisons

- Hopf: `Hopf/migrated_h4_expanded/native_attractor_audit_20260723_v1/results/HOPF_NATIVE_ATTRACTOR_AUDIT.json`; checkpoint SHA matches the user-specified final.pt.
- Periodic: `periodic_specialist_r32/reproduction/best_epoch85_heldout_eval_v2/periodic_r32_multihorizon_evaluation.json`.
- Steady identity conflict: copied original `source/SPECIALIST_ABLATION_REPORT.md` and `source/NUMERICAL_EXPERIMENTS_REVISION8_DETAILED_CN.md`.

This inventory does not claim that all historical training/data provenance questions have been re-audited. The current task restores the specified original inference contracts and observes routing without modifying prediction mathematics.
