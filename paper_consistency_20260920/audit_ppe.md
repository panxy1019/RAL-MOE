# PPE provenance audit — 2026-09-20

Read-only remote inspection; no training, installations, or checkpoint changes. After this report was prepared, the parent explicitly authorized a minimal edit to the paper's `appendix/rom_derivation.tex` only; changes are recorded below. Remote root below is `/root/siton-data-ecd30b0dd03a4b6c86d4ad3b98458111/panxy/particalMOE` (SSH port 20381). Existing checkpoint-to-evaluation identities come from `RAL_MOE_ROM_ICLR2027/network_architecture_audit.md:26–38` and its `build/network_architecture_audit_20260912/remote_checkpoints.json`; those historical checkpoint hashes were not recomputed in this limited audit. Asset hashes and numerical identities below were recomputed live.

## Conclusion

The Circular Periodic pressure base is a **projected weak-form polynomial pressure map**, with a precomputed pseudoinverse and omitted boundary terms. Calling its filename `pressure_poisson_surrogate` does not make it a regression fit. The actual NPZ metadata, accompanying construction report, retained untransformed tensors, numerical identities, and archived runtime hash all agree.

Circular Hopf has a similarly supported projected origin, but its rank-32 runtime tensors are **truncated from an already inverted rank-80 map**. One must not silently equate this with inversion of the freshly truncated rank-32 pressure stiffness matrix.

Square builders likewise assemble projected pressure tensors. The Steady and Periodic packaging additionally fits/interpolates existing operator coefficients in the two-column basis `[1, 1/Re]`; this is not fitting a polynomial pressure response to pressure snapshots. Exact upstream generation provenance is less complete for Square Periodic/Steady than for Circular Periodic. Square Hopf has an exact source-to-runtime tensor copy, but its final data-only deployment path bypasses the PP pressure base.

## Actual assets, hashes, and checks

| Asset under remote root | Live SHA-256 | Evidence |
|---|---|---|
| `periodic_specialist_r32/assets/pressure_poisson_surrogate_periodic.npz` | `bf64858428733e90bda185b504ab98ef7fc96f5e0f3bb3d6d788d758e2b15a09` | Exact match to `MIGRATION_RUNTIME.sha256:6` and `TRANSFER_MANIFEST.sha256:13`. `metadata_json` explicitly records weak-form projection and boundary omission. |
| `Hopf/artifacts/hopf/pressure_poisson_surrogate_hopf.npz` | `a6646aa3eb3814d8648293ee3cd798a6abe268bf46a79ec0c0b8b17b30aaf06d` | Metadata records same projected construction, ranks 80/80, pseudoinverse rcond 1e-10. |
| `Hopf/migrated_h4_expanded/assets_r32/expanded_h4_trainonly_pressure_r32.npz` | `9df6f17413ff884a04b5bbba9e2b791563bde412b5d17649f9badda4e151e44c` | Every runtime `c_tilde_all`, `A_tilde_all`, `H_tilde` equals the corresponding source slice cast to float32, checked with `np.array_equal`. |
| `centered_square_periodic_v1/assets/pressure_poisson_surrogate_periodic.npz` | `2b69f2d1a137e7868e4bacb58df932afabaa0ac0b22547b9421e63aab86165da` | Exact match to `centered_square_periodic_v1/RUNTIME_INPUTS.sha256`. Source packer is included in that manifest. |
| `centeredsquare_steady_specialist_v1/source_artifacts/steady/pressure_poisson_surrogate_steady.npz` | `13c9c1d01e46a65e90bc39f6f77130d551c2876c095eea6e54e60aeb6b8f18d6` | Exact match to both `ASSET_AUDIT.json` and `runs/s2b_rank999/startup_manifest.json` pressure_rom path/hash. |
| `CenteredSquare_Hopf_H4_20260728/data/source_subset/pod/rom/rank999_ru11_rp11/pressure_poisson_surrogate_tensors_Re50_150_N100_rank999_ru11_rp11.npz` | `e5be152fd4bd32749abf88b93b950427d3be38f14779f54089f488d0f0f5abe8` | Retains L, L_pinv, H_p and transformed tensors. Source manifest identifies VTK cell derivative backend. |
| `CenteredSquare_Hopf_H4_20260728/assets_r11/centeredsquare_hopf_trainonly_pressure_r11.npz` | `bb7206d488dbcd0e5944f5516248824d125c528e7fada04534bc20e74eb1cc5d` | c/A/H arrays are all exact float32 copies of preceding rank-11 source arrays. |

Numerical tests used `einsum('pm,mij->pij', L_pinv, H_p)` and measured relative Frobenius error against H_tilde:

- Circular Periodic: `2.0828021819178786e-16`; first c_tilde = L_pinv c_p relative error `2.7409953757050665e-16`.
- Circular Hopf rank80 source: `1.4323692580188222e-15`; first c identity `3.860629023869356e-15`.
- Square Steady runtime asset: `0.0`.
- Square Hopf rank11 source: `6.804088164094154e-17`.
- Square Periodic runtime asset does not retain H_p, so the raw-to-transformed numerical identity cannot be tested from that file alone.

These identities corroborate algebraic transformation of stored raw tensors. They do not by themselves prove how those raw tensors were generated; construction metadata/reports/code supply that additional evidence.

## Source chain and limits

### Circular Periodic

Checkpoint inventory `remote_checkpoints.json:4918–4920` records the original `/cephfs/shared/V17_IndependentMoEROM_v2/artifacts/periodic_r32_handoff/periodic/pressure_poisson_surrogate_periodic.npz`; migrated asset metadata records the same original directory. Runtime transfer hashes establish the inspected migration file identity.

Local trainer `iclr_expert_routing_analysis/source/periodic_specialist_r32/code/train_periodic_moe.py:4562–4565` loads `args.pressure_surrogate_path`; `:2389–2431` builds and evaluates the static map. The actual file stores L, L_pinv, singular values, numerical rank, H_p, H_tilde, per-Re raw c_p/A_p, and per-Re transformed c_tilde/A_tilde.

`periodic_specialist_r32/assets/provenance/pressure_poisson_surrogate_periodic.md` identifies PyVista `UnstructuredGrid.compute_derivative`, point-area quadrature and weak-form projection. Its matrix convention is negative stiffness:

`L_mk = - integral grad(psi_m) dot grad(psi_k)`;
`c_p = integral grad(psi) dot [ (ubar dot grad) ubar + grad(pbar) ]`;
`A_p = integral grad(psi) dot [ (ubar dot grad) phi + (phi dot grad) ubar ]`;
`H_p = integral grad(psi) dot [ (phi_j dot grad) phi_k ]`.

It reports rank 32/32, rcond 1e-10, condition estimate 72.53541. The metadata explicitly says `boundary terms are neglected in the projected weak form`. Do not describe this as the exact finite-volume pressure solve or exact boundary closure.

### Circular Hopf

Checkpoint inventory `:112` records `assets_r32/expanded_h4_trainonly_pressure_r32.npz`. Local `iclr_expert_routing_analysis/source/Hopf/migrated_h4_expanded/code/build_expanded_h4_assets.py:100–111` selects training Re labels, truncates c/A/H after upstream inversion, and casts float32. Live bytewise array checks confirm this exact transformation from the rank80 source. The source construction MD/metadata says projected weak form, with condition estimate approximately 61633.94 at rank80. Thus a universal statement that every deployed pressure map is `(K_r^p)^dagger` formed directly in its deployed rank is not proved and is potentially incorrect for this chart.

### Square

Local `remote_scripts/build_centered_square_rom_tensors.py:421–444` is a concrete projected operator builder: L is negative weighted gradient Gram, L_pinv is `np.linalg.pinv(L)`, and c/A/H are contracted from means, modes, derivatives and cell weights. Its c/A include `-nu*cp_lap` and `-nu*ap_lap`; these should not be erased when describing this particular implementation. This builder file alone is candidate-code evidence, not proof it generated every final asset.

Square Hopf strengthens that link: its actual source `ROM_PhysicsGeneralizable_manifest.json` identifies `pyvista_vtk_compute_derivative_cell_data`, 9400 cells, matching 11/11 rank configuration, and raw/transformed tensor report. Live source-to-runtime arrays match exactly after float32 conversion. However, the final `make_ops` and evaluator path are data-only as already documented in `network_architecture_audit.md:131–136`; possessing PP tensors is not proof they affect final pressure predictions.

Square Periodic local `fusion_source_snapshot/centered_square_periodic_v1/code/prepare_square_periodic_assets.py:52–57,256–269` reads pre-existing `pressure_poisson_surrogate_tensors_Re50_150_N100_rank999_ru28_rp26.npz`, copies H_tilde/L/L_pinv, and applies least squares in `[1,1/Re]` to c_tilde_all/A_tilde_all before packaging. Runtime hash is verified; the original rank28/26 upstream NPZ was not found within the bounded remote search, so full source-array equality and original projected build identity remain unresolved.

Square Steady local `prepare_centeredsquare_steady_assets.py:344–371` likewise fits existing c/A arrays in `[1,1/Re]`, copies H_p/H_tilde/L/L_pinv, and packages per-Re values. Actual runtime asset matches the training startup hash and confirms the H pseudoinverse identity. Original c_p/A_p arrays are absent from runtime package, so the complete c/A upstream identity remains unresolved.

## Defensible LaTeX wording

```latex
For the archived circular-cylinder periodic specialist, the static pressure
base is a projected weak-form polynomial map. The offline asset stores
\(L\), \(L^\dagger\), and the projected tensors, with
\[
  \widetilde c=L^\dagger c^p,\qquad
  \widetilde A=L^\dagger A^p,\qquad
  \widetilde H=L^\dagger H^p,
  \qquad b_{\mathrm{base}}(a)=\widetilde c+\widetilde A a
       +\widetilde H(a,a).
\]
Here \(L_{mk}=-\langle\nabla\psi_m,\nabla\psi_k\rangle\)
follows the archived sign convention. Boundary integrals are omitted in
this approximate weak-form construction. The polynomial is therefore a
projected pressure surrogate; it is not a pressure-target regression or
an online solution of the full-order finite-volume pressure equation.
```

For appendix chart-specific implementation details:

```latex
The circular Hopf runtime map retains the first 32 velocity and pressure
coordinates of a pre-inverted rank-80 projected pressure map.
The square steady and periodic asset adapters represent precomputed
constant and linear pressure-map coefficients as affine functions of
\(1/\mathrm{Re}\), using least squares on the operator coefficients.
This parameter interpolation is distinct from fitting pressure snapshots.
The deployed square Hopf specialist instead uses its learned pressure
output directly, so its stored projected pressure tensors do not define
the final pressure-output path.
```

Do not promote the verified Circular Periodic construction into a universal exact-PPE guarantee. No evidence found here supports replacing all archived pressure bases with the description “quadratic regression fitted to pressure targets.” Full upstream Square S/P reconstruction and all paper-row-to-checkpoint identities remain explicit gaps.

## Authorized paper change

Only `C:/Users/panxy1019/Desktop/STABLEMOE/PMD_Galerkin_Pan_ICLR (3)/appendix/rom_derivation.tex` was edited. Existing Circular Periodic derivation/sign convention/omitted boundary terms were preserved. Added the live hash/identity verification, a chart-specific paragraph covering Circular Hopf truncation, Square coefficient interpolation and its provenance limits, and the Square Hopf data-only deployment exception. Qualified introductory/final universal wording so it no longer states every deployed specialist uses both operators. No other paper file was touched by this audit agent. Full document compilation is left to the parent's integrated validation.
