# CenteredSquare Three-Regime Data Construction Record

## Scope

This record describes the construction of the CenteredSquare CN09 graded-mesh CFD dataset, its Hopf-region refinement, and the three overlapping specialist datasets used for independent POD and VTK-derivative ROM assembly.

## CFD Source Data

- Solver: OpenFOAM 13 `icoFoam -parallel`
- Time integration: `CrankNicolson 0.9`
- Mesh: previously validated compatible graded mesh, `Nc = 9400` cells
- Case execution: 3 MPI ranks per case, at most 5 concurrent cases
- Viscosity: `nu = 1 / Re`
- Time interval: `t = 0` to `t = 500`
- Outputs: reconstructed OpenFOAM fields were converted directly to compressed NPZ. `foamToVTK` was not used during the sweep.
- Per case: `times`, `U[Ns,Nc,2]`, `p[Ns,Nc]`, `Re`, `nu`, and a separate full-time probe history were retained.

The formal base sweep used 100 strictly increasing Reynolds numbers over 50--150:

```text
concat(np.linspace(50,80,30,endpoint=False),
       np.linspace(80,100,40,endpoint=False),
       np.linspace(100,150,30,endpoint=True))
```

The Hopf refinement used 31 points:

```text
concat(np.arange(95.00,95.60+eps,0.05),
       np.arange(95.75,98.00+eps,0.25),
       np.arange(98.50,102.00+eps,0.50))
```

After resolving repeated Reynolds numbers in favour of the refined dataset, the combined library contains 120 unique Re cases.

## Regime Labelling And Overlap

Probe-envelope analysis brackets the Hopf onset at approximately `Re_H = 95.312`.

Each Re has one physical source label:

| Source label | Re range |
|---|---:|
| steady | `Re <= 95.30` |
| hopf | `95.35 <= Re <= 102.00` |
| periodic | `Re >= 103.448275862` |

The specialist domains intentionally overlap. `source_regime` remains unique, while `target_specialist` may contain the same Re in adjacent datasets.

| Specialist | Re domain | Cases |
|---|---:|---:|
| Steady | 50--95.40 | 69 |
| Hopf | 94--102 | 34 |
| Periodic | 98.5--150 | 37 |

The Steady/Hopf overlap is 94--95.40; the Hopf/Periodic overlap is 98.5--102. There is no direct Steady/Periodic overlap.

## Split Contract

Splits are by complete Reynolds-number case: no time window from a case is shared across train, validation, or held-out roles. A Re that belongs to overlapping specialist domains keeps the same role in each applicable domain.

| Specialist | Train cases | Validation cases | Held-out cases |
|---|---:|---:|---:|
| Steady | 60 | 5 | 4 |
| Hopf | 23 | 6 | 5 |
| Periodic | 27 | 6 | 4 |

Validation Re:

```text
Steady:   55, 75, 90, 94.5, 95.25
Hopf:     94.5, 95.25, 95.5, 97.5, 99, 101.5
Periodic: 99, 101.5, 110.344827586, 125.862068966, 141.379310345, 150
```

Held-out Re were frozen in `split_contract_resolved.json` and are not used to fit POD means, bases, spectra, ranks, or ROM tensors.

## POD Construction

Each specialist POD is fitted only on its train cases.

1. For every pressure snapshot, subtract the volume-weighted pressure mean.
2. Compute one train-only regime mean for velocity and pressure.
3. Apply volume-weighted L2 POD using `sqrt(cellVolume)` weights; velocity repeats the weight for both components.
4. Determine each field's own 99% and 99.9% cumulative-energy ranks.
5. Retain only the modes needed through the 99.9% rank, then project validation cases using the frozen basis.

| Specialist | Velocity r99/r999 | Pressure r99/r999 |
|---|---:|---:|
| Steady | 3 / 5 | 2 / 4 |
| Hopf | 5 / 11 | 5 / 11 |
| Periodic | 13 / 28 | 12 / 26 |

All POD arrays are finite. The maximum volume-weighted Gram-matrix deviation from identity is approximately `2.2e-9` or smaller.

## ROM Construction

For every specialist and both energy ranks, semi-intrusive velocity Galerkin and pressure Poisson-surrogate tensors were built from the matching local POD basis.

- Derivative backend: PyVista/VTK `compute_derivative()` on the retained internal-field reference VTK.
- VTK/POD cell-center alignment tolerance: `5e-6`.
- Reynolds-dependent viscous operators use `nu = 1/Re` for the train Re list.
- Pressure consistency residual `||L H_tilde - H_p|| / ||H_p||` is approximately `1e-15`.
- All tensor arrays are finite; every pressure operator `L` has full rank for its retained pressure rank.

No global POD basis, old ROM tensor, differently ordered mesh, or incompatible VTK derivative grid was reused.

## Artifact Layout

VM result root:

```text
/home/ray/Desktop/centeredSquare/three_regime_overlap_v1
```

Each specialist contains `cases_npz/` train-case hard links, `mesh/`, `reference_vtk/`, `manifest/`, `pod/`, rank99/rank999 ROM outputs, validation projections, and a run summary. The root includes the split configuration, resolved split contract, duplicate audit, and `final_artifact_inventory.json`.

The original raw snapshots are intentionally not part of the cluster result upload. The uploaded result bundle contains 76 POD/ROM/configuration/report files and the reference VTK, verified file-by-file using SHA-256.
