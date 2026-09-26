# H–P one-sided recovery report

## Decision

`FAIL_CLOSED_NO_HP_TRAINING`

The missing Periodic train/validation modal asset was repaired without reading
heldout physical fields or either mixed coefficient member.  However, the
formal `G_HP_native_FULL_K56` gate failed: the frozen Hopf specialist does not
support K56 rollout over the Periodic development domain.  Therefore no H–P
native cache was created and none of T2-C, RiskPrediction, or LookAhead was
trained.

## Asset-sealing repair

- Raw source: 63 independent per-Re
  `Re_*_uvp_pointData.npz` trajectories on the virtual machine.
- Frozen projection source: the exact Periodic checkpoint POD contract from the
  migrated project root, not the older all-Re/per-Re-mean POD.
- Projection contract:
  `fit_split=train`,
  `centering=single_train_only_regime_mean`,
  `pressure_gauge=subtract_area_mean_per_snapshot`.
- Explicitly excluded members: `coeff_uv`, `coeff_p`.
- Development bundle:
  8,539 train snapshots over 53 complete Re trajectories and 967 validation
  snapshots over 6 complete Re trajectories.
- Heldout raw files opened: none.
- POD orthogonality maximum absolute error:
  velocity `1.66893e-6`, pressure `8.34465e-7`.
- Validation K32 projection error averaged over Re:
  velocity field `0.00301084`, pressure field `0.0136647`.
- Bundle SHA256:
  train `38d1788b8edcf2018654cc8599e317791143fb65b7eb714bc7b535efa3e1ba20`;
  validation `3ccba21ca046770c48a1a5b9651d316454a4d42fb5030530e4aed6cbd4f61553`.
- No-coefficient projection contract SHA256:
  `663fe53aaed012ca4ec8ba9fd851136b1d6a596c9ea4de39896c829d5918aa76`.

An initial projection attempt using the older virtual-machine POD was aborted
before producing a usable bundle when its asset identity was found not to match
the frozen checkpoint.  The aborted directory is retained as audit evidence.

## Native preflight

The K16 smoke gate passed on three low-Re train windows and one validation
window: both Periodic and Hopf were finite, with a common clock, common
three-state physical history, Periodic indexed phase, and phase-free Hopf.

That result did not authorize training.  The formal gate was strengthened to
one complete K56 window for every development Re:

| Split | Re count | Periodic finite | Periodic divergent | Hopf finite | Hopf divergent |
|---|---:|---:|---:|---:|---:|
| train | 53 | 53 | 0 | 5 | 48 |
| validation | 6 | 6 | 0 | 0 | 6 |
| total | 59 | 59 | 0 | 5 | 54 |

The only jointly valid Re values were
`60.307747`, `61.755955`, `63.499817`, `65.259827`, and `68.649712`,
all in train.  The lowest validation trajectory, `Re=66.970116`, was already
non-finite for Hopf at K56.  Consequently there is no validation-supported H–P
band on which a gate or route checkpoint can be selected.  Restricting training
to the five successful train Re values would create an unvalidated method and
is not allowed.

The Periodic specialist remained finite and below the 10× divergence rule on
all 59 development trajectories.  Failure is localized to Hopf extrapolation,
not to time alignment, indexed phase, pressure gauge, POD projection, or the
Periodic wrapper.

## Test seal and produced artifacts

- Heldout Periodic modal/field data were not loaded.
- No heldout cache exists.
- No H–P route preregistration was activated after the failed formal gate.
- No H–P native output cache was created.
- No optimizer was started.
- No H–P best/last checkpoint exists.
- The already frozen S–H directory was not modified.

Formal gate SHA256:
`ed90e8a090a6b5dca486f986f76500fd530a784c574d0a9991ca32c699dbfd02`.

Failure analysis SHA256:
`08412f3c57e0ccbfb2564f6f5e0648d8a433b7e284decc2990b1dd5ae8a96e4e`.

## Scientifically valid next routes

1. Keep H–P disabled: use native-time hard routing with P-only throughout the
   Periodic domain.  This is the only route supported by the current frozen
   validation contract.
2. If H–P blending remains necessary, warm-start the Hopf checkpoint and
   re-certify it on a common H–P overlap set with independent train and
   validation Re trajectories near `Re≈60–70`.  Preserve the existing POD,
   scaler, history, pressure, and native-time contracts where possible; do not
   randomly initialize a new specialist.  Only after K56 validation support is
   restored should `G_HP_native` and the three route experiments be retried.

The K16 smoke result must not be cited as long-horizon H–P support.
