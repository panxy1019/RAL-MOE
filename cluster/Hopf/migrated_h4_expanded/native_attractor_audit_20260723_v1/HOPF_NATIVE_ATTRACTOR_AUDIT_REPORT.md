# Frozen Hopf specialist native-domain attractor audit

Date: 2026-07-23  
Status: complete, read-only  
Checkpoint: `best_validation.pt` at optimizer step 7200  
Checkpoint SHA256: `02148741ed8bc9fbec88709f69b10492e263514b2edcee76652485b399962235`

## Decision

The frozen expanded-H4 Hopf specialist is **numerically stable on its own
native train/validation/held-out domain, but it is not globally
attractor-preserving under the preregistered strict K56 criterion**.

This distinction is important:

- Native K56 finite rollout and zero divergence pass at every audited Reynolds
  number.
- Native K128 finite rollout and zero divergence pass at every Reynolds number
  for which a K128 window exists.
- Strict K56 attractor preservation passes only 3/29 train Reynolds numbers,
  0/2 validation Reynolds numbers, and 0/3 held-out Reynolds numbers.
- The dominant failures are fluctuation amplitude and normalized orbit shape,
  not NaN/Inf, numerical divergence, total-field energy drift, or pressure
  instability.

Therefore the Hopf specialist may be described as a finite and stable native
Hopf-chart rollout model. It must not be described as having demonstrated
uniform attractor preservation across its native domain.

## Evaluation contract

- The checkpoint, POD assets, scaler/statistics contract, Galerkin terms,
  pressure closure, feature builder, and native integrator were left frozen.
- No optimizer or training code was invoked.
- Each Reynolds number used up to five evenly spaced legal K56 windows,
  initialized from the exact native three-step history.
- Rollouts were state-autonomous after initialization and did not consume
  future truth.
- K128 used one legal window per Reynolds number when the stored trajectory was
  long enough.
- Train results are in-sample diagnostics.
- Validation was used to select the checkpoint and is development evidence,
  not independent generalization evidence.
- Held-out results are post-freeze generalization evidence. This run did not
  tune any threshold, checkpoint, or model component.

The strict K56 conjunction requires:

- finite fraction 1.0 and zero divergent windows;
- velocity and pressure physical relative L2 no greater than 5%;
- RMS and peak-to-peak fluctuation-amplitude errors no greater than 10%;
- frequency relative error no greater than 5%;
- absolute terminal phase drift no greater than 0.25 cycle;
- normalized orbit distance no greater than 0.10;
- velocity and pressure energy drift no greater than 10%;
- no false growth.

## Split summary

| Split | Re count | Strict K56 preserved | K56 all finite / zero divergence | K128 available | K128 all finite / zero divergence |
|---|---:|---:|---:|---:|---:|
| Train | 29 | 3 (10.3%) | 29 / 29 | 23 | 23 / 23 |
| Validation | 2 | 0 (0.0%) | 2 / 2 | 1 | 1 / 1 |
| Held-out | 3 | 0 (0.0%) | 3 / 3 | 3 | 3 / 3 |

The three train Reynolds numbers satisfying the full conjunction are
49.300000, 49.599998, and 50.000000.

## Failure-mode counts at K56

Counts may overlap because a Reynolds number can violate more than one
threshold.

| Split | RMS amplitude | Peak-to-peak amplitude | Frequency | Phase drift | Orbit distance | Physical u/p | Energy u/p | False growth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train (29) | 22 | 20 | 5 | 1 | 18 | 0 / 0 | 0 / 0 | 0 |
| Validation (2) | 2 | 2 | 0 | 0 | 1 | 0 / 0 | 0 / 0 | 0 |
| Held-out (3) | 2 | 3 | 1 | 0 | 2 | 0 / 0 | 0 / 0 | 0 |

## Validation and held-out per-Re results

Errors are dimensionless relative errors. `phase` is terminal drift in cycles.

| Split | Re | u field | p field | RMS amp | P2P amp | Frequency | Phase | Orbit | Strict pass | Failed attractor clauses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Validation | 46.700001 | 0.000194 | 0.003425 | 2.590 | 29.52 | 0.0130 | 0.0678 | 1.548 | No | RMS, P2P, orbit |
| Validation | 56.543247 | 0.001125 | 0.004604 | 0.6005 | 0.7304 | 0.00614 | 0.0475 | 0.0428 | No | RMS, P2P |
| Held-out | 47.081356 | 0.000032 | 0.000355 | 0.00537 | 0.2481 | 0.1774 | 0.00389 | 0.0512 | No | P2P, frequency |
| Held-out | 49.022354 | 0.000288 | 0.004321 | 0.7397 | 3.781 | 0.0111 | -0.0665 | 0.4689 | No | RMS, P2P, orbit |
| Held-out | 51.786449 | 0.000061 | 0.000414 | 0.2398 | 0.3408 | 0.0118 | -0.0900 | 0.2572 | No | RMS, P2P, orbit |

The small total physical-field errors do not contradict the failed attractor
metrics. The reconstructed total field is dominated by the mean component,
whereas the attractor diagnostics measure the centered and normalized
fluctuation plane. A prediction can be close in total-field norm while
attenuating or distorting a relatively small oscillatory component.

## Consequence for the frozen unified system

The admissibility gate should retain two separate concepts:

1. **Rollout admissibility:** finite, non-divergent, contract-compliant,
   validation-supported native rollout.
2. **Attractor fidelity:** amplitude, frequency, phase, orbit, and fluctuation
   dynamics satisfy their scientific thresholds.

The present Hopf checkpoint passes the first concept on its native split but
does not pass the second uniformly. This does not change the already established
H–P fail-closed decision: the H specialist remains inadmissible on P-native
development trajectories. It does require a narrower statement for the H-only
core:

> H-only is the frozen native Hopf-chart baseline and is numerically stable,
> but its cross-Re attractor amplitude/orbit fidelity is incomplete.

Any claim that the complete frozen system preserves the Hopf attractor should
be reported per Reynolds number and per metric, rather than as a global binary
claim.

## Artifacts

- Structured result: `HOPF_NATIVE_ATTRACTOR_AUDIT.json`
- Audit implementation: `audit_hopf_native_attractor.py`
- Script SHA256:
  `2a26492fdd40f0c97f7f973aaea336bb430a795f898d3c63db6265b614415d56`
- JSON SHA256:
  `de6211930abd0eff9815cb45eba6b39b396bb64c2e45d7c3024b8820645499d9`

The structured JSON contains every train, validation, and held-out Reynolds
number, all K1/K4/K8/K16/K48/K56 metrics, the optional K128 diagnostic, and the
exact threshold contract.
