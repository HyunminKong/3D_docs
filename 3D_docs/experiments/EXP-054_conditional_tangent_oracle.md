# EXP-054 — Conditional Tangent Oracle Premise

Status: Registered; no-fit train-only experiment
Purpose: Determine whether spatially conditioned weighting of the same eight
TTT3R code axes can resolve metric conflict before fitting a conditioner

## Question

Does EXP-053 fail because one global tangent metric is shared across different
geometry-token contexts, and does token-local axis selection contain realized
absolute-3D benefit that cannot be explained by a scene-global mask or a
spatial shuffle?

## Protocol

Reuse exactly the 16 already exposed EXP-052 train anchors: four anchors in
each of `pumpkin`, `heads`, `chess`, and `stairs`. Every anchor is a four-frame
reset segment in official TTT3R mode. Validation and both terminal partitions
remain closed. No parameter is updated and no checkpoint is written.

At zero code, compute the deployed online-consistency gradient `g_on` and the
offline median-scale-aligned relative-3D gradient `g_metric` in the identical
8-D per-token code coordinates. Compare four one-step policies at the same
normalized step size:

1. `global`: all token axes active;
2. `axis_oracle`: axis `j` is active everywhere iff the token-summed product
   `sum_n g_on[n,j] * g_metric[n,j]` is positive;
3. `token_axis_oracle`: coordinate `(n,j)` is active iff
   `g_on[n,j] * g_metric[n,j]` is positive;
4. `spatial_shuffle`: deterministically permute the token dimension of the
   token-axis oracle mask while preserving its per-axis activation count.

The masks are offline algebraic capacity diagnostics. They use RGB-D labels
and are not deployable routing inputs. For each mask, the code gradient and
decoded code are both scaled, matching a conditional tangent metric rather
than post-hoc gradient replacement.

The implementation also instantiates the proposed zero-initialized
`768 -> 8` conditioner and verifies that it reproduces the global path exactly
and has a finite nonzero exact meta-gradient. It is not optimized.

## Registered gates

- exact 16-anchor/four-scene coverage, finite values, and zero-code parity at
  most `1e-5`;
- the zero-initialized conditional module reproduces the global adapted
  prediction within `1e-5`;
- token-axis oracle online loss descends in every scene;
- token-axis oracle metric gain is positive in every scene;
- token-axis oracle beats the global policy and spatial shuffle in every
  scene, with positive anchor-bootstrap 95% intervals for both paired gains;
- token-axis oracle harm is at most 25%;
- token-axis oracle mean gain is greater than the scene-global axis oracle;
- the conditioner exact meta-gradient is finite and nonzero at every anchor.

Passing supports H15's capacity premise and authorizes one fixed train-only
conditioner fit. It does not authorize validation. Failure rejects this minimal
diagonal conditional tangent before training and returns the project to a
method-level decision.

## Artifacts

- Config: `configs/EXP-054_conditional_tangent_oracle_v10.yaml`
- Runner: `revisit3d/scripts/evaluate_exp054_conditional_tangent_oracle.py`
- Result: `revisit3d/results/EXP-054/conditional_tangent_oracle_v10.json`
