# EXP-064 — Chronological Geometry-Consensus Direction

## Status

Preregistered; not yet run.

## Question

Does the single chronological prediction have a geometry-healthy local
direction toward the consensus of identical-evidence recurrent paths?

## Protocol

- Reuse exactly the 16 EXP-062/063 train contexts and six fixed-anchor orders.
- Divide each query self-view pointmap by its median predicted depth on the
  common valid mask, then take the arithmetic six-order point consensus.
- For every order, evaluate the fixed interpolation
  `P_0.1 = P + 0.1 * (P_consensus - P)`. The coefficient is a preregistered
  trust-region diagnostic and is not tuned or proposed as an inference value.
- Primary deployable-path comparison: chronological base versus chronological
  `P_0.1`, independently aligned by target median depth.
- Spatial control: deterministically permute the chronological consensus
  residual across valid pixels, preserving its vector payload and norm, then
  apply the same 0.1 coefficient.
- Report the mean result over all six orders as secondary evidence. RGB-D is an
  offline metric only; consensus and control use predictions alone.

## Frozen success gate

All must hold:

1. EXP-063 six base predictions reproduce within `1e-5` EPE;
2. chronological consensus gain is positive in every scene with a positive
   stratified context-bootstrap 95% lower bound;
3. chronological consensus beats spatial shuffle in every scene with a positive
   stratified context-bootstrap lower bound;
4. consensus harms at most 25% of chronological contexts;
5. the mean six-order consensus gain is positive in every scene.

Failure closes H20 as a paper-method candidate. Success authorizes only a fresh
trainability/architecture decision; validation remains closed.

## Artifacts

- Config: `configs/EXP-064_geometry_consensus_direction_v10.yaml`
- Result: `revisit3d/results/EXP-064/geometry_consensus_direction_v10.json`
