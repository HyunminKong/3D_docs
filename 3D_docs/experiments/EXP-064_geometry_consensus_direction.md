# EXP-064 — Chronological Geometry-Consensus Direction

## Status

Completed; gate failed. No model was fit and H20 is closed as a paper-method
candidate.

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

## Result

The immutable EXP-063 base predictions reproduced to `5.55e-17` maximum EPE
difference. The fixed 10% consensus direction changed mean chronological EPE
from `0.0799140` to `0.0798371`, a gain of only `7.69e-5`. Its stratified
context-bootstrap 95% interval was `[-1.12e-4, 2.71e-4]`, so the local benefit
is uncertain.

The chronological gain was positive in `chess`, `heads`, and `pumpkin`, but
negative in `stairs` (`-2.30e-4`). It harmed 7/16 contexts (`43.75%`), exceeding
the frozen 25% maximum. More importantly, the consensus direction beat its
spatially shuffled residual by only `1.01e-4`; that control interval
`[-6.65e-5, 2.79e-4]` crosses zero and the effect is again negative in
`stairs`.

As secondary evidence, applying the same direction to all six order paths
improved mean-order EPE by `1.44e-4` and was positive in all four scenes. This
preserves EXP-063's descriptive observation that an output ensemble has weak
denoising value. It does not establish a safe or deployable correction of the
single chronological path.

Frozen gates:

- base reproduction: pass;
- chronological gain in every scene: fail;
- positive chronological bootstrap lower bound: fail;
- spatial-control gain in every scene: fail;
- positive spatial-control bootstrap lower bound: fail;
- chronological harm at most 25%: fail;
- all-order mean gain in every scene: pass.

## Interpretation

Order sensitivity is a genuine recurrent-geometry failure mode, but the
scale-quotiented output consensus does not define a reliably healthy local
direction for the deployed chronological prediction. The similar effect of a
spatially shuffled residual shows that the small aggregate gain cannot be
attributed specifically to correcting the measured path disagreement.

The preregistered stop rule therefore applies. The project will not tune the
interpolation coefficient, select favorable orders, add a confidence gate, or
fit a decoded-commutator loss on these exposed contexts. H20 remains useful
failure-mode evidence, not an active method claim.

## Verification

```bash
PYTHONPATH=. /home/khm/anaconda3/envs/tttlrm/bin/python \
  revisit3d/experiments/exp064_geometry_consensus_direction.py \
  --config configs/EXP-064_geometry_consensus_direction_v10.yaml
```

- peak allocated GPU memory: `4,951,565,312` bytes;
- fit performed: no;
- validation accessed: no;
- terminal accessed: no;
- EXP-063 source SHA-256:
  `70264df08acda90a5f6731c9136e196c5e9e9439dad75d58fbef1205b4db854a`;
- result SHA-256:
  `dd17e4db842b16397bd752c2510421b4ec93b752ca73ae3d23ec13caf1c23c5a`.

## Artifacts

- Config: `configs/EXP-064_geometry_consensus_direction_v10.yaml`
- Result: `revisit3d/results/EXP-064/geometry_consensus_direction_v10.json`
