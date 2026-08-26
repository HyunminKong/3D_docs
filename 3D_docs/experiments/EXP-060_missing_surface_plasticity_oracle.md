# EXP-060 — Missing-Surface Plasticity Oracle

Status: Completed; registered gate failed

Protocol revision v1.1 changes only one unsupported PyTorch API call after the
v1.0 runner stopped on its first anchor before computing a metric or writing a
result. `torch.flatnonzero(mask)` is replaced by the equivalent supported
`torch.nonzero(mask).flatten()`; method, data, randomness, controls, and gates
are unchanged.

## Question

When current evidence is deliberately removed, does a spatially local TTT code
computed while that same surface was visible add unique geometry value beyond
an equal second current TTT step?

## Protocol

Reuse the exact 16 exposed EXP-057/058 anchors, carrier, central erasure,
evaluation support, generic 8-D basis, normalized step, and relative 3D EPE.
Process the first three observations cleanly with official TTT3R recurrence.
At the third clean observation, compute one local-code step against the second
observation's predicted canonical patch points and store only that 8-D code and
its predicted canonical patch locations.

At the erased fourth observation, compute one current local-code step. Transport
the stored source code by nearest predicted canonical 3D patch, retain it only
at target patches whose centers lie inside the supplied erasure mask, and add
it to the one-step current code. No source RGB, depth, pointmap, pose, feature,
or future observation enters the target readout after transport.

Compare:

1. erased official TTT3R;
2. one and two current local-code steps;
3. one current step plus correctly transported past local code;
4. one current step plus the untransported source-grid code;
5. one current step plus a within-erasure spatial permutation of the identical
   transported code payload;
6. per-pixel best memory/current diagnostic;
7. EXP-058 predicted-surface fusion only as an immutable information upper
   reference, not as an online input.

The source and target codes use the same fixed generic basis, loss, norm, and
step size. No parameter, strength, router, threshold, address, or memory bank is
fit. The paired previous frame is an oracle address; passing would establish
representation value, not deployable retrieval.

## Registered gates

- exactly 16 anchors/four scenes, finite values, exact support and erased-base
  reproduction against EXP-058, and no validation/terminal access;
- the source and both current local steps reduce their deployed consistency
  objectives in every anchor;
- transported past code beats second-current TTT, untransported code, and
  spatial shuffle in every scene, with positive paired anchor-bootstrap
  intervals for all three comparisons;
- transported-code reuse harms at most 25% of anchors.

Passing supports H18 at oracle address and authorizes one minimal observable
address test. Failure rejects the present local adaptation-experience object
under missing current evidence and prevents router, bank, or validation work;
the explicit-surface result remains a separate prior-art-colliding fact.

## Artifacts

- Config: `configs/EXP-060_missing_surface_plasticity_oracle_v10.yaml`
- Runner: `revisit3d/scripts/evaluate_exp060_missing_surface_plasticity_oracle.py`
- Result: `revisit3d/results/EXP-060/missing_surface_plasticity_oracle_v10.json`

Result SHA-256:
`346dc5160824f2d641738cc8cac2cd6c913e01bad674817ffa4b155871a5747e`.

## Result

All common guards passed. The 16 anchor identities, evaluation supports, and
erased baselines reproduce EXP-058 exactly. Source adaptation, the first target
step, and the second target step each reduce their deployed consistency loss at
every anchor, so the code-generation mechanism is active and well formed.

The memory premise fails:

- transported past code versus second-current gain is `4.36e-6`, CI
  `[-4.08e-6, 1.41e-5]`;
- the scene mean is negative in `pumpkin` and `stairs`;
- transported code is worse on average than untransported code by `1.79e-6`;
- transported code is worse on average than spatial shuffle by `5.63e-7`;
- both spatial-control intervals cross zero and the all-scene gates fail;
- past-code reuse harms 56.25% of anchors, above the 25% maximum;
- an offline per-pixel best fallback has only `2.03e-5` mean headroom, despite
  EXP-058 explicit surface fusion gaining `0.3940` under the same task.

## Conclusion

H18 is rejected for the fixed generic local code. The failure is not caused by
an inactive online objective, missing evaluation support, or a broken current
step. A past update generated while the surface was visible carries no robust,
spatially specific absolute-geometry value at the erased target: correct
transport cannot be distinguished from grid identity or payload shuffle.

No learned address, bank, visibility head, step-size change, transport change,
basis refit, or validation experiment is authorized for this object. The large
explicit-surface result and the failed compact-code result must remain separate;
the former cannot be presented as evidence for adaptation-experience memory.
