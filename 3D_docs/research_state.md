# Current Research State

Last updated: 2026-08-25 (after EXP-020)

## First objective

Produce one compact CVPR-first paper on streaming static 3D revisits. The paper
must establish novelty, absolute geometry benefit, causal/source-safe
generalization, and efficiency without accumulating heads, losses, thresholds,
or unrelated 4D tasks. A broader dissertation architecture is deferred.

## Research question

Can a streaming 3D model store a spatially local **adaptation experience**, move
that correction into a later observation's token frame, and retrieve it by
expected geometric utility rather than by place appearance alone?

## Defensible novelty boundary

Generic TTT memory, fast-weight scene compression, and retrievable gradients
are not novel claims because of direct overlap with tttLRM, ZipMap, TTT3R,
Mem3R, and ReGrad. The defensible conjunction is:

1. a per-token local plasticity code rather than model-wide gradients or scene
   content;
2. explicit cross-view transport of that code;
3. an address supervised by causal future geometry utility rather than place
   identity or RGB similarity;
4. physical-revisit evaluation with same-bank random-address controls.

## Compact frozen candidate evaluated in EXP-020

- frozen VGGT/FastVGGT geometry backbone;
- one 3D-track online loss, one local-code step, `eta=0.0125`;
- 8-D per-token code and visual transport with residual `alpha=0.10`;
- one factorized 64-D Ridge utility address, top `K=5`;
- parameter-free current-geometry agreement reranking with coarse fallback;
- deterministic reservoir capacity `C=64` per official-location stream;
- no fine router, risk head, learned threshold, learned eviction, second TTT
  step, pose adaptation, or dynamic 4D state.

## Evidence that survives

- EXP-009: the earlier source-safe bounded system beat matched random addressing
  on a terminal 22-component self-supervised future-utility test. Reservoir did
  not beat FIFO significantly and no retention-policy superiority is claimed.
- EXP-011: before atom meta-refitting, one absolute 3D frozen-track loss at
  `eta=0.0125` improved SILog, aligned AbsRel, and 3D EPE on 218 train targets
  and a one-shot 103-target/17-component validation.
- EXP-015: the auxiliary-free core atom produced +1.048% OOF oracle reuse
  utility and positive selection headroom on 25 train components.
- EXP-019: one utility-MIPS address plus parameter-free agreement fallback beat
  coarse and matched random proxy utility on train folds.
- EXP-020: the frozen full method improved aligned AbsRel over current-only TTT
  by 0.00165 with component 95% CI `[0.00078, 0.00276]`; proxy utility beat
  matched random by 0.00286 with CI `[0.00120, 0.00459]`.

## Decisive failure

EXP-020 rejected the frozen paper model. On 103 targets/17 unseen components,
current-only TTT from the EXP-015 refit worsened mean SILog and 3D EPE versus no
TTT. Full memory did not obtain a positive primary LiDAR interval over random,
and proxy harm was 33.01% versus the registered 20% maximum. The model improves
its utility proxy and aligned relative depth, but does not support a broad 3D
reconstruction claim.

This localizes the bottleneck to **metric alignment and negative-transfer
calibration after meta-training**, not to lack of proxy retrieval signal. The
EXP-015 checkpoint and EXP-019 address are frozen historical candidates, not an
accepted final architecture.

EXP-022 further localized the failure. The final zero-code head exactly equals
foundation depth; damage is created by the learned one-step direction. Its
future track3D gain is significantly anti-correlated with SILog and 3D EPE gain.
The proxy cannot remain the sole offline utility target.

## Data-use boundary

- The original EXP-005 six-episode test is closed.
- The EXP-009 22-component terminal test is closed.
- EXP-020 validation is now exposed and cannot select a replacement.
- Any changed model must be selected by source-safe OOF on existing train data
  and evaluated once on the newly locked EXP-021 terminal benchmark.
- Query/future observations and LiDAR remain offline labels only and may never
  enter online TTT, memory write, retrieval, or routing.

## Immediate next step

EXP-021 has frozen an untouched terminal benchmark with 214 directional
episodes, 96 scenes, 29 components, and three locations. Its manifest hash is
`22f7ec04caf83ead7efef828dab3231c7919757d13f88509b66ea0257ea95d61`;
no sensor or model output has been opened. EXP-023 passed its no-fit feasibility
gate: its oracle improved SILog, aligned AbsRel, and 3D EPE over both
current TTT and the old proxy oracle on the registered component tests. EXP-024
is therefore authorized to replace the EXP-015 meta objective with the equal
mean of current and best-reuse evaluations of this one metric loss. No other
loss or module is added.
