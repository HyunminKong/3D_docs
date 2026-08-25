# Current Research State

Last updated: 2026-08-25 (EXP-031 terminal protocol registered)

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

EXP-024 then improved SILog/EPE and oracle reuse but failed current aligned
AbsRel, so its one-loss checkpoint was rejected. One terminal EXP-025 objective
is authorized: a fixed equal mean of aligned log-depth and aligned relative-depth
residuals. The online method remains one loss/one step; failure ends atom
objective development for this paper.

EXP-025 failed terminally: it improved AbsRel but significantly worsened
SILog/EPE, the inverse of EXP-024. No atom checkpoint was accepted. Scalar
atom-objective development is stopped. The empirical reusable-memory oracle
remains positive, but learning a Pareto-healthy current direction is unresolved.

The constrained/Pareto multi-objective branch is now explicitly approved. The
generic idea of gradient consensus is not novel—CoCo-MT-TTA, GraTa, PCGrad,
CAGrad, Aligned-MTL, and ConFIG directly constrain that claim—so it is treated
as an offline geometry-health mechanism rather than the standalone novelty.

EXP-026 passed every zero-fit premise gate on 675 anchor/episode evaluations.
The learned anchors exhibited 27–35% metric-gradient conflict and 23–33% raw
equal-average endpoint sacrifice, while the parameter-free unit-normalized
bisector was non-degenerate common descent in every case.

EXP-027 failed without a checkpoint. It improved SILog/EPE and retained
significant three-metric reuse headroom, but mean AbsRel worsened slightly and
AdamW preserved common descent on only 72.54% of component-balanced steps.
This localizes the remaining variable to offline optimizer rotation rather than
another loss or architecture.

EXP-028 passed every OOF gate. Current one-step TTT improved SILog, aligned
AbsRel, and 3D EPE together; oracle reuse further improved all three with
positive intervals. The parameter-free safeguard intervened on 29.28% of
steps, achieved 100% realized common descent, and changes no inference work.
The accepted checkpoint hash is
`3ebf194f3a28876014e46d1d3bbdbcd1422cfb8ebdba48f3d16635520ca787ae`.

EXP-029 passed every address gate on 13,631 causal pairs. Its single factorized
Ridge obtained positive held-location association everywhere, +0.00320 metric
utility, 11.44% harm, and positive component intervals over matched random and
appearance. The frozen artifact hash is
`d8b81fff36d5cb5635c194a63b422edf700c0683b7f7eb2d477be67091430984`.

Before opening EXP-021, EXP-030 must perform a no-fit full-system OOF audit of
the selected candidates on SILog, aligned AbsRel, and 3D EPE directly. No
parameter, threshold, or policy may change.

EXP-030 passed that frozen audit on 217 targets/25 components. Full metric
memory beat current-only, matched random, and appearance on SILog, aligned
AbsRel, and 3D EPE, with positive intervals for every comparison. Development
model selection is closed. EXP-031 is authorized to open the untouched
EXP-021 benchmark once using the immutable checkpoint/address/protocol.

EXP-031 is now registered before access: 96 selected official-test scenes, 214
directional episodes, 29 components, deterministic reservoir capacity 64 per
location, write-after-predict, top-1 semantic-zero metric address, and frozen
random/appearance/current controls. Terminal output cannot change any method or
gate and will be reported whether it passes or fails.
