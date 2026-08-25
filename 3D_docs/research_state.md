# Current Research State

Last updated: 2026-08-25 (EXP-041 localized failure to the raw update coordinate)

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

## Selected compact paper candidate

- frozen VGGT/FastVGGT geometry backbone;
- one 3D-track online loss, one local-code step, `eta=0.0125`;
- 8-D per-token code and visual transport with residual `alpha=0.10`;
- one factorized 64-D Ridge metric-utility address, top-1 with semantic-zero
  fallback;
- deterministic reservoir capacity `C=64` per official-location stream;
- no fine router, risk head, learned threshold, learned eviction, second TTT
  step, pose adaptation, or dynamic 4D state.

## Current decisive evidence

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
- EXP-028: the safeguarded Pareto-trained atom improved current TTT and oracle
  reuse on SILog, aligned AbsRel, and 3D EPE in component OOF.
- EXP-029/030: one metric-utility Ridge address selected broadly useful local
  corrections and the frozen full system beat current, same-bank random, and
  appearance on all three geometry means across 25 development components.
- EXP-031: on untouched official-test scenes, full memory again improved all
  three means over all three controls. Eight of nine metric intervals were
  positive; the ninth, aligned AbsRel versus appearance, had positive mean and
  crossed zero. All registered comparison-family gates passed.

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

## Terminal qualification

EXP-031 did not pass its complete registered gate. Its 190-target coverage
minimum was infeasible: the 214 directional episodes contain 188 unique target
contexts, and one location begins with a target before any causal memory exists.
EXP-032 verified that the evaluator included the maximum 187 eligible targets,
zero were lost to LiDAR validity, and all 29 components were represented. The
model result is therefore qualified positive evidence, not a literal
preregistered pass. No terminal rerun, threshold repair, or model tuning is
permitted.

## Completed selection path

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

EXP-031 then evaluated the immutable system once. On 187 causally eligible
targets and 29 components, full memory improved SILog, aligned AbsRel, and 3D
EPE over current-only, same-bank random, and appearance means. Development and
terminal model selection are closed.

## Completed paper-evidence path

EXP-033 passed its reporting gate. On an A100 and eight 224×224 views, the
method adds about 1.996 ms over 292.328 ms of separate frozen foundation passes
(0.68%); exact bank-64 address search is 0.002 ms CPU. The learned additions
contain 288,386 parameters. Reservoir-64 tensor payload is 38.52 MiB, dominated
by the per-token visual key, and is the main efficiency cost.

The selected model remains frozen. EXP-034 found one local independent dataset:
TUM RGB-D supplies 223 causal contexts and 111 physical-revisit targets across
three indoor sequences. Because 98 targets lie in one sequence, this cannot
replace paper-level multi-component inference.

EXP-035 passed every descriptive gate without TUM fitting. Full memory improved
sequence-balanced SILog by 0.08042, aligned AbsRel by 0.000998, and 3D EPE by
0.002172 m over current-only, and all three means also beat random and
appearance. Current-only improvement was positive within every sequence. Only
three imbalanced sequences are available, and the address accepted 100%, so
this is transfer evidence rather than a general safety result.

The final candidate is frozen. No further architecture, loss, address,
threshold, seed, capacity, or dataset-specific variant is permitted under the
closed-paper protocol. EXP-036 then evaluated official CUT3R and TTT3R modes on
the exact TUM causal event order with query updates disabled.

## Top-tier submission blocker

EXP-036 completed on all 111 TUM targets. TTT3R achieved 15.727 SILog, 0.0781
aligned AbsRel, and 0.2246 m 3D EPE; CUT3R achieved 16.607, 0.0812, and 0.2394
m. The frozen Revisit3D full result is 28.462, 0.2301, and 0.4589 m. Different
resolution/training prevents a controlled capacity claim, but the roughly
1.8–3.0× error gap makes the custom head noncompetitive as a broad CVPR 3D
reconstruction framework.

The memory hypothesis remains supported relative to identical-backbone
current/random/appearance controls. The current candidate is a completed
mechanism proof, not a paper-ready SOTA framework. The scientifically preferred
next branch is to attach the same compact local-code/utility-address principle
to a competitive CUT3R/TTT3R-class state and rebuild development/held-out
evidence from fresh data. That is a material scope reopening and requires an
explicit project decision; it cannot be inferred from the closed-model mandate.

## Active next branch

D112 records the explicit decision to preserve the complete EXP-036 candidate
and reopen only the geometry-carrier scope. The archived candidate remains the
mechanism-proof baseline and must not be repaired on exposed nuScenes/TUM data.

EXP-037 completed the no-fit carrier diagnostic on the already exposed TUM
causal query protocol. This is engineering selection evidence, not a final
paper test. Official FastVGGT improved the archived custom head by
36.2% SILog, 51.4% aligned AbsRel, and 35.3% 3D EPE, but remained 1.43 times
TTT3R AbsRel and 1.32 times TTT3R EPE. It failed the registered carrier gate.
D113 therefore selects a CUT3R/TTT3R-class recurrent carrier.

The immediate next step is an interface audit and minimal integration proof:
identify one spatial state on which a local transported plasticity residual can
act without changing the competitive base prediction at zero residual. The
paper method remains exactly three conceptual operations--write a local
adaptation code, transport it to a revisit, and retrieve it by expected future
geometry utility. New data partitions must be frozen before fitting the
integrated residual or utility address.

EXP-038 v1.1 passed every interface gate. A per-patch 8-D code injected only at
the last decoder-token level has exact zero-code parity with official CUT3R,
receives a finite gradient from one symmetric canonical-3D consistency loss,
and transports exactly under self-correspondence using predicted 3D nearest
neighbors. The shared basis has 6,144 parameters and each 768-token float32 atom
is 24 KiB.

No learned integrated model exists yet. The accepted implementation is a
differentiable scaffold, not evidence of future utility or reconstruction gain.
The next mandatory step is a metadata-first inventory and freeze of genuinely
new development and held-out revisit partitions. Only then may one basis be
trained and the causal sequence `current TTT -> store -> revisit transport ->
future geometry utility` be evaluated.

EXP-039 v1.1 completed that requirement without decoding pixels or accessing a
model. It locked 63 train scenes/982 pairs, 14 validation scenes/213 pairs, and
14 terminal scenes/224 pairs. The roles are scene-disjoint. The terminal
manifest hash is
`49e6c389048fb41194970538a021f5345ae3006ac306d7bbc70fe62b591b89d6`
and remains unopened.

The next experiment is a train-only 32-pair oracle premise. With the fixed
orthonormal 8-D scaffold and one normalized code step, it must test whether
current TTT lowers adjacent canonical-point consistency and whether the
correctly transported source code improves target future consistency beyond
current TTT and a spatially shuffled control. No basis or address is fit yet.

EXP-040 failed that premise. Current TTT lowered source and target consistency,
but correctly paired nearest-3D reuse worsened the target mean, lost to spatial
shuffle, and harmed 68.75% of pairs. Thus the v2 scaffold currently supports a
new TTT coordinate but not continual reuse. No address or memory bank should be
built yet.

The immediate next step is a train-only carrier decomposition on the same 32
pairs: compare no transport, cosine visual-token transport, predicted-3D
transport, and spatial shuffle, while measuring agreement with the target
current code. This determines whether the failure is the 3D correspondence or
the update coordinate itself. Validation and terminal remain closed.

EXP-041 found no eligible raw carrier. Untransported, visual, and canonical-3D
source codes had negative target-code agreement and all worsened current TTT.
The broad raw-update reuse hypothesis is rejected for CUT3R. The failure is the
plasticity coordinate itself, not merely correspondence.

D119 authorizes one compact revision: train only the existing 6,144-parameter
shared basis offline so the same online loss induces revisit-compatible 8-D
codes. The offline objective is the equal mean of the same consistency loss
after current-code and oracle visual-reuse application; it is not a second loss.
The next experiment must fit on a fixed subset of EXP-039 train scenes and test
on disjoint train scenes. No validation, terminal, address, or bank access is
allowed until this learned-coordinate oracle premise passes.
