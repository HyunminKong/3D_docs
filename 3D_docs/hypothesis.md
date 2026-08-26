# Research Hypotheses

Status values are `Supported`, `Partially supported`, `Open`, or `Rejected`.

## H1 — Locality of reusable adaptation

Reusable streaming adaptation is better represented by spatially local fast codes than by a model-wide update, provided the local code is transported into the current token frame.

Status: **Supported only for the tested FastVGGT depth/point setting; open on the competitive carrier.** Global and slot states failed context selectivity, whereas transported visual local codes improved all primary means in EXP-030/031 and again over current-only within every TUM sequence in EXP-035. EXP-040 showed that a raw 8-D CUT3R code transported by predicted canonical 3D is harmful, so cross-backbone reuse is not established.

## H2-P — Predicted geometry as the update carrier

Predicted 3D alignment transports a reusable update better and more safely than appearance-only correspondence.

Status: **Rejected.** Predicted geometry and geometry+appearance transport did not improve over visual transport in the FastVGGT branch, reduced coverage, and increased harm. EXP-040 independently found that nearest canonical-3D transport on frozen CUT3R harmed 68.75% of train pairs and lost to a spatial shuffle. Predicted geometry may remain routing evidence, but it is not the accepted update carrier.

## H2-E — Predicted geometry as routing evidence

Predicted alignment quality is necessary primary evidence for candidate utility/risk even when geometry is not the update carrier.

Status: **Rejected as a primary input.** No-alignment observable routing generalized, while geometry-only routing was weaker and less safe. Alignment statistics remain diagnostic ablations.

## H3 — Online utility observability

Current/source descriptors, current-only loss changes, update statistics, and transport evidence predict the future utility of a past local correction without accessing future frames online.

Status: **Supported in the tested static-revisit setting.** EXP-029 learned a source-safe metric-utility address without online query access. The frozen address generalized in EXP-030/031 and retained positive sequence-balanced means in zero-shot TUM EXP-035. TUM acceptance was 100%, so cross-domain rejection calibration remains unsupported.

## H4-U — Learnable utility addressing and routing

A trainable candidate/current utility model can retrieve and apply more useful adaptation than identity, appearance/place similarity, current-loss heuristics, and matched random addressing.

Status: **Supported in the tested static-revisit setting.** The single factorized metric-utility Ridge address beat matched random and appearance means in development, untouched nuScenes terminal evaluation, and the three-sequence TUM descriptive transfer. This does not establish reliable per-sample harm rejection or transfer beyond the FastVGGT feature space.

## H4-R — Learnable negative-transfer risk

A separate risk classifier can generalize rejection of harmful memories across independent physical contexts.

Status: **Rejected for the tested classifier.** Neural risk AUROC was nontrivial, but hard routing did not improve selected harm over the compact utility router. Safety in the final model comes from utility selection, current-only fallback, and a fixed 0.10 residual—not a separately claimed risk classifier.

## H5 — Continual local-code consolidation

A causal, capacity-bounded memory can retain most of an unbounded utility-addressed bank's revisit benefit.

Status: **Partially supported for bounded deployment.** Reservoir-64 retained a positive metric-memory effect on the frozen EXP-031 location streams. It was not compared with an unbounded metric-addressed bank on that terminal test and has never significantly beaten FIFO. No reservoir-superiority or universal-capacity claim is made.

## H6 — Extension to dynamic 4D

The same utility-routed local-memory principle can attach to tracked dynamic points or motion-conditioned regions to improve reappearance and occlusion recovery.

Status: **Open; outside the completed static-revisit milestone.**

## H7 — Pareto-healthy offline plasticity

The aligned log-depth and aligned relative-depth meta objectives expose
different but jointly necessary geometry signals. A parameter-free common
descent update applied only during offline head meta-training can preserve
improvement in both, while deployment remains one-loss/one-step TTT.

Status: **Supported for the selected offline head.** EXP-026 exposed local
gradient conflict, EXP-027 localized AdamW rotation, and EXP-028's
parameter-free feasible-displacement safeguard achieved 100% realized common
descent while improving all three OOF geometry means. The frozen head retained
the full-system benefit in EXP-030/031. This is a training-health mechanism,
not the paper's standalone novelty claim or a general Pareto theorem.

## H8 — Learned revisit-compatible plasticity coordinate

A single low-dimensional basis can be trained offline so that gradients of one
online geometry-consistency loss become both current-useful and compatible
across physical revisits, while the foundation carrier and online loss remain
frozen.

Status: **Rejected for the tested compact first-order basis.** EXP-038 proved
the coordinate is differentiable, but EXP-040/041 rejected reuse under a generic
orthonormal basis: every simple carrier had negative future gain and negative
source/target code agreement. EXP-042's one trained 6,144-parameter basis
substantially increased current-step improvement,
but failed the registered revisit-compatibility gate: mean transported-code
agreement remained negative, the reuse gain was only `5.96e-6`, and its
scene-bootstrap interval crossed zero. This rejects the authorized compact v2
realization; it does not prove that every larger or differently supervised
plasticity representation is impossible.

## Rejected or narrowed hypotheses

- A small global/slot fast-weight vector is sufficiently context-selective for retrieval.
- More optimization alone fixes update-direction collapse.
- Cosine similarity of raw gradients is a reliable proxy for causal future utility.
- The paired physical revisit is always the uniquely correct or most useful memory.
- A parameter-free current-loss threshold is a sufficient negative-transfer safeguard.
- Predicted 3D correspondence should be the primary carrier of the fast update.
- A physical revisit plus nearest predicted canonical-3D transport is sufficient to make a CUT3R local update reusable.
- Raw source TTT directions are naturally compatible with target TTT directions on a competitive recurrent carrier.
- One fixed-pass first-order shared basis trained only through current/revisit consistency is sufficient to make CUT3R codes robustly revisit-compatible.
- Geometry-alignment failure should hard-reject a candidate that still has valid visual transport.
- On the archived FastVGGT mechanism proof, a second current TTT step replaces
  memory reuse. This rejection does not transfer to the competitive CUT3R
  branch: EXP-048 found that the second current step decisively dominates
  FIFO-16 memory there.
- Generic DINOv2 place compatibility is a causal adaptation-utility address.
- Past utility history is a reliable learned eviction priority at capacity 8.
- Capacity 8 is a general sufficient bound.
- Reservoir sampling is demonstrably superior to FIFO at capacity 64; the final test does not support this stronger claim.

## H9 — Future-utility-differentiated plasticity

The same compact basis can become revisit-compatible when offline training
differentiates future consistency through the online source and target code
creation steps, rather than treating those generated codes as fixed.

Status: **Rejected for ungated reuse; routing sub-hypothesis open.** EXP-043's
exact meta-gradient strongly improved current adaptation, but ungated reuse and
correct-over-shuffle confidence intervals crossed zero. Post-hoc analysis found
large oracle-abstention headroom and strong association between online
current/memory code agreement and utility. EXP-044 records that lead; only a
pre-frozen validation policy can support the routing sub-hypothesis.

## H10 — Parameter-free descent-agreement routing

The algebraic sign of the cosine between current and transported memory codes
is sufficient online evidence to reject most harmful reuse without a learned
router or tuned threshold.

Status: **Supported on scene-disjoint validation for a supplied revisit
candidate.** EXP-045 passed every gate: positive reuse in all 14 scenes,
positive confidence bounds over current, ungated reuse, and independently
gated shuffle, with 3.76% harm. Retrieval from a causal multi-candidate bank
remains open.

## H11 — Parameter-free agreement addressing

Among multiple causal memory records, maximum current/memory code agreement
selects more useful adaptation than frozen appearance or random addressing.

Status: **Supported as a cached-direction ranking signal, rejected as a
continual-memory advantage on the competitive full-stream carrier.** EXP-046
showed that agreement beats appearance/random in a curated causal bank.
EXP-047 preserved that ordering under every-frame writes, but selected recent
FIFO records. EXP-048 then showed that FIFO memory loses to an equal normalized
second current step by `4.40e-4`, with a wholly negative confidence interval
and zero favorable scenes out of 14. Thus agreement identifies useful update
directions, but the tested memory supplies no demonstrated information beyond
additional current optimization. Terminal generalization is not authorized.

## H12 — Information-insufficient revisit advantage

When the current adjacent observation has substantially weaker geometric
baseline than a physically corresponding past observation, transported past
adaptation contains useful information that cannot be replaced by an equal
additional current TTT step.

Status: **Rejected for natural low parallax and the tested update-code memory.**
EXP-049 used low target translation (`<=0.5` scene-median steps), a supplied
past source of at least `1.0` step, and a within-scene motion-sufficient control.
Even future-oracle application lost to second-current TTT in all 24 low-parallax
scenes, and the low-versus-sufficient interaction was not positive. This does
not rule out explicit past geometry under true occlusion; it does reject the
claim that weak parallax makes the existing adaptation-direction code uniquely
useful.

## H13 — Exact-meta local coordinate for current-only 3D TTT

A 6,144-parameter exact-meta low-dimensional coordinate makes a single
self-supervised local TTT step improve absolute depth and point accuracy on a
frozen competitive recurrent carrier, beyond both zero-code CUT3R and an equal
generic coordinate.

Status: **Rejected for the frozen EXP-043 coordinate.** Corrected EXP-050
established exact official replay parity, then found that one exact-meta step
worsens SILog in all three TUM sequences and yields no significant aligned
AbsRel or 3D EPE gain. It also fails every capacity-matched generic-coordinate
comparison and is substantially worse than matched official TTT3R. The online
consistency gain does not imply absolute geometry gain. This rejects the
current realization, not every possible metric-aligned TTT objective.

## H14 — Metric-aligned current-only plasticity on TTT3R

A shared low-dimensional plasticity basis can be trained on fresh RGB-D scenes
so that one online self-supervised local-code step improves absolute geometry
on top of the stronger official TTT3R recurrent update, while preserving one
online loss, one step, and zero additional recurrent or retrieval modules.

Status: **Rejected for the fixed EXP-053 shared-basis realization; broader
formulation open.** EXP-051 established exact TTT3R-mode carrier parity and
EXP-052 showed that the compact code contains metric-useful directions with
finite exact meta-gradients. EXP-053 then moved held-train-scene mean utility in
the intended direction, but both confidence intervals crossed zero and final
harm remained 56.25%. Thus one 48-step AdamW pass on a single global shared
basis is not sufficient for reliable metric alignment. Validation was not
opened. Continuing requires a newly justified training formulation or
representation, not post-hoc tuning of this fit.

## H15 — Geometry-token-conditioned tangent metric

A single geometry-decoder-conditioned module can make the same one-step,
one-loss TTT3R update metric-useful by changing the relative contribution of
the eight shared tangent axes at each spatial token, without changing the
official recurrent update or adding a learned optimizer, memory, gate, or
online objective.

Status: **Supported at oracle capacity; rejected for the token-only linear
conditioner.** EXP-054 passed every zero-fit gate. Its token-axis oracle improved the metric on all 16
anchors with mean gain `2.00e-5` and 0% harm, versus `1.96e-6` for the global
basis and `1.71e-6` for the same mask after spatial shuffle. Both paired 95%
intervals were wholly positive, and a scene-global axis mask was substantially
weaker. This establishes that spatially local axis selection is the missing
capacity on the exposed anchors. EXP-055 answered no for the registered realization: final gain and paired
improvement intervals crossed zero, harm remained 43.75%, and the conditional
model did not beat EXP-053's learned global basis. The oracle result therefore
identifies a spatial selection opportunity, but frozen current tokens alone do
not make the required selection learnable under the compact fit.

## H16 — Pairwise geometry residual observability

The source-safe residual between the current predicted canonical point and its
nearest previous predicted point contains spatial evidence about which shared
plasticity axes align the online consistency step with absolute geometry,
beyond the frozen current decoder token alone.

Status: **Rejected for the registered minimal residual.** EXP-056's scene-OOF
combined balanced accuracy was `50.20%`, below the `50.24%` spatial-shuffle
control and essentially chance. Its realized gain was positive on average but
negative in `pumpkin`, harm was 50%, and paired intervals over global and
shuffled geometry crossed zero. Only the comparison with token-only was
positive. Thus the nearest-point residual does not expose the oracle axis label
reliably across scenes and cannot justify another conditioner fit.

## H17 — Explicit past-surface advantage under missing current evidence

When a target image region is deliberately erased but its static surface was
visible in the immediately preceding clean observation, explicitly retained
past surface geometry contains information that cannot be recovered by the
frozen recurrent TTT3R state plus an equal second current local-TTT step.

Status: **Supported under controlled erasure; predicted-pose deployment evidence
is qualified positive.** EXP-057
passed every gate. Erasure increased error in every train scene, while
predicted past-surface fusion beat second-current TTT by `0.408` mean relative
3D EPE with CI `[0.292, 0.525]`, beat spatial shuffle by `0.129` with CI
`[0.091, 0.174]`, and harmed 0% of anchors at 89.5% mean coverage. GT pose,
scale, and visibility were then removed from the fusion policy in EXP-058.
Predicted-pose/native-scale fusion retained 97.6% of the oracle gain, beat
second-current and spatial shuffle in every scene, and harmed 0%. Its literal
gate failed only because repeated local TTT missed the `1e-5` EXP-057
reproduction guard by at most `1.97e-5`; EXP-059 localized that drift after
adaptation while data, support, and the erased baseline matched exactly. A
known synthetic erasure mask remains, and no natural visibility rule or
multi-frame address is established.

## H18 — Reusable local plasticity under missing current evidence

When a static surface is visible in a clean source observation, the spatially
local TTT code induced by that observation retains surface-specific adaptation
experience. After predicted-3D transport to a later observation where the same
surface evidence is erased, that code improves target geometry beyond both an
equal second current local-TTT step and a spatially shuffled identical code.

Status: **Rejected for the fixed generic local code.** EXP-060 reproduced the
same 16 anchors and scoring support exactly, and source/current consistency
steps descended at every anchor. Nevertheless transported past code gained only
`4.36e-6` over second-current TTT with CI `[-4.08e-6, 1.41e-5]`, was negative
in two of four scenes, lost on average to both untransported and spatially
shuffled controls, and harmed 56.25% of anchors. Correct spatial transport is
therefore not observably useful. This rejects the current adaptation object,
not the general possibility of a radically different learned representation;
such a redesign is not authorized on these exposed anchors.

## H19 — Hierarchical gauge/local reliability

For streaming pointmap reconstruction, prediction error contains two
empirically separable components: a frame-shared, time-varying Sim(3) gauge
error and a local surface residual after that gauge is removed. A per-point
confidence alone cannot fully characterize native-coordinate risk because the
shared gauge latent induces correlated error across all points in a frame.

Status: **Selected, premise not yet tested.** EXP-061 is a zero-fit train-only
error-anatomy experiment. Each Sim(3) is estimated on one checkerboard half of
dense valid points and scored on the disjoint half. It compares one transform
for the four-frame context, one transform per frame, and cyclically reassigned
per-frame transforms. The analysis is repeated on only the top-confidence
quartile. No uncertainty head, calibration model, loss, or validation access is
authorized until the magnitude, temporal specificity, and high-confidence
persistence gates pass.
