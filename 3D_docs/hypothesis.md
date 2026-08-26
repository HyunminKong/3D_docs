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

Status: **Rejected as a central paper hypothesis for the registered short-stream
carrier/protocol.** EXP-061 found a real temporal gauge component: matching
per-frame Sim(3) beat one context transform and cyclic reassignment with wholly
positive intervals. Its magnitude was only 3.66% of total relative 3D error,
however, versus the preregistered 15% minimum. On the native-confidence top
quartile it explained 6.47%, missed the 10% gate, and was slightly negative in
`stairs`. This is useful evaluation anatomy but too small and heterogeneous to
justify a new hierarchical uncertainty head. No threshold, context-length, or
carrier repair is authorized under H19.

## H20 — Geometry-relevant noncommutativity of recurrent updates

Let `U_x` denote the frozen persistent-state update induced by observation `x`.
For static-scene observations `a` and `b`, the commutator

\[
[U_a,U_b](s)=U_b(U_a(s))-U_a(U_b(s))
\]

is nonzero and large enough to cause material absolute-geometry variation at a
fixed later query, even when the initial anchor and complete observation set are
identical. Local path inconsistency is therefore a distinct source of
long-horizon instability that can in principle be reduced by an offline
commutator-consistency objective without changing causal inference.

Status: **Empirical premise supported; paper-method candidate rejected.** EXP-062 passed
every gate. Identical evidence orders span 12.58% of chronological query EPE on
average; 12/16 contexts exceed 5%; every scene has positive range; exact replay
is bitwise; and label-free geometric output dispersion predicts metric range
with Spearman `0.835`. Chronological order is best in only 5/16 contexts and
worst in 4/16, so no fixed favorable ordering explains the effect. Generic RNN
swap regularization is prior art (SIRE, NeurIPS 2020), however. EXP-063 must
establish that a geometry-decoded commutator is more informative than latent
state distance and that a symmetric state lies in a geometry-healthy region
before a 3D-specific method can be proposed.

EXP-063 **rejects latent-state symmetrization but preserves one output-space
sub-hypothesis.** Geometry dispersion retains Spearman `0.835` with metric
range, while normalized latent-state dispersion is `-0.012`; the association
gap is `0.847`. Arithmetic state barycentering worsens mean EPE by `0.001162`
versus the six-order mean and by `0.002365` versus chronological, with negative
gains in three scenes. In contrast, the preregistered descriptive output-
pointmap barycenter improves the mean-order EPE in all 16 contexts. EXP-064 is
authorized only to test whether a fixed small geometry-consensus direction is
also healthy for the single chronological path. Failure ends H20 as a method
candidate; success authorizes trainability design, not a latent commutator.

EXP-064 failed five of seven gates. The 10% geometry-consensus step yields only
`7.69e-5` chronological EPE gain with CI `[-1.12e-4, 2.71e-4]`, is negative in
`stairs`, harms 43.75% of contexts, and does not significantly beat a spatially
shuffled residual. Its all-order average remains weakly positive in every
scene, so output ensembling has descriptive denoising value, but the measured
commutator does not supply a healthy deployable chronological correction.
Coefficient tuning, order selection, confidence gating, and training on these
exposed contexts are prohibited. H20 is retained as evaluation anatomy only.

## H21 — Persistent calibration-shock contamination

In a recurrent streaming pointmap model, an observation is written into a
persistent scene state without an explicit camera-coordinate input. A transient
change in focal length/FoV can therefore be entangled with scene structure:
even after the camera returns to its original regime and one clean observation
is written, later clean-query geometry remains worse than if the zoomed frame
had been skipped.

The claim is specifically a difference-in-differences state effect, not the
immediate difficulty of a cropped image. A valid premise must exceed a
full-FoV resampling control and a missing-periphery control, remain after one
clean recovery write, and be positive across scenes.

Status: **Rejected as a central paper premise for the registered shock.** The
mean persistent difference-in-differences penalty is positive in every scene
but is only 1.48% of clean EPE, with CI `[-0.000351, 0.002366]` and 68.75%
positive contexts. It does not significantly exceed either the resampling or
missing-periphery control, and loses to the latter in `stairs`. The immediate
effect is significant, but persistent camera-coordinate contamination is too
small and confounded to justify a method. No stronger zoom, context selection,
camera-prior encoder, or calibration gate is authorized under H21.

## H22 — Causal observation support predicts ray-query geometry risk

After a recurrent pointmap carrier has consumed only a causal RGB history, an
RGB-free camera-ray query contains geometry supported by surfaces visible in
the history and geometry completed without direct historical evidence.
Unsupported patches should have higher absolute 3D error, and their risk should
be observable from predicted history/query geometry beyond native confidence.

The fixed zero-fit provenance signal is nearest predicted 3D distance from a
query patch to the union of predicted history pointmaps, normalized by predicted
query range. Ground-truth visibility is only an offline label. The camera pose
used to form the ray query is controlled oracle input and cannot be described
as deployable pose estimation.

Status: **Rejected for the registered carrier and signal.** EXP-066 obtains an
aggregate +8.62% unsupported error gap, below the 20% gate, with a confidence
interval crossing zero and negative scene gaps in `pumpkin`, `heads`, and
`chess`. Predicted provenance has error Spearman 0.196 versus 0.343 for native
confidence; its paired advantage interval is wholly negative. Equal-rank fusion
worsens AURC by 7.68%, also with a wholly negative gain interval. Exact replay
passes and no query RGB or update is used. Thus binary historical visibility is
not a stable risk partition here, and nearest predicted-history geometry does
not add complementary reliability. No alternate-distance or learned-head repair
is authorized on these contexts.

## H23 — Function-space transport resolves plasticity-coordinate mismatch

Let `P_x(z)` be the pointmap decoded at observation `x` from local adaptation
coordinate `z`. Directly transporting a source update `delta z_s` assumes that
the source and target decoder Jacobians assign it the same geometric meaning,
which EXP-040--060 contradict. The induced source displacement

`delta P_s = P_s(delta z_s) - P_s(0)`

is instead a function-space object. Transporting this displacement by predicted
3D correspondence and pulling it back with one target gradient should preserve
the intended 3D change despite observation-dependent coordinate charts.

Status: **Rejected for the frozen operator.** EXP-067's gain over equal-compute
second-current TTT is only `2.04e-6` (0.00299% relative), with CI
`[-1.12e-6, 5.22e-6]`, a negative `pumpkin` mean, and 43.75% harm. It loses on
average to direct code and untransported function payloads and is tied with
spatial shuffle. The fixed normalized pull-back step also increases its own
function residual in all pairs because the transported displacement is much
smaller than that code step. The preregistered stop rule prohibits step or
solver repair. Compact reusable TTT experience is therefore closed on this
carrier in both coordinate and function-space forms.

## H24 — Cross-clip query-equivalence residual

For a fixed physical point, target time, and camera reference in the overlap of
two video clips, a query-based 4D reconstruction model should return the same
3D point independent of the surrounding clip used to encode the scene. The
tested hypothesis is that OpenD4RT violates this equivalence through a material
non-rigid residual that cannot be explained by global Sim(3) or depth-layer
scale mismatch and is poorly exposed by pointwise APD.

Status: **Rejected as the registered paper premise; diagnostic residual
supported.** EXP-068 finds a 2.8292%-of-scale residual after held-out four-layer
Sim(3), with positive CI, 34.51% retention from raw disagreement, and a
1.8999%-of-scale advantage over the one-frame context shift in all 16 premise
sequences. Pair-distance disagreement also remains positive. Exact replay is
bitwise and no validation, terminal, or model fit is involved.

The complete gate fails because mean absolute A/B APD difference is `0.06051`,
above the frozen `<0.05` blindness threshold. Thus the window-dependent
non-gauge phenomenon is real, but it is not sufficiently distinct from an
ordinary pointwise accuracy change to carry the intended novelty. Generic
window alignment and relational regularization are already occupied, so no
equivalence loss, new head, memory, or validation run is authorized from these
exposed premise sequences. The immutable v1.0 gate-counting bug is corrected
separately to 16/17; it does not change this rejection.

## H25 — Pointwise ranking does not measure query integrity

For semantically identical 4D point queries, an irrelevant encoder-context
change can produce a larger non-gauge structural residual while pointwise APD
remains indifferent or improves. Pointwise accuracy and counterfactual query
integrity are therefore empirically distinct model-selection axes.

Status: **Rejected as the registered evaluation-paper premise; complementary
diagnostic supported.** EXP-069 reproduces positive large-minus-adjacent damage
in all 10 evaluable confirmation sequences and APD still prefers the large
shift in 16/30 target rows. Structural damage and APD gain remain weakly
associated (Spearman `0.123`). However, mean signed APD gain is `-0.01593` with
CI `[-0.08154, 0.04212]`, failing both preregistered mean/interval gates.

Thus pointwise APD can locally misrank query integrity but does not
systematically reward the less stable context. No benchmark expansion, metric
repair, equivalence loss, or terminal access is authorized. The structural
residual from H24/25 remains useful diagnostic evidence only.
