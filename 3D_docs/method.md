# Current Method Specification

## Archived v1 status

EXP-028 + EXP-029 is the selected frozen paper candidate. EXP-030 passed the
complete development geometry audit. EXP-031 produced positive terminal method
comparisons on every primary mean, although its overall registered gate failed
because an infeasible 190-target coverage threshold exceeded the evaluator's
187-target maximum. EXP-032 preserves that qualification without repair.

## Frozen selected candidate

```text
streaming RGB context
        ↓
frozen FastVGGT features, predicted geometry, and tracks
        ↓
8-D per-token plasticity code initialized at zero
        ↓
one local-code step on L_track3D, eta=0.0125
        │
        ├── current-only fallback
        │
64-D pooled descriptor → factorized metric-utility Ridge MIPS top-1
        ↓
positive score? otherwise current-only fallback
        ↓
visual token correspondence transports the selected stored code
        ↓
z_out = clamp(z_current + 0.10 z_memory, -1, 1)
        ↓
depth / point readout

predict before write → deterministic reservoir, capacity 64/location
```

The only learned inference-time additions are the 288,193-parameter plasticity
head and 193-parameter factorized linear address. Semantic-zero fallback and
reservoir retention are parameter-free. There is no fine router, risk head,
learned threshold, learned eviction, pose update, second TTT step, or dynamic
state.

## Online objective

The deployed objective is exactly one absolute frozen-track 3D consistency
loss:

\[
\mathcal L_{\mathrm{TTT}}=\mathcal L_{\mathrm{track3D}}.
\]

Only the local code is differentiated, for one step. Query/future observations
are never online inputs. EXP-028's final offline training preserves broad
metric health for this deployed step.

## Offline atom training

Offline head meta-training differentiates two sparse future-geometry
objectives separately: median-aligned absolute log-depth and aligned absolute
relative-depth. Their unit-normalized gradient bisector is common descent before
the optimizer. EXP-028 preserves AdamW's proposal when its realized displacement
is common descent; otherwise it uses the same bisector direction with the
proposal norm. This safeguard has no coefficient, solver, line search, extra
head, or inference operation. Sparse LiDAR is offline supervision and is never
an online TTT or routing input.

Generic Pareto/gradient consensus is established prior art and is a training
health mechanism, not the standalone novelty claim. The contribution is the
transported and utility-addressed local adaptation experience.

## Plasticity record and causal contract

Each record contains a local 64-D visual key, an 8-D per-token code, geometry
support tensors, a pooled 64-D descriptor, timestamp, and stream partition. A
target is predicted before its record is written. Future/query frames may
produce offline meta labels and evaluation metrics only. Source-safe folds
remove held physical entities from both target and memory-source training rows.

## Utility address

For current/source descriptors `c_t,c_i`, one Ridge score uses

```text
[c_t, c_i, c_t-c_i, c_t*c_i].
```

The score factorizes exactly into a current query and source memory vectors.
The highest-scoring bank record is used only when its score is positive;
otherwise the prediction remains current-only. There is no appearance reranker
or calibrated decision threshold in the selected model. The offline target is
the single median-aligned absolute log-depth future-utility label.

## Theoretical rationale

For an `L`-smooth future loss `F` and transported residual `delta`,

\[
F(z)-F(z+\delta)\ge
-\langle\nabla F(z),\delta\rangle-\frac{L}{2}\lVert\delta\rVert^2.
\]

Transport and metric-utility addressing seek a favorable first-order term; the
fixed 0.10 residual and clamp bound second-order damage. This motivates the
method but is not a worst-case safety theorem. EXP-023's high raw-candidate harm
and the remaining terminal per-sample failures rule out a formal safety claim.

The offline feasible-displacement safeguard separately enforces local common
descent for both sparse geometry objectives at each training step. It explains
why scalarized EXP-024/025 and unconstrained-AdamW EXP-027 failed, but it does
not imply global Pareto optimality.

## Efficiency

EXP-033 measures approximately 1.996 ms for the complete method after frozen
foundation outputs, versus 292.328 ms for the current separate foundation
passes on an A100. Reservoir-64 stores 38.52 MiB of tensor payload; 83.1% is the
64-D per-token key. The learned method is compute-light but not storage-free.

## Claim boundary

Supported in the tested static nuScenes/FastVGGT setting: the transported local
correction improves SILog, aligned AbsRel, and 3D EPE over current-only TTT;
metric-utility addressing beats same-bank random and appearance means; bounded
causal reservoir deployment retains the effect; method compute overhead is
small relative to the foundation.

EXP-035 additionally supports descriptive zero-shot transfer to TUM RGB-D with
the same FastVGGT stack: full memory beats current/random/appearance on all
sequence-balanced primary means and beats current within every sequence.

Unsupported: reliable per-sample negative-transfer rejection, pose/4D claims,
reservoir superiority, universal capacity, worst-case safety, or transfer to a
second backbone. EXP-031 must be described as qualified terminal evidence
because its impossible coverage gate technically failed; EXP-035 has only
three imbalanced sequences and cannot carry a broad generalization claim.

EXP-036 further shows that the selected custom reconstruction head is not
competitive in absolute TUM quality: official CUT3R/TTT3R reduce the three
primary errors substantially. The frozen candidate is therefore a mechanism
proof, not a state-of-the-art reconstruction system.

## Active v2 experimental specification

The v1 method above is immutable. The active paper branch replaces only its
noncompetitive geometry carrier:

```text
RGB stream -> frozen official CUT3R recurrent state and DPT geometry head
           -> 8-D code per 16x16 decoder patch
           -> shared 8->768 residual basis before the final DPT token level
           -> one normalized code step on symmetric canonical-point consistency
           -> visual token transport at a physical revisit
           -> current code + transported memory code
```

Zero code is exactly the official CUT3R prediction. The pose token, recurrent
state, pose memory, encoder, decoder, and geometry head remain frozen. The only
candidate learned component currently authorized is the 6,144-parameter shared
plasticity basis. Online deployment still has one loss and one step.

This is not an accepted method. EXP-040/041 showed that a generic
orthonormal basis produces useful current TTT but incompatible source/target
codes under untransported, visual, and 3D carriers. Therefore no v2 address or
memory bank exists. EXP-042 then trained the one authorized shared basis on a
train-internal scene split. It strengthened current adaptation but failed the
pre-registered revisit-compatibility gate and did not show robust reuse beyond
shuffle. Consequently the compact v2 method is stopped: no validation run,
utility address, or bank is authorized from this checkpoint.

## Approved exact-meta revision

D121 reopens only offline differentiation. The active EXP-043 candidate is
identical at inference, but its offline objective retains the computation graph
through source and target one-step code generation. Thus the shared basis is
optimized for how it induces an update as well as how it decodes that update.
No module, loss family, or deployment operation is added. This candidate is not
accepted unless it passes the scene-bootstrap functional reuse gate on the last
15 unopened train scenes.

EXP-043 rejected unconditional reuse but revealed an online-observable decision
variable: cosine agreement between the current descent code and transported
memory code. The only candidate routing revision is the semantic zero rule
`use memory iff cosine > 0`; zero means the two descent directions cease to
agree. It has no learned parameter or calibrated threshold. Because it was
identified on an exposed audit, its development result is post-hoc and the rule
must be immutable before validation.

EXP-045 validates this rule on unseen scenes. The active candidate therefore
adds exactly one online decision to the exact-meta local code: transport a
candidate, compute its mean cosine with the current code, and apply it only when
positive. The decision has no trained weights or scalar hyperparameter. A
supplied-candidate result is not yet a deployable memory system; causal bank
construction and candidate selection remain to be established.

## Fresh-data TTT3R redesign status

D133 removed memory from the active redesign and selected official TTT3R
recurrence plus one local-code consistency step. EXP-051 established exact
native/step-wise parity and froze scene-disjoint 7Scenes roles. EXP-052 verified
that the 8-D code contains absolute-3D-improving directions and exposes finite
exact meta-gradients.

EXP-053 rejected the first learned realization: one global 6,144-parameter
basis trained by one exact relative-3D outer objective. Although its mean moved
in the desired direction, its held-train-scene confidence intervals crossed
zero and 56.25% of anchors remained harmful. There is currently no accepted v3
model or checkpoint, and validation/terminal access is prohibited pending a new
method decision.

## Proposed conditional-tangent revision

D137 authorizes only the following candidate interface:

\[
\Delta h_n = B\left(s(h_n)\odot z_n\right),\qquad
s(h_n)=1+\tanh(A\,\mathrm{LN}(h_n)).
\]

Here `B` is the existing shared `8 -> 768` basis, `h_n` is a detached frozen
final decoder patch token, and `A` is one bias-free `768 -> 8` map initialized
to zero. `B` and `A` together form one 12,288-weight conditional tangent
module. The recurrent state, TTT3R state-update rule, DPT head, online
consistency loss, code dimension, and one normalized code step do not change.
At `z=0`, the official prediction is exact for any scale; at `A=0`, the
generic shared-basis code path is exact.

This is a proposal, not an accepted method. EXP-054 must first show with a
train-only metric-gradient oracle that token-conditioned axis weighting has
useful capacity not explained by a global axis subset or a spatially shuffled
mask. No learned conditioner, validation result, or v3 checkpoint exists.
