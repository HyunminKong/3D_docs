# Paper Scope — Revisit3D

Last updated: 2026-08-25 after EXP-035

## Decision status

The EXP-028 atom plus EXP-029 address is the **final frozen paper model**.
Development selection, untouched nuScenes terminal evaluation, efficiency, and
zero-shot TUM transfer are complete. No further architecture, loss, threshold,
seed, or dataset-specific model variant is allowed for this paper.

The controlled method evidence is strong enough for a mechanism paper draft,
but EXP-036 shows that the custom reconstruction head is far behind CUT3R and
TTT3R in absolute TUM geometry. The current candidate is therefore not ready
for a broad CVPR reconstruction-framework submission. ICLR-level generality is
also unsupported because a second backbone/task has not been validated.

## Working title

**Revisit3D: Utility-Addressed Test-Time Plasticity for Streaming 3D Reconstruction**

## One-sentence thesis

Streaming 3D systems can remember **how local geometry adapted**, transport
that correction into a later observation, and retrieve it by causal future
geometry utility rather than by appearance alone.

## Frozen method

The inference graph has two learned additions to a frozen FastVGGT/custom
geometry stack:

1. an 8-D per-token plasticity code updated once by `L_track3D` at
   `eta=0.0125`;
2. one 64-D factorized Ridge address trained on offline future metric utility.

The top-1 record is reused only above semantic zero. Visual correspondence
transports its code and applies a fixed 0.10 residual after current TTT. A
deterministic reservoir stores at most 64 records per stream. There is no risk
head, fine router, learned threshold, learned eviction, pose update, second TTT
step, or dynamic state.

Offline atom training uses aligned-log and aligned-relative depth endpoint
gradients with the parameter-free EXP-028 feasible-displacement safeguard.
This makes the deployed one-loss/one-step direction metric healthy; generic
gradient consensus is prior art and not the novelty claim.

## Visible method constants

- `eta=0.0125`: one online local-code step;
- `alpha=0.10`: bounded memory residual;
- `C=64`: per-stream reservoir capacity;
- code dimension 8 and address dimension 64.

Top-1 and semantic zero are fixed decisions, not sweepable hyperparameters.
Ridge alpha 1 is an implementation constant. The paper must not present a large
hyperparameter search.

## Formal learning problem

For current context `x_t`, local adaptation is

\[
z_t=-\eta\nabla_z\mathcal L_{\mathrm{track3D}}(x_t,z)\rvert_{z=0}.
\]

For stored record `i`, visual transport gives

\[
\delta_{t,i}=\alpha T(z_i;x_i\rightarrow x_t).
\]

Offline future utility on disjoint query observations is

\[
U_{t,i}=F_t(z_t)-F_t(z_t+\delta_{t,i}).
\]

The address estimates this utility from current/source descriptors. Query RGB,
depth, and LiDAR never enter online adaptation, addressing, bank writes, or
retention.

## Evidence summary

| Requirement | Evidence | Status |
|---|---|---|
| metric-healthy current TTT | EXP-028 OOF | passed |
| reusable local correction | EXP-028 oracle reuse | passed |
| utility address over random/appearance | EXP-029 | passed |
| full absolute geometry | EXP-030, 217 targets/25 components | passed |
| untouched terminal geometry | EXP-031, 187 targets/29 components | qualified positive; impossible coverage gate failed |
| coverage accounting | EXP-032 | all 187 causally eligible targets included |
| efficiency/storage | EXP-033 | 1.996 ms method overhead; 38.52 MiB bank-64 |
| independent dataset | EXP-035 TUM zero-shot | descriptive pass on 111 targets/3 sequences |
| second backbone/task | none | open; not required for the CVPR-first claim |
| matched external SOTA baselines | EXP-036 CUT3R/TTT3R | completed; exposes major absolute-quality gap |
| fixed qualitative results | absent | required before submission |

## Claim boundary

Claim:

- local transported adaptation improves SILog, aligned AbsRel, and 3D EPE over
  current-only TTT;
- one metric-utility address beats same-bank random and appearance means;
- the effect survives a bounded causal stream and descriptive indoor zero-shot
  transfer;
- learned compute overhead is small relative to the frozen foundation.

Do not claim:

- camera-pose, point-tracking, dynamic-4D, or second-backbone improvement;
- reliable per-sample negative-transfer rejection or worst-case safety;
- reservoir superiority over FIFO or universal capacity 64;
- a fully self-supervised training pipeline, since sparse future geometry is
  offline supervision;
- that EXP-031 literally passed every preregistered gate.

## Remaining submission work

1. Decide whether to reopen method scope around a competitive CUT3R/TTT3R-class
   backbone or narrow the current work to a non-SOTA mechanism study.
2. If the current paper continues, export deterministic earliest-target qualitative depth/point results for
   every test component/sequence before viewing improvements; show fixed
   examples and failure cases.
3. Assemble main tables, ablation table, efficiency table, method figure, and
   causal protocol diagram from committed machine-readable results.
4. Write the CVPR paper with EXP-031's coverage qualification and EXP-035's
   three-sequence limitation stated explicitly.

## Stop rule

The final model is closed. A baseline or qualitative failure can narrow the
paper claim or motivate future work, but cannot trigger another model variant
on the exposed nuScenes/TUM evaluations.
