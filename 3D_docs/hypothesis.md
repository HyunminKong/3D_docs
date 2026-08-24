# Research Hypotheses

Status values are `Supported`, `Partially supported`, `Open`, or `Rejected`.

## H1 — Locality of reusable adaptation

Reusable streaming adaptation is spatially local and is better represented by 3D-addressed atoms than by a global parameter update.

Status: **Supported in controlled development probes.**

## H2 — Geometry-conditioned transport

Transport using shared 3D coordinates and local appearance correspondence preserves useful adaptation better than raw vector reuse or appearance-only transport.

Status: **Supported under oracle-coordinate probes; predicted-coordinate validation remains open.**

## H3 — Online utility observability

Current self-supervised geometry evidence contains information about whether a past atom will improve future reconstruction.

Status: **Partially supported.** Average future utility improved, but individual negative transfers remain.

## H4 — Learnable risk-aware routing

A candidate/current utility head meta-trained with future frames can outperform heuristic current-loss and pose-distance gates while learning to reject harmful reuse.

Status: **Open; target of EXP-006.**

## H5 — Continual atom consolidation

Merging compatible local atoms and preserving their uncertainty/utility statistics can bound memory while retaining revisit benefits.

Status: **Open; memory-bank experiments are gated on H4.**

## H6 — Extension to dynamic 4D

The same atom mechanism can attach to tracked dynamic points or motion-conditioned local regions to improve reappearance and occlusion recovery.

Status: **Open; outside the first static-revisit milestone.**

## Rejected hypotheses

- A small global/slot fast-weight vector is sufficiently context-selective for memory retrieval.
- More optimization alone fixes update-direction collapse.
- Cosine similarity of raw gradients is a reliable proxy for causal future utility.
- The originally paired episode is always the uniquely correct retrieval target.
- A parameter-free current-loss threshold is a sufficient negative-transfer safeguard.
