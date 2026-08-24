# Current Method Specification

## Working architecture: utility-routed local plasticity memory

```text
streaming RGB context
        ↓
frozen foundation encoder (VGGT first)
        ↓
dense tokens + predicted depth/pose + frozen track evidence
        ↓
custom local plasticity head
        ↓
exactly one current-context TTT step on per-token code z_t
        │
        ├──────── current-only prediction / reject path
        │
frozen token-set consolidation key ──> capacity-bounded candidate bank
        │                                      │
memory candidates ── learned visual correspondence ──> transported local codes
        │
        └── current/source adaptation-history statistics
                         ↓
regularized future-utility router
                         ↓
select one candidate or reject all
                         ↓
z_out = clamp(z_t + 0.10 z_memory, -1, 1)
```

The foundation weights, base geometry head, plasticity decoder, and router are frozen online. Only the local fast code is updated by TTT.

## Plasticity memory object

The provisional local record contains:

- a normalized local appearance key;
- an 8-D per-token fast code;
- predicted 3D anchor, scale, and confidence as context metadata;
- online update/loss statistics;
- future utility and risk history only after causal evaluation becomes available;
- optional motion state in the later 4D extension.

Predicted `xyz` is not used to move the code or route memory in the locked primary path. Predicted alignment remains an ablation. Spatial locality is preserved by per-token visual addressing without assuming that a noisy cross-traversal predicted coordinate gauge is the correct update coordinate system.

## Online TTT objective

The deployable online objective may use only current observations and predictions:

- frozen-track 3D consistency;
- depth/point consistency where objective-health checks pass;
- edge-aware smoothness and bounded code regularization;
- later, pose/cycle consistency after independent health validation.

Held-out future frames generate meta-labels during training and evaluate utility, but never enter online adaptation, alignment, retrieval, or router features.

Exactly one current TTT step is used in the current design. EXP-006 found that a second step increased future loss and caused 60% harm, while bounded memory reuse improved it safely.

## Transport, evidence, and routing

1. A local visual key proposes a small candidate set.
2. Source codes are read at current tokens by appearance correspondence.
3. Online loss changes, source/current track residual history, code statistics, descriptors, and visual-transport statistics enter a regularized utility router.
4. Predicted alignment validity/inliers/residual/coverage are computed only in the geometry ablation.
5. The router selects one candidate or returns current-only TTT.
6. Accepted code is applied after current TTT with fixed residual strength 0.10.

An invalid predicted-geometry alignment does not hard-mask a visually transportable candidate. Paired place identity and appearance similarity are controls, not correctness labels. Correctness is measured by future-utility regret and negative transfer.

## Continual-learning role

EXP-007 selects a provisional two-address continual store:

- **Transport address:** the learned per-token key remains local and moves a stored 8-D code to current tokens.
- **Consolidation address:** separately normalized frozen foundation token sets predict redundant/place-compatible records. The transport key must not be reused for this role.
- **Retention statistic:** past-only predicted utility history prioritizes records under a capacity bound.
- **Retrieval:** consolidation prefiltering produces a small candidate set; the utility router ranks transported candidates.
- **Safety:** the fixed 0.10 residual bounds damage. Consolidation similarity is not an accept/reject gate, and learned rejection is still unresolved.

Generic parameter-protection methods remain ablations rather than the central mechanism.

## Current evidence boundary

The local reuse choice is based on exact train-only OOF estimates from 19 overlap components plus one locked two-component validation. EXP-007 adds train-only pseudo-stream evidence: a strict crossfit frozen consolidation key retained 97.9% of the oracle-scene utility at capacity 8 and exceeded a matched permutation null (p=0.00699). Its same-scene AUC was only 0.650 and its harm was not lower than random-score merging, so it is a provisional prefilter, not a safety or place-recognition claim. The current utility router was trained on fixed K=5 pools and exhibits winner-distribution shift as banks grow; a future bank-aware router must be trained on causal candidate sets. Real timestamp order, independent locations, learned rejection, pose adaptation, and dynamic 4D remain unverified.
