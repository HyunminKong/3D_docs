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
memory candidates ── visual token correspondence ──> transported local codes
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

After routing generalizes, continual learning will manage the long-term local-code store:

- write only useful, confident adaptations;
- cluster by adaptation utility/regime, not place identity alone;
- merge visually/semantically compatible local codes;
- preserve frequently useful memories;
- age or evict low-utility records;
- reactivate or split records when new evidence contradicts consolidated state.

Generic parameter-protection methods remain ablations rather than the central mechanism.

## Current evidence boundary

The architecture choice is based on exact train-only OOF estimates from 19 overlap components. The explicit neural risk head was identifiable after expansion but did not reduce selected harm, so it is not part of the locked model. The regularized utility router passed its one-shot 14-episode/two-component descriptive validation gate at +1.79% utility and zero deadband harm, but accepted every episode. Thus candidate ranking and bounded reuse are supported; learned rejection and broad generalization are not. EXP-007 uses train-only causal streams for continual-bank design, and both the EXP-006 validation and exposed test split are closed to its model selection.
