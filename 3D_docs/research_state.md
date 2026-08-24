# Current Research State

Last updated: 2026-08-25

## Research goal

Develop a streaming 3D/4D reconstruction framework that learns from current geometric evidence at test time and reuses useful local adaptation experience when a related physical context reappears, without catastrophic interference or uncontrolled memory growth.

The target outputs are depth, point cloud/pointmap, camera pose, and ultimately dynamic point tracking / 4D geometry.

## Current central claim

Reusable adaptation is not well represented by a model-wide gradient or a small global fast-weight vector. It should be represented as spatially addressable local plasticity atoms anchored in a persistent 3D coordinate system and transported by geometry plus appearance correspondence.

## Current method direction

- Frozen foundation backbone initially based on VGGT.
- New trainable geometry/plasticity head; no tttLRM wrapper.
- Local test-time updates on 3D-addressed residual atoms.
- Visual/local relocalization retrieves a small candidate set.
- 3D coordinate and appearance correspondence transport atoms into the current context.
- A learned future-utility/risk head selects, mixes, or rejects candidates.
- Continual consolidation merges useful overlapping atoms and preserves uncertainty/utility statistics.

See `3D_docs/method.md` for the full boundary.

## Evidence snapshot

- Global/slot learned update states collapsed to nearly identical directions and failed context-selective reuse.
- Raw gradient direction retained weak context information but did not identify causally useful memory.
- Dense visual transport produced a larger reuse effect than vector transport.
- Oracle 3D-coordinate plus appearance transport gave the strongest controlled effect on development validation.
- Current self-supervised geometry score predicted future candidate utility on average, including the locked test probe, but still caused negative transfer in some episodes.
- A hand-written threshold or scalar pose/loss fusion did not reliably remove negative transfer.

## Current benchmark

- Physical cross-episode nuScenes revisits in `A → B → A'` form.
- Development manifest: `revisit3d/manifests/nuscenes_revisit_dev.json`.
- Original six-episode test split was used once in EXP-005 and is now closed to tuning.

## Experiment status

- EXP-001 — tttLRM fast-weight premise probes: completed, negative.
- EXP-002 — independent Revisit3D benchmark and objective health: completed.
- EXP-003 — compact/global/slot adaptation reuse: completed, negative for context-selective reuse.
- EXP-004 — retrieval keys and learned local update routing: completed, partial/negative.
- EXP-005 — dense 3D plasticity transport and online utility: completed, central feasibility supported with safety caveat.
- EXP-006 — trainable 3D plasticity atom head and future-utility/risk meta-objective: Stage-0 v2.2 train cross-fit health gate passed; Stage-1 implementation next.

## Open questions

1. Can predicted pose/depth maintain a sufficiently stable shared coordinate system without oracle poses?
2. Can a rich candidate/current utility head generalize and reduce the observed negative-transfer rate?
3. What atom representation provides the best accuracy-memory-compute trade-off?
4. How should atoms be merged, aged, and reactivated under dynamic changes?
5. When should the method extend from static 3D revisits to dynamic point tracking and 4D reconstruction?

## Next step

Implement Stage 1 from `3D_docs/EXP-006 Implementation Brief.md` v2.2 using the passed `revisit3d/checkpoints/exp006_geometry_bootstrap_v22.pt`. Do not open validation yet. First implement and unit-test the spatial atom, predicted point construction, robust Sim(3), and visual/geometry transport on train-only smoke/cross-fit paths. The v2.1 identity-gate failure remains preserved as an objective-degeneracy diagnostic.
