# Research Goal and Scope

## Long-term goal

Create a general streaming 3D/4D reconstruction system that combines fast test-time geometric adaptation with long-term reuse of successful adaptation experience.

## First-paper scope

The first paper studies static or mostly static physical revisits and predicts depth, pointmap/point cloud, and camera pose. Dynamic point tracking and full 4D reconstruction are planned extensions after the reusable local-adaptation mechanism is established.

## Intended novelty

The method remembers **where, why, and how local geometry adapted**, rather than storing frames or whole-model parameter deltas. It couples:

- a local geometric TTT state;
- transport between observations through 3D and appearance correspondence;
- future-utility-aware memory routing;
- continual consolidation of spatial adaptation atoms.

## Non-goals for the first implementation

- Modifying tttLRM or presenting a memory layer on top of it.
- Full fine-tuning of a foundation backbone at test time.
- Treating oracle poses or foundation prediction heads as deployable outputs.
- Building a large memory bank before local reuse and routing safety are validated.
