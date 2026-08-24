# Current Method Specification

## Architecture

```text
streaming RGB observations
        ↓
frozen foundation encoder (VGGT first; D4RT/VGGT ablation later)
        ↓
dense tokens + initial depth/point/pose/track evidence
        ↓
new geometry/plasticity head
        ├── depth / point residuals
        ├── pose correction
        ├── tracking and geometry confidence
        └── sparse 3D-addressed plasticity atoms
                         ↑
visual relocalization → candidate atoms
                         ↓
3D + appearance transport
                         ↓
online geometry probe → learned utility/risk routing
                         ↓
local TTT update and continual consolidation
```

## Plasticity atom

The provisional atom contains:

- 3D anchor position and spatial scale;
- local frozen-feature key;
- compact residual/update code, initially a low-rank depth/point residual;
- observation count, age, uncertainty, and past utility statistics;
- optional motion state for the later 4D extension.

The atom is the fast state. Foundation and main head weights are not mutated online in the first implementation.

## Online objective

The deployable TTT objective may use only current observations and predictions:

- multi-view track 3D consistency;
- depth/point consistency;
- reprojection where objective-health checks confirm valid support;
- pose consistency;
- temporal/cycle consistency;
- confidence-weighted regularization.

Held-out future frames supervise the outer utility/risk objective during training but never enter the online update.

## Retrieval and routing

1. A learned local visual key proposes a small candidate set.
2. Candidate atoms are transported into current tokens using predicted 3D coordinates and local feature correspondence.
3. A cheap current geometry probe is computed for each candidate.
4. A utility/risk head predicts future benefit and negative-transfer probability.
5. The router rejects all candidates or softly mixes accepted atoms.

Pose-map proximity is an optional prior, not a sufficient selector by itself.

## Continual-learning role

Continual learning manages the long-term atom store:

- write only useful, confident adaptation;
- merge geometrically overlapping and compatible atoms;
- preserve frequently useful atoms;
- age or evict low-utility atoms;
- reactivate or split atoms when new evidence contradicts consolidated geometry.

Generic parameter-protection methods may be ablations, but they are not the central mechanism.

## Current oracle boundary

The positive 3D transport result used known poses as an upper bound. EXP-006 must replace this with predicted online pose/map alignment before the method can be called deployable.
