# Revisit3D v1: independent foundation-model streaming geometry

## Fixed first scope

The first paper does **not** modify tttLRM and does not begin with a memory
bank.  It studies cross-episode, physically revisited locations in a stream:

\[
A \rightarrow B \rightarrow A'
\]

where A and A' are separate captures of the same road segment and B is a
different temporal part of the source traversal.

The v1 output space is token-level pointmap, depth, confidence, and view-level
relative pose. Point tracking / dynamic 4D is an extension after the static
revisit mechanism is established.

## Architecture boundary

```text
Frozen VGGT patch tokens (no pretrained output heads)
        ↓
New StreamingGeometryHead(features, z)
        ↓
pointmap / depth / confidence / relative pose
        ↑
compact test-time state z only
```

`z` is a 16--32 dimensional, explicitly stored state. It FiLM-modulates only
the new head. Test-time adaptation does not mutate VGGT or head weights.
Therefore the state whose transfer is tested is exactly the state which could
later become a memory value; it is not an opaque full-model delta.

## What exists now

- `revisit3d/data/revisit_benchmark.py`: pose-overlap graph, component-safe
  train/validation/test split, directional A→B→A' manifest.
- `revisit3d/data/dataset.py`: RGB/intrinsic/camera loader that preserves the
  distinction among A, B and A' context/query frames.
- `revisit3d/backbones/vggt.py`: frozen 2048-D FastVGGT patch-token extractor.
- `revisit3d/models/geometry_head.py`: new compact-state geometry head and an
  adaptation interface that updates only `z`.

The current nuScenes subset produces 40 directional physical-revisit episodes
at <= 2 m overlap, with no scene in both train and test components.

## Next implementation gate

Before any retrieval/memory work, implement and evaluate two states on held-out
physical locations:

1. `z_current`: online self-supervised TTT from the current A'/context;
2. `z_revisit`: initialization/residual derived from the matched prior A under
   an oracle pairing.

The outer training objective is revisit-aware:

\[
\mathcal L_{\rm outer} = \mathcal L_{\rm current} +
\lambda_{\rm revisit}\mathcal L_{\rm geometry}(A'; z_{A\rightarrow B}+h_\phi(z_A,c_{A'})).
\]

`h_phi` begins as a signed low-rank residual map.  There is no retrieval key,
bank write rule, or continual-loss term until oracle matched reuse is both
positive and larger than foreign/random controls on held-out locations.

The first implementation uses **first-order meta-TTT**: gradients of the
photometric reprojection update are detached, but gradients from the transported
initial state to the held-out A' objective are preserved. Full second-order
MAML through CUDA `grid_sample` is not supported and is not required for this
initial go/no-go.

## Supervision discipline

Held-out depth/point/pose metrics may use nuScenes LiDAR and poses. They must
not be used by the online TTT objective. The latter will be limited to
cross-view reprojection, point/depth consistency, confidence regularisation and
track consistency when the 4D extension is introduced.
