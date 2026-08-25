# FastVGGT Mechanism Candidate Snapshot

Date: 2026-08-25
Status: Archived, immutable mechanism-proof baseline
Source commit before archival record: `a84df8b6f0dcffd4ee114c1f9e69f01d734b6782`

## Scope

This snapshot closes the first Revisit3D research branch: a compact streaming
3D adaptation-memory mechanism built on frozen FastVGGT features and a custom
geometry head. It is retained as a reproducible mechanism proof. It is not the
absolute-geometry carrier for the next paper branch.

## Frozen method

- frozen FastVGGT feature/tracker carrier at 224 x 224;
- one 3D-track online loss and one local-code step with `eta=0.0125`;
- one 8-D per-token plasticity code;
- visual cross-view transport with residual `alpha=0.10`;
- one factorized 64-D Ridge metric-utility address, top-1 retrieval, and
  semantic-zero fallback;
- deterministic reservoir capacity 64 per official-location stream;
- no fine router, risk head, learned threshold, learned eviction, second TTT
  step, pose adaptation, or dynamic 4D state.

## Frozen artifacts

| Artifact | SHA-256 |
| --- | --- |
| `revisit3d/checkpoints/exp006_geometry_bootstrap_v22.pt` | `d06f09ccb9353e8a9bdb131c0afd140a7cdbc7d77b1d72f0488ecae3e345b04d` |
| `revisit3d/checkpoints/exp028_safeguarded_pareto_atom_v10.pt` | `3ebf194f3a28876014e46d1d3bbdbcd1422cfb8ebdba48f3d16635520ca787ae` |
| `revisit3d/checkpoints/exp029_metric_utility_address_v10.joblib` | `d8b81fff36d5cb5635c194a63b422edf700c0683b7f7eb2d477be67091430984` |
| `FastVGGT/ckpt/model_tracker_fixed_e20.pt` | `b08a43baa2db1aad9718e71e098831b8ad32f6f6826c802e9eb714aa34420969` |

The FastVGGT source revision is
`6526e275a29572653a034762bb3c6c9ce280ff55`.

## Decisive evidence

- EXP-030: on 217 development targets and 25 components, full memory beat
  current-only, matched random, and appearance retrieval on SILog, aligned
  AbsRel, and 3D EPE with positive component intervals throughout.
- EXP-031/032: the immutable method evaluated the maximum 187 causally eligible
  official-test targets across all 29 components. Full memory improved all
  three mean metrics over all controls. The registered 190-target gate was an
  infeasible accounting requirement, so this remains qualified positive
  terminal evidence rather than a literal gate pass.
- EXP-033: learned additions contain 288,386 parameters. Post-foundation full
  memory costs about 1.996 ms, and a bank of 64 records stores 38.52 MiB tensor
  payload.
- EXP-035: without TUM fitting, full memory reached 28.4625 SILog, 0.230136
  aligned AbsRel, and 0.458924 m 3D EPE, improving current-only, random, and
  appearance means across the three-sequence descriptive transfer set.
- EXP-036: CUT3R reached 16.6067 / 0.081164 / 0.239378 m and TTT3R reached
  15.7271 / 0.078073 / 0.224615 m on the same 111 TUM query targets. The
  custom geometry head is therefore not competitive enough for a broad
  top-tier reconstruction-framework claim.

## Frozen interpretation

The experiments support the causal mechanism claim that spatially local,
transported adaptation experience addressed by future geometry utility can
improve an identical reconstruction carrier over current-only, random-address,
and appearance-address controls. They do not support a state-of-the-art
reconstruction, pose, dynamic-4D, general risk-calibration, or second-backbone
claim.

## Recovery

The archival Git ref is
`archive/revisit3d-fastvggt-exp036-20260825`. The independent recovery bundle
and binary artifacts are stored under
`/home/khm/3D_4D_backups/revisit3d_fastvggt_exp036_20260825/`.

The next branch must use a new experiment ID and fresh development/held-out
data. It may reuse the scientific principle and code interfaces, but it may not
reinterpret the exposed nuScenes or TUM results as new model-selection data.
