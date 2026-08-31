# EXP-070 — Spatiotemporally Distal Fast-Weight Interference

Status: **Completed; complete gate failed**

## Question

After a released 4D LaCET memory has enough evidence to answer a past-time
query, do later observations from a distant time change that identical query in
visually stable image regions, despite elastic consolidation?

## Why this experiment is necessary

The broad project idea is directly occupied by FSM: it already combines TTT
fast weights, EWC-inspired continual consolidation, and long-context 4D novel
view-time reconstruction. A new framework is unjustified unless the released
carrier exhibits a narrower functional failure not measured by FSM.

H26 tests functional output interference, not parameter drift. Failure stops
the direction before any locality mechanism, head, loss, bank, or router.

## Frozen source

- Dataset: fixed-camera PStudio NPZ files distributed with TAPVid-3D/OpenD4RT.
- Names available before deserialization: 156.
- The three files whose array schemas were inspected during feasibility work
  are excluded: `basketball_7`, `football_6`, `boxes_29`.
- SHA-256 ranking with seed `EXP-070-role-freeze-v1` freezes 24 premise, 16
  validation, and 16 terminal sequences.
- Manifest: `revisit3d/results/EXP-070/role_manifest_v10.json`.
- Canonical manifest SHA-256:
  `0e8d0e672260ee603bbe08caed82d8bcbe5dfb9a2c53e03a22526927e8c1f3fb`.
- Only the 24 premise files may be opened in EXP-070.

## Frozen carrier

- Official FSM repository commit
  `499464ecd971dc096cc9a27d197aa0b5995f123a`.
- Released 128px 4D-LVSM checkpoint SHA-256
  `4cedb490e4cbfcbade3ed26745b9c63f32d7d37bbaf15df600708571c0a48ee4`.
- Fast-weight chunk size `2048`, matching eight 128px views per chunk.
- Released LaCET settings: SI-style importance, proximal weight `0.5`, Fisher
  EMA `0.5`, streaming-EMA anchor `0.5`.
- LaCT control disables only EWC; weights, inputs, chunks, and decoder remain
  identical.

## Frozen A→B→A protocol

The camera is static, so all `c2w` matrices are identity and the supplied
intrinsics define the query rays.

- Query: RGB frame 32, never included in a write.
- Core A: 16 even frames from 0 through 30.
- Near C: 16 odd frames from 1 through 31.
- Distant B: 16 even frames from 64 through 94.

Five conditions are evaluated independently from identical initial weights and
fresh EWC buffers:

1. `A_only_LaCET`;
2. `A_plus_near_LaCET` (32-view compute/context control);
3. `A_plus_distant_LaCET` (retention condition);
4. `A_plus_distant_LaCT` (matched non-elastic control);
5. exact replay of condition 1.

No target RGB, future metric, or mask enters a fast-weight update. The target
pose/time-only token is apply-only, as in official FSM inference.

## Stable-region definition

For evaluation only, each target pixel receives the median RGB L1 difference
between target frame 32 and the 16 distant B frames. The bottom quartile is the
stable region and the top quartile is the changing region. These fixed
quantiles are not routing inputs and will not be searched.

## Primary measurements

- A-only PSNR and masked MSE;
- stable-region relative damage of distant versus A-only;
- stable-region relative damage of distant versus near;
- number of sequences with positive damage;
- stable output drift from the A-only prediction for distant versus near;
- stable-to-changing damage ratio;
- descriptive LaCET versus LaCT difference.

Sequence-bootstrap 95% percentile intervals use 10,000 resamples and seed
`70070`.

## Complete pass gate

All gates in
`configs/EXP-070_fastweight_distal_interference_v10.yaml` must pass. The central
requirements are at least 2% stable-region relative damage with a positive
interval against both A-only and matched near context, positive damage in at
least 18/24 sequences, and at least 1.25x distant/near stable output drift with
a positive drift-difference interval.

Success authorizes only one compact locality mechanism using the existing FSM
objective. Failure closes this paper premise without frame, quantile, EWC,
checkpoint, or sequence repair.

## Result

EXP-070 ran once on all 24 frozen premise sequences. No validation or terminal
file was deserialized, no model or threshold was fit, the checkpoint loaded
strictly, and exact replay had maximum absolute difference `0.0`.

| Measurement | Result |
|---|---:|
| Mean A-only PSNR | `32.7180 dB` |
| Stable damage, distant vs A-only | `-29.3231%` |
| 95% sequence-bootstrap CI | `[-38.1298%, -20.0596%]` |
| Sequences worsened by distant vs A-only | `3/24` |
| Stable damage, distant vs near | `+39.6957%` |
| 95% sequence-bootstrap CI | `[+5.2143%, +83.1891%]` |
| Sequences worsened by distant vs near | `13/24` |
| Distant/near stable output-drift ratio | `0.5514x` |
| Drift-difference 95% CI | `[-1.034e-4, -1.583e-5]` |
| Mean changing-region damage, distant vs A | `-16.8936%` |
| LaCET stable MSE advantage over LaCT | `-2.660e-6` |
| Complete gate | `8/15`, failed |

The central prediction is reversed. Distant observations improve stable-region
MSE relative to A-only by 29.3% on the sequence-balanced mean, and worsen only
3/24 sequences. Near evidence improves more (`-35.33%` mean relative damage),
so distant is worse than near on average, but this is evidence-quality
difference rather than retention loss. Distant output also moves only `0.551x`
as far from the A prediction as near evidence.

The registered stable-to-changing ratio is numerically inapplicable because
both registered damages are improvements; the immutable JSON reports the
signed denominator-clamped value. This accounting detail cannot rescue the
premise because all three direct distant-vs-A gates and both drift gates fail
in the opposite direction.

Conclusion: **H26 is rejected.** The released carrier does not exhibit the
required distal forgetting phenomenon under the frozen protocol. No local
fast-weight partition, router, bank, extra loss, EWC tuning, validation run, or
terminal run is authorized.

Artifacts:

- result:
  `revisit3d/results/EXP-070/fastweight_distal_interference_v10.json`
- result SHA-256:
  `918ee93ad7083400b6aef46d89aa29472137b40b23c94a81247974224431ba6f`
- config SHA-256:
  `0ae81c3204f93c1efc62cd220816c62f70af6264c415f9ce6fadb8926159fa05`
- evaluator:
  `revisit3d/scripts/evaluate_exp070_fastweight_distal_interference.py`
