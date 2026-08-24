# EXP-009 — Fully Unseen Paper-Scale Revisit Benchmark

Status: **Stage 6 completed; local reuse transfers, causal DINOv2 retrieval registered next.**

## Question

Can the selected dual-address local-plasticity architecture generalize to an independently constructed, component-disjoint benchmark whose scenes were never used in EXP-001–008?

## Stage-0 boundary

Stage 0 reads nuScenes metadata only. It does not open camera images, foundation features, depth predictions, utility labels, or any existing holdout result.

- Blacklist every scene directory under the three previously converted roots and every scene named by an existing manifest.
- Keep unseen CAM_FRONT scenes with at least 200 frames.
- Compare scenes only within the same official nuScenes location.
- Create an undirected edge when any two camera centers are within 2.0 m.
- Split only by connected overlap component; no scene may cross train/validation/test.
- Use deterministic greedy balancing of undirected edge counts within each location at 70/15/15.
- Record both a full ignored inventory and a compact tracked audit.

If connected components are too large for a credible split, Stage 0 must stop and redesign the sampling unit using metadata only. It may not inspect images or model performance to repair the split.

## Stage-0 result

After blacklisting 130 previous scenes, 719 unseen scenes remained. Metadata-only matching found 1,368 undirected overlap edges over 636 scenes and 65 connected components. The fixed split produced:

- train: 1,134 undirected / 2,268 directional episodes, 454 scenes;
- validation: 117 / 234 episodes, 86 scenes;
- locked test: 117 / 234 episodes, 96 scenes.

Scene and component intersections are zero. Validation and test each contain components from all four official locations. The largest 304-scene/936-edge component is isolated in train.

## Stage-1 manifest freeze

Generate both directions of every fixed edge with eight context and four disjoint query frames. A/A′ use a 15%-length window around the closest pose anchors; B uses the opposite temporal end of the source traversal. Before writing the manifest, assert unique episode IDs, valid/disjoint indices, blacklist exclusion, minimum split size, four-location holdout coverage, and zero scene/component intersections. No image or model output is read.

## Outputs

- Config: `configs/EXP-009_unseen_benchmark_inventory_v10.yaml`
- Script: `revisit3d/scripts/build_exp009_unseen_inventory.py`
- Full inventory: `revisit3d/cache/EXP-009/unseen_overlap_inventory_v10.json`
- Summary: `revisit3d/results/EXP-009/stage0_unseen_overlap_inventory_v10.json`
- Manifest config: `configs/EXP-009_unseen_manifest_v11.yaml`
- Frozen manifest: `revisit3d/manifests/nuscenes_revisit_unseen_exp009_v11.json`

## Stage-1 result

The frozen manifest passed every registered check: 2,268/234/234 directional episodes, 454/86/96 scenes, 26/17/22 components, four locations in every split, no blacklist intersection, and zero scene/component leakage. Its SHA-256 is `682cca8796e5cb321ae8f02efc90f8eea495bdb93e24a1db8afee9bc64d6e13f`.

## Stage-2 conversion boundary

Convert `opencv_cameras.json` metadata for exactly the 636 frozen-manifest scenes. The converter reads calibration, ego pose, timestamps, and file paths but does not decode image pixels. Creating metadata for validation/test scenes does not open those holdouts; subsequent feature extraction must explicitly restrict itself to the train split until a new model is locked.

## Stage-2 result

Exactly 636 scene metadata files were generated: 454 train, 86 validation, and 96 test, with no missing or extra scene. The audit records `image_pixels_accessed=false` and `model_output_accessed=false`. Validation/test pixels and features remain unopened.

## Stage-3 train-only key pilot

Before full geometry caching, select up to 64 undirected positive train edges per location by a fixed hash. Require at least 40 positives from every location. Pair each positive with one same-location negative whose scene bounding boxes are at least 20 m apart and which has no direct 2 m overlap edge. Use four fixed context views. This pilot chooses the long-term consolidation representation only; it does not change the learned local transport key.

The frozen selection contains 225 positive and 225 negative pairs over 247 train scenes. Compare four-view token-set statistics from VGGT mean-patch descriptors and DINOv2 ViT-L/14 CLS descriptors with leave-one-location-out logistic evaluation. DINOv2 is selected only if its OOF AUC exceeds VGGT by at least 0.03 and every held-out location reaches AUC 0.70; otherwise VGGT remains the consolidation fallback. Validation/test pixels remain unopened.

## Stage-4 result

- VGGT token-set OOF AUC: **0.744**; per-location range 0.702–0.772.
- DINOv2 token-set OOF AUC: **0.936**; per-location range 0.900–0.971.
- DINOv2 margin: **+0.193 AUC**.
- DINOv2 pooled-cosine AUC without the classifier: 0.917.

The registered DINOv2 gate passed. DINOv2 is selected only for long-term consolidation/prefiltering. VGGT remains the reconstruction/TTT backbone, and the learned plasticity key remains the local code-transport address. Stage 4 measures geometric place compatibility, not causal adaptation utility; the selected DINOv2 key must next improve a causal bank on new train scenes.

## Stage-5 locked local-reuse transfer

Freeze one direction of each of the 225 Stage-3 positive edges as a geometry pilot. Without retraining, apply the locked EXP-006 custom plasticity head, exactly one TTT step, visual code transport, 0.10 residual, and locked utility router. Candidate pools remain matched A, distant B, and three deterministic foreign contexts. This isolates whether local TTT/reuse and utility ranking transfer to fully unseen train scenes before DINOv2 is introduced into a causal bank.

The gate requires mean current/base ratio at most 1.05, visual-mean utility above 0.01 with less than 10% deadband harm, at least 10 physical components, and locked-router utility above visual mean without higher harm. Failure requires retraining the local head/router on new train; success authorizes isolating the DINOv2 bank contribution.

### Stage-5 result

Current TTT remained healthy (mean current/base 0.682), visual mean retained +0.01227 utility, and the candidate oracle reached +0.02600. The locked old router improved utility to +0.01695, but harm rose to 14.22% versus visual mean 11.11%. Thus local plasticity/reuse transferred, while the old router failed the new-distribution safety gate.

## Stage-6 nested router correction

Keep the head, TTT, candidate pool, features, and residual fixed. Retrain only the same compact PCA-16 Ridge utility router with outer leave-one-physical-component-out evaluation. For each outer fold, choose the acceptance threshold using inner leave-one-component-out predictions from outer-train components only. The threshold maximizes utility subject to no more harm than visual mean and at least 20% acceptance. The outer primary must beat visual mean without higher harm and have a positive component-bootstrap lower bound.

### Stage-6 result

Nested component calibration reduced harm relative to the ungated new router and the old locked router. The primary reached **+0.01475** mean utility, **10.67%** harm, and **83.56%** acceptance versus visual mean at **+0.01227** utility and **11.11%** harm. Its paired component-bootstrap difference over visual mean was +0.00238 with 95% CI **[-0.00014, +0.00465]**. The registered gate therefore failed only the strictly positive lower-bound check. No further threshold tuning on this candidate set is permitted.

## Stage-7 causal DINOv2 retrieval

Test the selected consolidation address by adaptation utility rather than place-pair AUC. Replay all unique A/B/A′ contexts from the 225 train episodes in true nuScenes capture-time order, evaluate each unique A′ target before writing it, and retrieve K=5 causal memories using frozen DINOv2, the learned VGGT-side transport key, FIFO, and a deterministic-random control. Query frames remain utility labels only and never enter retrieval, adaptation, or router features.

The primary diagnostic is oracle utility contained in each top-K set; the secondary diagnostic applies the already registered nested component router. DINOv2 passes only if its oracle top-K utility exceeds the VGGT transport-key top-K utility, its routed mean utility also exceeds VGGT, and its routed harm is no higher. This stage isolates address quality and does not yet impose a capacity-bounded consolidation policy.

### Stage-7 result

DINOv2 improved over the learned VGGT transport key: oracle top-K utility was **+0.02412** versus **+0.01958**, and routed utility was **+0.01208** versus **+0.00889**. The paired component intervals for DINO minus VGGT were [+0.00385, +0.00588] oracle and [+0.00125, +0.00744] routed. However, routed harm was 10.55% versus VGGT's 10.09%, failing the registered safety check by one target. More importantly, the single deterministic-random top-K reached +0.02632 oracle and +0.01663 routed utility, while DINO retrieval score had no positive association with candidate utility (Spearman -0.038, p=0.210). DINO is therefore better than the inappropriate transport-key address, but causal utility beyond chance is not established.

## Stage-8 matched random-retrieval null

For every Stage-7 target, draw a fixed uniform panel of up to 64 records from the exact causal bank and evaluate their locked transported-code utility. From each panel, simulate 2,000 independent random K=5 retrieval policies. This estimates the random-policy distribution without changing the DINO model, router, residual strength, stream, or target set.

DINO is supported as a causal utility address only if both its oracle-top-K and routed mean utility exceed at least 95% of matched random policies, both paired component-bootstrap differences over the per-target random expectation have positive lower bounds, and DINO routed harm does not exceed the median random-policy harm. Failure retires generic place compatibility as the consolidation objective and motivates a utility-supervised prefilter key.

### Stage-8 result

The matched null decisively rejected DINOv2 as a plasticity-utility address. Across 2,000 random K=5 policies, random oracle utility was **+0.02642** and routed utility **+0.01614**, both above DINO's +0.02412/+0.01208; both one-sided p-values were 1.0. Component-bootstrap DINO-minus-random intervals were **[-0.00474, -0.00019]** oracle and **[-0.00772, -0.00011]** routed. Harm matched the random median (10.55%), so safety did not rescue the weaker utility. Generic place compatibility remains an optional feature/control, not the long-term memory address.

## Stage-9 utility-supervised observable prefilter

Use the fixed Stage-8 panels to test whether future adaptation utility is predictable before expensive code transport. The input contains only frozen DINO pair statistics, current/source learned transport descriptors, and their past/current TTT objective histories; transported-code statistics and query/future quantities are excluded. Fit a fixed StandardScaler→Ridge(alpha=1) model by leave-one-physical-component-out cross-fitting. The all-observable 274-D input is primary; DINO-only, transport-descriptor-only, history-only, and DINO+history are fixed ablations.

Rank each panel by the OOF prediction, prefilter K=5, and evaluate both oracle utility in that set and the unchanged nested utility router. The primary passes only with positive OOF score/utility association, positive component-bootstrap lower bounds over the matched random expectation for oracle and routed utility, and routed harm no higher than the Stage-8 random median. This is a feasibility test of a utility-conditioned pair scorer, not yet a scalable dual-encoder address.
