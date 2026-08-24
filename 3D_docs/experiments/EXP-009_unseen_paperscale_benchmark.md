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

### Stage-9 result

The primary passed all checks. Its candidate OOF Spearman was **0.299**; top-K oracle utility was **+0.03253** and routed utility **+0.02012** versus matched-random +0.02642/+0.01614. Component-bootstrap improvements were [+0.00410, +0.00719] oracle and [+0.00164, +0.00586] routed, with routed harm equal to the random median at 10.55%. DINO-only was negatively associated with utility (-0.187), while the transport descriptor (+0.281) and adaptation history (+0.379) carried the signal. The transport-descriptor ablation also had the best routed point estimate (+0.02039) and lower harm (8.72%).

## Stage-10 source-entity leakage correction

Stage 9 held out target components, but one source memory can occur in training pairs for other targets. Correct this by leave-one-location-out evaluation: for a held location, remove every training pair whose target **or source** belongs to that location, then test all target pairs from the held location. This prevents direct source-entity exposure and tests geographic transfer.

The exactly factorable 256-D transport-descriptor pair features are primary; their linear score can later be compiled into a 64-D maximum-inner-product address. DINO, adaptation history, and all-observable features remain fixed ablations. The primary must have positive pooled and per-location utility association, positive component-bootstrap gains over matched random for oracle and routed utility, and no higher routed harm.

### Stage-10 result

The source-safe correction passed all checks. The transport-descriptor address retained pooled OOF Spearman **0.203**, with every unseen location positive (0.182–0.235). It achieved **+0.03149** oracle top-K and **+0.01933** routed utility with 9.17% harm. Relative to matched random, the component-bootstrap gains were [+0.00380, +0.00630] oracle and [+0.00037, +0.00512] routed. DINO-only remained negative. This authorizes the factorized utility address, but not yet a capacity policy.

## Stage-11 source-safe capacity-8 replay

Before opening validation, replay a separate true-time stream per official location. For each held location, train the transport-descriptor address and source-history retention score only on pairs whose target and source are both outside that location. Compare an unbounded utility-addressed bank with capacity-8 history retention, FIFO, and deterministic reservoir sampling. Every policy uses K=5 and the same transported code/router.

History retention passes only if it retains at least 90% of unbounded routed utility, beats FIFO and reservoir, has no more harm than FIFO, and has a positive component-bootstrap lower bound over FIFO. If it fails, choose the safest passing simple retention policy rather than tuning the history score.

### Stage-11 result

History retention failed: although it reached +0.01747 routed utility, harm was 12.84%, it lost to deterministic reservoir (+0.01768, 8.72% harm), and its FIFO interval crossed zero. Reservoir-8 was also above FIFO-8 by +0.00313 with component CI [+0.00078, +0.00590], and was descriptively above the unbounded bank while using eight records. Per the registered fallback, deterministic reservoir sampling is the locked retention policy; no learned eviction score is tuned further.

## Stage-12 deployable lock and validation freeze

Fit the transport-descriptor Ridge on every train-only Stage-8 pair and compile it exactly into a 64-D maximum-inner-product score. Fit the fixed PCA-16 Ridge router on all train-only Stage-5 candidates and select its single threshold from component-OOF train predictions under the visual-harm constraint. Freeze the atom checkpoint, 0.10 residual, reservoir capacity 8, K=5, artifact hash, and one canonical direction for each of the 117 unseen validation overlaps before any validation pixel is decoded.

The one-shot validation compares reservoir-8 utility-MIPS with FIFO-8, an unbounded utility-address bank, and random K=5 addressing within the same reservoir bank. The primary must keep current/base at most 1.05, exceed +0.01 routed utility, retain at least 90% of unbounded utility, beat FIFO without more harm, beat the matched random address, and have a positive 17-component interval over random. Failure is reported without validation tuning.

### Stage-13 one-shot validation result

The locked full-model gate failed, but isolated the failure to capacity. Current TTT was healthy (current/base **0.715**). The unbounded utility-MIPS bank generalized strongly at **+0.02642** routed utility and 5.83% harm. Reservoir-8 reached **+0.01937** and beat FIFO-8 (+0.01359), but retained only **73.3%** of unbounded utility, had slightly higher harm than FIFO (8.74% vs 7.77%), and its advantage over same-bank random addressing had a component interval crossing zero. No model or threshold is changed based on this result.

## Stage-14 validation capacity selection

Validation is now exposed for model selection; locked test remains unopened. Keep the atom, utility-MIPS address, visual transport, residual strength, router, threshold, retention algorithm, and K fixed. Evaluate deterministic reservoir and FIFO at capacities {8,16,32,64}, plus random K=5 addressing inside each reservoir bank. Select the smallest capacity with routed utility above 0.01, at least 90% retention of the same unbounded bank, harm no higher than unbounded, utility above same-capacity FIFO and random address, and a positive component-bootstrap lower bound over random. If none passes, bounded consolidation is not locked for test.

### Stage-14 result

Capacities 8, 16, and 32 retained 65.3%, 82.5%, and 89.1% of unbounded routed utility and failed. Capacity **64** was the smallest passing value: routed utility **+0.02647**, retention **100.2%**, harm **5.83%**, FIFO-64 +0.02606, random address +0.01923, and the component interval over random was **[+0.00026, +0.01097]**. Capacity 64 is now fixed; no model parameter changed.

## Stage-15/16 final test lock

Freeze one metadata-selected direction for each of 117 locked test overlaps across 22 components and four locations. Copy the already frozen train artifact while changing only the validation-selected capacity to 64, record a new hash, and commit the pilot/hash before decoding any test pixel. The final test uses the same gates as capacity selection plus current-TTT health and at least 20 components. This is a one-shot terminal evaluation; no test outcome may alter the method.

The pilot manifest SHA-256 was `eed965238af588a581874a0d5628641f1f4036caf01c22df8c303ec93727e440`. The serialized final artifact SHA-256 was `b5c7c1d0ce46ee078a1ea6d890a16e7bc5cc217a2a3777ee578d2b34ed760e98`. Both hashes, the metadata-only pilot audit, and the evaluator were committed before the test cache was created.

### Stage-16 terminal test result

The one-shot test evaluated 104 unique target contexts grouped into 22 physical overlap components across all four locations. Current TTT remained healthy with a mean current/base future-loss ratio of **0.7018**.

| Policy | Oracle top-K utility | Routed utility | Harm | Accept |
|---|---:|---:|---:|---:|
| Reservoir-64 utility address | +3.356% | **+2.088%** | **3.85%** | 86.54% |
| FIFO-64 utility address | +3.271% | +2.068% | 3.85% | 86.54% |
| Unbounded utility address | +3.280% | +2.068% | 3.85% | 86.54% |
| Same-reservoir random address | — | +1.602% | 5.77% median | 75.56% mean |

Reservoir-64 retained **100.98%** of unbounded routed utility. Its paired component difference over the matched random address was **+0.475 percentage points**, with 95% CI **[+0.086, +0.924]**. Every one of the eight registered terminal checks passed.

Reservoir exceeded FIFO by only **+0.020 percentage points**, with 95% CI **[-0.013, +0.059]**. Therefore Stage 16 supports utility-conditioned addressing in a bounded bank, but it does not support a claim that reservoir is intrinsically better than FIFO. One of 22 components was harmful after routing, so negative-transfer prevention is improved but not solved.

### Conclusion

EXP-009 is complete. The fully unseen terminal test supports the static-revisit thesis that observable, utility-supervised addressing can retrieve reusable local TTT experience beyond a matched random policy while keeping a causal capacity bound. It rejects generic place similarity as the memory objective and narrows the continual-learning contribution to bounded retention rather than learned eviction or a universally optimal retention rule.

No further tuning or selection is permitted on EXP-009 test. The next experiment must use a new development protocol to measure absolute reconstruction metrics, resource cost, pose adaptation, or dynamic 4D extension.

### Final artifacts

- Final config: `configs/EXP-009_final_test_v24.yaml`
- Metadata pilot: `revisit3d/manifests/exp009_test_pilot_v24.json`
- Lock audit: `revisit3d/results/EXP-009/stage15_final_test_lock_v24.json`
- Terminal compact result: `revisit3d/results/EXP-009/stage16_final_locked_test_v24.json`
- Large test geometry and candidate caches remain outside Git under `revisit3d/cache/EXP-009/`.
