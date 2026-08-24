# EXP-006 — Trainable 3D Plasticity Atoms and Risk-Aware Routing

## Status

In progress. Expanded train-only OOF model selection is complete and the v2.8 one-shot validation lock is active as of 2026-08-25. Exact OOF evidence supports visual local-code reuse and regularized utility routing, rejects predicted 3D alignment as carrier/primary router input, and rejects the current explicit neural risk classifier. Official validation remains unopened.

## Question

Can a trainable local plasticity code remain causally reusable without oracle state, and can current-observable appearance, adaptation, and geometry evidence route that memory with less regret and negative transfer than simple controls?

## Hypotheses

- H1: a transported local code is more reusable than global or untransported update state.
- H2-P: predicted geometry plus appearance outperforms visual-only transport. This was tested and rejected.
- H2-E: predicted geometry remains useful as routing evidence.
- H4-U/H4-R: learned utility/risk routing lowers grouped harm and future-utility regret without reject-all collapse.

## Authoritative protocol

[`EXP-006 Implementation Brief.md`](../EXP-006%20Implementation%20Brief.md), v2.7 architecture addendum over the preserved v2.6 protocol.

Key safeguards:

- Stage-0 predicted depth/pose/confidence health gate before atom training.
- K=5 utility-conditioned Stage-1 training; matched episode identity is not a label.
- First-order meta-gradient with detached discrete correspondence, inlier, Sim(3), and neighbor selection.
- Train-only neutral deadband, normalization, class weights, and grouped five-fold model selection.
- One-shot official validation; the exposed test split is rejected by the CLI.
- Undirected scene-pair/overlap-component grouped bootstrap inference.

## Data

- Manifest: `revisit3d/manifests/nuscenes_revisit_dev.json`
- Train: 20 directional episodes / 10 pairs / 8 overlap components.
- Validation: 14 directional episodes / 7 pairs / 2 overlap components.
- Test: 6 exposed directional episodes / 3 pairs / 1 component; closed and prohibited.

## Planned configuration and outputs

- Config: `configs/EXP-006_atom_utility.yaml`
- Geometry checkpoint: `revisit3d/checkpoints/exp006_geometry_bootstrap_v22.pt` (created; excluded from Git)
- Compact results: `revisit3d/results/EXP-006/`

## Planned primary metrics

- normalized future-loss change versus current-only TTT
- grouped-bootstrap confidence interval
- cluster and directional harm rates
- future-utility regret and accept/reject rate
- alignment validity, residual, inlier ratio, and transport retention
- runtime and peak GPU memory

## Result

### Stage-0 v2.1 identity-gate diagnostic — preserved failure

Result: `revisit3d/results/EXP-006/stage0_geometry_health_train_v21_identity_gate_failed.json`

- Passed 8/9 checks: finite/positive depth, view-0 identity, rotation, translation direction, translation scale, confidence correlation, and depth-residual gradient health.
- Cross-fit median rotation error: 0.446°.
- Cross-fit median translation-direction error: 4.598°.
- Cross-fit median scale-aligned translation error: 0.250.
- Component-mean confidence Spearman: 0.546.
- Predicted/identity track-loss ratio: 1.098, so the registered identity gate failed.
- A train-only counterfactual found the teacher/identity ratio was also greater than one. This identifies identity-pose degeneracy rather than failed pose distillation; see D011.

### Stage-0 v2.2 teacher-retention gate — passed

Result: `revisit3d/results/EXP-006/stage0_geometry_health_train_v22.json`

- All 9 corrected health checks passed on 20 directional train episodes grouped into 8 overlap components.
- Cross-fit median rotation error: 0.446°.
- Cross-fit median translation-direction error: 4.598°.
- Cross-fit median scale-aligned translation error: 0.250.
- Component-mean confidence Spearman: 0.546.
- Predicted/teacher track-loss ratio: 0.927, below the registered 1.05 ceiling.
- Teacher/identity ratio: 1.184, confirming the preserved identity degeneracy diagnostic.
- Positive depth fraction and healthy depth-residual gradient fraction: 1.0.

Stage 0 supports using the custom predicted pose/depth/confidence as the frozen EXP-006 base. It does not yet support H2-P or H4; those require Stage 1/2 and the unopened validation protocol.

### Stage-1 v2.2 cross-view-scale diagnostic — preserved failure

Result: `revisit3d/results/EXP-006/stage1_predicted_transport_health_train_v22_crossview_scale_failed.json`

- Predicted Sim(3) was evaluated only on the 20 directional train episodes; validation and the closed test split were not accessed.
- Matched A→A' alignment validity was 0.70, below the 0.80 preflight gate.
- Current-context→future-query alignment validity was 1.00 and every valid transport was finite.
- The failure was traced to pooling all eight overlapping views before median 8-NN scale estimation. Cross-view duplicate observations made the metric-normalization bandwidth artificially small.
- A train-only, single-variable counterfactual using within-view 8-NN yielded matched validity 1.00 without changing any Sim(3) threshold. D012 and protocol v2.3 register this definition before validation.

### Stage-1 v2.3 query-alignment diagnostic — preserved as invalid

Result: `revisit3d/results/EXP-006/stage1_predicted_transport_health_train_v23_query_alignment_prohibited.json`

- All 20/20 matched A→A' alignments and 20/20 current-context→future-query alignments were valid.
- Median matched correspondence count was 188.5, median inlier ratio 0.925, and median normalized residual 0.889 local spacings.
- All valid geometry+appearance transports were finite.
- Distant-B validity was 0.90 and deterministic foreign-candidate validity was 0.467. Alignment validity is therefore a geometric feasibility mask, not a relevance or utility label; candidate selection must remain utility-conditioned as specified by D005/D006.
- The diagnostic used train only and did not influence fitting or selection, but its query Sim(3) calculation violates F10. It is preserved for audit and is not EXP-006 evidence; see D013.

The v2.4 rerun removes query geometry entirely and must pass before atom training. It does not yet establish H2-P future utility or H4 routing safety.

### Stage-1 v2.4 source→current transport health gate — passed

Result: `revisit3d/results/EXP-006/stage1_predicted_transport_health_train_v24.json`

- Matched source→current validity was 1.00 (20/20), with median inlier ratio 0.925 and median normalized residual 0.889.
- All valid transports were finite and the diagnostic records `query_geometry_accessed=false`.
- Distant-B and deterministic foreign validity remained 0.90 and 0.467, respectively, so alignment remains only an eligibility mask.
- The Stage-1 training smoke produced finite outer gradients with approximately 685 MB peak allocated GPU memory. The repeated current-only null control was exactly zero for all 20 train episodes, fixing `epsilon_u=0.01` as pre-registered.

The leakage-safe transport and differentiable atom paths are healthy enough to begin train-only grouped CV. Validation remains unopened.

### Stage-1 v2.4 relative-objective diagnostic — preserved failure

Result: `revisit3d/results/EXP-006/stage1_atom_training_train_v24_relative_objective_failed.json`

- Train-only CV selected 1,000 steps because mean held-out-train best-valid utility rose from 0.501 at 500 steps to 0.664 at 1,000 steps.
- This was a degenerate relative improvement. Full-train mean current-query loss rose from 0.0610 at initialization to 0.5838, a 9.57× degradation, while apparent best-valid utility reached 0.777.
- The cause is gradient through the current reference in `softplus(L_benefit-ell_current)`, which rewards worsening the reference loss.
- The result cannot support H2-P and Stage 2/validation were not opened. D014 registers the v2.5 absolute-quality guard.

### Stage-1 v2.5 safe-objective diagnostics — partial, reuse application failed

Results:

- `revisit3d/results/EXP-006/stage1_atom_training_train_v25.json`
- `revisit3d/results/EXP-006/stage1_transport_ablation_train_v25.json`
- `revisit3d/results/EXP-006/stage1_atom_collapse_train_v25.json`
- `revisit3d/results/EXP-006/stage1_transport_kernel_train_v25.json`
- `revisit3d/results/EXP-006/stage1_reuse_strength_train_v25.json`
- `revisit3d/results/EXP-006/stage1_objective_health_train_v25.json`

Findings:

- The current/base guard worked: CV current/base was 0.835 at 500 and 0.741 at 1,000 steps; the 1,000-step full refit reached 0.436.
- Offline known-pose track loss also improved to 0.771× base on average and improved in 19/20 episodes, supporting the custom current TTT signal.
- Full-state geometry+appearance initialization was harmful: mean utility −0.399, 14% beneficial, and 82% harmful among valid candidates. Visual transport was also negative, while the global-vector baseline was +0.0249 mean utility with only 5% harm.
- The spatial code had not vanished (`code_token_std=0.116`, residual token std 0.271), but source global directions were highly aligned (pairwise cosine median 0.998). Harder k-NN transport worsened utility, so simple over-smoothing was not the main cause.
- Applying memory as a bounded residual after current TTT fixed much of the interference. At strength 0.10, geometry+appearance changed to +0.0103 mean utility and 10% harm. This motivates v2.6/D015; H2-P remains unsupported until the additive form is trained and compared.

### Stage-1 v2.6 train-only grouped CV — passed with a generalization warning

Results:

- `revisit3d/results/EXP-006/stage1_atom_training_train_v26.json`
- `revisit3d/results/EXP-006/stage1_crossfit_heads_train_v26.json`

The grouped five-fold selection chose 1,000 steps: mean held-out-component current/base was 0.879 and best-valid utility was 0.0264, versus 0.867 and 0.0240 at 500 steps. The fixed 1,000-step full-train refit reached current/base 0.389 and best-valid utility 0.0076.

The large full-fit/OOF gap means same-train refit ablations are not valid architecture evidence. D016 therefore makes exact fold-specific heads authoritative. Full-fit diagnostics are retained rather than overwritten.

### Stage-1 v2.6 exact OOF transport ablation — visual carrier selected

Results:

- `revisit3d/results/EXP-006/stage1_transport_ablation_crossfit_train_v26.json`
- `revisit3d/results/EXP-006/stage1_visual_pool_crossfit_train_v26.json`
- `revisit3d/results/EXP-006/stage1_router_features_crossfit_train_v26.json`

Each episode was evaluated with the head trained without its physical-overlap component. The mean current-only/base ratio was 0.8647.

| Reuse condition | Mean utility | Beneficial | Harmful | Coverage |
|---|---:|---:|---:|---:|
| global vector | +0.0021 | — | 0% | 100% |
| untransported local | −0.0014 | 32% | 34% | 100% |
| visual local transport | +0.0162 | 65% | 2% | 100% |
| predicted geometry | +0.0161 | 58.3% | 6.25% | 48% |
| predicted geometry+appearance | +0.0162 | 52.1% | 8.33% | 48% |
| five-candidate visual mean | +0.0165 | 80% | 0% | 100% |
| oracle best visual candidate | +0.0323 | 95% | 0% | 100% |

Predicted geometry neither improved the mean nor safety and lost more than half the candidate coverage. H2-P is rejected. Visual correspondence becomes the primary code carrier in v2.7; predicted alignment statistics remain router evidence under H2-E.

Candidate identity was also non-causal: matched A averaged +0.0095, while distant/foreign candidates averaged roughly +0.019 to +0.022. This strengthens D005/D018: correct retrieval means high future utility, not recovery of the designated scene pair.

### Geometry upper-bound diagnostics — preserved but non-authoritative

Results:

- `revisit3d/results/EXP-006/stage1_oracle_transport_gap_train_v26_metric_pose_mixed_gauge_invalid.json`
- `revisit3d/results/EXP-006/stage1_oracle_transport_gap_train_v26.json`
- `revisit3d/results/EXP-006/stage1_lidar_transport_gap_train_v26.json`
- `revisit3d/results/EXP-006/stage1_centered_atom_transport_train_v26.json`

The first diagnostic mixed metric nuScenes pose with arbitrary VGGT depth gauge and is explicitly invalid. Context-only pose-gauge calibration remained unstable because near-zero predicted translations produced scale ratios up to approximately 588. A leakage-safe LiDAR context-only upper-bound and source-centering counterfactual did not recover an advantage for geometry transport. These tests used a full-fit head on the same train episodes and are optimization diagnostics only; the exact OOF ablation is the decision evidence.

### Stage-1 adaptation-budget control — memory is not an extra gradient step

Result: `revisit3d/results/EXP-006/stage1_adaptation_budget_crossfit_train_v26.json`

- One current step/base: 0.8647.
- Two current steps/base: 0.9580; relative utility −0.120, benefit 40%, harm 60%.
- Five-memory visual mean after one step: +0.0165, benefit 80%, harm 0%.
- Five-memory mean before one current step: +0.0187, benefit 85%, harm 0% (secondary diagnostic; no learned router was tested in this order).
- Oracle candidate after one step: +0.0323, benefit 95%, harm 0%.

This supports D020: the reusable memory effect is not explained by simply spending another TTT step.

### Stage-2 fixed grouped-OOF router feasibility — supported, not a final model

Results:

- `revisit3d/results/EXP-006/stage2_router_similarity_controls_crossfit_train_v26.json`
- `revisit3d/results/EXP-006/stage2_router_bootstrap_crossfit_train_v26.json`

The router features contain 256 appearance interaction values and 16 current-observable scalars. Query/future quantities are targets only and are absent from input. Fixed PCA-16 + ridge(alpha=1) was evaluated OOF without hyperparameter selection.

| Selector | Selected utility | Harm | Regret vs oracle |
|---|---:|---:|---:|
| full observable ridge | +0.0224 | 0% | 0.0099 |
| full minus geometry | +0.0202 | 0% | — |
| online scalars | +0.0265 | 5% | 0.0059 |
| geometry only | +0.0205 | 0% | — |
| current-objective heuristic | +0.0165 | 5% | — |
| appearance similarity | +0.0075 | 5% | — |
| matched A | +0.0095 | 5% | — |
| random candidate expectation | +0.0162 | — | — |
| visual mean | +0.0165 | 0% | — |

Candidate-level Spearman was 0.246 for the full model and 0.436 for online scalars. The full model traded some utility for the observed zero-harm result. Geometry contributed a modest safe increment over the no-geometry feature set, supporting H2-E only provisionally.

An overlap-component bootstrap with 10,000 samples gave full-router mean utility 0.0223, 95% CI [0.0159, 0.0273]. Full minus random was +0.0060, CI [+0.0002, +0.0106]; full minus visual mean was +0.0057, CI [−0.0004, +0.0103]. Only 8 groups are available, so this is architecture feasibility rather than conference-level statistical evidence.

### Stage-2 risk-label health — failed; neural risk training gated

Result: `revisit3d/results/EXP-006/stage2_router_label_health_crossfit_train_v26.json`

At deadband 0.01, the visual candidate set contains 65 beneficial, 33 neutral, and 2 harmful labels. Both harmful rows are from episode `scene-0246__scene-0675` in fold 4. Consequently, the fold-4 router-training partition has zero harmful examples and all harmful evidence comes from one physical-overlap component.

The utility head is implementable and the fixed linear probe is informative, but a grouped neural risk result would not be identifiable. Under D019, validation and persistent-memory work remain closed while the train revisit benchmark is expanded.

## Current conclusion

EXP-006 has established three train-only facts:

1. One-step local TTT improves the custom base and stores a reusable signal.
2. Visual transport of that local signal is causal and safer than untransported reuse; predicted 3D alignment should condition routing rather than carry the signal.
3. Observable utility routing has headroom over simple retrieval controls, but current risk labels are too concentrated to train the final risk head honestly.

The next operation is benchmark expansion and exact repetition of the OOF label-health/visual-routing pipeline. Official validation will not be opened merely to obtain more harmful examples.

### v2.7 benchmark expansion — passed without holdout contamination

Results/configuration:

- `revisit3d/results/EXP-006/benchmark_expansion_train_v27.json`
- `revisit3d/manifests/nuscenes_revisit_expanded_v27.json`
- `revisit3d/manifests/nuscenes_exp006_all_locations.json`
- `configs/EXP-006_utility_router_v27.yaml`

Pose/location metadata from 130 already-converted nuScenes scenes produced 52 undirected overlaps in 22 components. All components touching original validation/test scenes were excluded from expanded training. The resulting split contains 76 directional train episodes across 19 components, while the original 14 validation and 6 exposed-test episodes are copied unchanged. Train/holdout scene intersection is empty. No held-out image or model output was read during expansion.

### Expanded v2.7 visual atom training and exact OOF reuse — passed

Results:

- `revisit3d/results/EXP-006/stage1_atom_training_expanded_train_v27.json`
- `revisit3d/results/EXP-006/stage1_crossfit_heads_expanded_train_v27.json`
- `revisit3d/results/EXP-006/stage1_router_features_crossfit_expanded_train_v27.json`

Grouped CV selected 1,000 steps: current/base improved from 0.780 at 500 steps to 0.680 at 1,000, and group oracle-best utility rose from 0.0132 to 0.0385. Exact fold-head evaluation yielded current/base 0.6622.

| Condition | Mean utility | Benefit | Harm | Coverage |
|---|---:|---:|---:|---:|
| global vector | +0.0027 | 1.05% | 0% | 100% |
| untransported local | +0.0045 | 42.37% | 22.37% | 100% |
| visual local | +0.0180 | 63.68% | 4.47% | 100% |
| predicted geometry | +0.0152 | 60.59% | 5.20% | 70.79% |
| geometry+appearance | +0.0164 | 61.71% | 8.55% | 70.79% |
| visual mean | +0.0184 | 75.0% | 3.95% | 100% |
| oracle best visual | +0.0331 | — | 0% | 100% |

This larger result confirms D017 more strongly: visual correspondence carries the code, while predicted geometry reduces coverage, utility, and safety.

### Expanded risk-label health — passed

Result: `revisit3d/results/EXP-006/stage2_router_label_health_crossfit_expanded_train_v27.json`

There are 242 beneficial, 121 neutral, and 17 harmful visual candidates. Harm occurs in seven episodes across three held-out folds, and every outer training partition contains benefit and harm. Risk is now identifiable as a supervised target, so Stage-2 failure cannot be attributed solely to absence of labels.

### Adaptation-history feature revision and regularized router — passed

Results:

- `revisit3d/results/EXP-006/stage2_router_feasibility_crossfit_expanded_train_v27.json`
- `revisit3d/results/EXP-006/stage2_router_bootstrap_crossfit_expanded_train_v27.json`

Eight leakage-safe source/current statistics were added: source TTT post/pre and loss drop, source track coverage/pre/post residual, and current coverage/pre/post residual. The locked no-alignment feature set contains descriptor interactions plus 20 visual/adaptation-history scalars.

| Selector | Utility | Benefit | Harm | Regret |
|---|---:|---:|---:|---:|
| locked no-alignment Ridge | +0.0280 | 82.89% | 2.63% | 0.0051 |
| same + predicted alignment | +0.0282 | 81.58% | 2.63% | 0.0049 |
| descriptor only | +0.0275 | 81.58% | 1.32% | 0.0057 |
| online/history only | +0.0293 | 84.21% | 7.89% | 0.0038 |
| geometry only | +0.0184 | 77.63% | 6.58% | 0.0147 |
| visual mean | +0.0184 | 75.0% | 3.95% | — |
| current objective | +0.0138 | 40.79% | 2.63% | 0.0193 |
| appearance similarity | +0.0159 | 55.26% | 7.89% | 0.0172 |
| matched identity | +0.0147 | 55.26% | 9.21% | — |

The locked router's gain over visual mean is +0.00985 with overlap-component bootstrap 95% CI [0.00634, 0.01452]. Its gain over random is +0.01018 [0.00661, 0.01497], and over matched identity +0.01350 [0.00888, 0.01988]. Predicted alignment adds only +0.00016 and no safety, so D022 excludes it.

### Neural utility/risk ablation — explicit risk claim rejected

Results:

- `revisit3d/results/EXP-006/stage2_neural_router_crossfit_expanded_train_v27.json`
- preserved initial contracts: files suffixed `_16scalar_initial.json` and `_two_variant_initial.json`

The neural appearance+adaptation-history router reached risk AUROC 0.688 but selected +0.0257 utility with 3.95% harm. Adding predicted alignment increased risk AUROC to 0.718 but reduced utility to +0.0247 and increased harm to 5.26%. Appearance-only was +0.0240 with 5.26% harm. Risk threshold 0.5 rejected almost nothing useful/harmful consistently.

The result supports utility observability but not a separate risk-classifier claim. The compact regularized utility model generalizes better and is the only model admitted to one-shot validation under D023.

## Pre-validation lock

The validation model, features, normalization, PCA rank, ridge alpha, utility threshold, candidate set, visual transport, one-step TTT, and reuse strength are frozen in D023 and the v2.8 brief. Validation may now be evaluated exactly once. No validation result may be used to modify this EXP-006 model.

D024 additionally freezes the descriptive go/no-go checks before validation cache/model-output access. The train-only final router artifact and its hashes must be recorded before the guarded validation command is run.
