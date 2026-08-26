# Novelty Audit — Revisit3D

Last updated: 2026-08-25

## Closest 3D systems

| Work | What it remembers | Main overlap | Remaining distinction |
|---|---|---|---|
| CUT3R (CVPR 2025) | persistent recurrent scene state | streaming geometry with fixed state | does not retrieve prior adaptation experiences by causal future utility |
| Point3R (NeurIPS 2025) | explicit world-space spatial pointers | explicit local 3D memory | stores scene content/features, not transported local test-time corrections |
| TTT3R (ICLR 2026) | relevance-weighted recurrent state update | casts state update as TTT and controls forgetting | changes how one accumulated state is written; no episodic plasticity bank or future-utility address |
| tttLRM / ZipMap / Scal3R (CVPR 2026) | observations compressed into fast-weight scene state | TTT as long-context 3D memory | fast weights represent scene content; ours retrieves reversible context-local corrections learned from prior adaptation |
| STAC (CVPR 2026) | compressed spatio-temporal KV cache | bounded streaming memory | token/cache retention rather than adaptation-experience retrieval |
| Mem3R (2026 preprint) | fast-weight pose memory plus explicit geometry tokens | hybrid TTT and explicit memory | task/state decoupling, not a bank addressed by cross-context adaptation utility |

Primary sources: [CUT3R](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.html), [Point3R](https://proceedings.neurips.cc/paper_files/paper/2025/hash/650db8e1b0b016dc270d51c1476e91cf-Abstract-Conference.html), [TTT3R](https://openreview.net/forum?id=aMs6FtNaY5), [tttLRM](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_tttLRM_Test-Time_Training_for_Long_Context_and_Autoregressive_3D_Reconstruction_CVPR_2026_paper.html), [ZipMap](https://openaccess.thecvf.com/content/CVPR2026/html/Jin_ZipMap_Linear-Time_Stateful_3D_Reconstruction_via_Test-Time_Training_CVPR_2026_paper.html), [STAC](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_STAC_Plug-and-Play_Spatio-Temporal_Aware_Cache_Compression_for_Streaming_3D_Reconstruction_CVPR_2026_paper.html), and [Mem3R](https://arxiv.org/abs/2604.07279).

## Closest adaptation/retrieval ideas

- T3AR (CVPR 2023) retrieves examples/features to construct a test-time adaptation objective. It does not retrieve learned local corrections or model their cross-context future geometry utility.
- Continual TTA methods such as PETAL, RoTTA, and parameter-selective adaptation preserve a shared model under distribution shift. They do not provide a spatially transported episodic adaptation object.
- ReGrad (2026 preprint) is a direct conceptual collision with the generic phrase “retrievable gradients”: it stores offline document gradients and meta-learns query-relevant temporary language-model adaptation. Revisit3D must not claim the generic retrievable-gradient paradigm as novel.

Primary sources: [T3AR](https://openaccess.thecvf.com/content/CVPR2023/html/Zancato_TrainTest-Time_Adaptation_With_Retrieval_CVPR_2023_paper.html), [PETAL](https://openaccess.thecvf.com/content/CVPR2023/html/Brahma_A_Probabilistic_Framework_for_Lifelong_Test-Time_Adaptation_CVPR_2023_paper.html), and [ReGrad](https://arxiv.org/abs/2606.15734).

## Multi-objective and gradient-consensus collision audit

The post-EXP-025 branch cannot claim generic gradient alignment, gradient
surgery, common descent, or plasticity-constrained TTA as new:

- PCGrad (NeurIPS 2020) projects conflicting task gradients; CAGrad (NeurIPS
  2021) optimizes worst local task improvement while retaining an average-loss
  convergence target; Nash-MTL (ICML 2022) derives a bargaining solution.
- Aligned-MTL (CVPR 2023) stabilizes multi-task gradients by aligning
  orthogonal components and evaluates depth and surface normals.
- GraTa (AAAI 2025) explicitly aligns pseudo and auxiliary gradients during
  TTA and uses their cosine to control the online learning rate.
- ConFIG (ICLR 2025 Spotlight) constructs updates with positive dot product
  against every loss-specific gradient.
- CoCo-MT-TTA (AAAI 2026) is the closest naming and conceptual collision: it
  performs multi-task test-time gradient consensus inspired by CAGrad and adds
  a second-moment plasticity constraint.

Primary sources: [PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html), [CAGrad](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html), [Nash-MTL](https://proceedings.mlr.press/v162/navon22a.html), [Aligned-MTL](https://arxiv.org/abs/2305.19000), [GraTa](https://ojs.aaai.org/index.php/AAAI/article/view/32244), [ConFIG](https://arxiv.org/abs/2408.11104), and [CoCo-MT-TTA](https://ojs.aaai.org/index.php/AAAI/article/view/40006).

The remaining distinction is precise: Revisit3D uses multiple metric endpoint
gradients only offline to meta-learn one local code-to-geometry plasticity map.
Online inference still has one self-supervised gradient and no endpoint labels,
consensus solver, or extra task. Consequently, the optimizer is an enabling
mechanism; the paper-level novelty remains spatial adaptation objects,
cross-view transport, causal future-utility addressing, and physical-revisit
evaluation.

## Defensible novelty

The paper should claim only the conjunction below:

1. **Spatial adaptation object:** a reusable correction is a per-token local plasticity code, not a whole-model gradient or scene feature.
2. **Cross-view transport:** the correction is explicitly moved into the current token coordinate system before application.
3. **Utility address:** correctness is learned from causal future geometry improvement, not place identity, RGB similarity, gradient cosine, or reconstruction confidence.
4. **Revisit evaluation:** physical-overlap components and matched random-address controls test whether the system reuses adaptation rather than generic warm starts.

No single item is sufficient alone. The novelty is the geometry-specific formulation and its empirical/theoretical validation as one mechanism.

## Reviewer attack surface

- “This is only retrieval over features.” Counter only if utility address beats same-bank random and appearance/place controls on absolute geometry.
- “This is tttLRM/TTT3R with an external memory.” Counter by showing stored local corrections are reversible episodic experiences, not one accumulated scene state, and by cross-context transfer ablations.
- “Future frames leak into inference.” Counter with immutable query-read-only contracts and source-safe folds.
- “The method improves its own proxy but not reconstruction.” EXP-010 is the decisive test.
- “Too many heuristics.” Remove rejected branches, present one utility retrieval module, and run minimal-loss/hyperparameter sensitivity.
- “Continual learning is cosmetic.” Claim bounded causal reuse and retention only; do not claim catastrophic-forgetting theory or learned consolidation superiority.

## Venue assessment

- **CVPR path:** strongest if absolute depth/point/pose-compatible metrics, standard streaming baselines, qualitative reconstruction, and efficiency all hold.
- **ICLR path:** requires broader evidence that utility-addressed plasticity is a general learning principle, ideally a second backbone/task and stronger calibration/generalization analysis.

Given the current implementation and desire for one compact paper, the project proceeds on the CVPR path first.

## Current-only conditional-tangent audit (2026-08-26)

The post-EXP-053 current-only branch cannot claim any of the following ideas in
isolation:

- meta-learning a model so a self-supervised test-time step improves the target
  task (MT3; DocTTT);
- amortizing or learning an optimizer from test-time feedback (Rapid Network
  Adaptation);
- generating or correcting test-time gradients with a learned function and
  gradient history (MGTTA/GGTTA);
- continual test-time adaptation inside a low-dimensional parameter subspace
  (PACE);
- interpreting TTT as a recurrent streaming 3D state update (TTT3R).

Primary sources: [MT3](https://proceedings.mlr.press/v151/bartler22a.html),
[DocTTT](https://openaccess.thecvf.com/content/WACV2025/html/Gu_DocTTT_Test-Time_Training_for_Handwritten_Document_Recognition_using_Meta-Auxiliary_Learning_WACV_2025_paper.html),
[Rapid Network Adaptation](https://openaccess.thecvf.com/content/ICCV2023/html/Yeo_Rapid_Network_Adaptation_Learning_to_Adapt_Neural_Networks_Using_Test-Time_ICCV_2023_paper.html),
[MGTTA/GGTTA](https://arxiv.org/abs/2412.16901),
[PACE](https://arxiv.org/abs/2603.28678), and
[TTT3R](https://arxiv.org/abs/2509.26645).

The provisional distinction is therefore intentionally conjunctive: a frozen
streaming 3D recurrent model, one unchanged unlabeled geometry-consistency
step, and one per-spatial-token tangent metric at the official geometry
readout whose offline RGB-D supervision only shapes how that same step moves.
The candidate is not described as a learned optimizer because it neither
consumes gradients nor emits parameter updates. It is not described as generic
subspace TTA because its hypothesis concerns spatially varying geometry-token
tangent axes. EXP-054/055 must demonstrate that token conditioning, rather
than merely extra parameters, is causally necessary.
