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
