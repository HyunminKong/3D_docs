# Causal Fast-Weight 4D Novelty Audit

Last updated: 2026-08-27

## Question

Can the next paper claim a new combination of test-time training, continual
learning, persistent streaming state, and queryable 4D reconstruction?

## Direct collision

The broad combination is **not novel in August 2026**.

| Work | Occupied capability | Consequence here |
|---|---|---|
| [ZipMap, CVPR 2026](https://arxiv.org/abs/2603.04385) | TTT fast weights compress a long image collection into a fixed-size, queryable 3D state; a streaming variant is released. | TTT as a stateful 3D memory is occupied. |
| [tttLRM, CVPR 2026](https://arxiv.org/abs/2602.20160) | Long-context and autoregressive 3D reconstruction through TTT layers. | A static TTT reconstruction backbone is not a contribution by itself. |
| [Fast Spatial Memory / FSM, ECCV 2026](https://arxiv.org/abs/2604.07350) | LaCT fast weights plus EWC-style consolidation learn a long-context 4D representation and render arbitrary novel view-time combinations. | This directly occupies `TTT + continual consolidation + 4D`. |
| [Point4Cast, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Point4Cast_Streaming_Dynamic_Scene_Reconstruction_and_Forecasting_CVPR_2026_paper.html) | A persistent latent spacetime state is updated by streaming frames and queried for past, present, or future pointmaps and cameras. | Causal persistent state plus time-conditioned 4D readout is occupied without TTT. |
| [StreamVGGT, ICLR 2026](https://arxiv.org/abs/2507.11539) | Causal attention and historical KV cache provide real-time streaming 4D geometry. | Generic causal 4D streaming is occupied. |
| [4RC / S-4RC, ICML 2026](https://arxiv.org/abs/2602.10094) | Encode-once, query-anywhere/anytime 4D reconstruction; the supplement adds a causal online STream3R-based variant. | Arbitrary source/time querying and its streaming extension are occupied. |
| [UniQuery4R, 2026](https://arxiv.org/abs/2608.17283) | One clip encoding supports sparse continuous source-point/target-time queries, correspondence, 3D position, flow, depth, and per-view camera. | Sparse 4D point querying is occupied in the offline setting. |

FSM is the decisive collision. It explicitly describes fully plastic LaCT
updates as an inference-time catastrophic-forgetting problem, maintains fast
weights and anchor weights, tracks an online Fisher-style statistic, and uses a
streaming-EMA anchor. A new paper cannot present EWC, an EMA anchor, or generic
stability--plasticity balancing in a 4D TTT model as its novelty.

## Remaining narrow gap

FSM's consolidation is parameter-wise but **query independent**. Every
spatiotemporal write updates the same shared MLP matrices and the diagonal
importance/shrinkage is global to those parameters. The paper evaluates novel
view synthesis and camera-interpolation behavior, but does not audit whether a
later, spatiotemporally distant observation changes an already supported past
query in regions that are visually stable.

For an old query `a` and a write induced by evidence `b`, first-order functional
interference is

\[
\delta f(a) \simeq J_\theta f(a)\,\Delta\theta_b.
\]

Diagonal EWC bounds selected coordinates of `Delta theta`, but it does not make
the overlap `J_theta f(a) Delta theta_b` local in spacetime. This limitation is
not a new continual-learning principle: distal interference and functional
overlap are generic CL concepts. The potentially defensible contribution is
therefore narrower:

> establish and reduce **spatiotemporally distal functional interference in a
> released 4D fast-weight reconstruction model**, while keeping one fast state,
> one existing reconstruction objective, and fixed memory.

This is only a candidate boundary. It requires a fresh, no-fit premise before
any method is designed.

## What is prohibited

- Do not claim `TTT + CL + 4D` as the contribution.
- Do not rebuild Point4Cast, S-4RC, StreamVGGT, FSM, or UniQuery4R with renamed
  state tokens.
- Do not introduce EWC, Fisher importance, EMA anchoring, time embeddings, a
  memory bank, or sparse query decoding as standalone novelty.
- Do not infer locality failure from latent or parameter distance alone; the
  effect must appear in held-out rendered pixels at an identical query.
- Do not propose a local router or partitioned fast weights before the released
  FSM carrier exhibits material distal functional interference.

## Feasibility audit

- Official FSM code commit: `499464ecd971dc096cc9a27d197aa0b5995f123a`.
- Released `fsm_4dlvsm_patch8_res128.pth` loads strictly; SHA-256
  `4cedb490e4cbfcbade3ed26745b9c63f32d7d37bbaf15df600708571c0a48ee4`.
- The provided Stereo4D example produced mean PSNR `29.4240 dB` for eight
  128x128 queries and used `2765.83 MiB` peak allocated A100 memory.
- This is an engineering smoke test on an exposed demonstration, not scientific
  evidence and not an experiment result.

## Decision

Register one premise only: EXP-070 tests A-only, A-plus-near, and
A-plus-distant histories at an identical past-time query on fresh fixed-camera
PStudio videos, with released LaCET and matched LaCT. Failure closes this
candidate without a method. Success authorizes a single locality mechanism,
not a bank, router, new head, or additional loss.

EXP-070 subsequently failed in the opposite direction: distant evidence
improved stable-region MSE over A-only by 29.3% and produced less output drift
than near evidence. The narrow locality gap is therefore not an active method
claim under this protocol.
