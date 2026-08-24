# Revisit3D: framework 이전 가설 검증 결과

이 문서는 `revisit3d/`의 **memory-free** 사전 실험 기록이다. tttLRM의
fast-weight나 기존 bank를 사용하지 않았고, test split은 이 단계에서 사용하지
않았다.

## 현재 실험 조건

- frozen VGGT aggregator feature + 새 `StreamingGeometryHead(features, z)`;
- test-time에 갱신되는 객체는 32차원 compact state `z`뿐;
- 실제 cross-traversal `A → B → A'` (physical-overlap component split);
- depth/pose scale 퇴화를 분리하기 위한 controlled condition에서는 frozen VGGT의
  camera/intrinsics를 online reprojection에 사용했다. 이는 최종 deployment 설계가
  아니라 objective-health counterfactual이다.

## 관찰

| 질문 | 결과 | 해석 |
|---|---:|---|
| 무학습 custom head의 online loss가 식별 가능한가? | 실패 | predicted pose의 relative translation은 0.018, depth std는 0.0092였고, dataset-scale known pose와 depth 0.25–1 m에서는 valid reprojection이 0%였다. 유효 pixel 밖으로 투영하면 loss가 0이 되는 퇴화 해가 존재했다. |
| matched A–A′는 실제 geometric revisit인가? | 통과 | train에서 A→A′ 2 m camera-centre coverage 84.0%, A→B 25.0%; frozen feature cosine은 matched 0.9834 > B 0.9730 > scene-disjoint foreign 0.9614. |
| 새 head의 geometry prior를 bootstrap하면 online update가 생기는가? | 통과 | frozen VGGT depth pseudo-label bootstrap 32 step 후 depth std 0.2894, state update norm 0.0199. Pseudo-label은 초기화용이며 online input/최종 평가 target이 아니다. |
| raw matched update가 B/foreign보다 정렬되는가? | 실패 | train 32 directional episodes에서 matched cosine 0.1983, B 0.2166, foreign 0.2111. matched > B는 16/32뿐. |
| update가 저차원인가? | 통과 | realised 32-D updates의 PC1=79.1%, PC1–2=96.9%, effective rank=1.94. 하지만 이는 context-specific skill이 아니라 global online-loss mode일 수 있다. |
| global component 제거 뒤 signal이 남는가? | 탐색적 신호 | global mean을 제거한 train residual에서 matched–foreign cosine=+0.1918; PC1도 제거하면 +0.2198. 아직 causal utility를 뜻하지 않는다. |
| matched residual이 held-out A′ loss를 foreign보다 낮추는가? | 미통과/불충분 | val의 방향 중복 2개에서 mean-only residual은 −0.00168, PC1-removed residual은 +0.00475 (loss는 낮을수록 좋음). 표본이 너무 작고 PC1 결과가 반대이므로 claim 불가. |
| held-out utility로 직접 학습한 oracle transport가 통과하는가? | 실패 | frozen head에서 32-step transport 학습 뒤 val matched는 current보다 +0.00821, foreign보다 +0.000155 나빴다. matched/foreign transport prior cosine은 0.999895로, transport가 prior를 거의 무시한 global correction으로 collapse했다. |

관련 산출물:

- `revisit3d/results/reprojection_geometry_val.json`
- `revisit3d/results/depth_bootstrap_signal_val.json`
- `revisit3d/results/context_separation_train.json`
- `revisit3d/results/depth_bootstrap_alignment_train.json`
- `revisit3d/results/depth_bootstrap_update_subspace_train.json`
- `revisit3d/results/depth_bootstrap_residual_alignment_train.json`
- `revisit3d/results/oracle_{mean_,}residual_reuse_val.json`
- `revisit3d/results/oracle_utility_transport_epoch1_val.json`

## 수정된 가설

기존의 강한 가설인 *"유사 context의 raw TTT update는 재사용 가능하다"*는 이
설정에서 지지되지 않는다. 현재 유지할 수 있는 더 약한 관찰은 다음뿐이다.

> TTT update는 지배적인 global adaptation mode와 작은 context-dependent residual을
> 함께 포함할 수 있다. 그러나 residual similarity 자체는 reusable utility의 증거가
> 아니며, utility supervision만으로도 transport가 prior-agnostic global correction으로
> collapse할 수 있다.

## 다음 go/no-go gate

따라서 현 compact-state/reprojection-only formulation에서는 bank, key, retrieval,
CL consolidation을 만들면 안 된다. 먼저 TTT objective 자체를 바꿔야 한다.

다음 formulation이 통과해야 할 **memory-free oracle** gate는 다음과 같다.

1. generic photometric reprojection이 아니라 **tracked correspondence cycle,
   depth/point-map cross-view consistency, pose uncertainty**로 구성된 TTT objective가
   local geometric residual에 실제로 민감함을 먼저 보인다.
2. 새 objective의 realised update가 matched A–A′에서 B/scene-disjoint foreign보다
   더 유사하고, oracle injection이 `matched < current` 및 `matched < foreign`을
   validation에서 모두 만족해야 한다. 효과 크기와 paired CI를 보고한다.
3. 이 gate를 통과한 뒤에만 oracle matched prior를 geometry-conditioned retrieval로
   바꾸고, 마지막에 test split을 단 한 번 사용한다.

즉 다음 연구 질문은 "어떤 bank를 쓸까"가 아니라 **"어떤 geometry-aware online
objective와 adaptation coordinate가 local, reusable correction을 생성하는가"**다.
