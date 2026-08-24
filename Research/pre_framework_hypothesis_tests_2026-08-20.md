# Framework 전 가설 검증: streaming TTT adaptation reuse

실험 backbone은 tttLRM이며, 평가는 A 구간으로 적응한 뒤 B 구간을 통과했을 때 A의 held-out novel-view PSNR이 얼마나 변하는지로 한다. 모든 cross-episode 주입은 target의 A+B fast weight에 apply-only로 수행했다. 따라서 이는 memory/retriever 구현 결과가 아니라, 그것이 정당화되는지 묻는 oracle probe다.

## 결론

**현재의 greedily trained tttLRM fast weight를 그대로 장기 메모리 값으로 쓰는 가설은 기각한다.**

다만 update의 부호를 반전한 작은 rank-8 correction은 일부 실제 재방문 pair에서 matched source가 foreign source보다 좋은 현상을 보였다. 7개 방향의 pilot에서는 그 평균 이득이 matched 대비 foreign으로 +0.056 dB였지만, 95% bootstrap CI는 [-0.145, +0.245] dB이고 one-sided paired Wilcoxon p=0.234였다. 즉 **희미한 신호는 있으나, 재사용 가능성을 주장할 근거는 아직 없다.**

따라서 다음 연구 가설은 다음으로 제한해야 한다.

> Reusable adaptation may exist only as a **meta-learned, signed and context-conditioned residual coordinate**.  It is not an intrinsic property of the current TTT fast-weight delta.

이 결과로는 memory bank, online consolidation, retrieval network를 구현하지 않는다. 먼저 recurrent-revisit outer objective가 이 신호를 유의하고 큰 효과로 바꾸는지를 검증해야 한다.

## 1. Forgetting과 실제 revisit pair 검증

- 기존 oracle profile: DL3DV 35/35 scene에서 A→B 후 A의 held-out PSNR이 평균 약 4.47 dB 하락했다.
- nuScenes `scene-0001`과 `scene-0011`의 A camera-centre trajectory 최소 거리는 **0.14 m**이며 foreign `scene-0705`와는 약 518 m다.
- 28개의 nuScenes episode와 756 directional cross pairs를 실제 A trajectory 거리로 재분석했다. 2 m 이하 물리적 재방문은 36 directional pair다.

따라서 실험이 '같은 도시'가 아니라 실제로 같은 위치를 지나가는 episode를 포함한다는 점은 확인됐다.

## 2. Raw cross-episode update transfer

정방향 rank-8 prior update는 두 방향의 가장 명확한 revisit pair에서 모두 해로웠다.

| Target ← matched | λ=0.01 | λ=0.02 |
| --- | ---: | ---: |
| scene-0011 ← scene-0001 | −0.222 dB | −0.584 dB |
| scene-0001 ← scene-0011 | −0.075 dB | −0.331 dB |

rank 1, 4, 8, 16, 32 sweep도 `0001→0011`에서 matched 방향의 positive transfer를 만들지 못했다. 그러므로 실패 원인은 단순히 rank=8의 압축 오차가 아니다.

## 3. Layer-wise test

기존 within-stream localisation에서는 foreign correction의 손상이 layer 1, 2, 7에 집중됐다. 하지만 `0001→0011`의 cross-episode injection에서 이 layer만 쓴다고 matched transfer가 생기지 않았다.

- L1, λ=.05: matched −0.035 dB
- L7, λ=.05: matched −2.284 dB
- L1+L2+L7, λ=.05: matched −2.392 dB

즉, scene-specific information이 특정 layer에 있다는 사실과 그 layer의 prior update가 재사용 가능하다는 사실은 다르다.

## 4. Signed residual control

source update의 부호를 반전한 rank-8 residual을 λ=−.01로 넣으면 `0001↔0011`에서 positive transfer가 재현됐다.

| Target ← matched | matched | foreign | random | matched − foreign |
| --- | ---: | ---: | ---: | ---: |
| scene-0011 ← scene-0001 | +0.171 | −0.050 | +0.012 | +0.221 |
| scene-0001 ← scene-0011 | +0.145 | +0.037 | −0.000 | +0.108 |

그러나 물리적으로 재방문한 추가 5개 directional pair를 포함하면 matched 효과는 이질적이다.

- matched: 평균 +0.029 dB, 5/7 positive
- foreign: 평균 −0.027 dB, 4/7 positive
- matched − foreign: 평균 +0.056 dB, 4/7 win
- matched − foreign의 95% bootstrap CI: [−0.145, +0.245] dB

이는 **부호와 residual parameterisation이 중요한 후보 변수**임을 보이지만, 재사용 memory의 feasibility proof는 아니다. 일부 개선은 foreign/random도 공유하므로 global shrink/regularisation 효과일 수 있다.

## 5. Context key와 gate

기존 28-unit matrix에서 pose-regime descriptor distance는 raw borrowed penalty와 관계가 있었지만, target/source main effect를 제거한 interaction-only compatibility는 leave-one-unit-out linear key로 예측되지 않았다 (rho=−.102, p=.080).

또한 leave-one-out global prototype + regime residual control은 λ=.05에서:

- global: 평균 −0.266 dB
- global+correct-regime: 평균 −0.503 dB
- global+wrong-regime: 평균 −0.304 dB

으로 regime residual이 global prototype보다 나빴다. 현재 descriptor는 safe retrieval gate로 사용할 수 없다.

## 결정과 다음 go/no-go 실험

현재는 framework 제작 단계가 아니다. 다음 최소 intervention은 bank 없이, 반복 장소 stream을 직접 사용한 **revisit-aware meta-training**이다.

\[
\mathcal L = \mathcal L_{\mathrm{current}}(A,B) + \beta\,\mathcal L_{\mathrm{heldout}}(A'; W_{A\to B}+h_\phi(\Delta_A,c_{A'}))
\]

여기서 학습 대상은 작은 signed residual map \(h_\phi\), TTT projection/learning rate이고, retrieval bank는 아직 없다. 이 intervention은 다음을 동시에 만족해야 진행한다.

1. bank-off forgetting을 줄이는 것과 별개로 bank-on residual이 held-out A' 품질을 회복한다.
2. 동일 evaluation protocol에서 recovery가 사전등록 기준 20% 이상이다.
3. matched−foreign gain이 held-out physical locations에서 유의하게 양수다.
4. foreign penalty가 사라지지 않는다. 사라지면 모든 skill이 global constant로 collapse한 것이다.

그렇지 않으면 연구 단위를 parameter update에서 scene state, calibration, correspondence memory로 바꾼다.

## Reproducible artifacts

- `tttLRM/oracle/run_cross_episode.py`: layer subset, rank, signed scale oracle.
- `tttLRM/oracle/analyze_revisit_matrix.py`: scene label 대신 pose overlap을 쓰는 revisit matrix analysis.
- `tttLRM/oracle/results/cross_episode/`: raw, layer, rank, signed controls.
- `tttLRM/oracle/results/revisit_matrix_analysis.json`: 28-episode physical-revisit aggregate.
