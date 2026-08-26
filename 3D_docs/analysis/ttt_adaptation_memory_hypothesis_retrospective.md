# TTT Adaptation-Memory Hypothesis: Experimental Retrospective

Last updated: 2026-08-26

Scope: EXP-001--EXP-067

Decision status: compact TTT adaptation-experience reuse closed on the tested
competitive CUT3R/TTT3R carrier

## 1. 이 문서의 목적

이 문서는 처음 제안한 다음 가설이 왜 최종 논문 가설로 채택되지
못했는지를 실험적으로 추적한다.

> Streaming 3D reconstruction에서 현재 입력으로 생성한 TTT update에는
> 상황별로 재사용 가능한 adaptation experience가 들어 있다. 이를
> geometric context로 검색하고 continual-learning 방식으로 보존하면,
> 유사한 미래 상황에서 현재 TTT보다 빠르고 정확하며 안전하게 적응할
> 수 있다.

여기서 “가설이 틀렸다”는 표현은 모든 가능한 모델과 학습법에 대한
수학적 불가능성을 뜻하지 않는다. 정확한 결론은 다음과 같다.

> 우리가 정의하고 사전등록한 local-code, learned coordinate,
> exact-meta coordinate, agreement routing, missing-evidence transport,
> function-space displacement transport는 competitive CUT3R/TTT3R carrier에서
> 과거 adaptation experience의 고유하고 안정적인 미래 3D utility를
> 입증하지 못했다. 따라서 이 compact formulation을 CVPR/ICLR 논문의
> 중심 주장으로 유지할 근거가 없다.

개별 실험의 원문은 [Experiment Index](../experiments/index.md)와 각
`EXP-###` 문서에 보존되어 있다. 이 문서는 그 결과들을 하나의 인과적
논리로 재구성한 분석본이다.

## 2. 원래 가설을 검증 가능한 명제로 분해

시점 `t`의 geometric context를 `c_t`, online TTT가 만든 adaptation
record를 `a_t`, 미래 query에서의 utility를 `u(a_i,t)`라고 하자. 원래
아이디어가 성립하려면 아래 명제가 동시에 성립해야 한다.

| 하위 가설 | 필요한 현상 | 최종 판정 |
|---|---|---|
| P1: 현재 적응 가능성 | 한 번의 source-free online loss가 현재 3D를 개선해야 함 | 부분적으로 강하게 지지됨. Online loss descent는 반복 재현됨. 절대 3D 개선은 objective/head에 따라 달라짐. |
| P2: experience 특이성 | 서로 다른 상황의 update가 단순한 generic warm-start가 아니라 서로 다른 미래 utility를 가져야 함 | 초기 custom system에서는 일부 utility 분산이 있었으나 competitive carrier의 raw reuse는 실패. |
| P3: transport 가능성 | 과거 local update를 target의 대응 surface로 옮겼을 때 위치를 섞은 control보다 좋아야 함 | 최종적으로 기각. 여러 실험에서 correct transport가 shuffle과 동률 또는 열세. |
| P4: 현재 최적화 이상의 고유 가치 | memory reuse가 동일 계산량의 추가 current TTT보다 좋아야 함 | 명확히 기각. EXP-048/049에서 second-current가 memory 및 oracle fallback을 모든 scene에서 이김. |
| P5: 관측 가능한 retrieval | future/GT 없이 유용한 record를 선택하고 negative transfer를 억제해야 함 | Curated bank에서는 agreement routing이 작동했으나 full-stream/equal-compute에서 중심 효용이 사라짐. |
| P6: bounded continual retention | 제한된 bank가 과거의 유용한 experience를 보존해야 함 | 구현 가능했지만 long-term reservoir가 FIFO보다 열세. 보존 자체가 utility를 만들지 못함. |
| P7: absolute geometry | proxy loss가 아니라 SILog, AbsRel, 3D EPE를 개선해야 함 | v1은 within-backbone 개선을 보였지만 head가 공식 TTT3R보다 1.8--3.0배 나빴음. Competitive carrier의 새 code는 절대 3D에서 실패. |
| P8: information insufficiency 해결 | 현재 frame에 정보가 없을 때 과거 adaptation이 실제 관측 정보를 대체해야 함 | 기각. Explicit surface는 큰 효과가 있었지만 adaptation code는 거의 0 효과. |

원래 논문 가설에 가장 치명적인 것은 P4와 P8이다. 과거 update가 현재
frame의 추가 최적화보다 고유한 정보를 제공하지 못했고, 현재 관측에
정보가 실제로 없을 때도 update code는 그 정보를 복구하지 못했다.

## 3. 실험 설계 원칙

결론이 단순한 overfitting이나 leakage 때문이 되지 않도록 다음 원칙을
점진적으로 강화했다.

1. `A -> B -> A'`에서 future/query는 offline utility와 metric label에만
   사용하고 online TTT, write, retrieval 입력에서는 제외했다.
2. 같은 physical component가 train과 held-out role에 동시에 나타나지
   않도록 component/scene 단위 분리를 사용했다.
3. “원래 pair를 찾았는가”가 아니라 실제 future utility와 regret으로
   retrieval을 평가했다.
4. current-only, second-current, random, appearance, untransported, spatial
   shuffle을 동일 payload/acceptance/compute 조건으로 비교했다.
5. known pose, GT visibility, future oracle는 capacity evidence로만 기록하고
   deployable result와 분리했다.
6. 결과 확인 후 threshold, step, loss, frame, split을 바꾸지 않았다.
7. 각 실패가 발생할 때마다 더 복잡한 framework를 만들기 전에 다음
   prerequisite를 별도 EXP로 사전등록했다.

이 규칙 때문에 일부 결과는 평균이 양수여도 CI, harm, scene consistency,
control superiority 중 하나가 실패하면 채택하지 않았다.

## 4. 전체 실험 계보

### Phase A — 기존 tttLRM fast weight와 global update 가설: EXP-001--004

#### EXP-001: 기존 tttLRM update를 그대로 재사용할 수 있는가

- Fast-weight update에는 구조가 있었지만 matched update가 foreign update보다
  안정적으로 좋은 causal benefit을 보이지 않았다.
- 결론: tttLRM wrapper 위에 memory를 얹는 방향을 폐기하고 독립적인 fast
  state를 설계하기로 했다.

#### EXP-002: benchmark와 online objective가 유효한가

- nuScenes physical revisit episode와 query-leakage 방지 protocol을 만들었다.
- 단순 photometric reprojection은 zero-support/tiny-pose degeneracy를 보였다.
- Frozen track/camera prior를 사용하는 3D track consistency는 gradient
  probe용 online signal로 사용할 수 있었다.

#### EXP-003: global/slot compact state가 상황별 adaptation인가

- Generic past update는 current-only loss를 약 `3.5e-3` 개선했다.
- 그러나 matched/intervening/foreign update가 사실상 구분되지 않았다.
- 200-step meta-training 후 matched--foreign utility gap은 약 `1.48e-7`, code
  cosine은 거의 1이 되어 오히려 collapse가 심해졌다.
- 결론: generic warm-start와 context-specific memory를 구분해야 하며,
  global/slot vector를 central memory object로 사용할 수 없다.

#### EXP-004: context key와 update cosine으로 useful memory를 찾을 수 있는가

- Global key top-1: `0/14`, matched mean rank `5.36`.
- Local-token key: top-1 `2/14`, recall@3 `6/14`, recall@5 `7/14`.
- Learned update reranking top-1: `0/14`; positive/negative cosine이 모두 거의 1.
- Raw update cosine은 positive `0.9121`, negative `0.8554`였지만 top-1
  designated revisit은 여전히 `0/14`.
- 결론: key만의 문제가 아니라 update vector 자체의 causal discriminability가
  약했다.

### Phase B — Spatial plasticity atom의 초기 mechanism proof: EXP-005--009

#### EXP-005: 3D-addressed dense atom과 oracle utility

- Coordinate+appearance transport의 matched utility는 current보다
  `-3.76e-4`, foreign보다 `-1.06e-4`; designated utility top-1은 `4/14`.
- 하지만 전체 bank에서 online utility score를 사용하면 future loss를
  `6.05e-4` 낮췄다.
- 노출된 6-episode test에서 oracle-score selection은 `3/6` best를 선택하고
  `7.34e-4` 개선했지만 `2/6`을 해쳤다.
- 당시 해석: local transport에는 selection 가능한 utility가 있으나 안전한
  learned routing이 필요하다.
- 현재의 재해석: matched transport 자체가 좋은 것이 아니라 candidate
  panel 안에 generic하게 유용한 correction이 존재했을 가능성이 컸다.

#### EXP-006: trainable atom과 risk/utility routing

이 실험은 여러 stage를 거쳤고 초기 가설의 가능성을 가장 강하게 보였지만,
동시에 후속 실패의 원인을 이미 포함하고 있었다.

- Predicted geometry transport는 coverage와 safety를 잃었고 visual transport를
  이기지 못해 H2-P가 기각됐다.
- Exact OOF에서 visual local transport utility `+0.0162`, harm `2%`;
  five-candidate visual mean `+0.0165`, harm `0%`; oracle `+0.0323`.
- Untransported local은 `-0.0014`, harm `34%`로 local visual transport가
  분명 더 나았다.
- 그러나 matched A utility는 `+0.0095`인데 distant/foreign은 약
  `+0.019~+0.022`였다. “같은 장소의 update”가 가장 좋은 것은 아니었다.
- Source global directions의 pairwise cosine median은 `0.998`로 매우
  정렬되어 있었다. 이것은 상황별 update보다 generic correction에 가까운
  신호였다.
- One-shot v2.8 validation에서 router-minus-visual은 `+0.00378`이었지만
  router가 14개 episode를 모두 accept했다. Learned rejection/safety가
  검증된 것은 아니었다.

#### EXP-007--008: bounded bank와 true-time stream

- EXP-007 capacity-8 predicted history: utility `+0.02782`, harm `6.71%`;
  scene-latest `+0.02615`, harm `6.32%`. Utility는 높았지만 no-extra-harm
  조건을 통과하지 못했다.
- EXP-008 true-time primary: utility `+0.02650`, harm `5.63%`;
  appearance diversity `+0.02387`, harm `7.04%`.
- Primary는 matched compression null의 96.1 percentile(`p=0.03996`)이었다.
- 결론: causal bounded storage는 가능했으나, 이것은 stored adaptation의
  고유 정보성과 absolute geometry를 아직 증명하지 않았다.

#### EXP-009: paper-scale unseen proxy benchmark

- Source-safe split: train/validation/test `2,268/234/234` directional episodes,
  `26/17/22` components.
- Capacity 64가 smallest passing capacity였고 routed utility `+0.02647`,
  random address `+0.01923`.
- Terminal reservoir-64는 matched random보다 `+0.475` percentage points,
  CI `[+0.086,+0.924]`; 모든 등록 gate 통과.
- 그러나 reservoir superiority over FIFO는 입증되지 않았다.
- 중요 한계: primary endpoint가 self-supervised future utility였으며 절대
  point geometry가 아니었다.

### Phase C — Proxy utility와 absolute geometry의 충돌: EXP-010--020

#### EXP-010: 첫 absolute-geometry 검증

- Full memory는 aligned AbsRel을 current보다 target 평균 `0.00378`,
  component 평균 `0.00226`, CI `[0.00031,0.00460]` 개선했다.
- 반면 SILog는 `0.0577`, aligned RMSE는 `0.0336 m`, same-ray 3D EPE는
  `0.0118 m` 악화됐다.
- 결론: proxy utility 개선을 broad 3D reconstruction 개선으로 해석할 수
  없었다.

#### EXP-011: healthy online objective가 존재하는가

- 218 targets/25 components에서 하나의 frozen-track absolute 3D consistency,
  one step, `eta=0.0125`가 선택됐다.
- Train과 one-shot validation에서 SILog, aligned AbsRel, 3D EPE를 함께
  개선했다.
- 이것은 “TTT 자체가 불가능”한 것이 아님을 보여준다. 실패는 reuse와
  meta-objective에 있었다.

#### EXP-012--015: minimal reusable atom을 찾는 과정

- EXP-012 세 frozen-key variant 모두 실패. 가장 강한 ranking variant도
  oracle utility `+0.472%`로 1% gate 미달.
- EXP-013 trainable key: oracle `+0.521%`, mean candidate `-0.211%`, harm
  `45.38%`; overfit.
- EXP-014 1000-step budget: oracle `+0.9205%`, candidate mean `+0.4725%`,
  harm `27.33%`; 1% gate만 실패.
- EXP-015 combined objective: current/base `0.80194`, oracle utility
  `+1.04799%`, candidate mean `+0.52105%`, harm `26.37%`; 모든 OOF gate 통과.
- 당시에는 compact atom이 확립된 것처럼 보였지만 endpoint는 여전히
  track-based proxy였다.

#### EXP-016--019: observable utility address

- EXP-016 visual Ridge: utility `+0.771%`, harm `18.52%`, acceptance `94.72%`;
  random 대비 CI가 `[-0.00011,+0.00449]`로 실패.
- EXP-017 adaptation self-improvement scalar 추가: Spearman `0.1937`, harm
  `16.96%`; random superiority는 여전히 유의하지 않음.
- EXP-018 strict geometry agreement: harm `3.07%`지만 utility `+0.418%`로
  abstention이 지나침.
- EXP-019 fallback: utility `+0.8351%`, harm `18.45%`, acceptance `94.72%`;
  random 대비 CI `[+0.00041,+0.00512]`, coarse 대비
  `[+0.00010,+0.00147]`; 통과.

#### EXP-020: locked validation에서 v1 paper model 기각

103 targets/17 unseen components:

| Policy | SILog | aligned AbsRel | 3D EPE (m) |
|---|---:|---:|---:|
| no TTT | 47.2512 | 0.67589 | 5.57367 |
| current-only | 47.2797 | 0.66090 | 5.57931 |
| full memory | 47.2789 | 0.65925 | 5.57766 |
| random | 47.2820 | 0.65971 | 5.57868 |

- Full memory는 current 대비 AbsRel `0.00165` 개선, CI
  `[0.00078,0.00276]`.
- 어떤 primary LiDAR metric에서도 random 대비 positive interval을 얻지
  못했다.
- Proxy는 random보다 `0.00286`, CI `[0.00120,0.00459]` 좋았지만 harm이
  `33.01%`였다.
- Current TTT 자체도 no-TTT보다 모든 metric에서 건강하지 않았다.
- 결론: EXP-015+019 model은 논문 model로 기각됐다.

### Phase D — Metric-aligned v1 rescue: EXP-021--035

이 단계는 within-backbone memory phenomenon을 가장 엄격하게 살려냈다.
동시에 이후 EXP-036이 왜 이것만으로 논문이 될 수 없는지를 보여준다.

#### EXP-021--023: 독립 benchmark와 metric oracle

- EXP-021은 sensor/model access 없이 214 episodes, 96 scenes, 29 components의
  terminal manifest를 동결했다.
- EXP-022는 zero-code가 foundation과 정확히 같고 손상이 one-step direction에서
  생김을 확인했다. Final update의 proxy gain은 SILog gain과
  `rho=-0.285`, EPE gain과 `rho=-0.276`으로 유의하게 반상관이었다.
- EXP-023 metric oracle은 current보다 SILog, AbsRel, EPE를 모두 개선했지만
  1,125 raw candidate application 중 `68%`가 harmful이었다.

#### EXP-024--028: scalar objective conflict와 Pareto safeguard

- EXP-024 log objective: SILog/EPE 개선, AbsRel `0.00691` 악화.
- EXP-025 equal log+relative objective: AbsRel `0.02546` 개선, SILog
  `0.46068`, EPE `0.09945 m` 악화.
- EXP-026은 learned anchor에서 metric-gradient conflict `27--35%`, raw
  equal-average sacrifice `23--33%`를 확인했다. Unit-normalized bisector는
  675/675 common descent.
- EXP-027은 AdamW realized displacement common descent가 `72.54%`에 그쳐
  AbsRel mean을 살리지 못했다.
- EXP-028 safeguard는 step의 `29.28%`에서 개입해 realized common descent를
  `100%`로 만들었다.

EXP-028 OOF:

| Policy | SILog | aligned AbsRel | 3D EPE (m) |
|---|---:|---:|---:|
| foundation | 53.4149 | 0.81202 | 6.53079 |
| current TTT | 53.2398 | 0.80832 | 6.48049 |
| oracle reuse | 53.1757 | 0.80692 | 6.46334 |

#### EXP-029--031: source-safe address와 terminal evidence

- EXP-029: 13,631 causal pairs, OOF Spearman `0.2597`; metric address utility
  `+0.00320`, harm `11.44%`; random/appearance 대비 CI 모두 양수.
- EXP-030: full memory는 current보다 SILog `0.1265`, AbsRel `0.00314`, EPE
  `0.03486 m` 개선하고 random/appearance도 모든 mean과 interval에서 이김.
- EXP-031 untouched terminal 187 eligible targets:
  - current 대비 `+0.05492` SILog, `+0.001752` AbsRel, `+0.01446 m` EPE;
  - random 대비 `+0.02347`, `+0.000389`, `+0.00604 m`.
- 190 coverage gate는 실패했으나 EXP-032가 가능한 최대가 187임을 확인했다.
  결과는 qualified positive evidence이지 literal pass는 아니다.

#### EXP-033--035: 효율과 TUM zero-shot

- 추가 method latency 약 `1.996 ms`, frozen foundation `292.328 ms` 대비
  `0.68%`; learned parameter 288,386; reservoir-64 `38.52 MiB`.
- EXP-034는 223 contexts/111 targets의 causal TUM transfer benchmark를
  구성했지만 sequence 수와 balance가 부족해 descriptive evaluation으로
  제한했다.
- TUM 111 targets에서 full은 current보다 SILog `0.08042`, AbsRel `0.000998`,
  EPE `0.002172 m` 개선했다.
- 이 결과는 “동일한 약한 head 안에서 memory correction이 도움”을
  지지했지만 absolute competitiveness는 아직 비교하지 않았다.

### Phase E — Absolute competitiveness blocker: EXP-036--038

#### EXP-036: official CUT3R/TTT3R 비교

| Method | SILog | aligned AbsRel | 3D EPE |
|---|---:|---:|---:|
| Revisit3D full | 28.4625 | 0.230136 | 0.458924 m |
| CUT3R | 16.6067 | 0.081164 | 0.239378 m |
| TTT3R | 15.7271 | 0.078073 | 0.224615 m |

- Revisit3D full은 TTT3R보다 SILog `1.81x`, AbsRel `2.95x`, EPE `2.04x`
  나빴다.
- Capacity-matched 비교는 아니지만 broad reconstruction framework로 제출할
  수 없을 만큼 차이가 컸다.
- 핵심 교훈: within-backbone improvement는 competitive method가 아니다.

#### EXP-037--038: competitive carrier로 이전

- Official FastVGGT head도 TTT3R 대비 AbsRel `1.43x`, EPE `1.32x`로 실패.
- EXP-038은 official CUT3R에 8-D patch code를 넣는 최소 interface를
  확립했다: native/zero-code error `0`, gradient norm `9.224e-4`, online
  loss `0.232733 -> 0.232658`, identity transport error `0`.
- 이로써 이후 실패를 “wrapper가 동작하지 않아서”라고 설명할 수 없게
  됐다.

### Phase F — Competitive CUT3R에서 reuse의 직접 검증: EXP-039--049

#### EXP-039: fresh source-safe data

- DL3DV train/validation/terminal을 `63/14/14` disjoint scenes,
  `982/213/224` pairs로 pixel/model access 전에 동결했다.

#### EXP-040--043: raw, learned, exact-meta code 모두 ungated reuse 실패

- EXP-040 current TTT는 source/target 모두 loss를 낮췄다.
- Correct 3D reuse는 target loss를 `2.022e-5` 악화하고 spatial shuffle보다
  `2.036e-5` 나빴으며 harm `68.75%`.
- EXP-041 untransported/visual/3D carrier 모두 gain이 음수; code agreement
  `-0.0527/-0.0863/-0.0742`.
- EXP-042 learned 6,144-param basis는 current gain을 `9.72e-5 -> 3.59e-4`로
  키웠지만 reuse gain `5.96e-6`, CI `[-2.23e-5,3.16e-5]`, code agreement
  `-0.00479`.
- EXP-043 exact-meta basis는 current gain `6.42e-4`로 강화했지만 ungated
  reuse `9.30e-6`, CI `[-3.24e-5,5.16e-5]`; correct transport와 shuffle 동률.

이 단계에서 반복되는 패턴은 명확했다: training은 current adaptation은
강하게 만들 수 있지만 source-to-target transfer를 만들지 못했다.

#### EXP-044--046: agreement gating의 제한적 성공

- Post-hoc cosine agreement와 utility Pearson `0.7518`.
- Positive-sign gate는 exposed audit에서 gain `6.05e-5`, harm `1.67%`.
- EXP-045 untouched validation: gated reuse gain `7.35e-5`, 14/14 scenes,
  shuffle 대비 `5.29e-5`, harm `3.76%`; 통과.
- EXP-046 curated causal bank: current 대비 `1.96e-4`, appearance 대비
  `1.41e-4`, random 대비 `1.52e-4`, 모든 CI와 scene 양수.
- 하지만 selected source가 manifest pair인 비율은 `17.37%`뿐이었다. 이것은
  same-context experience retrieval보다 generic compatible correction을
  찾는 현상에 가까웠다.

#### EXP-047--049: full stream과 equal-compute control이 memory claim을 종료

- EXP-047 reservoir gain `2.76e-4`였지만 FIFO가 reservoir보다 모든
  14 scenes에서 좋았다; reservoir-minus-FIFO `-6.24e-5`, CI
  `[-8.87e-5,-4.06e-5]`.
- FIFO가 고른 record 평균 age는 `6.18` frames, reservoir는 `128.3` frames.
  장기 retention보다 최근 correction이 더 유용했다.
- EXP-048 second-current 추가 gain `7.7791e-4`; FIFO memory gain
  `3.3759e-4`. FIFO-minus-second는 `-4.4033e-4`, CI
  `[-6.1767e-4,-2.8623e-4]`, 0/14 scenes 승리.
- EXP-049 low-parallax에서도 future-oracle fallback memory가 second-current보다
  `5.18e-4` 나빴고 0/24 scenes 승리. Correct memory는 shuffle도 이기지
  못했다.

이 두 실험이 original continual-memory thesis를 가장 직접적으로 반박한다.
과거 experience의 이득처럼 보였던 것은 equal current optimization budget을
통제하면 사라졌다.

### Phase G — TTT3R에서 absolute-metric direction을 새로 학습: EXP-050--056

이 단계는 memory를 제거하고 “우리 목적에 맞는 TTT 자체”를 새로 만들 수
있는지를 검증했다.

#### EXP-050: surviving exact-meta CUT3R coordinate의 절대 3D

| Method | SILog | AbsRel | 3D EPE |
|---|---:|---:|---:|
| CUT3R | 17.70120 | 0.0823880 | 0.254219 m |
| exact-meta 1-step | 17.70143 | 0.0823876 | 0.254215 m |
| TTT3R | 15.24125 | 0.0775308 | 0.214267 m |

- Exact-meta는 CUT3R보다 SILog를 유의하게 악화하고 AbsRel/EPE 변화는
  사실상 0이었다.
- TTT3R보다 SILog `2.4602`, AbsRel `0.004857`, EPE `0.03995 m` 열세.

#### EXP-051--053: fresh 7Scenes와 metric-aligned basis

- 43,000 complete train frames를 scene-disjoint role로 동결하고 native
  step-wise TTT3R parity를 확립했다.
- EXP-052에서 online loss는 모든 scene에서 `4.92e-5` 감소했지만
  online/metric gradient cosine은 `0.0253`; 6/16 conflict, 6/16 metric harm.
- Same-norm metric oracle은 모든 scene에서 `5.43e-5` 개선해 code capacity는
  존재했다.
- EXP-053 learned basis는 absolute gain을 `-1.42e-6 -> +0.55e-6`로
  옮겼지만 CI `[-2.10e-6,3.43e-6]`, harm `56.25%`; checkpoint 없음.

#### EXP-054--056: spatial oracle는 존재하지만 observable하지 않음

- EXP-054 offline token-axis oracle: gain `2.0039e-5`, 16/16 anchors,
  harm `0%`; global/shuffle보다 유의하게 좋음.
- EXP-055 learned token conditioner: final gain `-0.34e-6`, CI
  `[-2.60e-6,2.11e-6]`, harm `43.75%`; global basis를 이기지 못함.
- EXP-056 token+pairwise geometry label accuracy `50.204%`, shuffled geometry
  `50.240%`; chance 수준. Realized gain control CIs도 0 포함, harm `50%`.
- 결론: useful tangent axis가 offline label로는 존재하지만 current observable
  feature로 scene-general하게 예측되지 않았다.

### Phase H — 정보가 실제로 없을 때 surface와 adaptation을 분리: EXP-057--060

이 단계는 원래 가설에 대한 가장 해석력이 높은 causal experiment이다.

#### EXP-057: explicit past surface oracle

- Current target 중앙 영역을 지우자 error가 `0.429`, CI `[0.312,0.546]`
  증가했다.
- 두 번의 current local TTT는 이를 거의 복구하지 못했다.
- GT surface fusion은 second-current보다 `0.480` 개선.
- Frozen predicted past surface는 `0.408`, CI `[0.292,0.525]` 개선;
  모든 scene/anchor 양수, harm `0%`, coverage `89.5%`.
- Correct addressing은 동일 payload spatial permutation보다 `0.129`, CI
  `[0.091,0.174]` 좋았다.

#### EXP-058--059: GT dependency 제거

- Predicted pose/native scale fusion은 second-current보다 `0.3940`, CI
  `[0.2866,0.5054]`; shuffle보다 `0.1188`, CI `[0.0815,0.1617]`.
- EXP-057 oracle gain의 `97.59%`를 유지하고 harm `0%`.
- Literal gate는 repeated TTT reproduction이 최대 `1.96695e-5`로 `1e-5`
  guard를 넘어서 실패했지만, drift는 fusion gain의 `0.00499%`뿐이고
  functional comparison 7개는 모두 통과했다.

#### EXP-060: 동일 task에서 past adaptation code

- Source/current update는 모든 anchor에서 각각의 online loss를 낮췄다.
- Transported past code의 second-current 대비 gain은 `4.36e-6`, CI
  `[-4.08e-6,1.41e-5]`.
- `pumpkin`, `stairs`에서 음수; untransported보다 `1.79e-6`, shuffle보다
  `5.63e-7` 나쁨; harm `56.25%`.
- Offline per-pixel best fallback headroom도 `2.03e-5`로 explicit surface의
  `0.3940`과 비교해 사실상 0.

이 비교는 중요한 결론을 준다.

> TTT update는 과거 관측에서 얻은 surface information의 sufficient
> statistic이 아니었다. 관측 정보가 현재 사라졌을 때, optimization
> direction을 보존하는 것으로는 content를 보존할 수 없었다.

### Phase I — Function-space로 표현을 바꾸면 되는가: EXP-067

Code가 coordinate-dependent라서 실패했다는 마지막 설명을 검증했다.
Source code가 실제 3D output에 만든 displacement를 저장하고, predicted-3D로
transport한 뒤 target Jacobian을 통해 target code로 pull-back했다.

| Quantity | Result |
|---|---:|
| second-current EPE | 0.0684103 |
| function gain vs second | `2.04e-6` |
| relative gain | 0.00299% |
| 95% CI | `[-1.12e-6,5.22e-6]` |
| harm | 43.75% |
| gain vs direct code | `-9.36e-7` |
| gain vs untransported | `-1.12e-6` |
| gain vs shuffle | `1.90e-7` |

- `pumpkin` scene mean이 음수였고 모든 comparison interval이 0을 포함했다.
- Source displacement mean norm은 `1.55e-4`; frozen normalized pull-back step은
  모든 pair에서 objective를 overshoot했다.
- 사전등록상 line search나 step repair는 금지되어 H23은 기각됐다.
- 이 결과 하나만으로 모든 자연-gradient/function-space 방법을 반박하지는
  않는다. 그러나 direct code, learned coordinate, exact meta, routing,
  missing evidence까지 누적된 실패를 구할 만큼의 effect는 보이지 않았다.

## 5. 왜 초기 결과는 성공처럼 보였는가

### 5.1 Proxy utility를 geometry accuracy로 해석했다

초기 endpoint는 future self-supervised track consistency였다. EXP-022에서
이 proxy gain은 SILog/EPE gain과 유의하게 반상관이었다. 따라서 retrieval이
proxy를 잘 최적화하는 것과 3D reconstruction을 개선하는 것은 다른 문제였다.

### 5.2 “과거 memory가 도움”과 “같은 상황의 adaptation이 도움”을 혼동했다

Matched source보다 distant/foreign source가 더 유용한 경우가 많았고,
agreement bank는 manifest-paired source를 17.37%만 선택했다. 이는
context-specific experience recall보다 generic residual dictionary 또는
regularization 효과에 가깝다.

### 5.3 약한 custom head 안의 상대 개선을 framework 경쟁력으로 해석할 위험

EXP-030/031/035의 within-backbone improvement는 실제였다. 그러나 EXP-036에서
official TTT3R가 absolute error를 1.8--3.0배 줄였다. 약한 base 위 0.1--1%
상대 개선은 top-tier reconstruction architecture의 충분한 근거가 아니었다.

### 5.4 Equal-compute control이 늦게 들어왔다

Memory는 one-current-step보다 좋아 보였지만 EXP-048에서 두 번째 current
step이 memory gain의 두 배 이상이었다. EXP-049의 low-parallax future oracle도
이를 이기지 못했다. 이 대조군이 adaptation memory의 고유 가치를 분리했다.

### 5.5 Curated candidate bank가 full stream 난이도를 숨겼다

Curated revisit-source bank에서는 agreement routing이 매우 잘 작동했다.
Every-frame write로 바꾸자 FIFO가 reservoir를 이겼고, 최근 6-frame correction이
128-frame-old memory보다 유용했다. Continual retention이 필요한 장기 경험보다
short-term smoothing 성격이 강했다.

## 6. 가설이 실패한 구체적 원인

### 원인 1 — Update는 scene information이 아니라 objective-local direction이었다

Online gradient는 현재 loss를 낮추는 방향이지, 관측된 surface content의
압축 표현이 아니다. EXP-057/060이 이를 직접 분리했다. Surface pointmap은
사라진 정보를 복구했지만 code는 못했다.

### 원인 2 — Update의 의미가 observation과 decoder Jacobian에 의존했다

같은 8-D coordinate도 frame마다 다른 decoder token/Jacobian을 통과한다.
EXP-041의 negative code agreement와 EXP-042/043의 current-gain/reuse-gain
분리가 이를 보였다. EXP-067에서 output-space로 옮겨도 robust effect는
나오지 않아, coordinate mismatch만 고치면 된다는 설명도 충분하지 않았다.

### 원인 3 — Online self-supervision과 absolute 3D metric이 정렬되지 않았다

EXP-022, 024, 025, 050, 052에서 반복됐다. 하나의 loss가 AbsRel을 개선하면서
SILog/EPE를 악화하거나 그 반대가 발생했다. Pareto safeguard로 v1 head는
고칠 수 있었지만 competitive carrier의 observable update로 일반화되지 않았다.

### 원인 4 — Useful direction은 존재하지만 현재 입력에서 식별되지 않았다

EXP-052 metric oracle, EXP-054 token-axis oracle는 강했다. 그러나 EXP-053,
055, 056은 그 방향을 current tokens와 pairwise predicted geometry로 예측하지
못했다. 즉 capacity 문제가 아니라 observability/generalization 문제였다.

### 원인 5 — 과거 experience의 conditional value가 current compute보다 작았다

Memory utility는 양수일 수 있지만 `u(memory | one current step)`가
`u(second current step | one current step)`보다 훨씬 작았다. CL이 아무리
잘 보존해도 저장 대상의 conditional value가 낮으면 paper contribution이
되지 않는다.

### 원인 6 — Continual learning은 “유용성”을 만들어 주지 않는다

Reservoir, FIFO, consolidation은 이미 유용한 record를 잊지 않게 할 수는
있다. 그러나 record가 spatial shuffle보다 낫지 않거나 second-current보다
약하면 retention algorithm은 근본 문제를 해결하지 못한다. 처음 아이디어는
CL의 stability가 adaptation experience의 semantic utility까지 보장할 것처럼
암묵적으로 가정했다.

## 7. 최종적으로 지지되는 주장과 기각되는 주장

### 지지되는 것

1. Frozen foundation 위 local code는 one-step online loss를 안정적으로
   낮출 수 있다.
2. 특정 custom head에서 utility-selected correction dictionary는 random과
   appearance보다 좋은 within-backbone effect를 만들 수 있다.
3. Online objective와 absolute geometry 사이에는 실제 gradient conflict가
   존재하며 optimizer displacement safeguard가 이를 완화할 수 있다.
4. Recurrent implicit state는 명시적으로 사라진 surface evidence를 완전히
   보존하지 못하며, explicit predicted surface는 이를 크게 복구한다.
5. 동일 evidence order만 바꿔도 recurrent geometry가 변한다(EXP-062).

### 기각되는 것

1. Similar geometric context이면 TTT update를 그대로 재사용할 수 있다는
   단순 명제.
2. Correct 3D transport만 제공하면 update reuse가 spatial control보다 좋아진다는
   명제.
3. Learned/shared/exact-meta 8-D coordinate가 revisit compatibility를 자동으로
   만든다는 명제.
4. Bounded continual bank의 장기 record가 recent FIFO보다 본질적으로 낫다는
   명제.
5. Past code가 equal-compute current TTT보다 고유한 미래 정보를 제공한다는
   명제.
6. Past adaptation code가 missing surface information을 보존한다는 명제.

### 아직 반박되지 않은 더 넓은 가능성

1. Backbone부터 end-to-end로 adaptation transport를 위해 학습한 완전히 다른
   architecture.
2. Update가 아니라 sufficient statistics, raw observation, explicit surface,
   track, feature를 저장하는 content memory.
3. 매우 긴 domain recurrence에서 task/domain-level TTA state를 재사용하는
   문제. 이는 현재 local 3D revisit 실험과 다르다.
4. Data-independent trust-region이나 exact Jacobian solver를 사용하는
   function-space method. EXP-067은 고정 step realization만 기각했다.

이 가능성들은 현재 논문을 계속 수정할 근거가 아니라 향후 별도 가설과
fresh data가 필요한 새로운 연구다.

## 8. EXP-061--066: 원 가설 종료 후 수행한 paper-problem 탐색

이 실험들은 adaptation-memory 가설의 직접 검증이 아니라 새로운 compact
paper question을 찾기 위한 pivot이었다.

| EXP | 질문 | 결과 |
|---|---|---|
| [061](../experiments/EXP-061_gauge_local_error_anatomy.md) | Gauge/local error가 중심 failure인가 | Gauge effect는 실재하지만 total EPE의 3.66%, top-confidence 6.47%로 너무 작음. |
| [062](../experiments/EXP-062_order_sensitivity_anatomy.md) | Evidence order만으로 geometry가 달라지는가 | 통과. Order range가 chronological EPE의 12.58%, geometry dispersion Spearman 0.835. |
| [063](../experiments/EXP-063_geometry_commutator_capacity.md) | Latent symmetry가 건강한 correction인가 | Geometry diagnostic은 유효하지만 latent barycenter가 EPE 악화. |
| [064](../experiments/EXP-064_geometry_consensus_direction.md) | Chronological path를 consensus로 옮기면 좋아지는가 | Gain `7.69e-5`, CI 0 포함, harm 43.75%, shuffle 우위 없음. |
| [065](../experiments/EXP-065_calibration_shock_anatomy.md) | 한 번의 focal/FoV shock이 state를 지속 오염시키는가 | Persistent penalty 1.48%, CI 0 포함, image controls와 분리 실패. |
| [066](../experiments/EXP-066_ray_query_provenance_anatomy.md) | History support가 ray-query risk를 confidence 이상 예측하는가 | Provenance Spearman 0.196 < confidence 0.343; fusion AURC 7.68% 악화. |

이 결과들은 “실패를 본 뒤 임의로 architecture를 늘리지 않고 먼저 현상을
검증한다”는 연구 절차의 기록으로 남긴다.

## 9. 전체 EXP ledger

아래 표는 누락 없이 EXP-001--067을 paper-logic 단계로 분류한다. 상세
protocol, config, result path, hash는 링크된 개별 문서를 따른다.

| 범위 | 역할 | 최종 상태 |
|---|---|---|
| [001](../experiments/EXP-001_tttlrm_fastweight_premise.md)--[004](../experiments/EXP-004_keys_and_update_routing.md) | tttLRM/global update premise, benchmark, key/routing | Existing/global update의 context selectivity 기각 |
| [005](../experiments/EXP-005_dense_3d_atom_transport.md) | Dense spatial atom oracle probe | Utility-selected local correction feasibility, safety 미확립 |
| [006](../experiments/EXP-006_trainable_3d_atom_risk_routing.md) | Trainable atom/transport/router | Visual local reuse descriptive support; predicted 3D carrier/risk rejection 미완료 |
| [007](../experiments/EXP-007_continual_atom_consolidation.md)--[009](../experiments/EXP-009_unseen_paperscale_benchmark.md) | Bounded/true-time/paper-scale proxy bank | Proxy benchmark 통과, absolute geometry 미검증 |
| [010](../experiments/EXP-010_paper_geometry_validity.md) | Absolute geometry gate | AbsRel 외 SILog/EPE 실패 |
| [011](../experiments/EXP-011_objective_health.md) | One-loss health | Healthy current TTT signal 통과 |
| [012](../experiments/EXP-012_paper_minimal_refit.md)--[015](../experiments/EXP-015_core_atom.md) | Minimal atom refit | EXP-015 proxy OOF gate 통과 |
| [016](../experiments/EXP-016_unified_utility_address.md)--[019](../experiments/EXP-019_agreement_fallback.md) | Observable utility address | EXP-019 통과 |
| [020](../experiments/EXP-020_paper_model_validation.md) | Locked validation | v1 paper model 기각 |
| [021](../experiments/EXP-021_independent_benchmark.md)--[023](../experiments/EXP-023_metric_utility_oracle.md) | Fresh terminal/metric oracle | Metric candidate headroom 확인 |
| [024](../experiments/EXP-024_metric_aligned_atom.md)--[028](../experiments/EXP-028_safeguarded_pareto_atom.md) | Metric/Pareto atom | Scalar objectives 실패 후 safeguarded head 통과 |
| [029](../experiments/EXP-029_metric_utility_address.md)--[032](../experiments/EXP-032_terminal_coverage_accounting.md) | Metric address/full/terminal/accounting | Within-backbone qualified positive terminal evidence |
| [033](../experiments/EXP-033_frozen_efficiency_audit.md)--[035](../experiments/EXP-035_tum_zero_shot_transfer.md) | Cost and zero-shot transfer | Efficient, descriptive transfer positive |
| [036](../experiments/EXP-036_cut3r_ttt3r_baselines.md)--[038](../experiments/EXP-038_recurrent_carrier_interface.md) | Competitiveness and carrier integration | v1 noncompetitive; exact CUT3R interface 통과 |
| [039](../experiments/EXP-039_dl3dv_source_safe_partition.md) | Fresh competitive-carrier splits | 통과 |
| [040](../experiments/EXP-040_cut3r_oracle_reuse_premise.md)--[043](../experiments/EXP-043_exact_meta_cut3r_plasticity_coordinate.md) | Raw/learned/exact-meta reuse | Current TTT만 강화, ungated reuse 기각 |
| [044](../experiments/EXP-044_posthoc_zero_agreement_routing.md)--[046](../experiments/EXP-046_causal_agreement_bank.md) | Agreement routing and curated bank | Unseen validation/curated bank 통과 |
| [047](../experiments/EXP-047_full_stream_bounded_bank.md)--[049](../experiments/EXP-049_low_parallax_oracle_premise.md) | Full stream, equal compute, low parallax | FIFO 장기보존 우위 없음; second-current가 oracle memory도 이김 |
| [050](../experiments/EXP-050_current_only_exact_meta_tum.md) | Absolute current-only competitiveness | Exact-meta와 competitiveness 모두 실패 |
| [051](../experiments/EXP-051_ttt3r_metric_aligned_prerequisites.md)--[053](../experiments/EXP-053_exact_metric_aligned_ttt3r_basis.md) | Fresh TTT3R metric basis | Capacity는 있으나 learned basis 실패 |
| [054](../experiments/EXP-054_conditional_tangent_oracle.md)--[056](../experiments/EXP-056_pairwise_geometry_observability.md) | Spatial tangent oracle/observability | Oracle 통과, learned observable realization 실패 |
| [057](../experiments/EXP-057_explicit_missing_surface_oracle.md)--[059](../experiments/EXP-059_exp058_reproduction_accounting.md) | Explicit missing-surface information | 매우 강한 qualified positive content-memory evidence |
| [060](../experiments/EXP-060_missing_surface_plasticity_oracle.md) | Same task adaptation code | 명확히 실패; adaptation-memory branch 종료 |
| [061](../experiments/EXP-061_gauge_local_error_anatomy.md)--[066](../experiments/EXP-066_ray_query_provenance_anatomy.md) | New paper-problem premise search | Order phenomenon만 강함; 제안된 correction/provenance는 실패 |
| [067](../experiments/EXP-067_function_space_plasticity_transport.md) | Final coordinate-independent reuse test | Negligible effect; compact TTT reuse 최종 종료 |

## 10. 향후 재분석할 때 확인할 질문

1. Early visual transport effect가 scene-specific adaptation이 아니라 generic
   low-frequency correction이었는가? EXP-006의 direction cosine `0.998`과
   foreign utility를 다시 분석할 수 있다.
2. Memory gain을 반드시 `same compute`, `same acceptance`, `same payload
   norm`, `spatial shuffle` 네 control과 비교했는가?
3. Online objective gain과 absolute metric gain의 sample-level correlation을
   먼저 확인했는가?
4. Stored object가 실제 관측 content의 sufficient statistic인가, 아니면
   단순 optimizer direction인가?
5. Oracle capacity와 observable/deployable signal 사이의 gap이 얼마나 큰가?
6. Within-backbone percentage보다 공식 foundation baseline의 absolute error가
   충분히 경쟁력 있는가?
7. Fresh data role을 architecture 결정 전에 동결했는가?

## 11. 보존해야 할 핵심 artifact

- 전체 실험 목록: [experiments/index.md](../experiments/index.md)
- 현재 공식 상태: [research_state.md](../research_state.md)
- 가설별 상태: [hypothesis.md](../hypothesis.md)
- 결정 이력: [decisions.md](../decisions.md)
- EXP-067 이후 paper pivot 판단:
  [pivot_decision_after_exp067.md](../paper/pivot_decision_after_exp067.md)
- 초기 장문 아이디어:
  `Research/ttt_continual_streaming_3d_research_idea.md`
- 초기 literature review: `Research/literature_review.md`

대용량 checkpoint와 외부 dataset은 Git source of truth가 아니며, 각 EXP
문서에 기록된 compact JSON result와 hash가 결과 대조의 기준이다.
