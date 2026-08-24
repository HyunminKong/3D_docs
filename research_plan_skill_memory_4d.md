# Regime-Conditioned Adaptation Skill Memory
### for Streaming 4D Reconstruction — 연구 계획서

---

## 0. 한 문장

> Streaming 4D reconstruction에서 **과거에 어떻게 적응했는지**를 재사용 가능한 형태로 저장해두고 비슷한 상황이 다시 오면 불러와 쓴다. **그리고 애초에 재사용 가능한 적응이 나오도록 모델을 학습시킨다.**

뒷문장이 핵심이다. 앞문장만이면 기존 연구(RE-GRAD, NSR)와 겹친다.

---

## 1. 문제 정의

### 1.1 관찰: fast weight는 write-only monotone memory다

Streaming 3D/4D의 state 관리 연구를 전부 나열하면:

| 계열 | 대표 | 제어하는 것 |
|---|---|---|
| Decay / gating | HorizonStream, TTT3R | 얼마나 **빨리** 잊을지 |
| Write selection | MeMix, MoRGS | **어디에** 쓸지 |
| Regularization | FSM (Fisher + EMA anchor) | 얼마나 **멀리** 갈지 |
| Content memory | Point3R, LingBot-Map | **무엇을 봤는지** |
| Reset / refresh | LongStream | **언제** 버릴지 |

**전부 write-side이고, 전부 시간에 대해 단조다.** 정보는 앞으로만 흐르고, 질문은 언제나 "얼마나 빨리 잃을 것인가"뿐이다. **한번 사라진 fast-weight 상태를 되찾는 경로가 존재하지 않는다.**

### 1.2 4D에서 이것이 특별히 아까운 이유

3D 장면은 **물리적으로 재방문된다.** 같은 표면이 다시 보이고, 같은 motion regime이 반복된다. 텍스트와 달리 잃어버린 것이 체계적으로 되찾을 수 있는 것들이다. 단조 망각은 여기서 순손실이다.

### 1.3 문제 문장

> Streaming 4D의 fast weight는 write-only monotone memory다. 우리는 여기에 **read-back 경로**를 만든다.

---

## 2. 연구 질문

> Streaming 4D 모델이 매번 처음부터 적응하는 대신, 반복되는 geometric regime에 대해 **과거에 어떻게 적응했는지**를 기억하고 재사용할 수 있는가?

저장하는 것은 장면이 아니라 **적응 전략**이다.

```
Content memory : "여기에 빨간 차가 있었다"        → Point3R 계열
Skill memory   : "낮은 parallax + textureless    → 본 연구
                  상황에선 이렇게 적응하면 됐다"
```

동일 **객체**를 다시 만날 필요가 없다는 점이 중요하다. 필요한 것은 동일 **처리 상황**의 재등장이고, 이는 시퀀스 내부뿐 아니라 **시퀀스 간에도 공유된다.**

---

## 3. 방법론

### 3.1 파이프라인

```
Frame t
   │
   ▼
Regime descriptor r_t  (t-1까지의 정보만, gauge-invariant)
   │
   ▼
Skill bank 검색  ─── 후보에 null skill(identity) 항상 포함
   │
   ├── confidence gate 미달 → no-op (vanilla TTT)
   │
   └── 통과
        │
        ▼
   retrieved skill = (adaptation direction, update mask)
        │
        ▼
   warm-start + proximal term → TTT refinement
        │
        ▼
   Consolidating write (online dictionary update)
```

### 3.2 Regime descriptor

forward pass에서 이미 계산되는 값만 사용한다. 추가 비용 ≈ 0.

구성: 상대 camera motion, parallax 통계, overlap ratio, geometry 통계

**두 가지 제약 (위반 시 방법이 무너짐):**

1. **인과성** — t−1까지의 상태로만 계산 가능해야 한다. TTT residual 스펙트럼 같은 *적응의 출력*을 넣으면 검색 시점(청크 t 적응 **전**)에 존재하지 않아 self-referential retrieval이 된다.
2. **Gauge invariance** — absolute camera pose나 world 좌표는 금지. streaming reconstruction 자체가 pose drift를 겪으므로 순환 논리가 된다. 상대/불변 정보만 쓴다.

### 3.3 Skill bank

- K = 64 슬롯
- 각 슬롯 = (key, low-rank delta U·Vᵀ, rank 8, update mask)
- 총 ≈ 2 MB

rank 예산 × 슬롯 수가 d를 크게 넘지 않도록 관리한다. 넘으면 bank가 사실상 dense가 되어 어떤 query든 여러 조합으로 근사되고, **검색이 하는 일이 없어진다.**

### 3.4 Read — 방향 + 마스크

retrieved skill은 방향만이 아니라 **(adaptation direction, update mask)** 쌍이다. regime별로 "어떤 자유도를 건드리고 어떤 걸 놔둘지"가 함께 나온다.

```
전진 모션 + 낮은 parallax  →  lateral 확정, depth 활성
빠른 회전                  →  depth가 오히려 잘 관측됨, 반대 패턴
dynamic 영역               →  canonical 확정, deformation 활성
```

**이 설계가 해소하는 문제들** (별도 DoF-commitment 모듈을 두었다면 전부 떠안았을 것):

- 정보행렬 Λ 누적 및 eigendecomposition 불필요 → 계산 비용 소멸
- block-diagonal Λ의 과신 문제 소멸 (Gaussian 간 상관 무시 → 조기 freeze → 복구 불가 drift)
- densification/pruning 하에서 Λ 정의 불명 문제 소멸
- **thaw 자기모순 해소** — "확정이 틀렸다"를 감지하려고 아끼려던 계산을 다시 해야 하는 순환이 사라진다. regime이 바뀌면 마스크가 바뀌는 것이 자연스러운 해제 신호다.

**두 가지 제약:**

1. 마스크는 **학습되어야** 한다. "전진 + 낮은 parallax → lateral 확정"을 손으로 쓰면 그건 엔지니어링이고, meta-objective 주장을 스스로 약화시킨다. 마스크가 skill의 일부로 meta-train되어야 §3.4와 §3.6이 하나의 기여가 된다.
2. 마스크는 **GS 파라미터에 직접 걸지 않는다.** gauge / loop closure 충돌이 되살아난다. 마스크가 제어하는 것은 refinement 모듈이 각 자유도에 실어주는 gradient 크기이고, GS 파라미터 자체는 계속 움직일 수 있어야 한다.

**적용 방식:** 하드 대입이 아니라 warm-start + proximal term. retrieved ΔW는 Wᵢ 근방에서 계산된 것이므로 W_t에 그대로 더하는 것은 이론적 보장이 없다.

**Null skill:** 후보 집합에 identity를 **항상** 포함한다. §3.6의 목적함수 해킹을 막는 핵심 장치.

**Confidence gate:** 검색 신뢰도가 낮으면 완전 no-op. negative transfer 방어.

### 3.5 Consolidating write

episode를 그대로 쌓지 않고 **online dictionary update**로 통합한다.

여러 경험을 하나의 basis로 손실 압축하면 bank가 더 이상 입력의 sufficient statistic이 아니게 되고, **"이건 그냥 attention over KV cache 아니냐"는 반박이 무력화된다.** (linear TTT에서 W_t = Σ ΔWᵢ 이므로, 압축 없는 ΔW 뱅크를 softmax로 읽으면 수식적으로 linear attention으로 환원된다.)

### 3.6 Reusability meta-objective — 핵심 기여

**문제:** 현재 TTT inner loop은 greedy하다. 매 청크에서 그 청크의 loss만 최소화하며, 그 업데이트가 나중에 재사용 가능한지는 전혀 고려하지 않는다. **재사용할 만한 ΔW가 애초에 만들어지지 않는데 검색만 붙이는 것은 순서가 틀렸다.**

**해법:**

> fast-weight 업데이트가 재사용 가능하도록 outer loop을 meta-train한다.

검색 메커니즘이 아니라 **적응 자체의 성질을 바꾸는 것**이다.

| | 말하는 것 |
|---|---|
| FSM (Fisher + EMA) | "덜 움직여라" |
| RE-GRAD / NSR | 사후에 "검색해서 재사용해라" |
| **본 연구** | **"재사용 가능한 방향으로 움직여라"** |

**목적함수 형태: contrastive (InfoNCE)**

regime r에 대해 후보 skill들을 적용한 loss로 contrastive를 건다 — 검색된 skill이 argmax-helpful이어야 한다.

**왜 이 형태여야 하는가 (붕괴 경로 차단):**

- ❌ *reuse 단독* → 자명한 최적해는 regime에 무관한 상수 ΔW. 그건 slow weight가 흡수하고, 정확히 FSM("덜 움직여라")과 같아진다. **논문의 유일한 novelty가 기존 방법으로 환원되는 경로가 목적함수 안에 열려 있다.**
- ❌ *reuse × 절대적 anti-transfer* ("A의 skill이 B에서 loss를 올려야 한다") → 가장 싼 만족법은 regime-특이적으로 유용해지는 것이 아니라 **아무 데나 해로운 성분을 붙이는 것**이다. `ΔW = useful_A + λ·junk` (junk는 A의 관련 부분공간에 직교하여 A에서는 무해, B에서는 무작위라 유해). 두 항을 동시에 만족시키면서 실제 specificity는 0이다. 게다가 confidence gate 때문에 cross-regime harm은 추론 시점에 애초에 발현되지 않는다 — 발현되지 않는 실패 모드를 학습 목적함수로 벌주면 용량만 쓰고 skill을 왜곡한다.
- ✅ *contrastive* → (a) 상수 해는 모든 skill이 동일 → margin 0 → 자동 배제, (b) 추론에서 실제로 쓰는 것(retrieval ranking)을 직접 학습, (c) gate가 같은 margin으로 캘리브레이션되어 목적함수와 추론 경로가 일치.
- ✅ **null skill(identity)을 후보에 포함** → junk 해킹의 최종 차단. 다른 skill을 나쁘게 만들어 margin을 벌 수는 있어도 identity는 나쁘게 만들 수 없으므로, 모든 skill이 "안 쓰는 것보다 낫다"를 실제로 증명해야 한다.

### 3.7 Gradient path

```
chunk t1 → ΔW_t1 → [bank write] → bank → [read at t2] → 적용 → loss_t2
```

skill은 **bank라는 지속 파라미터**에 쓰인다. 따라서 loss_t2에서 ΔW_t1로 가는 경로는 bank를 통과하지, **사이의 청크들을 통과하지 않는다.** 중간 구간의 fast-weight recurrence는 detach한다.

필요한 그래프는 **청크 2개 + 미분 가능한 write/read 연산**이지, N개 청크의 unrolled graph가 아니다. 이로써 두 등장 사이 전 구간 backprop이라는 go/no-go 리스크가 설계 선택으로 내려간다.

**대가 (논문에 명시):** 중간 구간의 적응이 skill 재사용에 미치는 상호작용이 gradient에서 빠진다. 편향이지만 수용 가능하다.

**참고:** 기여는 미분 차수가 아니라 목적함수다. 1차 근사로 가더라도 novelty가 하락하지 않는다.

---

## 4. Novelty 자기 채점

| 요소 | 등급 | 비고 |
|---|---|---|
| Regime descriptor | 낮음 | 엔지니어링 |
| Low-rank bank + retrieval | 중하 | RE-GRAD, NSR과 유사 |
| Consolidating write | 중 | attention 반박 방어용 |
| Read with regime-conditioned mask | 중 | DoF 확정을 skill에 흡수 |
| **Reusability meta-objective** | **높음** | **여기가 논문** |

**하나의 진짜 novelty에 네 개의 지지 구조.** contribution을 3개 동급으로 늘어놓으면 각각 얕아진다.

---

## 5. 검증 계획 — 아키텍처 설계 이전

> **원칙:** 완벽한 정보를 줬을 때의 천장을 모르는 채로 추정기를 설계하는 것이 이 프로젝트의 최대 위험이다. Oracle-first.

### Stage 1 (Day 1) — Perfect-memory oracle

**가장 싸고 가장 강한 kill test.** descriptor도, bank도, 압축도 필요 없다. 30줄 이내.

재방문 시점에 **세 조건을 같이** 잰다:

| 조건 | 내용 |
|---|---|
| (a) | 현재 W 그대로 (baseline) |
| (b) | W_t1 치환 |
| (c) | `α·W_t1 + (1−α)·W_t2` 의 α 곡선 |

**판정은 (b)가 아니라, 곡선이 α=0에서 개선 방향으로 기울어져 있는가로 내린다.**

> ⚠️ 단순 치환만 재면 중간 궤적에서 축적된 상태를 전부 파괴하므로, loss가 올라가도 "재사용 가치 없음"인지 "전체 상태 치환이 너무 거친 개입"인지 구분되지 않는다. **false kill이 난다.**

그리고 전체 W가 아니라 **ΔW_t1의 low-rank 투영만** 주입한다 — 그것이 실제 방법이 하려는 개입이고, 오라클과 방법 사이의 간극도 줄어든다.

**여기서 신호가 없으면 나머지는 전부 불필요하다.**

### Stage 2 — 학습 실현 가능성

"청크 2개 그래프 + differentiable bank write를 동시에 물고 1스텝 도는가"

거의 확실히 되지만, 방법론의 전제이므로 앞쪽에서 확인한다.

### Stage 3 — ΔW 구조

긴 시퀀스의 청크별 ΔW를 수집 → SVD / 클러스터링.

> **어려운 청크들이 regime descriptor 공간에서 뭉치는가?**

뭉치면 논문이고, 흩어지면 검색 자체가 불가능하다. 잘 나오면 **이 그림이 Figure 1**이 된다.

### Stage 4 — Static prototype baseline

regime별 **고정 ΔW prototype**(메모리 없이 학습된 상수 오프셋)을 붙여본다.

- memory 버전과 동등 → **"slow weight가 흡수 가능"이 증명된 것.** 종료.
- 못 따라옴 → 메모리가 필요하다는 증명. 그대로 논문의 필수 baseline이 된다.

> 재등장률 임계값(20%, 90% 등)을 추측하지 않는다. **이 baseline이 판정한다.**

### 관측량 주의

LaCT / tttLRM 계열의 inner loop은 **수렴까지 도는 루프가 아니라 청크당 1-pass delta rule**이다. "수렴 스텝 수" 히스토그램은 존재하지 않는다.

대체 관측량: **청크별 ‖ΔW‖, residual 감소율**

---

## 6. 실험 설계

### 6.1 통제축: param-matched (FLOP-matched 아님)

이 방법은 연산을 거의 쓰지 않는다(rank-8 apply ≈ 전체의 0.5% 미만, FLOP 기준). 따라서 **FLOP을 맞추는 것은 baseline에 0.5%를 더 주는 것일 뿐 통제가 되지 않는다.**

실제로 추가한 것은 FLOP이 아니라 **2MB의 파라미터**다. Amdahl 공격은 사라지지만 그 자리에 capacity 공격이 들어온다: *"메모리 구조가 아니라 그냥 파라미터를 늘린 것 아닌가."*

> **같은 2MB를 trunk의 slow weight / LoRA에 그냥 얹은 param-matched baseline이 반드시 필요하다. 이것이 없으면 메인 표가 무효다.**

FLOP-matched 표는 보조로 유지한다.

**0.5%는 FLOP 기준 추정치이므로 wall-clock으로 재측정해야 한다.** K=64 bank gather + rank-8 apply는 작은 연산이라 latency-bound일 수 있다.

### 6.2 Baseline

1. Vanilla TTT (tttLRM / LaCT)
2. **Exemplar replay** — 토큰을 저장하고 현재 W에서 다시 TTT. *가장 중요한 baseline.* ΔW 저장이 이것을 이기는 조건은 "ΔW를 만드는 데 비용이 많이 들었을 때"뿐이다.
3. FSM (EWC-on-fast-weights)
4. **Static prototype** (§Stage 4)
5. **Param-matched LoRA** (§6.1)

### 6.3 논문을 지탱하는 두 Ablation

| Ablation | 기대 결과 |
|---|---|
| **reusability objective 유무** | 없으면 효과 미미, 있으면 발생. **핵심 주장의 유일한 증거** |
| **negative transfer** (검색을 랜덤 교란) | confidence gate가 방어하는가 |

부가: null skill 제거, consolidating write → raw episode log, mask 제거(방향만).

### 6.4 데이터셋

| 세팅 | regime 다양성 | 판단 |
|---|---|---|
| 실내 스캔 (ScanNet / TUM) | 낮음 | 너무 균질 — slow weight가 흡수할 위험 |
| 주행 (Waymo / nuScenes) | 중 | 무난 |
| **Egocentric (Ego4D / Aria)** | **높음** | **주 타깃** |
| Kubric (합성) | 통제 가능 | toy / ablation |

Egocentric은 카메라 모션이 불규칙하고 조명·모션블러·시야 급변이 잦아 regime 다양성이 크면서, 사용자가 같은 공간을 반복 이동하므로 재등장도 확보된다.

### 6.5 메인 지표

- 긴 시퀀스 camera pose (ATE / RPE)
- Pointmap accuracy / completeness
- 재방문 구간 한정 지표 (첫 방문 대비 회복률)
- 파라미터 오버헤드, wall-clock latency

---

## 7. 중단 조건

하나라도 걸리면 접는다.

| 조건 | 판정 시점 |
|---|---|
| blend 곡선이 α=0에서 개선 방향으로 기울지 않음 | Stage 1 (Day 1) |
| ΔW가 descriptor 공간에서 뭉치지 않음 | Stage 3 |
| static prototype이 memory 버전과 동등 | Stage 4 |

**Stage 1에서 죽으면 3일, Stage 4에서 죽으면 방법론을 다시 짠다.** 둘 다 아키텍처를 쓰기 전에 알 수 있다.

---

## 8. 알려진 위험

| 위험 | 대응 | 상태 |
|---|---|---|
| linear TTT에서 attention으로 환원 | consolidating write (손실 압축) | §3.5 |
| ΔW의 local validity (Wᵢ ≠ W_t) | warm-start + proximal, 하드 대입 금지 | §3.4 |
| 목적함수 붕괴 (상수 해) | contrastive | §3.6 |
| 목적함수 해킹 (junk 성분) | null skill | §3.6 |
| Negative transfer | confidence gate + 교란 ablation | §3.4, §6.3 |
| Key drift | gauge-invariant descriptor | §3.2 |
| Self-referential retrieval | t−1 인과성 제약 | §3.2 |
| Capacity 공격 | param-matched baseline | §6.1 |
| bank가 dense화 → 검색 무의미 | rank × K < d 관리 | §3.3 |
| **delta rule이 이미 부분 방어** | Stage 1에서 확인 | ⚠️ 미해결 |

마지막 항목: TTT의 표준 업데이트는 Hebbian outer product가 아니라 **delta rule** `ΔW = η(v − Wk)kᵀ`로, 잔차만 기록하므로 중복 정보를 재기록하지 않는다. "TTT는 과거를 덮어쓴다"는 전제가 이미 update rule 수준에서 부분적으로 방어되어 있다. **"delta rule + gating이 있는데도 왜 여전히 잊는가"를 실측으로 보여야 한다.**

---

## 9. 포지셔닝

**"memory"보다 "3-timescale system"으로 서술한다.**

consolidating write는 결국 "느리게 갱신되는 파라미터 집합"이다.

```
slow weight  (학습 시 고정)      — 일반 지식
bank         (느리게 consolidate) — 재사용 가능한 적응 전략   ← 본 연구
fast weight  (청크마다)          — 현재 컨텍스트
```

이 프레이밍이 방어에 강하고, complementary learning systems 계열과의 관계도 정직하게 정리된다.

**차별화 요약:**

| 대상 | 차이 |
|---|---|
| Point3R, LingBot-Map | content vs skill — 무엇을 봤는지 vs 어떻게 적응할지 |
| FSM, MeMix, HorizonStream | write-side 단조 제어 vs **read-back 경로** |
| RE-GRAD, NSR | 사후 검색 vs **재사용 가능하도록 학습** |
| MoE-LoRA + router | 오프라인 고정 expert vs 온라인 consolidate + meta-objective |

---

## 10. 다음 액션

**이번 주:**
- [ ] Stage 1 — (a)(b)(c) 세 곡선. `ΔW_t1`의 low-rank 투영만 주입. **30줄.**
- [ ] Stage 2 — 청크 2개 그래프 + differentiable write 1스텝

**통과 시:**
- [ ] Stage 3 — ΔW SVD / 클러스터링 → Figure 1 후보
- [ ] Stage 4 — static prototype baseline

**Stage 1~4 통과 후에야 아키텍처를 설계한다.**

---

### 부록: 폐기된 방향과 이유

| 방향 | 폐기 이유 |
|---|---|
| DoF-wise commitment을 독립 contribution으로 | ①이 잘 자를수록 ②가 먹을 게 없어져 ablation에서 상호 상쇄. → **skill의 출력(mask)으로 흡수** |
| 정보행렬 Λ 누적 + eigendecomposition | block-diagonal 과신, densification 하 정의 불명, gauge 충돌. → mask 학습으로 대체 |
| FLOP-matched를 메인 통제축으로 | 이 방법은 FLOP을 안 쓰므로 통제 불성립 → param-matched |
| 재등장률 임계값(20~90%) 기준 | 근거 없는 추측 → static prototype baseline이 판정 |
| Reuse 단독 / 절대적 anti-transfer objective | 상수 해, junk 해킹 → contrastive + null skill |
| 두 등장 사이 전 구간 unrolled backprop | gradient path 오판. bank 경유 + gap detach로 청크 2개면 충분 |
| 순수 speedup claim | Amdahl 천장 1.07× (inner update = trunk latency의 21.8%) |
