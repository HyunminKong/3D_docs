# Streaming 4D Reconstruction with Regime-Conditioned Skill Memory
## 전체 시스템 설계 v1

---

# 0. 설계 제약과 그로부터 나온 결정

설계를 시작하기 전에 **A100 1장**이라는 제약이 무엇을 강제하는지부터 명확히 한다. 이 제약이 이후 모든 선택을 결정한다.

## 0.1 불가능한 것

| 항목 | 참고 규모 | 판정 |
|---|---|---|
| Foundation model 처음부터 학습 | VGGT급: 64×A100 × 수일 | ❌ 불가능 |
| 백본 전체 fine-tuning | 24층 full backward + GS rasterization | ❌ 메모리 초과 |
| 4개 task를 동시에 처음부터 | 각각 별도 데이터·손실·수렴 특성 | ❌ 한 사이클에 무리 |

## 0.2 그래서 채택하는 전략

```
사전학습 백본 (tttLRM, 고정)
        +
소수 파라미터만 학습 (TTT layer 일부 + bank + heads)
        +
Task head는 Gaussian 표현에서 파생 (재학습 최소화)
```

**핵심 관찰: 네 개 task 중 셋은 Gaussian에서 거의 공짜로 나온다.**

| Task | 획득 방법 | 추가 비용 |
|---|---|---|
| Depth | Gaussian rasterization (depth buffer) | **0** — 미분 가능, 이미 있음 |
| Point cloud | Gaussian center (world frame) 또는 depth unprojection | **0** |
| Camera pose | 별도 head (VGGT식 camera token) | 낮음 — ~10M params |
| **Point tracking** | **Gaussian 대응 + motion field** | **높음 — 동적 모델링 필요** |

따라서 **tracking을 나머지 셋과 분리**하는 것이 유일하게 현실적인 경로다. §11의 단계별 계획이 이 구분을 따른다.

## 0.3 학습 파라미터 예산

| 구성 | 파라미터 | 학습 여부 |
|---|---|---|
| Patch embed + 24층 trunk | ~300M | 🔒 고정 |
| **TTT layer W_init (L1, L2, L7)** | ~21M | ✅ 학습 |
| TTT projections (q/k/v, inner LR) | ~5M | ✅ 학습 |
| 전 층 LayerNorm | ~0.1M | ✅ 학습 (백본 적응) |
| **Skill bank** | ~3M | ✅ 학습 |
| Regime encoder | ~0.5M | ✅ 학습 |
| Gaussian head | ~15M | ✅ 학습 (미세조정) |
| Camera head | ~10M | ✅ 학습 |
| *(Phase C) Motion / Track head* | *~20M* | *✅ 학습* |
| **학습 대상 합계** | **~55M (Phase A/B)** | |

AdamW optimizer state(fp32 m·v) = 55M × 8B ≈ 440 MB. 여유 있다.

> **왜 LayerNorm을 푸는가:** L1,2,7의 TTT 거동을 바꾸면 나머지 층이 기대하던 분포가 달라진다. LayerNorm만 열어두면 최소 비용으로 백본이 재정렬된다. 이것도 부족하면 frozen block에 rank-4 LoRA를 추가한다.

---

# 1. 시스템 개요

```
                        ┌─── 스트림 ───┐
   frame t-2  frame t-1 │  frame t   │ frame t+1 ...
        └──── chunk c-1 ─┘  └── chunk c ──┘
                              │
   ┌──────────────────────────┼──────────────────────────┐
   │                          ▼                          │
   │   ①  Regime Encoder   r_c = φ(pose deltas of c-1)   │
   │                          │                          │
   │                          ▼                          │
   │   ②  Skill Bank      top-1 retrieval (K slots)      │
   │                          │                          │
   │                          ▼                          │
   │   ③  Read: W ← W + λ·ΔW_skill   (L1,L2,L7만)        │
   │                          │                          │
   │                          ▼                          │
   │   ④  TTT Chunk Update   W ← W − η∇ℓ(W; k,v)         │
   │                          │                          │
   │                          ▼                          │
   │   ⑤  Trunk Forward + Apply → per-view tokens        │
   │                          │                          │
   │        ┌─────────────────┼─────────────────┐        │
   │        ▼                 ▼                 ▼        │
   │   Gaussian Head    Camera Head      (Motion Head)   │
   │        │                 │                 │        │
   │        ▼                 ▼                 ▼        │
   │   ⑥  Persistent Gaussian Map (world frame)          │
   │        │                                            │
   │   ⑦  Write (학습 시에만): ΔW → 최근접 슬롯 EMA        │
   └────────┼────────────────────────────────────────────┘
            ▼
      ┌─────────────────────────────────────────┐
      │  Depth · PointCloud · Pose · Track · NVS │
      └─────────────────────────────────────────┘
```

**흐름 요약:** 기존 TTT(④)는 그대로 두고 **앞(①②③)과 뒤(⑦)에만** 개입한다. bank가 비활성이면 원본 tttLRM과 완전히 동일하게 동작한다 — 이것이 ablation의 기준선이 된다.

---

# 2. 입력 명세

## 2.1 스트림 단위

```python
Chunk = {
    "images":     Tensor[N, 3, H, W],      # RGB, N = 4~8
    "intrinsics": Tensor[N, 3, 3],         # 알려진 경우. 없으면 예측
    "timestamps": Tensor[N],               # 상대 시각 (초)
    "chunk_id":   int,
}
```

| 항목 | 값 | 근거 |
|---|---|---|
| chunk 크기 N | **4 (학습) / 8 (추론)** | 학습 시 2 chunk를 그래프에 물어야 하므로 절반 |
| 학습 해상도 | **256 × 448** | 메모리. patch 16 → 16×28 = 448 tok/view |
| 추론 해상도 | **512 × 896** | 평가용 full res |
| 스트림 길이 | 무제한 | §6.3 Gaussian 예산 관리 참조 |

> **해상도 전략:** meta-training은 저해상도, 평가는 고해상도. TTT layer는 토큰 수에 대해 O(1) 상태 크기라 해상도 전이가 비교적 안전하다. 단, **전이 손실을 반드시 측정**해야 한다 (§10.4).

## 2.2 입력에서 즉시 계산되는 것

frame별로 forward 이전에 계산 가능한 값들:

```python
PoseDelta = {
    "trans_norm":   ‖t_i − t_{i-1}‖,       # 프레임 간 이동량
    "rot_angle":    angle(R_i R_{i-1}ᵀ),   # 프레임 간 회전량
    "trans_dir":    normalize(t_i − t_{i-1}),
}
```

**이 값들이 regime descriptor의 원재료**다(§3).

> ⚠️ 여기 쓰는 pose는 **직전 chunk까지의 추정 pose**다. GT가 아니다. 오라클 실험에서는 nuScenes ego pose(GT급)를 썼으므로, 추정 pose로 바꿨을 때 regime 상관(ρ=−0.336)이 유지되는지 **학습 착수 전에 재확인**해야 한다.

---

# 3. Regime Encoder (①)

## 3.1 설계 근거

오라클 2×2 실험 결과:

| 축 | 편상관 ρ | p |
|---|---|---|
| **pose-only regime** | **−0.336** | 2.1e−21 |
| can_bus regime | −0.316 | 6.0e−19 |
| 공간 근접도 | +0.021 | 0.56 |

**상호 통제 시:**
- pose 통제 후 can_bus 고유 기여: −0.038 (p=0.29) → **없음**
- can_bus 통제 후 pose 고유 기여: −0.122 (p=0.0008) → **있음**

**결론: can_bus(차량 텔레메트리)는 불필요하다.** pose만 쓰는 쪽이 오히려 강하고, 이로써 논문 범위가 "주행 데이터셋"에서 **"스트리밍 3D 일반"**으로 넓어진다.

## 3.2 구조

```python
class RegimeEncoder(nn.Module):
    """직전 chunk의 pose delta로부터 regime 벡터 생성"""
    def forward(self, pose_deltas):        # [W, 3] 윈도우
        f = torch.cat([
            speed.mean(), speed.std(),      # 이동 속도
            yaw_rate.mean(), yaw_rate.std(),# 회전 속도
            stopped_frac,                   # 정지 비율 (speed < ε)
            trans_dir_consistency,          # 방향 일관성
            arc_over_net,                   # 궤적 길이 / 직선 거리
        ])
        return self.mlp(f)                  # → [d_r], d_r = 64
```

## 3.3 두 가지 제약 (위반 시 방법이 무너짐)

**① 인과성** — t−1까지의 정보만 사용. TTT residual 같은 *적응의 출력*을 넣으면 검색 시점(chunk c 적응 **전**)에 존재하지 않아 self-referential retrieval이 된다.

**② Gauge invariance** — 절대 pose·world 좌표 금지. streaming reconstruction 자체가 pose drift를 겪으므로 순환 논리가 된다. **모든 특징은 프레임 간 상대량**이어야 하며, 이 때문에 누적 drift에 둔감하다.

## 3.4 정의 고정

```
oracle/regime_spec.json  ← 7개 통계의 정확한 정의를 파일로 박아둔다
```

can_bus 5개 통계 조합은 사후 구성이었다. 정의를 고정하고 **미사용 씬 30개에서 ρ가 유지되는지 확인한 숫자**가 논문에 들어가야 한다.

---

# 4. Skill Bank (②③⑦) — 핵심 기여

## 4.1 저장 대상: 어느 층인가

층 국소화 실험 (DL3DV, out-of-sample 10씬):

| 그룹 | 층 수 | 잔존율 | foreign/random |
|---|---|---|---|
| L12–23 | 12 | **−1.2%** | 3.4× |
| L7 | 1 | 31.7% | **19.0×** |
| **L1, L2, L7** | **3** | **78.8%** | 7.9× |
| L0–11 | 12 | 98.6% | 10.6× |

**후반 12개 층은 신호가 0이다.** 그리고 상호작용이 L1,2,7 사이에 집중돼 있다 (L1–2 단독 16% + L7 단독 36% = 52% < 합쳐서 74~79%).

> ⚠️ **이 값은 greedy로 학습된 모델에서 잰 것이다.** meta-training은 TTT 갱신 방식 자체를 바꾸므로 **신호가 다른 층으로 옮겨갈 수 있다.**
> → **층 집합을 하드코딩하지 말고 설정값으로 둔다.** 재학습 없이 층만 바꿔 재측정할 수 있는 구조여야 한다.

## 4.2 자료구조

```python
class SkillBank(nn.Module):
    K = 16                                  # 슬롯 수
    layers = [1, 2, 7]                      # ← 설정값
    rank = 8                                # ← 학습 후 재스윕

    keys:   Parameter[K, d_r]                        # 검색 키
    U:      ParameterDict{l: [K, d_out, rank]}       # 저차원 인자
    V:      ParameterDict{l: [K, rank, d_in]}
    usage:  Buffer[K]                                # EMA 사용 빈도
```

**저장 비용:**

| rank | K | 크기 | trunk 대비 |
|---|---|---|---|
| 8 | 16 | **18 MB** | 1.05% |
| 16 | 16 | 36 MB | 2.1% |
| 32 | 16 | 72 MB | 4.2% |

> ⚠️ **rank-8은 확정값이 아니다.** 오라클에서 r128이 r8보다 나빴던 이유는 "정렬 안 된 방향을 많이 주입하면 더 해롭다"였는데, meta-training이 방향을 정렬시키면 **이 관계가 뒤집힐 수 있다.** 학습 후 재스윕하고, param-matched baseline도 그에 맞춰 재설정한다.

> ⚠️ **rank × K가 d를 크게 넘지 않도록 관리.** 넘으면 bank가 사실상 dense가 되어 어떤 query든 여러 조합으로 근사되고 **검색이 하는 일이 없어진다.**

## 4.3 Read (③) — top-1 + straight-through

```python
def read(self, r_c, W, tau):
    logits = (self.keys @ r_c) / temp                    # [K]
    logits = torch.cat([logits, self.null_logit])        # + null skill
    idx = logits.argmax()

    if idx == K:  return W, None                         # null → no-op
    if logits.softmax(-1)[idx] < tau:  return W, None    # gate → no-op

    p = logits.softmax(-1)
    onehot = F.one_hot(idx, K+1).float()
    w = onehot + p - p.detach()                          # straight-through

    for l in self.layers:
        dW = (self.U[l] * w[:K,None,None]).sum(0) @ \
             (self.V[l] * w[:K,None,None]).sum(0)
        W[l] = W[l] + self.lam * dW                      # ★ 대체 아닌 보정
    return W, idx
```

**왜 top-1인가.** softmax attention이면 "검색된 skill"이 혼합이 되어, contrastive의 *"검색된 것이 가장 도움이 되어야 한다"*가 정의되지 않는다. top-1이어야 슬롯 하나 = skill 하나라는 해석이 유지되고, **"몇 개의 skill을 학습했는가"를 셀 수 있다.**

**왜 보정인가 (대체 아님).** 실측으로 강제된 설계:

```
과거 W로 대체 (blend):
  α:     0.0    0.1    0.3    0.5    0.7    0.95   1.0
  PSNR: 21.11  19.56  17.94  17.70  17.73  23.35  23.98
                             ↑ 최저점이 양 끝점보다 3.4 dB 아래

차분을 얹기 (correction):
  계곡 소멸, λ=0 기울기 양수
```

두 절대 상태는 서로 다른 basin에 있어 선형 연결되지 않는다. 차분은 훨씬 작은 벡터라 국소 선형 영역 안에 있다.

**Null skill과 gate.** 잘못된 검색은 **적극적으로 해친다** (foreign −1.35 dB vs random −0.06 dB, 20배). gate는 선택이 아니라 필수다.

## 4.4 Write (⑦) — 학습 시에만

> **결정: bank는 학습 단계에서 구축되고 배포 시 동결된다.**

**왜 온라인 write를 포기하는가.** write 조건 ②("이후 k개 chunk의 품질이 실제로 좋아졌는가")를 판정하려면 **반사실**(그 write를 안 했을 때)이 필요하다. 학습 중에는 두 번 돌려 잴 수 있지만 **추론 중에는 정답이 없어 불가능**하다.

동결의 이점:
- 재현 가능
- 검증 세트에서 bank가 오염되지 않음
- **static prototype baseline과의 비교가 공정해진다** ← 최대 경쟁자

```python
def write(self, dW_chunk, r_c):
    # 조건 ①: 변화가 충분히 큰가
    if dW_chunk.norm() < self.thresh: return
    # 조건 ②: 반사실 대비 이후 k chunk 품질이 개선됐는가 (학습 시에만 계산 가능)
    if not self.counterfactual_gain(dW_chunk) > 0: return

    j = (self.keys @ r_c).argmax()               # 최근접 슬롯
    U_new, V_new = low_rank_factor(dW_chunk, self.rank)
    self.U[j] = (1-β)*self.U[j] + β*U_new        # EMA 통합
    self.V[j] = (1-β)*self.V[j] + β*V_new
    self.keys[j] = (1-β)*self.keys[j] + β*r_c
    self.usage[j] = (1-γ)*self.usage[j] + γ
```

**슬롯 붕괴 방지 (필수).** 최근접 EMA만 쓰면 초기에 우연히 가까웠던 소수 슬롯이 전부를 흡수하고 나머지는 초기값으로 남는다 — VQ-VAE 코드북 붕괴와 같은 형태다. top-1 read를 쓰면 더 심해진다.

```python
def revive_dead_slots(self):
    dead = self.usage < self.dead_thresh
    # 사용 빈도 최상위 슬롯 근처에서 재초기화 + 노이즈
    self.keys[dead] = self.keys[self.usage.argmax()] + noise
    self.usage[dead] = self.usage.mean()
```

**bank는 로그가 아니라 사전이다.** 슬롯 K개 고정, 스트림 길이와 무관. 슬롯 하나에 특정 chunk가 대응하지 않고 **여러 경험이 압축된 통계**가 남는다. 이것이 "장면이 아니라 skill을 저장한다"의 실제 구현이다.

---

# 5. 학습이 바꾸는 것 — 진짜 메커니즘

## 5.1 문제

사후 압축은 실패했다. 회수율 **1.6%**.

원인은 명확하다. 지금 TTT는 **그 순간의 loss만** 최소화한다. 나중에 재사용 가능한지는 목적함수에 아예 없다. 그래서 ΔW가 **stable rank 43.9**에 퍼져 있고, rank-8로는 30%밖에 안 담긴다.

```
❌ 적응 → 사후 저차원 변환 → 저장     (실측 1.6%)
✅ 애초에 재사용 가능하게 적응하도록 학습
```

## 5.2 학습 대상 — bank만이 아니다

```
학습:
  - regime encoder φ
  - bank keys / U / V
  - read gate 온도 τ, 보정 계수 λ
  - ★ TTT layer 자체: W_init, q/k/v projection, inner learning rate η
```

**마지막 줄이 실제 메커니즘이다.** ΔW가 저차원·재사용 가능해지는 것은 bank를 잘 만들어서가 아니라 **TTT가 그렇게 갱신하도록 학습되기 때문**이다. 이게 빠지면 그냥 "LoRA 뱅크 + 라우터"가 된다.

## 5.3 목적함수

```
L = L_recon + α·L_contrastive + β·L_rank
```

### L_contrastive (핵심)

regime r에 대해 후보 skill들(bank K개 + **null**)을 적용했을 때, 검색된 것이 **가장 도움이 되어야** 한다.

```python
gains = []
for j in range(K+1):                       # null 포함
    W_j = apply_skill(W, j)
    gains.append(-loss_recon(W_j))         # 높을수록 좋음
L_contrastive = F.cross_entropy(gains / temp, retrieved_idx)
```

### 왜 이 형태여야 하는가 — 붕괴 경로 차단

| 형태 | 실패 방식 |
|---|---|
| ❌ reuse 단독 | 자명한 최적해는 **regime-무관 상수 ΔW**. slow weight가 흡수하며 FSM과 같아진다. **novelty가 기존 방법으로 환원되는 경로가 목적함수 안에 열린다.** |
| ❌ reuse × 절대적 anti-transfer | 가장 싼 만족법은 `ΔW = useful_A + λ·junk` (junk는 A의 부분공간에 직교해 A에서 무해, B에서 무작위라 유해). 두 항을 만족하면서 specificity는 0. 게다가 gate 때문에 cross-regime harm은 추론 시 발현되지도 않는다. |
| ✅ **contrastive** | 상수 해 → 모든 skill 동일 → margin 0 → 자동 배제. 추론에서 실제로 쓰는 것(ranking)을 직접 학습. gate가 같은 margin으로 캘리브레이션됨. |
| ✅ **+ null skill** | junk 해킹 최종 차단. 다른 skill을 나쁘게 만들어 margin을 벌 수는 있어도 **identity는 나쁘게 만들 수 없다.** 모든 skill이 "안 쓰는 것보다 낫다"를 실제로 증명해야 한다. |

### L_rank (보조)

```python
L_rank = stable_rank(ΔW_chunk)   # 낮을수록 좋음, 약한 가중치
```

목표: stable rank 43.9 → 20 이하. 다만 **직접 최적화는 위험**하므로 β를 작게 두고, 주로 진단 지표로 쓴다. contrastive만으로 rank가 내려가는지 먼저 확인한다.

## 5.4 Gradient path

```
chunk t1 ──→ ΔW_t1 ──→ [write] ──→ bank ──→ [read at t2] ──→ loss_t2
                                     ↑
                        중간 chunk들은 detach
```

skill은 **bank라는 지속 파라미터**에 쓰인다. 따라서 `loss_t2 → ΔW_t1` 경로는 bank를 통과하지, 사이의 chunk들을 통과하지 않는다.

**필요한 그래프는 chunk 2개**지 N개 chunk의 unrolled graph가 아니다. 전 구간 backprop이라는 go/no-go 리스크가 설계 선택으로 내려간다.

**대가 (논문에 명시):** 중간 구간의 적응이 skill 재사용에 미치는 상호작용이 gradient에서 빠진다. 편향이지만 수용 가능하다.

**bank momentum:** bank는 지속 파라미터인데 한 스텝에서 한 chunk만 미분 가능하므로, 안정성을 위해 MoCo식 EMA 갱신을 병행한다.

---

# 6. 출력 — Gaussian 표현과 4개 Task

## 6.1 Gaussian Head

chunk의 각 view에서 per-pixel(또는 per-patch) Gaussian 예측:

```python
Gaussian = {
    "mu":      [M, 3],    # world frame 위치
    "scale":   [M, 3],
    "rot":     [M, 4],    # quaternion
    "opacity": [M, 1],
    "sh":      [M, C],    # color / SH coefficients
    # (Phase C) "motion_coef": [M, K_m]
}
```

**좌표계:** 첫 chunk의 첫 프레임을 world origin으로 고정. 이후 모든 pose는 이에 상대적.

## 6.2 네 개 Task의 획득

### ① Depth — 비용 0

```python
depth = rasterize_depth(gaussians, camera_i)   # gsplat의 depth buffer
```
미분 가능. 별도 head 불필요.

### ② Point cloud — 비용 0

두 경로 모두 지원:
- **Gaussian centers**: opacity 임계 이상인 μ를 그대로 사용
- **Depth unprojection**: `K⁻¹ · depth · [u,v,1]` → world

평가 지표에 따라 선택 (pointmap 계열 벤치는 후자, recon 계열은 전자).

### ③ Camera pose — 낮은 비용

VGGT식 camera token 방식:

```python
class CameraHead(nn.Module):
    """view별 camera token → pose"""
    def forward(self, view_tokens):
        cam_tok = self.cam_query.expand(N, -1)
        x = self.transformer(cam_tok, context=view_tokens)   # 4 layers
        return {
            "quat":  self.quat_head(x),      # 회전
            "trans": self.trans_head(x),     # 이동
            "fov":   self.fov_head(x),       # intrinsics 미지 시
        }
```

**대안(더 저렴):** 예측된 pointmap에서 PnP/Umeyama로 pose를 푼다. 학습 파라미터 0이지만 미분이 까다롭고 outlier에 약하다. → **head 방식 채택, PnP는 refinement로만.**

### ④ Point tracking — 높은 비용, 분리 필요

query `(u, v, t_src)` → 모든 t에서의 `(u', v', visible)`.

**정적 장면: 거의 공짜**

```python
g = nearest_gaussian(unproject(u, v, t_src))   # query → Gaussian 대응
for t in frames:
    u', v' = project(g.mu, camera_t)           # 그냥 투영
    visible = depth_test(g, camera_t)
```

**동적 장면: motion field 필요**

per-Gaussian 독립 motion은 수백만 개라 불가능. **저차원 motion basis**를 쓴다 (Shape-of-Motion / 4DGS 계열):

```python
class MotionHead(nn.Module):
    K_m = 20                                    # motion basis 개수
    def forward(self, t):
        basis = self.basis_mlp(time_embed(t))   # [K_m, 6]  (SE3 또는 translation)
        return basis

def deform(gaussian, t):
    basis = motion_head(t)                      # [K_m, 6]
    delta = gaussian.motion_coef @ basis        # [M, 6]
    return apply_se3(gaussian, delta)
```

per-Gaussian 저장은 계수 `[M, K_m]`뿐이라 감당 가능하다.

> **이 구조가 §12(Gaussian DoF)와 직접 연결된다.** canonical 파라미터(μ, scale, rot)와 motion 계수가 자연스럽게 분리된 DoF 그룹을 이룬다.

## 6.3 Persistent Gaussian Map — 스트림 예산 관리

**문제:** 10시간 스트림에서 per-pixel Gaussian을 계속 쌓으면 메모리가 터진다. 512×896 × 8 view = 3.7M Gaussian/chunk.

**정책 (순서대로 적용):**

| 정책 | 내용 |
|---|---|
| **Keyframe 생성** | 새 Gaussian은 keyframe에서만 생성. 나머지 프레임은 기존 Gaussian 갱신만 |
| **Voxel 병합** | 공간 해시 그리드. 같은 voxel + 유사 normal인 Gaussian 병합 |
| **Opacity pruning** | opacity < ε 제거 |
| **Visibility 카운팅** | k chunk 이상 안 보인 Gaussian → CPU offload 또는 제거 |
| **Active window** | 최근 W chunk에서 관측된 Gaussian만 갱신 대상 |

마지막 항목이 **§12의 진입점**이다.

**목표 예산:** 정상 상태에서 활성 Gaussian ≤ 2M (A100 40GB 기준 rasterization 여유 확보).

---

# 7. 손실함수 전체

```
L_total = λ_photo · L_photometric
        + λ_depth · L_depth
        + λ_pose  · L_pose
        + λ_track · L_track           (Phase C)
        + λ_reg   · L_gaussian_reg
        + α       · L_contrastive     ← 기여
        + β       · L_rank            ← 보조
```

| 항목 | 정의 | 비고 |
|---|---|---|
| L_photometric | `0.8·L1 + 0.2·(1−SSIM)` | LPIPS는 메모리상 선택적 |
| L_depth | scale-invariant log loss | GT 스케일 없을 때 |
| L_pose | `‖quat‖ geodesic + ‖trans‖₁` | GT pose 있는 데이터셋만 |
| L_track | `L1(2D) + BCE(visibility)` | Phase C |
| L_gaussian_reg | scale 정규화 + opacity 희소화 | 발산 방지 |
| **L_contrastive** | §5.3 | **핵심** |

**가중치 결정:** 각 손실을 초기 크기로 정규화한 뒤 grid가 아니라 **uncertainty weighting**(Kendall)으로 자동 조정. 1 GPU에서 손실 가중치 탐색에 시간을 쓸 수 없다.

---

# 8. 학습 전략 (A100 1장)

## 8.1 메모리 절감 기법 (전부 필수)

| 기법 | 효과 |
|---|---|
| **bf16 mixed precision** | 활성값 절반 |
| **Gradient checkpointing** (trunk 전체) | 활성값 ~1/√L |
| **백본 동결** | optimizer state 대폭 감소 |
| **저해상도 meta-training** (256×448) | 토큰 4배 감소 |
| **chunk 크기 4** (학습) | 2 chunk 그래프 감당 |
| **중간 chunk detach** | unrolled graph 방지 (§5.4) |
| **Gaussian 개수 상한** | rasterization 메모리 고정 |

## 8.2 3단계 학습 스케줄

### Stage 1 — Head 워밍업 (bank 없음)

```
동결: trunk 전체, TTT layer
학습: Gaussian head, Camera head, LayerNorm
목적: L_recon + L_depth + L_pose
기간: ~2일
```

이 단계가 **모든 baseline의 기준선**이 된다. 여기 체크포인트를 `base.pt`로 보관.

### Stage 2 — TTT + Bank meta-training ★

```
동결: trunk attention/MLP
학습: TTT layer (L1,2,7), bank, regime encoder, heads
목적: 전체 (L_contrastive 포함)
데이터: 재방문 쌍이 있는 시퀀스
기간: ~1주
```

**배치 구성이 핵심이다:**

```python
# 한 샘플 = (chunk_t1, [중간 chunk들], chunk_t2)
#   t1과 t2는 regime이 유사한 쌍
#   중간은 detach로 통과만
sample = {
    "t1": chunk_at(seq, i),          # write 대상
    "gap": chunks(seq, i+1, j-1),    # detach
    "t2": chunk_at(seq, j),          # read + loss
}
```

regime 유사 쌍은 §3의 descriptor로 사전에 인덱싱해둔다.

### Stage 3 — 고해상도 적응

```
동결: 대부분
학습: LayerNorm + Gaussian head만, 512×896
기간: ~2일
```

## 8.3 검증 분할 — 누수 방지

```
학습/검증/테스트를 씬 단위로 분할.
쌍 단위로 나누면 같은 씬이 양쪽에 들어가 누수가 생긴다.
```

nuScenes는 **지역(4개) 층화 분할**도 병행. boston/singapore가 한쪽에만 몰리면 안 된다.

---

# 9. 데이터셋

## 9.1 학습

| 데이터셋 | 성격 | 제공 GT | 용도 |
|---|---|---|---|
| **DL3DV** | 정적, 고품질 | pose, NVS | 재구성 품질, 오라클 기준 |
| **nuScenes** | 동적, 주행 | ego pose, LiDAR(희소) | **regime 반복** — 핵심 |
| **PointOdyssey** | 합성, 동적 | 완전 GT (depth·pose·track) | tracking 감독 (Phase C) |
| **Kubric** | 합성, 통제 가능 | 완전 GT | regime 통제 실험 |
| ScanNet++ | 실내, 정적 | depth, pose | 실내 일반화 |

**nuScenes가 중심인 이유:** 850개 씬이 4개 지역 68개 세션에서 나와 **같은 동네를 여러 번 지나간다.** "같은 장소×다른 regime"과 "다른 장소×같은 regime"을 둘 다 만들 수 있는 유일한 데이터셋이고, 2×2 검증이 여기서 통과했다.

## 9.2 평가

| Task | 벤치마크 | 지표 |
|---|---|---|
| Depth | Sintel, Bonn, KITTI, ScanNet | AbsRel, δ<1.25 |
| Pose | Sintel, TUM-dynamics, ScanNet | ATE, RPE |
| Point cloud | 7-Scenes, NRGBD, DTU | Accuracy, Completeness, Chamfer |
| Tracking | TAPVid-3D, PointOdyssey | AJ, δ_avg, OA |
| NVS | DL3DV, RealEstate10K | PSNR, SSIM, LPIPS |

---

# 10. 평가 프로토콜

## 10.1 표준 벤치마크 (§9.2)

경쟁 대상: CUT3R, StreamVGGT, Point3R, TTT3R, LongSplat 등 스트리밍 계열.

## 10.2 우리 기여 전용 지표 ★

이쪽이 논문의 본체다.

### 회수율 — **분모 정의가 결정적**

```
❌ 잘못된 정의:  회수량 / greedy 모델의 간섭 4.6 dB
✅ 올바른 정의:  회수량 / 같은 학습 모델에서 bank-off로 잰 간섭
```

**왜 중요한가.** 학습이 TTT를 바꾸면 간섭 자체가 달라진다. 분모가 움직이면 "20%"가 무의미해진다. 올바른 정의를 쓰면 **"학습으로 간섭이 줄었다"와 "bank가 회수했다"가 분리된다** — 전자만으로 좋아졌다면 bank가 불필요하다는 뜻이고, 반드시 구분해야 할 실패 모드다.

```python
interference_off = psnr_A_only - psnr_A_after_B(bank=False)
interference_on  = psnr_A_only - psnr_A_after_B(bank=True)
recovery_rate    = (interference_off - interference_on) / interference_off
```

### 진단 지표 (붕괴 감시)

| 지표 | 현재 | 목표 | 미달 시 의미 |
|---|---|---|---|
| **회수율** | 1.6% | **≥ 20%** | 핵심 주장 실패 |
| **stable rank** (차분) | 43.9 | **≤ 20** | 정리가 안 됨 |
| **foreign penalty** | −1.35 dB | **유지** | 사라지면 = **regime-무관 상수로 붕괴** |
| **memory gain의 층 분포** | 미측정 | **L1,2,7에 ≥74%** | 층 선택 재수행 필요 |
| **슬롯 활용률** | — | **≥ 70%** | 코드북 붕괴 |
| **검색 정확도** | — | regime 유사 쌍에서 top-1 일치 | 검색 무의미 |

> **세 번째가 특히 중요하다.** meta-training이 skill을 상수로 붕괴시키면 foreign penalty가 사라진다. 그것이 §5.3에서 경고한 붕괴의 **관측 가능한 징후**다.

> **네 번째는 결과가 나오는 즉시, 다른 어떤 분석보다 먼저 돌린다.** 회수율이 올라간 기쁨에 묻히기 쉽다.

## 10.3 Baseline

| # | Baseline | 무엇을 반박하는가 |
|---|---|---|
| 1 | Vanilla TTT (bank off, 같은 학습 모델) | 기준선 |
| 2 | **Exemplar replay** (토큰 저장 후 재계산) | *"그냥 토큰 저장하면 되지 않나"* |
| 3 | FSM / EWC on fast weights | *"정규화로 충분하지 않나"* |
| 4 | **Static prototype** (regime별 고정 ΔW, 메모리 없음) | *"slow weight가 흡수 가능하지 않나"* — **최대 경쟁자** |
| 5 | **Param-matched LoRA** (같은 18MB를 slow weight에) | *"메모리 구조가 아니라 파라미터를 늘린 것 아닌가"* |

**통제축은 FLOP이 아니라 param이다.** 이 방법은 연산을 거의 안 쓰므로(rank-8 apply, FLOP 기준 0.5% 미만) FLOP을 맞추는 건 baseline에 0.5%를 더 주는 것일 뿐이다. 실제로 추가한 것은 **파라미터**이고, #5가 없으면 메인 표가 무효다.

## 10.4 Ablation

| Ablation | 확인 대상 |
|---|---|
| **L_contrastive 제거** | **핵심 주장의 유일한 증거** — 없으면 효과 미미해야 함 |
| null skill 제거 | junk 해킹 발생 여부 |
| 검색 랜덤 교란 | gate의 negative transfer 방어 |
| 층 집합 변경 (L1,2,7 vs 전체 vs L0-11) | 국소화 유효성 |
| rank 스윕 (4/8/16/32) | 학습 후 적정 rank |
| K 스윕 (4/8/16/32) | 슬롯 수 |
| top-1 vs softmax read | 설계 선택 검증 |
| 해상도 전이 (256→512) | Stage 3 필요성 |

---

# 11. 단계별 구현 계획

## Phase A — 최소 검증 (3~4주) ★ 최우선

**목표: 회수율 1.6% → 20%**

```
Task:   depth + pose + pointcloud만  (tracking 제외)
데이터: nuScenes + DL3DV
Regime: 학습 없이 pose 통계 직접 사용
Bank:   K=16, rank=8, L1/L2/L7
학습:   bank + TTT inner LR + W_init(L1,2,7)
```

**여기서 회수율이 안 오르면 나머지를 정교하게 만들어도 소용없다.**

주차별:

| 주 | 내용 | 산출 |
|---|---|---|
| 1 | Stage 1 워밍업, `base.pt` 확보 | baseline 성능표 |
| 2 | Bank + read/write 구현, 그래프 검증 | "chunk 2개 + differentiable write가 도는가" |
| 3 | Stage 2 meta-training | 회수율 곡선 |
| 4 | 진단 지표 전량 측정 (§10.2) | **go/no-go 판정** |

## Phase B — 확장 (3~4주)

Phase A 통과 시:

- Regime encoder 학습화
- Consolidating write 정교화 (dead slot revival, 빈도 균형)
- Stage 3 고해상도 적응
- 전체 baseline + ablation
- 표준 벤치마크 평가

## Phase C — Tracking (4주+)

**분리 이유:** 동적 모델링(motion basis) + tracking 데이터셋 + tracking 지표가 전부 추가된다. Phase A/B와 독립적으로 성패가 갈리므로 묶으면 위험이 곱해진다.

- Motion basis head (K_m=20)
- PointOdyssey 학습 파이프라인
- TAPVid-3D 평가
- **regime × tracking 상호작용:** 동적 regime에서 skill이 motion 예측도 돕는가

## Phase D — Gaussian DoF (§12)

Phase B/C 이후. 상세는 §12.

---

# 12. [향후] Gaussian 자유도 선택적 갱신

> **작성 시점: Phase D. 지금은 설계 고려사항만 기록한다.**

## 12.1 아이디어

Gaussian을 통째로 고정/갱신하지 않고, **파라미터 자유도 단위**로 나눈다.

```
전진 모션 + 낮은 parallax  →  lateral 억제, depth 활성
빠른 회전                  →  depth가 오히려 잘 관측됨, 반대 패턴
동적 영역                  →  canonical 억제, motion 계수 활성
```

## 12.2 어디에 붙는가 — 재정의 필요

> ⚠️ **초기 설계는 "refinement 모듈의 gradient 크기를 제어한다"였으나, tttLRM에는 별도 refinement 모듈이 없다.** Gaussian이 디코더 헤드에서 바로 나온다.

**두 가지 부착점이 가능하다:**

**(a) Fast weight 부분공간 마스크** — skill이 출력하는 것에 마스크를 포함시켜, ΔW의 어느 부분공간을 갱신할지 regime별로 결정. low-rank 구조와 자연스럽게 맞고 §4 설계에 그대로 얹힌다. **"Gaussian DoF"라는 이름은 부정확하므로 논문에서는 쓰지 않는다.**

**(b) Persistent Gaussian Map의 갱신 마스크** — §6.3의 active window를 자유도 단위로 세분화. Gaussian 자체에 마스크를 거는 방식이며, **refinement 단계를 추가할 경우에만 성립**한다.

## 12.3 왜 독립 contribution에서 내렸는가

원래 설계(정보행렬 Λ 누적 → eigendecomposition → 확정 방향 freeze)를 skill 출력으로 흡수하면서 **어려운 문제 세 개가 함께 사라졌다:**

| 문제 | 내용 |
|---|---|
| block-diagonal Λ의 과신 | Gaussian 간 상관을 무시해 확신을 과대평가 → 조기 freeze → 복구 불가 drift |
| densification 하 정의 불명 | Gaussian이 split/clone되면 누적 Λ의 상속 규칙이 없음 |
| **thaw 자기모순** | "확정이 틀렸다"를 감지하려면 아끼려던 계산을 다시 해야 하는 순환 |

세 번째가 특히 중요하다. **regime이 바뀌면 마스크가 바뀌는 것이 자연스러운 해제 신호**이므로 별도 thaw 감지가 필요 없다.

또한 매 프레임 정보행렬을 쌓고 eigendecomposition하는 비용이 0이 된다 — 검색 결과에서 마스크를 읽기만 하면 된다.

## 12.4 제약 두 개

**① 마스크는 학습되어야 한다.** "전진 + 낮은 parallax → lateral 억제"를 손으로 쓰면 그것은 엔지니어링이고, meta-objective 주장을 스스로 약화시킨다. 마스크가 skill의 일부로 meta-train되어야 §5와 §12가 하나의 기여가 된다.

**② world 좌표를 직접 freeze하지 않는다.** loop closure와 충돌한다 — streaming에서 pose 자체가 drift하는데 world 좌표를 고정하면 잘못된 전역 정렬을 교정할 수 없다. SLAM이 확정된 landmark도 나중에 움직일 수 있게 설계된 이유다.

→ 마스크는 **로컬 anchor 상대 좌표** 또는 **fast weight 부분공간**에 건다.

## 12.5 Phase C와의 연결

Motion basis 구조가 자연스러운 DoF 그룹을 만든다:

```
canonical:  μ, scale, rot        ← 정적 구조. 조기 확정 가능
motion:     motion_coef [M, K_m] ← 동적. 계속 활성
appearance: opacity, SH          ← 조명 변화 시 갱신
```

동적 영역에서 canonical만 확정하고 motion을 활성으로 두는 정책이 가장 자연스럽다.

## 12.6 평가 방법 (미리 정해둘 것)

- 자유도별 갱신 빈도 vs 최종 품질
- 마스크를 무작위화했을 때의 성능 하락 (마스크가 의미 있는지)
- 고정 규칙 마스크 vs 학습된 마스크
- **oracle 마스크 상한**: 사후적으로 최적 마스크를 줬을 때의 성능 → 이 상한을 먼저 재고 추정기를 만든다

---

# 13. 위험 목록

| 순위 | 위험 | 조기 감지 | 대응 |
|---|---|---|---|
| 1 | meta-training이 stable rank를 못 낮춤 | Phase A 3주차 | L_rank 가중치 상향, 실패 시 근본 재검토 |
| 2 | **static prototype이 동등** | 착수 전 분산분해 | 주효과 비중 확인. 지배적이면 novelty 위협 |
| 3 | memory gain이 L1,2,7에 없음 | Phase A 4주차 | 층 재선택 (설정값이므로 가능) |
| 4 | 슬롯 붕괴 | 슬롯 활용률 모니터 | dead slot revival, 빈도 균형화 |
| 5 | rank-8이 학습 후 부적합 | rank 스윕 | 저장 크기 재계산, baseline 재설정 |
| 6 | 추정 pose에서 regime 상관 소멸 | **착수 전** | 다른 descriptor 재탐색 |
| 7 | 해상도 전이 손실 | Stage 3 | 고해상도 학습 비중 상향 |
| 8 | Gaussian 메모리 폭발 | 장시간 스트림 테스트 | §6.3 정책 강화 |

---

# 14. 착수 전 체크리스트

```
□ SUCCESS_CRITERIA.md 갱신
    □ regime = pose-only (can_bus 제거 확정)
    □ 회수율 분모 = 같은 학습 모델의 bank-off 간섭
    □ write = 학습 시에만, 추론 시 bank 동결
    □ read = top-1 + straight-through
    □ 층 집합 = 설정값, 재도출 가능
    □ bank = EMA + dead slot revival

□ regime_spec.json — 7개 통계 정의 고정

□ 추정 pose에서 regime 상관 재확인 (GT pose → 추정 pose)

□ 미사용 씬 30개에서 ρ 유지 확인

□ nuScenes 분산분해 재실행 (소스 주효과 제거 후에도 regime 유의한가)

□ 씬 단위 학습/검증/테스트 분할 + 지역 층화

□ base.pt (Stage 1) 확보 — 모든 baseline의 기준선
```

**특히 회수율 분모와 write 정책은 코딩 시작 전에 문서에 박아야 한다.** 나중에 정하면 결과에 맞춰 기준이 움직인다.

---

# 부록 A. 지금까지의 실측 근거 요약

| 항목 | 결과 | 출처 |
|---|---|---|
| 재방문 망각 실재 | nuScenes 90/90, 평균 5.79 dB / DL3DV 35/35, 4.43 dB | E1, E6 |
| 동적 물체 오염 | 2.5% (무시 가능) | 마스킹 실험 |
| fast weight가 유일한 view 간 통로 | α=1 재현 오차 0.0000 dB, 전 씬 | 하네스 검증 |
| 상태 특이성 | self −0.28 vs 타인 −1.35 (20/20) | 교차 주입 |
| 선형 상태 혼합 불가 | blend 최저점이 양 끝점보다 3.4 dB 아래 | E1 |
| 차분이 저차원 아님 | stable rank 43.9 (원본 35.1보다 높음) | E4 |
| 사후 압축 회수율 | 1.6% | E5 |
| 신호의 층 국소화 | L1,2,7 = 78.8%, L12–23 = 0% | 층 실험 |
| **호환성을 regime이 지배** | **regime ρ=−0.336 vs 공간 ρ=+0.021** | **2×2** |
| can_bus 불필요 | pose 통제 후 can_bus 고유 기여 없음 (p=0.29) | 재분석 |

# 부록 B. 폐기된 설계

| 방향 | 이유 |
|---|---|
| DoF commitment을 독립 contribution으로 | ablation 상호 상쇄 → skill 출력으로 흡수 |
| 정보행렬 Λ 누적 + eigendecomposition | 과신, densification, thaw 자기모순 |
| FLOP-matched 통제축 | 이 방법은 FLOP을 안 씀 → param-matched |
| 선형 상태 혼합 (blend) | 계곡, 3.4 dB 하락 |
| 사후 저차원 변환 저장 | 1.6%, rank 올리면 악화 |
| 전 구간 unrolled backprop | bank 경유 + gap detach로 chunk 2개면 충분 |
| 순수 speedup claim | Amdahl 천장 1.07× |
| soft attention read | contrastive의 "검색된 것"이 정의되지 않음 |
| 추론 시 online write | 반사실 계산 불가 |
| can_bus 기반 regime | pose가 완전 대체, 더 강함 |
| 기하 descriptor (parallax_ratio 등) | 호환성 예측 실패 (ρ=−0.04) |