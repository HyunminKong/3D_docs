# Depth-Normalized Progressive Commitment for Streaming 4D Reconstruction

> **프로젝트 스펙 문서 (v1.0)**
> 이 문서는 Claude Code 에이전트에게 전달되는 연구 명세다.
> **§10 작업 지시**를 먼저 읽고, 현재는 **Stage 0만** 수행한다.

---

## 0. 한 줄 요약

Streaming 4D reconstruction에서 각 Gaussian의 "확정(commit)" 시점을 **깊이로 정규화된 삼각측량 정보 충분성**으로 결정하고, 확정을 **자유도(DoF)별 이방성**으로 점진 수행하며, 확정 과정에서 남는 잔차를 **동적 객체 탐지 신호**로 재활용하는, **연산 예산 제어형(compute-budgeted)** 프레임워크.

---

## 1. 배경과 문제 정의

### 1.1 상황

Online / streaming 3D-4D reconstruction에서는 프레임이 순차적으로 들어오고, 각 시점마다 Gaussian 집합을 갱신해야 한다. 모든 Gaussian을 매 프레임 최적화하면 연산이 선형적으로 누적되어 실시간성이 무너진다.

### 1.2 기존 접근의 한계

- **Age / keyframe window 기반**: 오래된 것 또는 현재 윈도우 밖의 것을 최적화에서 제외. 단순하지만 "실제로 잘 결정되었는지"와 무관하다.
- **RTG-SLAM (SIGGRAPH 2024)**: Gaussian을 stable/unstable로 나누고 unstable만 최적화 + unstable이 점유한 픽셀만 렌더링. **아이디어의 직접적 선행연구다.** 단, (a) RGB-D 입력의 opacity 누적 휴리스틱에 의존, (b) Gaussian 단위 이진 판정, (c) static scene 가정.
- **불확실도 기반 (FisherRF, PUP 3D-GS)**: per-Gaussian Fisher/Hessian으로 불확실도를 계산하지만, offline pruning 또는 view selection 용도이고 streaming commitment 스케줄과 결합되지 않았다.

### 1.3 우리가 메우는 gap

> **단안(monocular) streaming 환경에서, RGB-D 깊이 센서 없이 "이 Gaussian을 확정해도 되는가"를 원리적으로 판정하고, 그 판정을 동적 장면까지 확장한다.**

---

## 2. 핵심 아이디어

### 2.1 발상 전환

❌ **하지 않을 것**: "카메라에 가까우면 confidence 점수를 +α 준다"
→ streaming에서 근거리는 frustum을 빨리 벗어나므로 **거리와 관측 횟수가 역상관**이다. 두 항을 가중합하면 서로 상쇄된다. 또한 ad-hoc heuristic으로 분류되어 리뷰에서 방어 불가.

✅ **할 것**: **거리를 confidence의 "분모(요구 정보량)"로 승격**

$$
c_i \;=\; \frac{\text{누적된 기하 정보}}{\text{그 깊이에서 요구되는 정보}}
$$

거리와 관측량이 **분자/분모로 짝지어져** 상쇄 문제가 사라지고, $c_i$가 무차원이 되어 씬 전역 비교가 가능해진다.

### 2.2 물리적 근거 (2개 채널)

**채널 1 — 삼각측량 정밀도.** baseline $B$, 초점거리 $f$, disparity 잡음 $\sigma_d$:

$$
\sigma_z \approx \frac{z^2 \sigma_d}{fB}, \qquad \sigma_\perp \approx \frac{z\,\sigma_u}{f}
$$

불확실도 타원체는 ray 방향으로 늘어나고 **종횡비 ≈ $z/B$**. 목표 정밀도 $\epsilon$에 필요한 baseline: $B^*(z) = z^2\sigma_d/(f\epsilon)$ → **요구 정보량이 $z^2$로 증가**.

**채널 2 — 광학적 증거 밀도.** 표면이 차지하는 입체각 $\propto 1/z^2$ → 감독 픽셀 수 $\propto 1/z^2$ → photometric SNR $\propto 1/z$.

### 2.3 정식화

**(A) 간이 형태 (Stage 0 검증용)**

$$
c_i = \frac{B_i^{\text{acc}}}{\kappa \cdot z_i^{\,p}}
$$

**지수 $p$의 물리적 의미 — 이것이 핵심 실험 변수다:**

| $p$ | 의미 | 대응하는 고전 기준 |
|---|---|---|
| 0 | raw baseline, 정규화 없음 | — |
| **1** | **triangulation angle** $\alpha \approx B/z$ | **COLMAP의 고전적 point 품질 기준** |
| **2** | metric depth 정밀도 기준 ($\sigma_z \propto z^2$) | — |

> ⚠️ **$p=1$이 곧 고전 SfM 기준이다.** 따라서 "$p>1$이 필요하다"를 실증하면 그 자체가 **"고전 기준은 streaming reconstruction에 불충분하다"**는 기여가 된다. 반대로 $p\approx1$이 나오면 novelty가 약해지므로, 그 경우 지표별 최적 $p$ 차이(§2.4)로 방향을 전환한다.

**(B) 엄밀 형태 (Stage 1 구현용)** — information filter

$$
\Lambda_i^{(t)} = \Lambda_i^{\text{prior}} + \sum_{t' \le t} J_{t'}^\top R^{-1} J_{t'}, \qquad c_i = \lambda_{\min}(\Lambda_i)
$$

- $z$가 Jacobian 안에서 **자동으로** 나온다 (별도 heuristic 항 불필요)
- $\lambda_{\min}$: "가장 덜 결정된 방향까지 결정되었을 때만 확정" — 그 최악 방향은 항상 radial
- $\Lambda^{\text{prior}}$에 **feed-forward 모델(VGGT / CUT3R 계열)의 per-pixel confidence를 사전 정보로 주입** 가능 → 학습된 prior × 기하 증거의 Bayesian 융합. RGB-D 의존 선행연구가 구조적으로 할 수 없는 부분.

### 2.4 예상되는 반론과 대응 (미리 알고 있을 것)

> 삼각측량 오차가 novel view에 유발하는 픽셀 오차는 $(B'/B)\sigma_d$로 **depth에 무관**하다. 즉 $z^2$ 논변은 **기하/metric 지표에서 강하고, 순수 PSNR에서는 약하다.**

**대응**: (a) 평가에 depth/mesh 정확도 지표를 **반드시** 포함한다. (b) "목적 지표에 따라 최적 $p$가 달라진다"를 실험으로 보이는 것 자체를 기여로 삼는다. 예측: 기하 지표 → $p \to 2$, 광학 지표(PSNR) → $p \to 1$.

---

## 3. Novelty 축 (선행연구와 갈리는 지점)

### 축 A — 이방성 확정 (Anisotropic Commitment)

단일 뷰에서 **bearing(방위)은 이미 정확하고 depth만 불확실**하다. 게다가 bearing 오차는 $z$와 무관하게 약 1픽셀이다. 기존 연구는 전부 Gaussian **통째로** 이진 판정한다.

| DoF | 확정 시점 | 근거 |
|---|---|---|
| Lateral 2 DoF, $P^\perp = I - v_iv_i^\top$ | 거의 즉시 | 단일 뷰 bearing 정밀 |
| Radial 1 DoF, $P^\parallel = v_iv_i^\top$ | $c_i > \tau$ 이후 | parallax 누적 필요 |
| Scale / rotation | radial 확정 이후 | 형상은 위치 확정 후 |
| SH 고차항 | 최후 | 관측 각도 다양성 필요 |

구현: gradient 마스킹 $\tilde{g}_i = P_i g_i$ ($v_i$ = 첫 관측 ray 방향).
부수 효과: 오확정 시에도 틀릴 수 있는 자유도가 1개뿐 → error absorption 대폭 감소.

### 축 B — 동적 객체 탐지로의 역전

> parallax가 충분한데($c^{\text{geo}}$ 높음) 잔차가 지속적으로 높다 ⟹ 기하가 어려운 게 아니라 **움직이는 것**이다.

근거리가 기준선을 가장 빨리 넘으므로, **근거리 동적 객체를 가장 이른 시점에 분리**하게 된다. "근거리=동적이라 위험"이라는 약점이 탐지 신호로 전환된다.

Factorized commitment: $c_i = c_i^{\text{geo}} \cdot c_i^{\text{static}}$

| $c^{\text{geo}}$ | $c^{\text{static}}$ | 처리 |
|---|---|---|
| 높음 | 높음 | 전면 확정 (연산 제외) |
| 높음 | 낮음 | **canonical(모양·색)만 확정, deformation 계수는 유동** |
| 낮음 | — | active |

세 번째 행이 "4D에서 확정이란 무엇인가"에 대한 답이며, static 가정 GS-SLAM 계열 전체와 갈리는 지점.

### 축 C — 정확도 이득과 연산 이득의 분리

| | 개수 | 픽셀 점유 | 프레임 간 화면 이동 | 확정 시점 |
|---|---|---|---|---|
| 근거리 | 적음 | 큼 | 큼 | 이름 |
| 원거리 | 많음 | 작음 | 작음 | 늦음 |

- **근거리 확정 → 정확도 이득.** 단안에서 metric scale은 근거리가 지배 → 스케일 앵커 조기 고정 → drift 억제. (연산 논변 아님, **정확도 논변으로 판매**)
- **원거리 확정 → 연산 이득.** 개수가 많고 화면상 이동이 작으므로, 확정된 원경을 **per-tile RGBA proxy로 pre-composite** 후 재사용. 유효 윈도우가 같은 기하에서 유도됨: $\Delta t(z) \propto z\,\tau_{\text{pix}} / (f\|v_{\text{cam}}\|)$ → **멀수록 오래 유효**.

> **논문의 통일성**: 하나의 $z$ 수학에서 서로 다른 두 이득이 나온다. "왜 거리인가"에 대한 답이 두 번 나온다.

### 축 D — 연산 예산 스케줄러

고정 임계값 $\tau$ 대신, 매 프레임 예산 $C_{\text{budget}}$을 받아 $c_i$ **순위** 상위부터 예산이 찰 때까지 확정.

- confidence의 **절대값 보정 불필요, 순위만 맞으면 됨** → miscalibration에 강건
- **anytime 성질**: "12 ms/frame 안에 돌려라"가 파라미터
- 주 결과가 단일 operating point가 아니라 **Pareto 곡선**
- A100 1장 제약이 약점이 아니라 세일즈 포인트로 전환

*(참고: LLaDA의 low-confidence remasking은 고정 임계값이 아니라 스케줄 기반 top-k다. 우리가 계승하는 것은 이 스케줄러이지 "확정=연산절감"이 아니다 — LLaDA는 매 step full forward를 돌리므로 확정으로 연산을 줄이지 않는다. 이 점을 논문에서 혼동하지 말 것.)*

---

## 4. 검증 가능한 가설

| ID | 가설 | 검증 방법 | 실패 시 |
|---|---|---|---|
| **H1** | 최종 기하 오차는 $z$ 단독이나 $B^{\text{acc}}$ 단독보다 **$B^{\text{acc}}/z^p$의 단일 함수로 훨씬 잘 설명된다** | Stage 0, depth-bin stratified collapse | 프로젝트 근거 붕괴 → §9 대응 |
| **H2** | 최적 $p^* > 1$ (고전 triangulation angle 기준으로 불충분) | Stage 0, $p$ sweep | novelty 약화 → H3로 전환 |
| **H3** | 최적 $p^*$가 **평가 지표에 따라 달라진다** (기하 지표 $p\to2$, PSNR $p\to1$) | Stage 0, 지표별 sweep | — |
| **H4** | 제안 confidence는 동일 freeze rate에서 **age-based freezing을 능가** | Stage 1 | 논문 성립 불가 |
| **H5** | 이방성 확정이 이진 확정보다 동일 연산에서 우수 | Stage 1 ablation | 축 A 철회 |
| **H6** | $c^{\text{geo}}$ 高 + 잔차 高 ⟹ 동적, 판정이 GT motion mask와 상관 | Stage 2 | 축 B 철회 |

---

## 5. 실험 계획 (단계별)

| Stage | 내용 | 산출 | 상태 |
|---|---|---|---|
| **0** | **가설 검증 probe + age baseline** | Figure 1, $p^*$ 결정 | **← 지금 여기** |
| 1 | Depth-normalized sufficiency + 이방성 확정 (static) | 핵심 기여 1·2. **여기까지가 최소 논문** | 대기 |
| 2 | $c^{\text{static}}$ 분해 + canonical/deformation factorized freezing | 4D 확장 | 대기 |
| 3 | 연산 예산 스케줄러 + 원거리 proxy 캐싱 + memory compaction | 시스템 기여, Pareto | 대기 |

> Stage 3의 proxy 캐싱은 CUDA 작업량이 크다. 시간이 부족하면 gradient/optimizer state 절감만 보고하고 forward 절감은 future work로 남긴다. **어설픈 캐싱은 wall-clock이 안 줄어 논문을 오히려 약화시킨다.**

---

## 6. Stage 0 상세 명세 (Claude Code가 지금 구현할 것)

### 6.1 목표

**"$B^{\text{acc}}/z^p$가 최종 기하 오차를 설명하는 정규화 좌표인가"** 를 확인하고 $p^*$를 결정한다.

### 6.2 파이프라인

```
[GT pose + GT depth 데이터셋]
        ↓
[프레임 순차 투입 3DGS 학습]  ← 개조 대상: instrumentation만 추가, 알고리즘 변경 없음
        ↓
[per-Gaussian 로그]  z_first, B_acc(t), n_obs(t), α_max(t), err_final
        ↓
[분석 스크립트]  p sweep → collapse 정량화 → Figure 1
```

### 6.3 로깅할 per-Gaussian 통계

| 필드 | 정의 |
|---|---|
| `gid` | Gaussian id (densification 후에도 추적 가능한 lineage id) |
| `z_first` | 첫 관측 프레임에서 카메라 중심까지 거리 |
| `z_mean` | 관측 전체 평균 거리 |
| `n_obs` | 유효 관측 프레임 수 (frustum 내 + 가시성 임계 이상) |
| `B_acc` | 관측한 카메라 중심들의 최대 pairwise 거리 |
| `alpha_max` | 최대 triangulation angle: $\max_{t,t'} \angle(\mu_i - c_t,\; \mu_i - c_{t'})$ |
| `err_final` | 최종 기하 오차 (아래 6.4) |
| `first_frame`, `last_frame` | 관측 구간 |
| `contrib` | 누적 렌더 기여도 (weight 합) — 저기여 Gaussian 필터링용 |

> `alpha_max`는 이미 $B/z$로 정규화된 무차원량이다. 즉 **`alpha_max` 자체가 $p=1$ 케이스**이고, 일반 $p$는 `B_acc / z^p`로 계산한다. 두 경로가 $p=1$에서 일치하는지 sanity check로 확인할 것.

### 6.4 오차 정의 (`err_final`)

- **주 지표**: GT depth/LiDAR로 만든 point cloud 또는 mesh에 대한 **point-to-plane 거리** (mesh 없으면 kNN plane fit)
- **보조 지표**: 해당 Gaussian이 지배적으로 기여한 픽셀들의 rendered depth vs GT depth 절대오차 (가중 평균)
- **정규화 버전 병기**: `err_final / z_mean` (상대 오차) — H3 검증에 필요

### 6.5 데이터셋

| 우선순위 | 데이터셋 | 이유 |
|---|---|---|
| 1 | **Waymo Open Dataset** (또는 KITTI) | LiDAR GT, **깊이 범위 1–80 m로 넓음** → $p$ sweep에 이상적 |
| 2 | **Replica** | GT 완벽, 노이즈 적음, 빠른 iteration |
| 3 | ScanNet++ / TUM-RGBD | 실내 실촬, 일반화 확인 |

각 데이터셋 **3–5개 시퀀스**로 시작. 전체 벤치마크는 Stage 1 이후.

### 6.6 $p$ sweep

- $p \in \{0,\ 0.5,\ 1.0,\ 1.5,\ 2.0,\ 2.5,\ 3.0\}$
- 각 $p$에 대해 depth bin으로 층화: 예) Waymo `[0,5), [5,10), [10,20), [20,40), [40,80)` m
- x축 = $B^{\text{acc}}/z^p$ (log scale), y축 = median `err_final`, bin별 곡선

### 6.7 Collapse 정량화 — **Figure 1의 핵심**

눈으로 "겹쳐 보인다"는 불충분하다. 아래 중 최소 2개를 계산:

1. **Bin 간 분산**: 공통 x-grid에서 bin별 곡선의 pointwise 분산 평균. $p$의 함수로 플롯 → **최소점이 $p^*$**
2. **단일 곡선 $R^2$**: 모든 bin 데이터를 합쳐 단조 함수 하나를 적합, $R^2$를 $p$의 함수로 플롯 → **최대점이 $p^*$**
3. **Spearman $\rho$**: $\rho(B^{\text{acc}}/z^p,\ \text{err})$ 를 $p$의 함수로 → 절댓값 최대점

**$p=0$ 대비 $p^*$에서 bin 간 분산이 유의하게(예: 40% 이상) 감소하면 H1 성립.**

### 6.8 추가 산출 — AUSE

$c_i$를 uncertainty score로 보고 **AUSE (Area Under Sparsification Error)** 를 계산한다. 반드시 **depth bin별로 분리 보고** — "근거리뿐 아니라 원거리에서도 정렬이 좋다"를 보여야 한다.

### 6.9 Age-based freezing baseline

Stage 0에서 **함께** 구현한다. 이게 공짜인데도 강력해서, 이걸 못 이기면 나머지가 무의미하다(H4).

- 규칙: 마지막 관측 후 $k$ 프레임 경과 시 동결. $k \in \{5, 10, 20, 50\}$
- 측정: freeze rate, wall-clock ms/frame, peak memory, 품질 지표
- **Stage 1의 비교 기준선이므로 인터페이스를 공통화**해 둘 것

### 6.10 산출 파일

```
outputs/stage0/
  logs/{dataset}_{seq}_gaussian_stats.parquet
  figs/fig1_collapse.png            # bin별 곡선, p=0 / p=1 / p* 3-panel
  figs/p_sweep_metrics.png          # 분산·R²·Spearman vs p
  figs/ause_by_depth_bin.png
  tables/p_sweep.csv
  tables/age_baseline.csv
  REPORT.md                          # 결과 요약 + H1/H2/H3 판정
```

### 6.11 성공 기준 (Go / No-Go)

| 판정 | 조건 | 다음 행동 |
|---|---|---|
| **Go** | H1 성립 & $p^* > 1.3$ | Stage 1 진행 |
| **조건부 Go** | H1 성립 & $p^* \approx 1$ | H3(지표별 $p^*$ 차이)로 논지 전환 후 Stage 1 |
| **No-Go** | H1 불성립 (collapse 없음) | §9 리스크 대응 실행 |

---

## 7. 평가 지표 (전 Stage 공통)

**품질**
- 광학: PSNR, SSIM, LPIPS
- **기하(필수)**: depth abs-rel, RMSE, $\delta_{1.25}$; mesh 있으면 accuracy / completion / chamfer
- 불확실도 품질: AUSE (depth bin별), ECE (참고용)

**연산 — FLOPs 금지, 아래만 보고**
- **wall-clock ms/frame** (forward / backward / densification 분해)
- peak GPU memory
- 활성 Gaussian 비율 (freeze rate)
- Gaussian 총 개수

**필수 산출: Pareto 곡선.** freeze rate 또는 $C_{\text{budget}}$을 sweep해 quality vs wall-clock 곡선. 단일 operating point는 설득력이 없다.

---

## 8. Baseline 목록 (Stage 1에서 전부 필요)

| Baseline | 설명 | 목적 |
|---|---|---|
| **No freezing** | 전량 최적화 | 품질 상한 / 속도 하한 |
| **Age-based** | $k$ 프레임 후 동결 | **가장 중요. 반드시 이겨야 함** |
| **Random** | 동일 freeze rate로 무작위 동결 | confidence가 정보를 담는지 |
| **Distance-only** | $z$만으로 판정 | 정규화의 기여 분리 |
| **Count-only** | $n_{\text{obs}}$만으로 판정 | 정규화의 기여 분리 |
| **Triangulation angle** ($p{=}1$) | 고전 SfM 기준 | $p^*>1$ 주장의 직접 비교군 |
| RTG-SLAM 류 | opacity 누적 휴리스틱 | 선행연구 비교 |

**필수 ablation**: 정규화 지수 $p$를 0으로 보내도 성능이 유지되면 거리 항은 장식이다. 이걸 **우리가 먼저** 확인한다.

---

## 9. 리스크 및 대응

| 리스크 | 징후 | 대응 |
|---|---|---|
| **H1 불성립** | 어떤 $p$에서도 collapse 없음 | motion blur로 근거리 실효 $\sigma_d$가 커진 것이 원인일 가능성 → blur metric으로 층화 재분석. "streaming에서는 이론적 $z^2$가 성립하지 않으며 원인은 motion blur"는 **negative-but-interesting 결과**로 발표 가치 있음 |
| **$p^* \approx 1$** | 고전 기준과 동일 | H3(지표별 $p^*$ 분화)로 논지 전환. 축 A/B/D는 그대로 유효 |
| **age baseline을 못 이김** | H4 실패 | 정보량 기준을 정확도 축(스케일 앵커/drift)으로 재프레이밍. 연산 절감 주장 철회 |
| **freeze해도 wall-clock 안 줄어듦** | speedup ≈ 0 | 확정/비확정 Gaussian이 메모리에 섞여 warp divergence 발생. **주기적 compaction/partitioning 필수** (pruning 문헌의 "sparsity ≠ speedup" 문제) |
| **오확정 후 error absorption** | 남은 Gaussian이 왜곡 흡수 | 이방성 확정(축 A) + hysteresis thaw (확정 $\tau_{\text{high}}$, 해제 $\tau_{\text{low}} < \tau_{\text{high}}$) + free-space violation 기반 무효화 |
| **thaw 후 optimizer state stale** | 해제 직후 불안정 | Adam moment 리셋 정책 명시 및 ablation |

---

## 10. Claude Code 작업 지시

### 10.1 지금 할 일

**Stage 0만 구현한다.** Stage 1 이상의 코드는 작성하지 않는다.

순서:

1. **환경 조사 보고 먼저.** 기존 3DGS 구현 후보(원본 3DGS / gsplat / MonoGS 등) 중 어느 것을 개조할지, 각각의 장단점과 instrumentation 난이도를 정리해 **먼저 제시하고 승인을 받는다.** 코드를 바로 짜지 말 것.
2. 데이터셋 로더 확인 — Waymo(또는 KITTI) + Replica. GT depth/pose 접근 경로 확인.
3. `stage0_probe.py` — 프레임 순차 투입 학습 루프 + per-Gaussian 통계 로깅 (§6.3). **알고리즘은 손대지 않는다. 계측만 추가한다.**
4. `stage0_analyze.py` — $p$ sweep, collapse 정량화(§6.7), Figure 1 생성
5. `baselines/age_freeze.py` — §6.9
6. `outputs/stage0/REPORT.md` — H1/H2/H3 판정 포함

### 10.2 하지 말 것

- ❌ Stage 1–3 시스템을 미리 구현
- ❌ 새로운 confidence 항을 임의로 추가/변형
- ❌ FLOPs로 연산량 보고 (wall-clock만)
- ❌ 단일 operating point만으로 결론
- ❌ 시각적 인상("겹쳐 보인다")으로 collapse 판정 — §6.7 정량 지표 필수

### 10.3 코딩 규약

- **Instrumentation과 알고리즘을 분리**한다. probe는 baseline 3DGS 동작을 바꾸지 않아야 하며, 계측 on/off로 결과가 동일한지 검증한다.
- Gaussian **lineage id**를 유지한다. densification(split/clone) 시 부모 id를 상속해 통계 추적이 끊기지 않게 한다.
- 로그는 **parquet**으로 저장. 시퀀스당 수십만 행이 나올 수 있다.
- 모든 실험은 **seed 고정 + config 파일 기반**. config를 outputs에 복사 저장.
- 저기여 Gaussian(`contrib` 하위 X%)은 분석에서 제외 — 옵션으로 두고 임계값 민감도를 확인한다.
- A100 **1장** 기준. OOM 시 씬을 줄이지 말고 batch/resolution을 먼저 조정하고 보고한다.

### 10.4 중간 보고 시점

각 단계 완료 시 **멈추고 보고**한다:
- (1) 베이스 구현 선정안 → **승인 대기**
- (3) 첫 시퀀스 로깅 성공 → 통계 샘플 100행 + 분포 요약 제시
- (4) 첫 collapse 플롯 → **$p^*$ 후보와 정량 지표 제시, 해석은 함께 논의**

### 10.5 산출물 최종 체크리스트

- [ ] `fig1_collapse.png` — 3-panel ($p{=}0$ / $p{=}1$ / $p{=}p^*$), depth bin별 곡선
- [ ] `p_sweep_metrics.png` — 분산·$R^2$·Spearman vs $p$, $p^*$ 표시
- [ ] `ause_by_depth_bin.png`
- [ ] `p_sweep.csv`, `age_baseline.csv`
- [ ] `REPORT.md` — H1/H2/H3 판정 + Go/No-Go 권고

---

## 11. 포지셔닝 요약표 (논문 Table 1 초안)

| | RTG-SLAM | GS-SLAM 계열 | **제안** |
|---|---|---|---|
| 입력 | RGB-D | 주로 RGB-D | **단안 streaming** |
| 확정 기준 | opacity 누적 휴리스틱 | keyframe window / age | **depth-normalized information sufficiency (유도됨)** |
| 확정 단위 | Gaussian 이진 | window 이진 | **DoF별 이방성, 점진적** |
| 씬 가정 | static | 대부분 static | **동적 ($c^{\text{geo}} \times c^{\text{static}}$)** |
| 스케줄 | 고정 규칙 | 고정 window | **연산 예산 기반 top-k** |
| 연산 절감 | unstable-only + 픽셀 게이팅 | local BA | + **깊이 계층 temporal proxy** |

---

## 12. 참고 문헌 (확인 필요)

- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023
- Peng et al., *RTG-SLAM: Real-time 3D Reconstruction at Scale using Gaussian Splatting*, SIGGRAPH 2024 — **최우선 정독. stable/unstable 메커니즘 상세 파악 필수**
- Jiang et al., *FisherRF: Active View Selection and Uncertainty Quantification for Radiance Fields using Fisher Information*, ECCV 2024
- Hanson et al., *PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting*, CVPR 2025
- Nie et al., *Large Language Diffusion Models (LLaDA)*, 2025 — 스케줄러 참조. **"확정=연산절감"이 아님에 유의**
- Dynamics-Aware Gaussian Splatting Streaming (arXiv 2411.14847) — 동적/정적 분리 streaming 비교군
- Schönberger & Frahm, *Structure-from-Motion Revisited* (COLMAP) — triangulation angle 기준의 출처

> 서지 정보는 **직접 확인 후 인용**할 것. 위 목록은 시작점이며 완전하지 않다.

---

## 13. 한 줄 피치 (초록 초안용)

> Streaming 4D reconstruction에서 Gaussian의 확정 시점을 **깊이로 정규화된 삼각측량 정보 충분성**으로 결정하고, 확정을 **자유도별 이방성**으로 점진 수행하며, 확정 여유분을 **동적 객체 탐지 신호**로 재활용하는 **연산 예산 제어형** 프레임워크.
