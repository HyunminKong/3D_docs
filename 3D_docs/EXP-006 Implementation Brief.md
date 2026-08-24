# EXP-006 Implementation Brief
**Utility-Routed Local Plasticity Memory with Geometry Evidence**

**Revision:** v2.8, 2026-08-25

**Status:** the v2.8 validation lock below is implementation-authoritative. The v2.7 correction and complete v2.6 protocol remain preserved as historical/reproduction documentation. When they conflict, v2.8 and D021–D023 take precedence.

# v2.8 one-shot validation lock

Expanded train-only OOF evaluation fixes the validation model before any validation image/output access:

```text
fast code carrier       visual token correspondence
current adaptation      exactly one local-code TTT step
memory application      z_current + 0.10 * z_memory, clamped to [-1,1]
router descriptors      current, candidate, difference, product
router scalars          20 visual/current/source adaptation-history values
excluded router input   predicted-alignment validity/inlier/residual/coverage
utility model           train-only StandardScaler → PCA(16) → Ridge(alpha=1)
routing                 top predicted utility if > 0; otherwise current-only
candidate count         5
```

The 20 scalar inputs are the original non-alignment indices 0–11 plus source/current history indices 16–23. The four predicted-alignment values at indices 12–15 are an ablation only. Query/future quantities remain labels/evaluation only.

The neural utility/risk MLP is also an ablation. Although harmful labels expanded from 2 to 17 across three folds, its explicit risk output did not reduce selected harm. EXP-006 therefore does not claim a successful separate risk classifier.

One-shot validation cannot tune any feature, normalization, PCA rank, regularization, threshold, reuse strength, candidate construction, or ordering. The exposed test split remains prohibited.

The v2.8 descriptive decision rule is fixed in D024 before validation output access. The router must exceed the 0.01 mean-utility deadband and every registered control, have zero harmful validation components, no greater directional harm than visual mean, and at least 0.20 acceptance. Any failed item rejects the strict validation claim; the two-component result remains feasibility evidence only.

---

# v2.7 architecture correction

## Why v2.6 was revised

Exact fold-specific evaluation changed the conclusion obtained from full-fit train diagnostics:

```text
visual local transport                  +0.0162 utility, 100% coverage, 2% harm
predicted geometry transport            +0.0161 utility, 48% coverage, 6.25% harm
predicted geometry+appearance transport +0.0162 utility, 48% coverage, 8.33% harm
```

Predicted geometry therefore does not support H2-P as a deployable update carrier. It still contains routing evidence: a fixed full observable OOF router achieved +0.0224 utility with 0% harm, versus +0.0202 after removing geometry statistics.

The current risk labels are not sufficient for neural risk training: only 2/100 candidates are harmful, both in one overlap fold. Validation remains unopened while the train benchmark is expanded.

## Superseding method contract

```text
frozen VGGT/current evidence
        ↓
custom local plasticity head
        ↓
one current-context TTT step
        │
visual transport of K past local codes
        │
predicted geometry alignment ──> evidence only
        ↓
observable utility/risk router
        ↓
select one or reject all
        ↓
clamp(z_current + 0.10 * z_memory, -1, 1)
```

The following rules supersede v2.6:

1. `visual_transport` is the primary fast-code transport.
2. Sim(3) validity, inlier ratio, residual, and coverage are router scalar inputs only.
3. Failed predicted alignment does not invalidate a candidate with valid visual transport.
4. The router receives pooled current/candidate descriptors, their difference/product, current online statistics, visual-transport statistics, and geometry evidence. It receives no query/future quantity.
5. The primary route is hard select-or-reject. Accepted code is a fixed-strength 0.10 residual after exactly one current TTT step.
6. A second current TTT step is prohibited in the primary method; it is a negative control.
7. Five-candidate visual mean remains a required safe control. It is not silently used as the learned method.
8. Architecture decisions must use exact held-out-overlap fold heads. Full-fit same-train results are diagnostics only.
9. Neural risk training and official validation are gated until harmful labels occur in at least two independent overlap folds and every grouped training partition contains both benefit and harm.
10. Persistent bank/consolidation remains out of scope until the utility/risk gate passes.

## v2.7 model interface

`ObservableUtilityRiskRouter` consumes:

```text
current pooled descriptor                 64
candidate pooled descriptor               64
shared projection of each                 32
current, candidate, difference, product  128 total
normalized observable scalar evidence     16 minimum
```

The 16-scalar feasibility contract contains current/base and current-post ratios, candidate current-objective behavior, code magnitude/spatial statistics, visual entropy/max-weight/coverage, appearance similarity, and predicted-geometry validity/inlier/residual/correspondence coverage. Production training may extend the scalar list only through a new documented protocol revision using train-only normalization.

The output remains `(utility_hat, risk_logit)`. Candidate availability refers to actual visual-code availability, not predicted-geometry validity.

## Active go/no-go sequence

```text
expand train overlap components and adaptation regimes
        ↓
repeat exact OOF visual utility + label-health audit
        ↓
risk identifiable in every grouped fold?
   NO → expand data; do not open validation
   YES
        ↓
train neural utility/risk router with fixed train-only protocol
        ↓
beat visual mean/current-loss controls under grouped OOF?
   NO → H4 unsupported; no continual bank
   YES
        ↓
one-shot validation, then EXP-007 continual consolidation
```

---

# Preserved v2.6 protocol (historical/reproduction only)

## Status convention

- **[Repository fact]**: 현재 repository 문서/코드에 이미 명시된 사항.
- **[EXP-006 decision]**: EXP-006을 실제 구현 가능하게 만들기 위해 본 brief에서 새로 고정하는 설계.
- **[Diagnostic only]**: 결과 해석에는 사용할 수 있으나 deployable method에는 포함하지 않는 항목.

---

# 0. Experiment boundary

### [Repository fact]

현재 연구의 central memory object는 global/slot fast weight가 아니라 **spatially addressable 3D plasticity atom**이다. 첫 backbone은 frozen VGGT이며, 새로운 geometry/plasticity head에서 local TTT를 수행한다. Retrieval 이후 3D coordinate + appearance correspondence로 atom을 현재 context에 transport하고, learned utility/risk head가 candidate를 select/reject/mix하는 방향이 채택되어 있다.

EXP-005에서는 per-token log-depth residual을 local atom으로 사용했고, visual transport보다 oracle 3D coordinate transport가 강했으며 geometry+appearance가 가장 좋은 controlled effect를 보였다. 하지만 heuristic online score는 일부 episode에서 negative transfer를 발생시켰다.

EXP-006의 목표는 이를 **trainable atom + predicted geometry + learned utility/risk routing**으로 바꾸는 것이다. 기존 six-episode test split은 이미 EXP-005에서 사용되었으므로 EXP-006에서 절대 사용하지 않는다.

### [EXP-006 decision]

EXP-006의 범위는 다음 식으로 고정한다.

```text
Frozen VGGT feature tokens
        ↓
Custom frozen base geometry head
(depth + relative pose)
        ↓
Trainable local plasticity decoder
        ↓
Per-token 3D plasticity atom
        ↓
Predicted local 3D geometry
        ↓
Appearance-assisted Sim(3) alignment
        ↓
Geometry / geometry+appearance transport
        ↓
1-step local TTT on atom code
        ↓
Utility / risk prediction
        ↓
Select candidate or reject to current-only TTT
```

**Long-term memory bank, merge, eviction, aging, reactivation은 구현하지 않는다.**

---

# 1. 검증할 가설

## H2-P — Predicted geometry-conditioned transport

### [Repository fact]

H2는 oracle coordinate에서는 supported이지만 **predicted-coordinate validation은 아직 open**이다.

### [EXP-006 hypothesis]

> Custom head가 예측한 depth와 relative pose로 구성한 local 3D geometry를 appearance correspondence로 cross-traversal align하면, visual-only 또는 untransported local reuse보다 높은 future utility를 유지할 수 있다.

핵심 비교:

```text
visual transport
       vs
predicted geometry transport
       vs
predicted geometry + appearance transport
```

Oracle pose coordinate transport는 upper bound만 제공한다.

---

## H4 — Learnable risk-aware routing

### [Repository fact]

현재 geometry score는 future utility와 평균적으로 관계가 있지만 harmful reuse를 안정적으로 제거하지 못했다. 따라서 future-frame supervision을 이용한 utility/risk head가 필요한 상태다.

### [EXP-006 hypothesis]

> Current context에서 관측 가능한 geometry/appearance/transport statistics만으로 future benefit과 negative-transfer risk를 예측하여, heuristic current-loss selector보다 낮은 regret과 낮은 harm rate를 달성할 수 있다.

---

# 2. 최소 구현 범위

### 구현한다

1. Frozen VGGT patch feature extractor 유지.
2. 기존 custom geometry head에서 base depth와 relative pose 사용.
3. 새로운 per-token plasticity code.
4. plasticity code → local log-depth correction을 만드는 trainable decoder.
5. predicted depth + predicted relative pose로 local 3D point 생성.
6. source/current geometry 사이의 predicted Sim(3) alignment.
7. visual / geometry / geometry+appearance transport.
8. candidate별 current-context geometry statistics.
9. utility regression + risk classification head.
10. reject 가능한 router.
11. EXP-006 train/validation script.
12. required ablation evaluator.

### 구현하지 않는다

- persistent large memory bank
- FAISS 등의 scalable retrieval engine
- atom merge
- atom eviction
- age policy
- long-term utility statistics update
- dynamic-object motion state
- 4D tracking memory
- online backbone update
- online main geometry-head weight update
- oracle pose를 production path에서 사용하는 기능

---

# 3. 3D plasticity atom tensor 정의

현재 VGGT wrapper는 frozen patch token을

```text
F: [B, V, P, 2048]
```

형태로 출력한다. 224×224 입력과 patch size 14를 사용할 경우 token grid는 16×16이므로 `P=256`이다. VGGT prediction heads는 feature extractor path에서 disable되어 있다.

### [EXP-006 decision]

하나의 atom set을 다음 dataclass로 고정한다.

```python
@dataclass
class PlasticityAtom:
    xyz: Tensor          # [B, V, P, 3]
    scale: Tensor        # [B, V, P, 1]
    key: Tensor          # [B, V, P, 64]
    code: Tensor         # [B, V, P, 8]
    confidence: Tensor   # [B, V, P, 1]
```

의미는 다음과 같다.

| Tensor | 정의 |
|---|---|
| `xyz` | 해당 token의 predicted 3D anchor, segment-local reference coordinate |
| `scale` | 동일 view 안에서 계산한 token별 predicted point cloud median 8-NN spacing; transport kernel bandwidth로 실제 사용 |
| `key` | frozen feature에서 projection한 normalized appearance key |
| `code` | online TTT에서 실제로 update되는 fast state |
| `confidence` | Stage-0에서 보정한 base confidence와 detached track support의 geometric mean, online update 대상 아님 |

`observation_count`, `age`, `past_utility`는 EXP-006에서는 persistent tensor로 구현하지 않는다. 생성 시 `observation_count=1`, `age=0`인 ephemeral metadata로만 둔다.

`confidence`는 임의의 untrained head 출력을 사용하지 않는다.

```text
c_base  = sigmoid(confidence_head(token))
c_track = token-wise aggregate(visibility × tracker_confidence)
confidence = sqrt(clamp(c_base, 1e-4, 1) × clamp(stopgrad(c_track), 1e-4, 1))
```

`c_base`는 Stage 0의 train-only teacher-confidence distillation으로 보정한다. FastVGGT DPT confidence는 `[0,1]` 확률이 아니라 `1+exp(logit)`이므로 직접 clamp하지 않는다. train context에서만 raw logit과 robust quantile을 fit한다.

```text
ell_conf = log(clamp(c_teacher_raw - 1, min=1e-6))
q05, q95 = train-only quantile(ell_conf, [0.05, 0.95])
c_teacher = clamp((ell_conf - q05) / (q95 - q05), 0, 1)
```

`c_track`은 frozen tracker의 camera output이 아니라 track evidence만 사용한다. 둘 중 하나가 구현되지 않았으면 confidence를 router input이나 Sim(3) weight에 넣지 않으며, 해당 run은 full A7이 아니다.

`key_projection`은 random projection 상태로 Sim(3) correspondence에 바로 사용하지 않는다.

```text
key_projection = Linear(2048, 64, bias=False)
initialization = top-64 PCA components of cached train-only VGGT tokens
key = L2_normalize(key_projection(LayerNorm(F)))
```

PCA는 train context에서 deterministic reservoir sampling한 최대 100,000 visible token으로 fit한다. PCA mean/components, sampling seed와 feature checkpoint hash를 config에 저장한다. Stage 1에서는 §6.1.1의 `L_key`로 projection을 계속 학습하지만 validation correspondence를 initialization이나 threshold tuning에 사용하지 않는다.

`xyz`와 `scale`은 code-corrected depth가 아니라 frozen `d_base`와 frozen predicted pose에서 생성하고 atom 수명 동안 detach한다. Adapted depth로 anchor 자체를 다시 움직이는 circular transport는 EXP-006에서 금지한다. EXP-006의 fast code는 depth residual만 바꾸며 camera pose는 Stage-0 base prediction으로 고정된다. 따라서 결과에서 pose adaptation을 claim하지 않는다.

---

## Atom code가 geometry에 미치는 영향

기존 EXP-005는 scalar log-depth residual

```text
d = d_base · exp(atom)
```

을 사용했다.

EXP-006에서는 이 아이디어를 유지하되 trainable하게 만든다.

```text
q = Linear64(LayerNorm(F))

h1 = MLP([q, z])
h0 = MLP([q, 0])

δlogd = 0.5 · tanh(h1 - h0)

d = d_base · exp(δlogd)
```

여기서

```text
q : [B,V,P,64]
z : [B,V,P,8]
δlogd : [B,V,P,1]
```

이다.

`h1-h0` 구조를 사용하므로 **z=0이면 residual=0**이 자동으로 보장된다.

즉 atom을 사용하지 않은 결과는 항상 기존 base geometry prediction과 정확히 일치한다.

### Online-adaptable parameter

EXP-006 online TTT에서는 오직

```text
PlasticityAtom.code
```

만 update한다.

VGGT backbone, base geometry head, key projection, plasticity decoder, utility/risk head는 online에서 전부 freeze한다.

---

# 4. Predicted geometry 기반 transport

## 4.1 Predicted geometry 생성

현재 custom geometry head는 depth와 6-DoF relative pose를 출력하며, 기존 loss code에는 predicted twist를 relative `w2c` matrix로 변환하는 `relative_w2c_from_twist()`가 이미 존재한다.

### [EXP-006 decision]

EXP-006 production path에서는

```python
w2c_pred = relative_w2c_from_twist(pred["relative_pose"])
```

만 사용한다.

**`FrozenVGGTGeometryTracker["w2c"]`를 geometry prediction으로 사용하면 안 된다.**

Frozen VGGT tracker는 다음 세 signal만 online objective용 evidence로 사용한다.

```text
track
visibility
confidence
```

Frozen tracker의 camera prediction은 버린다.

이는 foundation output head가 Revisit3D의 online prediction처럼 보이는 것을 막기 위한 것이다. 기존 tracker 코드도 이를 controlled geometry prior라고 명시하고 있다.

---

## 4.2 Local 3D point 생성

각 token center `(u,v)`와 predicted depth `d`에 대해

```text
X_cam =
[
 (u-cx)/fx · d,
 (v-cy)/fy · d,
 d
]
```

를 계산하고,

```text
X_ref = inv(w2c_pred) · X_cam
```

으로 segment 첫 view 기준 local coordinate에 올린다.

Dataset의 calibrated intrinsics는 입력 camera metadata로 사용한다.

결과:

```text
X_pred: [B,V,P,3]
```

---

# 4.3 Cross-traversal predicted alignment

A와 A'는 서로 다른 local coordinate gauge를 가지므로 단순히 두 `xyz`를 비교해서는 안 된다.

### [EXP-006 decision]

source와 current 사이의 coordinate transform은 **appearance correspondence + weighted Sim(3) alignment​**로 추정한다.

### Step 1 — appearance correspondence

Flatten:

```text
Ks : [Ns,64]
Kt : [Nt,64]

Ns = Vs × Ps
Nt = Vt × Pt
```

cosine similarity:

```text
S = Kt · Ksᵀ
```

mutual nearest-neighbor pair만 사용한다.

기본 filtering:

```text
cosine similarity >= 0.60
minimum correspondences = 32
```

32개 미만이면

```text
alignment_valid = False
```

로 표시한다.

MNN index와 correspondence set은 discrete operation이므로 뒤의 meta-gradient에서 detach한다.

---

### Step 2 — weighted Sim(3)

Correspondence `(Xs_j, Xt_i)`에 대해 weighted Umeyama를 수행한다.

weight:

```text
w_ij =
    clamp((cos_ij - 0.60) / 0.40, 0, 1)
    * sqrt(confidence_s,j * confidence_t,i)
```

추정:

```text
Xt ≈ s R Xs + t
```

첫 추정 이후 각 correspondence의 normalized residual을 계산한다.

```text
rho_ij = ||Xt_i - (s R Xs_j + t)|| / max(ell_t,i, 1e-6)
rho_ij <= 2.5
```

인 correspondence만 남겨 한 번 재추정한다. `ell_t,i`는 target token의 median 8-NN spacing이다. 따라서 `2.5`는 meter가 아니라 **2.5 local spacings**인 dimensionless threshold다.

다음 조건을 모두 통과할 때만 `alignment_valid=True`다.

```text
number of MNN correspondences >= 32
number of second-pass inliers >= 32
second-pass inlier ratio >= 0.25
source and target effective rank >= 2
lambda_2 / max(lambda_1, 1e-8) >= 0.01
0.1 <= Sim(3) scale <= 10.0
det(R) > 0
all transform values finite
normalized median residual <= 2.5
```

scale bound는 train split의 depth gauge가 이 범위를 벗어난다는 Stage-0 evidence가 있을 때만 training 전에 한 번 넓힐 수 있다. validation 결과를 보고 바꾸지 않는다.

두 번째 추정이 최종 alignment다.

Sim(3)를 쓰는 이유는 monocular predicted depth/pose의 cross-traversal scale gauge 오차를 SE(3)보다 안전하게 처리하기 위해서다.

---

# 4.4 Transport kernel

Aligned source:

```text
X̄s = s R Xs + t
```

각 target token에 대해 aligned source 중 3D nearest `k=8`개를 사용한다.

local spatial scale은 target마다 하나의 global scalar가 아니라 token별로 계산한다. 각 view는 같은 segment-local 좌표계로 backproject되지만, scale의 8-NN 후보는 **해당 token과 동일한 view의 point만** 사용한다. 여러 view를 합치면 동일 표면을 관측한 cross-view near-duplicate가 bandwidth를 인위적으로 0에 가깝게 만들어 metric-normalized residual을 왜곡하므로 금지한다.

```text
sigma_t,i = median distance to target 8-NN
sigma_s,j = median distance to aligned-source 8-NN
```

---

## Geometry-only

```text
logit_ij =
  - ||Xt_i - X̄s_j||²
    / (2 (sigma_t,i² + sigma_s,j² + 1e-8))
```

---

## Geometry + appearance

EXP-005의 coordinate+appearance condition을 계승해

```text
logit_ij =
    - ||Xt_i - X̄s_j||²
      / (2 (sigma_t,i² + sigma_s,j² + 1e-8))
    + 5 · cos(Kt_i, Ks_j)
```

로 고정한다.

weights:

```text
W_ij = softmax_j(logit_ij)
```

transport:

```text
z_transport_i = Σ_j W_ij z_source_j
```

---

## Visual-only baseline

EXP-005와 동일한 concept을 유지한다.

```text
logit_ij = cos(Kt_i,Ks_j) / 0.07
```

3D coordinate는 사용하지 않는다. 기존 EXP-005 visual transport도 frozen-feature attention으로 residual을 이동했다.

---

## Invalid predicted alignment

`alignment_valid=False`일 때:

- `geometry`
- `geometry+appearance`

candidate는 **mask 처리**한다.

Visual transport로 자동 fallback하지 않는다.

그렇게 해야 predicted geometry failure가 visual transport 성능으로 숨겨지는 것을 막을 수 있다.

---

# 5. Utility/risk head 입력과 출력

## Candidate pool

Large bank는 만들지 않는다.

각 `A → B → A'` training episode에서 A'에 대해 정확히 **5개의 candidate**를 만든다.

```text
C0 = atom from A
C1 = atom from B
C2 = foreign atom 1
C3 = foreign atom 2
C4 = foreign atom 3
```

foreign atom은 동일 split에서 현재 episode의 source/target scene과 scene을 공유하지 않는 episode에서 선택한다.

선택은 reproducibility를 위해

```text
sorted(eligible_episode_id)
rotated by int(SHA256(f"{seed}:{current_episode_id}")[:8], 16)
```

방식으로 고정한다.

Python process마다 달라질 수 있는 built-in `hash()`는 사용하지 않는다. 선택된 candidate episode ID 다섯 개를 result row에 저장한다.

**A가 정답이라는 label을 절대 주지 않는다.**

D005에 따라 실제 future utility가 candidate correctness를 정의한다.

---

## Input

각 candidate마다:

```text
shared-projected current descriptor   : 32
shared-projected candidate descriptor : 32
current - candidate                   : 32
current * candidate                    : 32
observable scalar statistics       : 24
--------------------------------------
total                               152
```

따라서

```text
utility_input: [B, K, 152]
```

이다. 64-D mean descriptor를 shared `Linear(64,32) → GELU → LayerNorm(32)`로 먼저 압축한다. 단순 mean descriptor 두 개만 주는 대신 difference와 element-wise product를 명시적으로 추가하여 candidate/current interaction을 표현하면서, 작은 train set에서 router parameter 수가 과도해지는 것을 막는다.

`current descriptor`와 `candidate descriptor`는 각각 64-D local key의 mean pooling이다.

### 24 scalar statistics

순서를 고정한다.

```text
0  current_pre_online_loss
1  current_post_online_loss
2  current_loss_drop = current_pre - current_post
3  candidate_pre_online_loss
4  candidate_post_online_loss
5  candidate_loss_drop = candidate_pre - candidate_post
6  candidate_post_online_loss - current_post_online_loss

7  current_track_coverage
8  candidate_track_coverage
9  current_mean_3d_residual
10 candidate_mean_3d_residual

11 alignment_valid
12 alignment_inlier_ratio
13 normalized_alignment_median_residual
14 mean_correspondence_cosine

15 normalized_transport_entropy
16 mean_transport_max_weight
17 transport_coverage

18 mean_abs_transported_code
19 rms_transported_code
20 transported_code_spatial_std
21 matched_key_difference_mean
22 matched_key_difference_std
23 online_code_gradient_cosine
```

`candidate_pre_online_loss`는 transported initialization에서 online step을 수행하기 전의 current-context loss이고, `candidate_post_online_loss`는 정확히 한 step 후의 loss다. `online_code_gradient_cosine`은 current-only zero initialization의 code gradient와 candidate initialization의 code gradient 간 cosine이며 둘 다 current context에서만 계산한다.

현재 `track_3d_consistency_loss(..., return_stats=True)`에서 track coverage, mean track weight, mean 3D residual을 이미 반환할 수 있다.

모든 scalar normalization mean/std는 **train split에서만 계산**한다.

Predicted-geometry candidate가 `alignment_valid=False`이면 router가 무조건 reject한다. 해당 candidate는 utility regression/risk BCE에서 제외하고 invalid-alignment rate로 별도 보고한다. invalid flag 자체를 risk label처럼 학습시키지 않는다.

---

## Head architecture

```text
152
 ↓
Linear(152,64)
GELU
LayerNorm(64)
 ↓
Linear(64,32)
GELU
 ↓
Linear(32,2)
```

output:

```text
utility_hat : [B,K]
risk_logit  : [B,K]
```

`utility_hat`는 predicted normalized future improvement.

`sigmoid(risk_logit)`은 negative-transfer probability.

---

# 6. Online loss와 future meta-objective

## 6.1 Online TTT loss

### [Repository fact]

현재 검증 가능한 geometry-only signal은 frozen track correspondence를 이용한 3D consistency이며, naïve reprojection은 objective-health probe에서 degenerate behavior가 확인됐다.

현재 코드 역시 `track_3d_consistency_loss`와 edge-aware depth smoothness를 이미 제공한다.

### [EXP-006 decision]

EXP-006 main online loss는 **reprojection을 제외**한다.

```text
L_online =
    L_track3D
    + 1e-3 L_smooth
    + 1e-4 L_code
```

where

```text
L_code = mean(z²)
```

중요:

`L_track3D`의 camera pose는

```text
predicted relative_pose
```

를 사용한다.

Frozen VGGT tracker의 `w2c`는 사용하지 않는다.

---

### Online update

EXP-005의 one-step normalized update를 유지한다. 기존 probe도 gradient magnitude가 storage object 자체가 되는 것을 막기 위해 per-segment normalization을 사용했다.

```text
g = ∂L_online / ∂z

ḡ = g /
     (mean(|g| over V,P,R) + 1e-6)

z₁ = clamp(z₀ - 0.05 ḡ, -1, 1)
```

main protocol:

```text
TTT steps = 1
TTT step size = 0.05
```

모든 ablation에서 동일하게 유지한다.

---

### 6.1.1 First-order meta-gradient contract

EXP-006은 full MAML이 아니라 다음 경계를 갖는 first-order meta-learning으로 고정한다.

```text
online gradient g and its normalization denominator : detach
MNN indices / correspondence indices                : detach
second-pass inlier mask                              : detach
estimated Sim(3) transform                           : detach
3D k-NN neighbor indices                             : detach
appearance logits inside fixed neighbors             : differentiable
transported code and plasticity decoder outer path   : differentiable
future query loss through decoder                    : differentiable
```

즉 Hessian을 만들지 않지만, fixed correspondence/neighbor 안에서 appearance transport weight와 decoder가 future outer loss로 학습될 수 있다. 구현은 `create_graph=False`를 강제하고 online gradient tensor만 detach한다. source/current/query feature extractor와 base geometry head는 항상 `eval()`과 `requires_grad_(False)` 상태다.

Discrete MNN만으로 key projection을 학습시키지 않는다. Stage 1에는 current/query를 사용하지 않는 auxiliary key loss를 추가한다.

```text
L_key = symmetric InfoNCE(projected token keys)
temperature = 0.07
```

positive pair는 context frame 안의 frozen tracker correspondence 또는 **unprojected frozen VGGT feature**로 미리 정한 detached MNN pair이며, negative는 같은 batch의 다른 visible token이다. A' query feature는 positive mining과 `L_key`에 절대 사용하지 않는다.

---

## 6.2 Future utility target

A' context는 online adaptation에 사용한다.

A' query는 **절대로 online update에 사용하지 않는다.**

이는 repository의 explicit protocol이다.

A' query에서:

```text
LQ,current
LQ,j
```

를 계산한다.

### Query readout boundary

Current context에서 얻은 code를 query token에서 평가할 때는 `A' current key → A' query key`의 **visual-only transport**를 사용한다. 이 readout은 모든 current/candidate path에 동일하게 적용하며 query의 predicted `xyz`, `scale`, pose로 Sim(3)을 추정하지 않는다. Query feature는 read-only future prediction과 outer loss의 미분 경로에만 참여하고 source→current alignment, online update, candidate selection, utility/risk 입력에는 참여하지 않는다. 따라서 H2-P의 predicted geometry transport 비교는 오직 source context→A' current context 구간에 해당한다.

candidate j의 normalized utility:

```text
u_j =
(LQ,current - LQ,j)
/
(|LQ,current| + 1e-6)
```

따라서

```text
u_j > +epsilon_u  → beneficial reuse
|u_j| <= epsilon_u → neutral
u_j < -epsilon_u  → harmful reuse
```

`epsilon_u`는 validation을 보기 전에 train-only null control로 고정한다.

```text
epsilon_u = max(
    0.01,
    percentile_95(abs(u_null))
)
```

`u_null`은 같은 current-only zero atom을 candidate path에 넣어 서로 다른 dataloader/compute seed로 두 번 평가한 normalized utility 차이다. 즉 1% 이하이거나 반복 측정 잡음 범위 안의 변화는 beneficial/harmful label로 만들지 않는다. 이 값은 Stage 1/2 및 모든 validation ablation에 동일하게 사용하고 결과를 보고 재조정하지 않는다.

risk target:

```text
r_j = 1[u_j < -epsilon_u]
risk BCE mask = 1[abs(u_j) > epsilon_u]
```

---

## 6.3 Utility/risk meta-objective

```text
L_utility =
SmoothL1(utility_hat, stopgrad(u))
```

```text
L_risk =
masked_BCEWithLogits(risk_logit, stopgrad(r))
```

class imbalance용 `pos_weight`는 neutral과 invalid candidate를 제외한 train candidate utility에서 한 번 계산하고 고정한다.

최종:

```text
L_router =
L_utility
+ L_risk
```

Future query는 **label 생성과 outer supervision에만 사용**한다.

Utility/risk head 입력에는 query에서 계산된 어떤 quantity도 포함하지 않는다.

---

## 6.4 Routing rule

Primary EXP-006 evaluation은 hard routing으로 고정한다.

candidate j가 eligible하려면:

```text
utility_hat_j > epsilon_u
AND
sigmoid(risk_logit_j) < 0.5
AND
alignment_valid_j == True
```

단 visual-only condition에서는 `alignment_valid` 조건을 사용하지 않는다.

eligible candidate가 없으면:

```text
current-only TTT
```

를 반환한다.

있으면 가장 큰 `utility_hat`을 가진 candidate 하나를 선택한다.

---

## Soft mixing

구조적으로 지원은 하되 main result가 아니다.

eligible candidate들에:

```text
w_j = softmax(utility_hat_j / 0.1)
```

을 적용하여 transported code를 mix한다.

**Hard routing이 primary, soft routing은 secondary ablation​**으로 고정한다.

---

# 7. 학습 순서

## Stage 0 — Predicted geometry bootstrap

### 목적

EXP-006 시작 전에 base custom head의 depth와 relative pose가 같은 gauge에서 non-degenerate해야 한다.

현재 depth bootstrap은 frozen VGGT pseudo depth를 offline calibration 용도로 사용하며, 이는 online signal이나 final score가 아니다.

### [EXP-006 decision]

시작 checkpoint와 head type을 다음으로 고정한다.

```text
checkpoint = revisit3d/checkpoints/depth_bootstrap_anchored_dev_epoch1.pt
head_type  = anchored
base prediction state = all-zero anchored state
foundation checkpoint = FastVGGT/ckpt/model_tracker_fixed_e20.pt
```

`anchored_router_dev_epoch1.pt`는 과거 router training의 영향을 받았으므로 Stage-0 시작점으로 사용하지 않는다. `anchored` state는 새로운 central memory로 재사용하는 것이 아니라, zero state에서 depth-bootstrap base prediction만 읽기 위한 호환 layer다.

기존 `pretrain_depth_bootstrap.py`는 depth loss만 학습하므로 그 checkpoint의 `pose_head`와 `confidence_head`를 calibrated output으로 간주하면 안 된다. Stage 0에서 token trunk와 depth head를 freeze하고 다음 두 head만 **train split에서만** distill한다.

```text
trainable: pose_head, confidence_head
frozen: VGGT feature extractor, token trunk, depth_head, point_head
```

teacher camera는 view 0을 identity로 만든 relative transform으로 변환한다.

```text
T_teacher_rel[v] = T_teacher[v] @ inverse(T_teacher[0])
T_pred_rel       = relative_w2c_from_twist(relative_pose)

L_pose =
    rotation_geodesic(T_pred_rel, T_teacher_rel)
  + 0.5 * translation_direction_loss
  + 0.1 * SmoothL1(log(||t_pred|| + 1e-6), log(||t_teacher|| + 1e-6))

L_conf = SmoothL1(c_base, stopgrad(c_teacher))
L_stage0 = L_pose + 0.1 * L_conf
```

translation-direction term은 teacher translation norm이 train median의 1%보다 큰 view에만 적용한다. teacher confidence는 patch grid로 resize한 뒤 위 train-only `log_expm1 + robust quantile` target으로 변환한다. Teacher는 training 및 Stage-0 health evaluation에서만 사용하며 runtime EXP-006에서는 instantiate하지 않는다.

Stage-0 optimizer/protocol은 다음으로 고정한다.

```text
AdamW(lr=1e-4, weight_decay=1e-4)
steps=500
gradient_clip=1.0
5-fold grouped train cross-fitting for health metrics
then 500-step refit on all train groups
```

validation으로 step이나 checkpoint를 선택하지 않는다.

출력 checkpoint는 다음으로 고정한다.

```text
revisit3d/checkpoints/exp006_geometry_bootstrap_v22.pt
```

### Stage-0 geometry health gate

Stage 1로 넘어가기 전에 train-only grouped cross-validation prediction에서 다음을 모두 만족해야 한다.

```text
all depth/pose/confidence values finite
depth > 0 for at least 99.9% of valid tokens
view-0 relative transform is identity within 1e-4
median relative rotation error <= 15 degrees
median translation-direction error <= 30 degrees
median scale-aligned translation error <= 0.50
Spearman(c_base, c_teacher) >= 0.30
predicted-pose track objective <= 1.05 * teacher-pose track objective
track objective has finite, non-zero gradient w.r.t. a zero log-depth residual field
    on >= 95% of train episodes
```

첫 Stage-0 train-only run에서 teacher pose조차 identity pose보다 track loss가 낮지 않다는 것이 확인됐다. 이 3D residual은 pose를 0으로 두면 parallax가 사라져 인위적으로 작아질 수 있으므로 `predicted < identity`는 pose-health criterion이 될 수 없다. Identity ratio는 degeneracy diagnostic으로 계속 보고하되, gate는 custom pose가 offline teacher pose의 geometry behavior를 5% 이내로 보존하는지 검사한다. Pose는 online에서 freeze하며 track objective로 pose를 최적화하지 않는다.

마지막 두 조건은 custom pose가 teacher geometry behavior를 보존하고 depth fast state에 실제 gradient를 제공하는지 확인한다. 하나라도 실패하면 Stage 1을 실행하지 않고 Stage-0 failure로 기록한다. threshold는 validation을 보고 완화하지 않는다.

scale-aligned translation error는 episode의 valid view 전체에서 scalar `alpha`를 least-squares로 구한 뒤 `||alpha*t_pred-t_teacher||/(||t_teacher||+1e-6)`로 계산한다. confidence Spearman은 valid patch 전체가 아니라 episode별 값을 먼저 계산한 후 group mean으로 집계한다.

Base geometry bootstrap 이후:

```text
FrozenVGGTFeatures      freeze
StreamingGeometryHead   freeze
```

한다.

---

## Stage 1 — Plasticity atom meta-training

학습 대상:

```text
key_projection
plasticity_decoder
```

base geometry head는 freeze.

각 train episode에서 Stage 2와 동일한 K=5 candidate pool을 만든다.

```text
A / B / three foreign source contexts
    ↓
z_j = online_update(zero), j=0...4

A' context
    ↓
current = online_update(zero)
candidate_j = clamp(current + 0.10 * transport(z_j → A'), -1, 1)

A' query (outer supervision only)
    ↓
LQ,current and LQ,j
    ↓
u_j and beneficial / neutral / harmful masks
```

**A가 matched episode라는 이유로 positive label을 주지 않는다.** D005에 따라 candidate identity와 무관하게 현재 decoder에서 측정된 future utility만 사용한다.

v2.5 train-only application diagnostic에서 transported code 전체를 current TTT initialization으로 사용하면 local pattern이 current gradient를 압도했다. v2.6은 source atom을 current-only update 이후의 residual memory로 사용한다. Current context에서는 여전히 정확히 1-step TTT만 수행하며, fixed `reuse_strength=0.10`은 모든 A1–A5 transport ablation에 동일하게 적용한다. Strength는 v2.5 train-only sweep `{0.05,0.10,0.25,0.50,1.0}`에서 geometry+appearance의 positive mean utility와 10% harm을 동시에 보인 가장 작은 non-trivial setting으로 고정했으며 validation에서는 바꾸지 않는다.

```text
B = {j | alignment valid and stopgrad(u_j) >  epsilon_u}

if B is not empty:
    S = B
else if any candidate is alignment-valid:
    S = {argmax_j stopgrad(u_j)}
else:
    S = empty

H = {j | alignment valid and stopgrad(u_j) < -epsilon_u and j not in S}

ell_j       = LQ,j / (stopgrad(abs(LQ,base)) + 1e-6)
ell_current = LQ,current / (stopgrad(abs(LQ,base)) + 1e-6)
```

`S`가 비어 있지 않으면:

```text
w_j = softmax(stopgrad(u_j) / 0.10), j in S
L_benefit = sum_j w_j * ell_j
```

모든 candidate가 아직 harmful인 initialization에서도 가장 덜 해로운 valid candidate 하나는 개선 대상으로 남겨 cold-start 학습 신호를 만든다. valid candidate가 하나도 없으면 `L_benefit=ell_current`로 두어 current-only decoder quality를 학습한다. 선택되지 않은 harmful transported initialization은 current context에서 generic bias로 작동하지 않도록 online step 전 decoder residual을 중립화한다.

```text
L_neutral = mean_{j in H} mean(abs(delta_log_depth_j_pre_update))
L_center  = mean_source square(mean_tokens(delta_log_depth_source))
```

`H`가 비어 있으면 `L_neutral=0`이다. 최종 outer loss는 다음으로 고정한다.

```text
L_atom =
    ell_current
  + L_benefit
  + softplus(L_benefit - stopgrad(ell_current))
  + 0.10 * L_neutral
  + 0.01 * L_center
  + 0.05 * L_key
```

candidate utility mask와 weight는 stop-gradient다. online update와 transport의 미분 경계는 §6.1.1을 따른다.

`LQ,base`는 query에서 zero code를 평가한 frozen base-geometry loss다. v2.4 train-only failure에서는 relative margin의 `ell_current`에 gradient가 남아 있어 current-only loss를 키우는 것으로 candidate 상대 utility를 부풀릴 수 있었다. v2.5는 relative margin 안의 current reference를 detach하고 `ell_current`를 직접 최소화한다. 500/1000-step train-CV checkpoint가 eligible하려면 overlap-component mean `LQ,current/LQ,base <= 1.05`를 먼저 만족해야 하며, 그 안에서 held-out-train mean best-valid utility가 큰 step을 선택한다. 두 checkpoint 모두 guard를 위반하면 Stage 1은 실패로 종료한다.

필수 collapse monitoring:

```text
per-token decoder residual variance
per-token transported-code variance
global-vector vs local-transport future-loss gap
matched-A / B / foreign utility distributions (identity label은 metric only)
beneficial / neutral / harmful candidate counts
```

Default optimizer:

```text
AdamW
lr = 1e-4
weight_decay = 1e-4
candidate max_steps = {500, 1000}
gradient_clip = 1.0
```

validation으로 early stopping하거나 hyperparameter를 고르지 않는다. 20개 directional train episode를 **8개 physical-overlap component**로 묶어 5-fold grouped CV를 수행한다. 위에 고정한 optimizer/loss coefficient는 바꾸지 않고 500/1000 step 중 하나만 train fold의 held-out group mean utility로 선택한 뒤, 선택된 고정 step으로 전체 train split에서 다시 학습한다. official validation은 최종 checkpoint에 대해 한 번만 평가한다.

---

## Stage 2 — Utility/risk meta-training

Stage 1의

```text
base head
key projection
plasticity decoder
```

를 freeze한다.

candidate pool K=5를 구성하고 train query future loss로 `u_j`, `r_j`를 생성한다.

Utility/risk head만 train한다.

```text
AdamW
lr = 3e-4
weight_decay = 1e-4
candidate max_steps = {500, 1000}
gradient_clip = 1.0
```

Stage 2도 같은 8-component, 5-fold train-only CV를 사용한다. train-fold checkpoint selection은 다음 lexicographic rule로 고정한다.

```text
1. minimum held-out-train-fold cluster harm_rate
2. tie → minimum mean_selected_minus_current
3. tie → minimum mean_regret
```

선택된 step과 설정으로 전체 train split에서 router를 재학습하고 official validation을 한 번만 평가한다. validation은 checkpoint selection, threshold tuning, normalization fitting에 사용하지 않는다.

### Frozen-output cache

Stage 0 이후 train/validation의 다음 output을 episode/frame/checkpoint hash와 함께 disk cache한다.

```text
frozen VGGT patch features
frozen tracker tracks / visibility / confidence
base depth / relative pose / confidence
Stage-0 teacher targets (train cache only)
```

K=5 transport는 blockwise cosine similarity를 지원해야 하며 dense `[Nt,Ns]` matrix의 peak memory와 episode wall-clock time을 기록한다. 이 cache와 blockwise 구현은 scalable long-term retrieval을 주장하기 위한 것이 아니라 EXP-006 반복 실험의 재현성과 비용 통제를 위한 것이다.

---

# 8. 필수 ablation

모든 조건에서 같은:

```text
backbone
base geometry checkpoint
atom decoder
data
candidate pool
TTT loss
TTT step size
TTT steps
```

를 사용한다.

## A0 — Current only

```text
z0 = zero
A' context에서 1-step TTT
```

memory reuse 없음.

---

## A1 — Global vector

source atom:

```text
zg = mean(z_source over V,P)
```

target 모든 token에 broadcast한다.

이는 이미 retired된 global fast state를 **baseline으로만** 사용한다.

---

## A2 — Untransported local

```text
z_target[v,p] = z_source[v,p]
```

correspondence 없음.

---

## A3 — Visual transport

appearance key만 사용.

```text
temperature = 0.07
```

---

## A4 — Predicted geometry transport

predicted Sim(3) + 3D distance.

appearance term 없음.

---

## A5 — Predicted geometry + appearance

full proposed transport.

```text
geometry + β appearance
β = 5
```

---

## A6 — A5 + utility only

utility regression head만 사용.

risk output 사용 안 함.

---

## A7 — A5 + utility/risk

**EXP-006 full method.**

---

## D1 — Oracle geometry + appearance

### [Diagnostic only]

known camera poses를 이용한 world-coordinate transport.

model selection에 사용하지 않는다.

predicted transport가 oracle transport effect를 얼마나 보존했는지 확인하는 upper bound다.

EXP-005도 known pose를 바로 이 목적으로만 사용했다.

---

## D2 — Current-loss heuristic selector

EXP-005와 동일 concept:

```text
current online loss가 가장 작은 candidate 선택
```

utility/risk head가 heuristic보다 실제로 안전한지 비교하기 위한 baseline이다. 기존 parameter-free current-loss gate는 negative transfer 제거에 충분하지 않았다.

---

# 9. Baseline과 metric

## Primary baselines

```text
Current-only TTT
Global-vector reuse
Untransported local reuse
Visual transport
Predicted geometry transport
Predicted geometry+appearance
Current-loss heuristic routing
Utility-only routing
Utility+risk routing
```

Oracle coordinate transport는 diagnostic upper bound.

---

## Primary utility metrics

각 episode에 대해:

```text
Δfuture =
L_selected_query - L_current_query

delta_future_normalized =
Delta_future / (abs(L_current_query) + 1e-6)
```

낮을수록 좋다.

보고:

```text
mean_selected_minus_current
median_selected_minus_current
mean_normalized_selected_minus_current
median_normalized_selected_minus_current
```

---

## Negative-transfer rate

```text
harm_rate =
mean(Δfuture > epsilon_u * (abs(L_current_query) + 1e-6))
```

가장 중요한 safety metric이다.

수치 잡음까지 숨기지 않도록 `raw_sign_harm_rate = mean(Delta_future > 0)`도 함께 보고하되 success gate에는 deadband가 적용된 harm을 사용한다.

---

## Future-utility regret

candidate set에서 실제 best candidate를 알고 있다고 할 때:

```text
L_oracle_best = min(L_current, L_candidate1 ... L_candidateK)

regret =
L_selected - L_oracle_best

regret_normalized =
regret / (abs(L_current) + 1e-6)
```

보고:

```text
mean_regret
median_regret
mean_normalized_regret
median_normalized_regret
```

D005에 따라 episode identity가 아니라 이 utility가 retrieval correctness의 기준이다.

---

## Router metrics

```text
oracle_utility_top1
oracle_utility_top3_coverage
utility Spearman correlation
risk AUROC
risk AUPRC
risk false-negative rate
reject_rate
```

기존 EXP-005 code도 oracle utility top-1, top-3 coverage, regret을 사용하고 있다.

---

## Transport metrics

```text
alignment_valid_rate
alignment_inlier_ratio
median_alignment_residual
transport_entropy
transport_coverage
peak_gpu_memory_mb
transport_wall_time_ms_per_candidate
full_episode_wall_time_s
```

성능 수치는 CUDA warm-up 후 동일 hardware에서 측정하고, dense와 blockwise matching 중 실제 사용한 path와 block size를 config/result에 기록한다.

---

## Predicted geometry diagnostic

offline validation에서만:

```text
relative pose rotation error
relative pose translation error
```

known pose는 여기서 ground truth metric 계산에만 사용한다.

routing이나 online TTT에는 넣지 않는다.

---

# 10. Train / validation split 규칙

### [Repository fact]

현재 development manifest는:

```text
revisit3d/manifests/nuscenes_revisit_dev.json
```

이다.

Dataset은 각 A/B/A'에 대해 context와 query를 별도로 반환한다.

Development manifest는 physical-overlap component 전체를 train→validation으로 이동시키는 방식으로 만들어 scene/location leakage를 막는다.

현재 directional/undirected group 수를 명시한다.

```text
train: 20 directional episodes = 10 pairs = 8 overlap components
val:   14 directional episodes =  7 pairs = 2 overlap components
test:   6 directional episodes =  3 pairs = 1 overlap component (closed)
```

양방향 episode는 독립 표본이 아니다. 모든 confidence interval과 hypothesis test의 resampling unit은

```text
group_id = tuple(sorted(source_scene, target_scene))
```

인 undirected pair다. 더 큰 physical-overlap component가 여러 pair를 연결하면 그 component ID를 우선 cluster로 사용한다. directional episode 평균은 descriptive metric으로만 함께 보고한다.

---

## EXP-006 rules

### Train

```text
split == "train"
```

다음에 사용 가능:

```text
model fitting
atom meta-training
utility/risk training
scalar normalization fitting
class-weight fitting
hyperparameter development
5-fold grouped CV checkpoint/step selection
```

---

### Validation

```text
split == "val"
```

사용 가능:

```text
one-shot final ablation comparison
one-shot success/failure gate
```

validation query frame은 evaluation/outer target으로만 사용.

validation은 early stopping, router threshold, `epsilon_u`, scalar normalization, class weight, loss coefficient 선택에 사용하지 않는다. validation을 확인한 후 어떤 설정을 바꾸면 새로운 experiment ID가 필요하다.

---

### Test

**코드에서 EXP-006 script가 `split="test"`를 받지 못하게 한다.**

즉 `evaluate_exp006.py` CLI:

```text
choices = ("train", "val")
```

로 제한한다.

기존 six-episode test split에 접근하면 해당 EXP-006 run은 **invalid**로 처리한다.

이 규칙은 repository의 D007과 AGENTS.md에서 명시되어 있다.

---

# 11. Success / failure threshold

EXP-006은 두 gate를 모두 통과해야 성공이다.

## Gate A — H2 predicted transport

같은 validation protocol에서 episode-normalized improvement를 정의한다.

```text
i_oracle =
(L_current - L_oracle) / (abs(L_current) + 1e-6)

i_pred =
(L_current - L_pred) / (abs(L_current) + 1e-6)
```

transport retention은 다음 oracle denominator condition이 먼저 성립할 때만 정의한다.

```text
mean(i_oracle) > epsilon_u
AND grouped-bootstrap 95% CI lower bound of mean(i_oracle) > 0

R_transport = mean(i_pred) / mean(i_oracle)
```

condition이 성립하지 않으면 ratio를 계산하지 않고 Gate A를 **INCONCLUSIVE**로 기록한다. 작은/음수 denominator에 epsilon을 더해 큰 retention을 만드는 것을 금지한다.

### PASS

모두 만족해야 한다.

```text
mean(i_pred) > epsilon_u

grouped-bootstrap 95% CI lower bound of mean(i_pred) > 0

R_transport >= 0.30

predicted geometry+appearance
    better than visual-only in mean future loss

alignment_valid_rate >= 0.80
```

즉 EXP-005에서 관측한 oracle coordinate advantage의 최소 30%는 predicted geometry에서도 유지되어야 한다.

### FAIL

다음 중 하나면 H2 predicted form을 통과하지 못한 것으로 본다.

```text
oracle denominator condition fails
OR
mean(i_pred) <= epsilon_u
OR
R_transport < 0.30
OR
alignment_valid_rate < 0.80
```

이 경우 utility/risk 결과가 좋아도 large memory bank로 넘어가지 않는다.

---

## Gate B — H4 risk-aware routing

Utility+risk router가 validation에서 다음을 모두 만족해야 한다.

```text
mean_normalized_selected_minus_current < 0

cluster_harm_rate <= 0.15

cluster_harm_rate
    <= 0.5 × heuristic_cluster_harm_rate

mean_normalized_regret
    <= 0.75 × heuristic_mean_normalized_regret
```

추가로 trivial reject-all solution을 막기 위해:

```text
accept_rate >= 0.20
```

를 요구한다.

`cluster_harm_rate`는 각 physical-overlap component의 mean normalized `Delta_future`가 `epsilon_u`보다 큰 component의 비율이다. validation에는 독립 component가 2개뿐이므로 `cluster_harm_rate <= 0.15`는 harmful component가 0개여야 한다. directional/pair-level `harm_rate`도 반드시 함께 보고한다.

---

## Strong success

EXP-006 내부의 강한 feasibility 신호는:

```text
95% grouped bootstrap CI of
mean_normalized_selected_minus_current
lies entirely below zero

AND

cluster_harm_rate <= 0.10
```

로 정의한다. bootstrap은 overlap component를 10,000회 resample하고 seed를 config에 저장한다. 단, validation의 2개 component로 계산한 CI는 descriptive feasibility diagnostic일 뿐 유효한 paper-level inferential evidence로 해석하지 않는다. 새로운 closed benchmark에서의 최종 검증은 별도 EXP ID로 수행한다.

---

# 12. 수정하거나 새로 만들 파일

## New

```text
revisit3d/models/plasticity_atom.py
```

포함:

```text
PlasticityAtom
SpatialPlasticityHead
local code decoder
atom construction
online code update
```

---

```text
revisit3d/models/geometry_transport.py
```

포함:

```text
backproject_tokens()
predicted_pointmap()
mutual_feature_matches()
weighted_sim3()
robust_sim3()
visual_transport()
geometry_transport()
geometry_appearance_transport()
```

---

```text
revisit3d/models/utility_risk.py
```

포함:

```text
UtilityRiskHead
RouterFeatureBuilder
hard_route()
soft_route()
```

---

```text
revisit3d/losses/meta_utility.py
```

포함:

```text
normalized_future_utility()
utility_risk_loss()
future_regret()
```

---

```text
revisit3d/scripts/bootstrap_exp006_geometry.py
```

역할:

```text
existing depth-bootstrap checkpoint load
assert head_type == anchored and source path matches §7
train split only
offline teacher pose + confidence distillation
Stage-0 geometry health gate
save custom geometry-head checkpoint
```

---

```text
revisit3d/scripts/train_exp006_atom.py
```

Stage 1.

K=5 utility-conditioned atom objective, train-only grouped CV, and full-train fixed-step refit을 포함한다.

---

```text
revisit3d/scripts/train_exp006_router.py
```

Stage 2.

deadband labels, invalid-candidate mask, train-only grouped CV, and full-train fixed-step refit을 포함한다.

---

```text
revisit3d/scripts/evaluate_exp006.py
```

필수 A0–A7 + D1–D2 비교.

`test` split CLI option 금지.

---

```text
revisit3d/scripts/exp006_smoke_test.py
```

검증:

```text
tensor shapes
z=0 identity
transport normalization
Sim3 recovery
invalid alignment masking
query leakage guard
router reject behavior
gradient only on code during online TTT
first-order detach boundary and no Hessian
key projection receives gradient through fixed-neighbor appearance weights and L_key
metric/local-spacing Sim3 threshold (meter threshold 금지)
Sim3 degeneracy and scale checks
neutral-risk masking
grouped bootstrap reproducibility
EXP-006 CLI rejects test split
```

---

```text
configs/EXP-006_atom_utility.yaml
```

모든 pre-registered threshold, CV fold assignment, random seed, cache/checkpoint hash, optimizer, router threshold, bootstrap seed를 저장한다.

---

```text
revisit3d/scripts/cache_exp006_frozen_outputs.py
```

frozen feature/track/base-geometry output cache를 만들며 validation query는 cache하더라도 evaluation namespace에서만 접근할 수 있게 분리한다.

---

```text
3D_docs/experiments/EXP-006_trainable_3d_atom_risk_routing.md
```

실험 protocol, config, metrics, result path, interpretation 기록.

---

## Modify

```text
revisit3d/models/__init__.py
```

새 class export.

현재 factory는 global/slot/anchored head만 expose하고 있으므로 spatial atom class는 별도 export로 추가한다.

---

```text
revisit3d/losses/__init__.py
```

meta utility loss export.

---

```text
revisit3d/losses/geometry.py
```

가능하면 기존 loss 의미는 바꾸지 않고:

```text
token backprojection helper
pose diagnostic helper
```

만 추가한다.

---

실험 완료 후:

```text
3D_docs/research_state.md
3D_docs/experiments/index.md
```

를 갱신한다.

Methodological conclusion이 바뀌었을 때만:

```text
3D_docs/hypothesis.md
3D_docs/decisions.md
```

를 수정한다.

이는 AGENTS.md workflow를 따른다.

---

# 13. 예상 failure mode

## F1. Predicted pose/depth gauge inconsistency

증상:

```text
Sim3 match residual 증가
alignment_valid_rate 감소
geometry < visual
```

대응:

먼저 pose bootstrap을 점검한다.

**oracle pose로 production result를 대체하지 않는다.**

---

## F2. Repetitive texture로 잘못된 Sim3

예:

```text
도로
건물 창문
차선
비슷한 차량
```

증상:

appearance cosine은 높지만 geometric residual도 높음.

대응:

mutual matching + robust second-pass Sim3가 이를 제거해야 한다.

EXP-006에서 추가 RANSAC system까지 만들지는 않는다.

---

## F3. Low texture / small parallax

3D geometry 자체가 under-constrained되어 alignment가 불안정할 수 있다.

이 경우 candidate를 억지로 reuse하지 않고:

```text
alignment_valid=False
→ reject
```

하게 한다.

---

## F4. Plasticity code가 global bias로 collapse

증상:

```text
global vector ≈ local transport
foreign candidate도 비슷하게 도움
atom spatial variance 감소
```

필수 monitoring:

```text
per-token code variance
matched - foreign utility
global-vs-local gap
```

---

## F5. Utility head가 current loss 하나만 복사

증상:

heuristic current-loss selector와 거의 동일한 ranking.

검증:

```text
utility/risk vs heuristic
```

ablation에서 improvement가 없어야 탐지된다.

---

## F6. Risk head의 reject-all collapse

증상:

```text
harm_rate = 0
accept_rate ≈ 0
```

그래서 success gate에

```text
accept_rate >= 0.20
```

를 둔다.

---

## F7. Harmful soft mixing

각각은 유용한 atom이지만 동시에 mix했을 때 서로 충돌할 수 있다.

따라서 primary protocol은 hard selection이다.

Soft mixing은 secondary ablation까지만 수행한다.

---

## F8. Frozen track prior dependency

현재 online geometry objective는 frozen foundation track correspondence에 의존한다.

이는 EXP-006에서 허용되는 controlled foundation evidence이지만 최종 4D model의 독립적인 tracking capability를 증명하는 것은 아니다.

결과 해석에서 명시해야 한다.

---

## F9. Dynamic objects

nuScenes 내 차량/보행자 등의 dynamic region은 static shared-coordinate assumption을 깨뜨릴 수 있다.

EXP-006에서는 motion state를 도입하지 않는다.

이러한 영역에서 발생하는 risk는 utility/risk router가 reject하도록 두고 failure statistics를 기록한다.

---

## F10. Query leakage

다음 중 하나라도 발생하면 전체 run을 폐기한다.

```text
A' query RGB → online update
A' query loss → router input
A' query geometry → Sim3 alignment
A' query feature → candidate selection
```

query는 오직 future outer supervision/evaluation용이다.

---

# 14. EXP-007로 넘겨야 할 범위

EXP-006이 H2-P와 H4를 모두 통과한 경우에만 EXP-007을 시작한다.

## EXP-007: Continual 3D Atom Memory

다음 항목을 EXP-007로 넘긴다.

```text
persistent memory bank
write policy
memory capacity
geometric merging
duplicate removal
uncertainty accumulation
utility-history accumulation
age
eviction
reactivation
atom splitting
long-sequence memory growth
retrieval scalability
memory/computation trade-off
```

이는 H5인 continual atom consolidation을 직접 검증하는 단계다. H5는 현재 H4가 통과하기 전에는 수행하지 않도록 repository에서 명시되어 있다.

EXP-007에서도 아직 다음은 넣지 않는다.

```text
dynamic-point-specific memory
motion-conditioned atoms
long-term 4D tracking
occlusion reappearance memory
```

이들은 H6이며 static 3D revisit milestone 이후 단계로 유지한다.

---

# Final go/no-go logic

```text
EXP-006
│
├─ Stage-0 geometry health gate passes?
│      │
│      ├─ NO → STOP
│      │        pose/confidence bootstrap 또는 online objective health 문제
│      │
│      └─ YES
│
├─ Predicted geometry transport valid?
│      │
│      ├─ NO → STOP
│      │        pose / geometry / alignment 문제 해결
│      │
│      └─ YES
│
├─ Geometry+appearance > visual?
│      │
│      ├─ NO → H2 predicted form unsupported
│      │
│      └─ YES
│
├─ Utility/risk router improves safety?
│      │
│      ├─ NO → H4 unsupported
│      │        memory bank 구현 금지
│      │
│      └─ YES
│
└─ EXP-007
       Continual atom bank + consolidation
```

EXP-006의 최종 claim은 **“큰 memory를 만들었다”가 아니다.**

성공 시 주장할 수 있는 것은 정확히 다음 두 가지다.

> **(1) Spatially local adaptation learned on an earlier traversal remains reusable after transport based on predicted geometry rather than oracle poses.**

> **(2) A learned current-context utility/risk model can exploit that reusable state while substantially reducing negative transfer relative to heuristic selection.**

이 두 claim이 먼저 성립해야 continual memory bank를 만드는 것이 연구적으로 정당화된다.
