가능해. 다만 지금 아이디어를 그대로 **“TTT + CL”**이라고만 잡으면 범위가 너무 넓고, 최근 streaming 3D 연구와 겹칠 가능성이 높아. 연구 문제를 아래처럼 더 날카롭게 정의하는 게 좋다.

> **현재 관측으로부터 TTT가 만든 “유용한 parameter-update experience”를 장기 메모리에 축적하고, 이후 유사한 geometric context가 다시 등장하면 이를 검색하여 현재 TTT를 가속·안정화한다.**

즉, **frame을 기억하는 것이 아니라 “어떻게 적응했는가”를 기억하는 것**이다.

---

# 1. 먼저 TTT와 CL의 역할을 정확히 나누자

네 해석은 대체로 맞는데 조금 다듬으면 다음과 같다.

**TTT**
- 지금 들어온 observation \(x_t\)에 맞춰 fast weight \(W_t\)를 업데이트
- 현재 환경에 빠르게 적응
- **plasticity 담당**

**Continual Learning**
- 시간에 따라 새로운 정보를 계속 학습하면서 과거에 얻은 유용한 지식을 파괴하지 않음
- stability–plasticity trade-off를 관리
- **long-term stability 담당**

그래서 streaming 3D에서는 다음과 같은 2-timescale 구조가 자연스럽다.

\[
\boxed{
\text{Short-term TTT}
\quad + \quad
\text{Long-term Adaptation Memory}
}
\]

최근 tttLRM은 이미 여러 observation을 TTT fast weights에 압축해 long-context 3D reconstruction을 수행하고, streaming online variant도 제공한다.

또 Mem3R은 더 직접적으로 **TTT fast-weight memory를 camera tracking에, explicit recurrent memory를 geometry에** 사용하는 hybrid 구조를 제안했다.

따라서 단순히

> "TTT로 현재 학습 + CL로 과거 저장"

만으로는 Mem3R 등과 차이가 약하다.

---

# 2. 네 연구에서 기억해야 하는 건 `past state`가 아니라 `past adaptation`

여기가 핵심이다.

일반적인 streaming reconstruction은

\[
I_t
\rightarrow
S_{t-1}
\rightarrow
S_t
\]

처럼 **scene state**를 기억한다.

네 방식은 여기에 별도의 메모리를 추가하는 거다.

\[
\boxed{
\mathcal M
=
\{
(k_i,\Delta W_i,q_i)
\}
}
\]

각 memory entry는 예를 들어 다음처럼 구성한다.

- \(k_i\): 그 당시 상황을 나타내는 **geometric context key**
- \(\Delta W_i\): 그 상황에서 효과적이었던 **TTT update**
- \(q_i\): 그 update가 얼마나 신뢰할 만했는지 나타내는 quality/confidence

즉,

\[
\underbrace{k_i}_{\text{어떤 상황이었나}}
\rightarrow
\underbrace{\Delta W_i}_{\text{그때 어떻게 적응했나}}
\]

를 저장하는 것이다.

이게 네 연구의 가장 중요한 conceptual distinction이다.

---

# 3. 그런데 \(\Delta W\) 자체를 저장하면 너무 크다

모델의 전체 gradient나 parameter difference를 프레임마다 저장하면 메모리가 터진다.

그래서 **update direction을 low-rank subspace로 압축**하는 방식이 좋다.

예를 들어 TTT update가

\[
\Delta W_t = -\eta g_t
\]

라면 과거 모든 \(g_t\)를 저장하지 않고

\[
U_j \in \mathbb{R}^{d\times r},
\qquad r\ll d
\]

같은 작은 update basis를 저장한다.

메모리는

\[
\boxed{
M_j=(k_j,U_j,q_j)
}
\]

가 된다.

이 아이디어는 CL의 GPM과 철학적으로 잘 맞는다. GPM은 과거에 중요했던 gradient/representation subspace를 compact basis 형태로 기억하고 이후 gradient를 projection해 interference를 줄인다. OGD 역시 parameter-space에서 gradient 방향을 제한하여 과거 지식에 대한 interference를 줄인다.

다만 네 경우에는 반대로,

> **과거 gradient와 겹치지 않도록 만드는 것만이 아니라, 과거에 유용했던 gradient direction을 적극적으로 다시 활용한다.**

는 점이 재미있다.

---

# 4. 내가 제안하는 전체 architecture

가칭으로

### **Continual Test-Time Adaptation Memory for Streaming 3D Reconstruction**

정도로 생각해보자.

전체 흐름은 이렇다.

```text
                  ┌─────────────────────────────┐
                  │ Long-term Adaptation Memory │
                  │                             │
                  │ k1 → U1                     │
                  │ k2 → U2                     │
                  │ k3 → U3                     │
                  └──────────────┬──────────────┘
                                 │ retrieve
                                 ↓
Frame I_t → Encoder → Context Descriptor k_t
                        │
                        ├──── similarity ────→ Top-K update memories
                        │
                        ↓
                  Current TTT Gradient g_t
                        │
                        ↓
           Retrieval-guided TTT Update
                        │
                        ↓
                 Fast Weight W_t
                        │
                        ↓
         Point Cloud / Pose / Point Track
```

기존 TTT에 **retrieval branch 하나를 추가**하는 형태다.

---

# 5. 가장 중요한 것이 memory key다

여기서 단순 image feature similarity만 쓰면 안 된다.

예를 들어

> 같은 복도 이미지

라고 해도

- 앞으로 이동하는 중
- 회전 중
- 거의 정지
- low parallax
- motion blur
- dynamic object 존재

상황에서는 필요한 업데이트가 다를 수 있다.

따라서

\[
k_t =
f(
z_t,
\Delta P_t,
D_t,
C_t,
M_t
)
\]

처럼 만드는 것을 추천한다.

여기서

- \(z_t\): visual feature
- \(\Delta P_t\): camera motion / relative pose
- \(D_t\): depth / geometry feature
- \(C_t\): reconstruction confidence
- \(M_t\): motion / tracking information

이다.

즉,

### RGB similarity

가 아니라

### **Geometric Adaptation Context Similarity**

를 사용한다.

이게 논문의 꽤 중요한 contribution이 될 수 있다.

---

# 6. 새로운 frame이 들어왔다고 해보자

시간 \(t\)에서

\[
I_t
\]

가 들어온다.

먼저 현재 context

\[
k_t
\]

를 구한다.

그리고 memory에서

\[
j^*
=
\operatorname{TopK}
\operatorname{sim}(k_t,k_j)
\]

로 과거 유사 상황을 찾는다.

---

# 7. 과거 update를 그냥 복사하면 위험하다

예를 들어 과거에

\[
\Delta W_{old}
\]

가 좋았다고 해서 지금 그대로 적용하면 안 된다.

따라서 **현재 gradient와 과거 update가 동의하는지** 확인하는 게 좋다.

현재 TTT gradient를

\[
g_t=\nabla_W L_{\text{TTT}}
\]

라고 하자.

retrieved update direction을

\[
d_{\text{mem}}
=
\sum_j a_jd_j
\]

라고 하면

\[
A_t
=
\cos(-g_t,d_{\text{mem}})
\]

를 계산할 수 있다.

### \(A_t\)가 높음

과거 경험과 현재 TTT가

> "이 방향으로 움직여야 한다"

고 동의.

→ 과거 adaptation 적극 활용.

### \(A_t\)가 낮거나 음수

현재 상황이 실제로는 다름.

→ memory를 무시하고 ordinary TTT.

이렇게 해야 **negative transfer**를 막을 수 있다.

---

# 8. 그래서 실제 update는 이렇게 만들 수 있다

가장 간단한 버전은

\[
\boxed{
\Delta W_t
=
-\eta
\left[
(1-\lambda_t)g_t
+
\lambda_t g_{\text{mem}}
\right]
}
\]

여기서

\[
\lambda_t =
f(
\text{context similarity},
\text{gradient agreement},
\text{confidence}
)
\]

이다.

즉,

### 익숙한 상황

\[
\lambda_t \rightarrow 1
\]

과거 adaptation 적극 재사용.

### 새로운 상황

\[
\lambda_t \rightarrow 0
\]

현재 TTT 중심으로 학습.

이게 바로 네가 원하는

> "과거와 비슷한 input이 들어오면 그때 TTT update 방향을 기억했다가 사용"

을 수식으로 만든 형태다.

---

# 9. 나는 오히려 `Update Subspace` 방식이 더 좋다고 본다

gradient 하나를 저장하기보다,

예전에 비슷한 상황에서 관측된 gradient들을 모아

\[
G_j =
[g_1,g_2,\cdots,g_n]
\]

SVD 등을 통해

\[
G_j \approx U_j\Sigma_jV_j^\top
\]

로 만든다.

그리고 \(U_j\)만 저장한다.

현재 gradient를

\[
g_t
\]

라고 하면

\[
g_{\parallel}
=
U_jU_j^\top g_t
\]

를 계산한다.

이건

> 과거 비슷한 상황에서 자주 유용했던 update space 안에 현재 gradient가 얼마나 들어가는가

를 의미한다.

그리고

\[
g_{\perp}
=
g_t-g_{\parallel}
\]

도 얻는다.

그러면

\[
\boxed{
\Delta W_t
=
-\eta
(
g_{\parallel}
+
\gamma_tg_{\perp}
)
}
\]

같이 할 수 있다.

---

# 10. 이 식이 상당히 의미가 있다

익숙한 상황이면

\[
\gamma_t \ll 1
\]

로 둔다.

즉,

> 예전에 잘 작동했던 adaptation subspace 안에서만 주로 움직인다.

반대로 새로운 환경이면

\[
\gamma_t \approx 1
\]

로 둔다.

그러면

> 새로운 update direction도 충분히 학습한다.

즉,

\[
\boxed{
\text{Familiar}
\rightarrow
\text{Reuse}
}
\]

\[
\boxed{
\text{Novel}
\rightarrow
\text{Learn}
}
\]

이라는 아주 명확한 원리가 생긴다.

---

# 11. 여기서 Continual Learning이 하는 역할

CL module은 크게 세 가지를 하면 된다.

### ① Consolidation

매 frame의 gradient를 전부 저장하지 않는다.

현재 TTT가 실제 reconstruction을 개선했다면

\[
\mathcal M
\leftarrow
\mathcal M\cup
(k_t,U_t)
\]

로 memory에 consolidate.

---

### ② Merge

새로운 context가 기존

\[
k_j
\]

와 매우 비슷하면 memory를 추가하지 않고

\[
U_j \leftarrow
\operatorname{Merge}(U_j,U_t)
\]

한다.

그래서 memory size가 무한히 증가하지 않는다.

---

### ③ Preserve

자주 등장하면서 유용한 adaptation direction은 importance

\[
q_j
\]

를 높여서 쉽게 overwrite되지 않게 한다.

즉,

\[
q_j
=
\text{frequency}
\times
\text{utility}
\times
\text{confidence}
\]

같은 형태다.

이 부분이 continual consolidation이 된다.

---

# 12. 그런데 어떤 TTT update가 "좋은 update"인지 어떻게 아느냐?

GT가 없으니 이게 매우 중요하다.

현재 update 전

\[
L_{\text{geo}}^{before}
\]

와 update 후

\[
L_{\text{geo}}^{after}
\]

를 비교한다.

예를 들어

\[
Q_t
=
L_{\text{geo}}^{before}
-
L_{\text{geo}}^{after}
\]

로 정의한다.

여기에 streaming 3D라면

- reprojection consistency
- multi-view geometric consistency
- depth consistency
- point track cycle consistency
- pose consistency

등을 사용할 수 있다.

그래서

\[
Q_t> \tau
\]

인 adaptation만 memory에 저장한다.

즉,

> **TTT가 했다고 무조건 기억하는 것이 아니라 실제 geometry를 개선한 adaptation만 consolidation한다.**

이게 굉장히 중요하다.

---

# 13. Point tracking까지 넣으면 더 재미있어진다

dynamic reconstruction에서는 key에

\[
k_t =
[
\text{appearance},
\text{camera motion},
\text{scene geometry},
\text{object motion}
]
\]

을 넣을 수 있다.

예를 들어 사람이 걷다가 occlusion됐다가 다시 등장한다고 하자.

과거에 유사한

- motion
- appearance
- depth
- camera movement

조건이 있었다면 과거 adaptation direction을 retrieve한다.

그러면 TTT가 매번

> "이런 motion에서는 correspondence를 어떻게 조정해야 하지?"

를 처음부터 다시 배우지 않아도 된다.

D4RT는 이미 depth, camera parameters, spatio-temporal correspondence를 하나의 queryable model에서 처리하며 4D reconstruction과 point tracking을 함께 수행하므로, 장기적으로는 이런 동적 reconstruction backbone으로 확장하기 좋은 후보다.

다만 첫 논문부터 D4RT를 뜯는 것보다는 아래 순서가 훨씬 현실적이다.

---

# 14. 연구를 실제로 한다면 나는 이렇게 시작할 것 같다

### Phase 1 — CUT3R / TTT3R 기반

가장 먼저

\[
\text{CUT3R/TTT3R}
+
\text{Update Memory}
\]

부터 한다.

output:

- camera pose
- depth
- pointmap
- point cloud

여기서

**“retrieved past adaptation이 long-sequence reconstruction을 개선하는가?”**

만 검증한다.

CUT3R 계열은 persistent state를 사용하는 대표적인 streaming reconstruction 계열이고, TTT3R 자체가 이를 online-learning 관점에서 재해석하기 때문에 네 아이디어를 검증하기 가장 자연스러운 출발점이다.

---

# 15. 그다음 Phase 2

성공하면

\[
\text{Streaming 4D}
+
\text{Point Tracking}
\]

으로 간다.

그때

\[
M_{\text{static}}
\]

과

\[
M_{\text{dynamic}}
\]

을 나누는 것도 가능하다.

예를 들어

```text
Long-term adaptation memory

      ┌──── Static Geometry Memory
M ────┤
      └──── Dynamic Motion Memory
```

static memory는

- depth
- camera pose
- surface geometry

관련 update를 저장하고,

dynamic memory는

- motion
- correspondence
- tracking

관련 update를 저장한다.

이건 상당히 4D reconstruction다운 extension이다.

---

# 16. 이 접근법이 최근 연구와 정확히 어디가 다른가

여기가 중요하다.

### TTT3R

현재 observation과 memory의 관계를 보고

> **현재 state를 얼마나 업데이트할까?**

를 결정한다.

### ReCal3R

state reliability, alignment, residual 등을 이용해

> **현재 token의 learning rate를 얼마나 크게 할까?**

를 조절한다.

### Mem3R

TTT fast-weight pose memory와 explicit geometric memory를 분리해서

> **현재까지의 정보를 어디에 저장할까?**

를 개선한다.

### 네가 노려야 하는 것

> **과거에 비슷한 geometric situation에서 어떤 adaptation이 성공했는지를 기억했다가, 새로운 TTT의 update prior로 재사용한다.**

즉 질문 자체가 다르다.

\[
\boxed{
\text{How much to update?}
}
\]

가 아니라

\[
\boxed{
\text{How should I update, based on past adaptation experience?}
}
\]

이다.

이 차이가 매우 중요하다.

---

# 17. 하지만 novelty 측면에서 주의할 것도 있다

“과거 update direction을 기억하고 retrieval해서 재사용한다”는 발상 자체는 3D 외 분야까지 보면 완전히 새로운 개념이라고 주장하면 안 된다.

예를 들어 CL에서는 GPM/OGD처럼 gradient subspace를 기억하거나 보호하는 계열이 이미 존재하고, 최근에는 과거 correction direction을 memory prototype으로 저장해서 inference-time adaptation에 사용하는 방향도 다른 분야에서 나타나고 있다.

따라서 contribution은

**“update memory” 자체**보다

\[
\boxed{
\text{Geometry-aware retrieval}
+
\text{TTT}
+
\text{continual consolidation}
+
\text{streaming 3D/4D}
}
\]

의 결합으로 가져가는 것이 안전하다.

---

# 18. 최종적으로 논문의 핵심 아이디어를 한 문장으로 정리하면

나는 이렇게 잡는 게 가장 좋다고 본다.

> **Streaming 3D reconstruction에서 TTT를 통해 얻어진 successful adaptation directions를 geometry-aware long-term memory에 지속적으로 consolidate하고, 새로운 observation이 과거 geometric context와 유사할 경우 해당 adaptation subspace를 retrieval하여 현재 TTT update를 guide함으로써, 과거 정보를 직접 replay하지 않고도 장기 sequence에서 빠른 적응과 안정적인 기억을 동시에 달성한다.**

조금 더 쉽게 말하면,

> **TTT는 “지금 어떻게 바뀔지” 배우고, CL memory는 “예전에 이런 상황에서 어떻게 바뀌었더니 잘됐는지” 기억한다. 비슷한 상황이 다시 오면 처음부터 적응하지 않고 과거의 성공한 업데이트 방향을 꺼내 쓴다.**

이 형태라면 네가 생각한 **TTT + continual learning의 역할이 겹치지 않고 서로 명확하게 보완적**이다.

그리고 이 아이디어의 가장 핵심적인 실험은 정확도만 보는 게 아니라 **① 긴 sequence에서 forgetting/drift 감소, ② recurring scene에서 필요한 TTT step 수 감소, ③ 새로운 scene에서는 adaptation 능력 유지, ④ memory 크기는 bounded**라는 네 축으로 잡는 게 좋다. 이 네 가지가 동시에 나오면 논문의 스토리가 상당히 명확해진다.

원한다면 다음 단계로는 **이걸 실제 논문 Method처럼 3~4개 모듈로 쪼개서, 각 모듈의 입력/출력과 수식까지 포함한 전체 프레임워크를 설계**해볼 수 있어.
