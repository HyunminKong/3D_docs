2026년 8월 19일 기준으로 CVPR/ICLR/ICCV/NeurIPS/ICML의 관련 흐름을 다시 교차해서 보면, 처음 이야기했던 **“TTT update direction을 CL로 기억한다”**는 출발점은 좋지만, 그대로는 CVPR/ICLR 메인 contribution으로는 부족하다. 2026년에는 이미 TTT를 3D memory로 사용하는 tttLRM·ZipMap·Scal3R, online 3D의 state forgetting을 다루는 TTT3R, explicit spatial memory를 쓰는 Point3R, KV cache를 압축하는 STAC까지 등장했기 때문이다. ([openreview.net](https://openreview.net/forum?id=aMs6FtNaY5&utm_source=chatgpt.com))

내가 지금 이 분야에서 박사학위 연구 축을 잡는다면, 주제를 한 단계 추상화해서 다음 질문으로 가져가겠다.

> **“Streaming 3D/4D perception에서 모델은 과거 scene을 기억하는 것을 넘어, 과거에 어떤 geometric/motion context에서 어떻게 적응해야 했는지를 장기간 학습할 수 있는가?”**

즉 **scene memory가 아니라 plasticity memory**, 다시 말해 **“무엇을 봤는가”가 아니라 “어떻게 바뀌어야 하는가”를 기억하는 continual test-time geometry**를 연구의 중심으로 두는 방향이다.

---

# 1. 먼저 TTT라는 용어부터 정확하게 분리해야 한다

이 부분은 논문 formulation에서 매우 중요하다. 현재 문헌에는 사실상 두 계보가 있다.

전통적인 **Test-Time Training / Test-Time Adaptation**은 pretrained model을 현재 unlabeled test distribution에 맞춰 inference 시점에 self-supervised objective로 수정하는 것이다. 2023년 SAR은 dynamic/wild stream에서 adaptation collapse를 줄이는 안정성을, RoTTA는 correlated하고 계속 변화하는 test stream을, EcoTTA는 continual adaptation의 memory efficiency를 핵심 문제로 다뤘다. ([openreview.net](https://openreview.net/forum?id=g2YraF75Tj&utm_source=chatgpt.com))

반면 2025년 ICML의 **Learning to (Learn at Test Time)** 계열은 더 구조적인 의미를 갖는다. RNN의 hidden state 자체를 작은 ML model로 두고, 들어오는 sequence로 그 hidden model을 self-supervised update한다. 즉,

\[
x_t
\rightarrow
\text{update }W_t
\rightarrow
W_{t+1}
\]

에서 \(W_t\)가 sequence memory 역할을 한다. 긴 sequence를 KV cache처럼 명시적으로 모두 저장하는 대신 **정보를 fast weights에 압축한다**는 발상이다. ([proceedings.mlr.press](https://proceedings.mlr.press/v267/sun25h.html?utm_source=chatgpt.com))

네 연구에는 **두 번째 의미의 TTT가 훨씬 중요하다.** 그리고 첫 번째 계보에서 연구된 stability, collapse, distribution drift 문제를 가져오는 것이 좋다.

이 조합 자체가 이미 중요한 힌트다.

\[
\boxed{
\text{TTT as Memory}
+
\text{Continual Adaptation Stability}
}
\]

---

# 2. 2023: 아직 문제들이 비교적 분리되어 있었다

2023년을 보면 3D 분야는 지금처럼 하나의 foundation geometry model이 모든 것을 해결하는 구조가 아니었다.

LivePose는 monocular video가 들어오면서 camera pose도 계속 변하는 **online dense 3D reconstruction** 문제를 직접 다뤘다. TAPIR은 queried point를 video 전체에서 추적하는 tracking-any-point 문제를 per-frame initialization과 temporal refinement로 해결했다. 즉 depth, reconstruction, camera pose, tracking은 여전히 상당히 독립적인 task였다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2023/html/Stier_LivePose_Online_3D_Reconstruction_from_Monocular_Video_with_Dynamic_Camera_ICCV_2023_paper.html?utm_source=chatgpt.com))

동시에 TTA 쪽에서는 문제가 아주 명확했다.

\[
\text{new distribution}
\rightarrow
\text{adapt}
\]

하는 것만으로는 부족하고,

\[
\text{adapt repeatedly}
\rightarrow
\text{collapse / forgetting / error accumulation}
\]

이 발생한다는 것이 주요 연구 질문으로 올라왔다. SAR, RoTTA, EcoTTA가 각각 adaptation stability, temporally correlated stream, memory-efficient continual adaptation을 다룬 것이 대표적이다. ([openreview.net](https://openreview.net/forum?id=g2YraF75Tj&utm_source=chatgpt.com))

따라서 2023년의 핵심 질문을 한 줄로 만들면

> **“Streaming input에 어떻게 안전하게 적응할 것인가?”**

였다.

하지만 이 시점에는 이 문제가 foundation 3D reconstruction과 본격적으로 결합되지는 않았다.

---

# 3. 2024: 3D representation이 크게 바뀐다 — DUSt3R

2024년에서 가장 중요한 전환점은 **DUSt3R**라고 보는 것이 맞다.

DUSt3R는 image pair로부터 각 pixel에 대응하는 3D pointmap을 직접 예측하면서 기존에 개별적으로 처리하던 monocular/multi-view depth와 relative camera pose 같은 geometry 문제들을 공통 표현 안으로 통합했다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.html?utm_source=chatgpt.com))

이게 중요한 이유는 3D reconstruction을

\[
\text{matching}
\rightarrow
\text{pose}
\rightarrow
\text{triangulation}
\rightarrow
\text{optimization}
\]

이라는 classical pipeline으로만 볼 필요가 없어졌기 때문이다.

\[
\boxed{
I
\rightarrow
\text{Geometry Representation}
}
\]

이라는 learned geometry prior가 등장했다.

MASt3R는 이 계보에 dense matching/local feature 기능을 강화했고, SpatialTracker는 2D queried pixel을 monocular depth와 결합해 3D space에서 추적하는 방향으로 발전했다. SpatialTracker는 특히 **tracking representation이 2D optical flow에서 3D trajectory로 넘어가는 흐름**을 잘 보여준다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_SpatialTracker_Tracking_Any_2D_Pixels_in_3D_Space_CVPR_2024_paper.pdf?utm_source=chatgpt.com))

Depth 분야에서도 Depth Anything이 대규모 unlabeled data를 이용한 robust monocular depth foundation model을 제시했다. 즉 depth 역시 특정 dataset용 network에서 **general geometric prior**로 이동한다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Depth_Anything_Unleashing_the_Power_of_Large-Scale_Unlabeled_Data_CVPR_2024_paper.html?utm_source=chatgpt.com))

흥미롭게도 같은 시기 Continual Learning에서는 **parameter-efficient subspace**가 강하게 등장한다. InfLoRA는 low-rank parameter space를 이용하여 sequential task 사이 interference를 줄이는 방향을 제시했다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html?utm_source=chatgpt.com))

따라서 2024년에는 서로 독립적으로 다음 두 흐름이 만들어졌다.

\[
\text{3D}
:
\quad
\text{task-specific geometry}
\rightarrow
\text{shared geometry representation}
\]

\[
\text{CL}
:
\quad
\text{full parameter retention}
\rightarrow
\text{compact adaptation subspace}
\]

이 두 흐름이 네 연구에서 결국 만나게 된다.

---

# 4. 2025: “좋은 geometry model”에서 “지속되는 geometry state”로 이동

2025년은 streaming 3D 관점에서 결정적인 시기다.

VGGT는 하나의 feed-forward model이 camera parameters, depth, pointmaps 등 주요 3D attributes를 함께 예측할 수 있음을 보여줬다. 즉 representation unification이 더 강해졌다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html?utm_source=chatgpt.com))

하지만 VGGT 같은 global transformer에는 근본적인 문제가 있다.

\[
N \uparrow
\Rightarrow
\text{attention computation/memory} \uparrow
\]

그래서 **long streaming sequence를 어떻게 유지할 것인가**가 다음 문제가 된다.

CUT3R는 persistent recurrent state를 사용하여 새로운 frame이 올 때마다 state를 update하는 continuous 3D perception을 제안했다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.html?utm_source=chatgpt.com))

Point3R는 이보다 더 직접적인 문제를 제기한다. implicit memory는 용량이 제한되어 이전 frame 정보가 사라질 수 있으므로, world coordinate의 3D position과 연결된 **explicit spatial pointer memory**를 유지한다. 논문 자체가 earlier-frame information loss를 문제로 명시한다. ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2025/hash/650db8e1b0b016dc270d51c1476e91cf-Abstract-Conference.html))

MASt3R-SLAM과 SLAM3R 역시 learned reconstruction prior를 실제 online dense SLAM/streaming reconstruction으로 연결했다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html?utm_source=chatgpt.com))

즉 2025년부터 질문이

> “정확한 3D를 어떻게 예측할까?”

에서

> **“긴 sequence에서 geometry state를 무엇으로, 어떻게 유지할까?”**

로 명백히 바뀐다.

---

# 5. 동시에 2025년에는 3D에서 4D로 중심이 이동하기 시작한다

여기서 박사논문 관점에서는 이 흐름을 반드시 봐야 한다.

SpatialTrackerV2는 monocular video에서 **scene geometry + camera motion + 3D point trajectories**를 함께 출력하도록 문제를 확장한다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2025/papers/Xiao_SpatialTrackerV2_Advancing_3D_Point_Tracking_with_Explicit_Camera_Motion_ICCV_2025_paper.pdf?utm_source=chatgpt.com))

St4RTrack은 world coordinate frame 안에서 reconstruction과 dynamic point tracking을 동시에 수행한다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2025/papers/Feng_St4RTrack_Simultaneous_4D_Reconstruction_and_Tracking_in_the_World_ICCV_2025_paper.pdf?utm_source=chatgpt.com))

Dynamic Point Maps는 기존 static pointmap 자체를 확장해 motion segmentation, scene flow, 3D tracking 등을 표현할 수 있는 dynamic 3D representation을 제안한다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2025/papers/Sucar_Dynamic_Point_Maps_A_Versatile_Representation_for_Dynamic_3D_Reconstruction_ICCV_2025_paper.pdf?utm_source=chatgpt.com))

Shape of Motion은 world coordinate 안의 **persistent 3D motion trajectory**를 explicit representation으로 둔다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Shape_of_Motion_4D_Reconstruction_from_a_Single_Video_ICCV_2025_paper.html?utm_source=chatgpt.com))

그래서 연구의 representation이

\[
P(u,v)
=
(x,y,z)
\]

에서 점점

\[
P(u,v,t_s,t_t)
=
(x,y,z)
\]

또는 trajectory

\[
\mathcal T_i
=
\{
p_i^1,\ldots,p_i^T
\}
\]

로 넘어가기 시작한다.

즉 **3D point를 기억하는 문제 → 4D trajectory를 기억하는 문제**로 커진다.

---

# 6. 그리고 2025년 TTT가 이 문제와 정확하게 맞물리기 시작한다

앞서 말한 ICML 2025 TTT layer는 hidden state를 model로 바꾸고 현재 input을 통해 weight를 지속적으로 update한다. ([proceedings.mlr.press](https://proceedings.mlr.press/v267/sun25h.html?utm_source=chatgpt.com))

이것은 streaming 3D와 굉장히 잘 맞는다.

기존에는

\[
M_t=
\{K_1,V_1,\ldots,K_t,V_t\}
\]

처럼 과거를 직접 저장했다면,

TTT에서는

\[
W_t
=
\operatorname{Compress}
(x_{1:t})
\]

라는 형태가 가능하다.

다시 말해

> **memory를 token storage에서 learned state transition으로 바꾼다.**

이게 2026년에 폭발한다.

---

# 7. 2026: TTT가 실제 3D long-context memory가 됐다

여기가 네 연구에서 가장 중요하다.

### TTT3R

TTT3R는 3D recurrent reconstruction을 **online learning / test-time training** 관점으로 다시 해석한다. 현재 observation과 accumulated state의 alignment를 이용하여 adaptive update를 수행하고, 긴 sequence에서 forgetting 문제를 완화하는 방향이다. ICLR 2026에 발표됐다. ([openreview.net](https://openreview.net/forum?id=aMs6FtNaY5&utm_source=chatgpt.com))

즉 TTT3R의 질문은 대략

\[
\boxed{
\text{현재 state를 얼마나 강하게 update할 것인가?}
}
\]

에 가깝다.

---

### tttLRM

CVPR 2026 tttLRM은 TTT layer를 long-context/autoregressive 3D reconstruction에 직접 사용한다. 여러 observations를 fast weights에 압축하여 긴 sequence를 처리하는 방향이다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_tttLRM_Test-Time_Training_for_Long_Context_and_Autoregressive_3D_Reconstruction_CVPR_2026_paper.html?utm_source=chatgpt.com))

---

### ZipMap

ZipMap은 TTT를 이용해 image collection을 compact state로 압축하고 linear-time stateful 3D reconstruction을 수행한다. 즉

\[
\text{images}
\rightarrow
\text{TTT hidden state}
\]

라는 구조를 아주 직접적으로 사용한다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Jin_ZipMap_Linear-Time_Stateful_3D_Reconstruction_via_Test-Time_Training_CVPR_2026_paper.html?utm_source=chatgpt.com))

---

### Scal3R

Scal3R 역시 long-video / large-scale reconstruction에서 test-time adapted global context를 사용한다. 즉 pretrained feed-forward model만으로 끝내는 것이 아니라 **test time에 scene-specific information을 지속적으로 축적**한다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Xie_Scal3R_Scalable_Test-Time_Training_for_Large-Scale_3D_Reconstruction_CVPR_2026_paper.html?utm_source=chatgpt.com))

---

### StreamVGGT / STream3R

반면 ICLR 2026 StreamVGGT는 causal transformer와 historical KV memory를 이용해 VGGT 계열 geometry model을 streaming으로 만든다. STream3R는 pointmap reconstruction을 decoder-only autoregressive sequence problem으로 재구성한다. ([openreview.net](https://openreview.net/forum?id=5APgTKsnx8&utm_source=chatgpt.com))

즉 TTT만이 유일한 답은 아니고,

\[
\text{Streaming Memory}
=
\begin{cases}
\text{KV cache}\\
\text{recurrent state}\\
\text{explicit spatial memory}\\
\text{fast weights}
\end{cases}
\]

가 경쟁하고 있는 상황이다.

---

### STAC

그리고 CVPR 2026 STAC은 spatio-temporal cache compression을 이용해 streaming reconstruction에서 cache 자체를 훨씬 작게 유지한다. 즉 단순히

> “KV cache가 너무 크니까 TTT를 쓰자”

만으로는 이제 충분한 motivation이 아니다. KV-cache 기반 방법 자체도 빠르게 발전하고 있다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_STAC_Plug-and-Play_Spatio-Temporal_Aware_Cache_Compression_for_Streaming_3D_Reconstruction_CVPR_2026_paper.pdf?utm_source=chatgpt.com))

이건 네 연구 방향을 잡을 때 상당히 중요하다.

---

# 8. 2026 4D에서는 D4RT가 representation을 다시 한번 바꾼다

D4RT는 CVPR 2026에서 depth, spatio-temporal correspondence, camera parameters를 하나의 transformer representation으로 통합한다. 핵심은 dense output을 task별로 전부 decode하는 것이 아니라, 특정 pixel·source time·target time·camera reference를 query하여 해당 3D 위치를 얻는 방식이다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Efficiently_Reconstructing_Dynamic_Scenes_One_D4RT_at_a_Time_CVPR_2026_paper.html?utm_source=chatgpt.com))

즉

\[
q=(u,v,t_s,t_t,t_c)
\]

를 입력하면

\[
F(q)\rightarrow\mathbf p^{3D}
\]

를 얻는다.

이 하나의 query로

- depth
- point cloud
- point tracking
- camera geometry

를 표현할 수 있다. ([arxiv.org](https://arxiv.org/html/2512.08924v2?utm_source=chatgpt.com))

Any4D 역시 dense metric 4D reconstruction을 feed-forward하게 통합하면서 scene flow와 3D point tracking까지 연결한다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/papers/Karhade_Any4D_Unified_Feed-Forward_Metric_4D_Reconstruction_CVPR_2026_paper.pdf?utm_source=chatgpt.com))

따라서 앞으로는

\[
\text{Depth / Pose / Tracking / Reconstruction}
\]

을 완전히 독립적인 네 task로 연구하기보다,

\[
\boxed{
\text{Unified Spatio-Temporal Geometry}
}
\]

안에서 연구하는 것이 훨씬 강한 방향이라고 판단한다.

---

# 9. Continual Learning 쪽에서도 같은 방향의 힌트가 있다

CL을 단순히

> “old data를 안 잊도록 replay한다”

정도로 보면 안 된다.

최근 foundation model 시대의 CL은 **어떤 parameter subspace를 업데이트할 것인가**가 중요한 문제가 됐다.

2024 InfLoRA는 low-rank adaptation을 interference control과 연결했다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html?utm_source=chatgpt.com))

2025 LoRA Subtraction은 exemplar 없이 feature drift에 강한 parameter space를 만드는 방향을 연구했다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_LoRA_Subtraction_for_Drift-Resistant_Space_in_Exemplar-Free_Continual_Learning_CVPR_2025_paper.pdf?utm_source=chatgpt.com))

2026 KeepLoRA는 이전에 사용된 update directions와의 interference를 줄이기 위해 residual gradient subspace에서 새로운 LoRA update를 학습한다. ([openreview.net](https://openreview.net/forum?id=T3Vc5fkTzV&utm_source=chatgpt.com))

더 직접적으로 ICLR 2025에는 **Test-Time Training for Continual Learning**이 등장하여 continual learning과 test-time correction/retention을 직접 연결했다. ([openreview.net](https://openreview.net/forum?id=9bLdbp46Q1&utm_source=chatgpt.com))

그래서 네가 생각한

> “과거 TTT update 방향을 기억한다”

라는 직관은 연구 흐름과 맞다.

하지만 **그 자체는 이제 충분히 새롭지 않다.**

---

# 10. 2023 → 2026 흐름을 압축하면 이렇게 된다

| 시기 | 3D/4D의 핵심 변화 | TTT / CL의 변화 | 핵심 질문 |
|---|---|---|---|
| 2023 | online depth/reconstruction, tracking이 task별로 발전 | dynamic TTA stability | 현재 input에 안전하게 적응할 수 있는가? |
| 2024 | DUSt3R pointmap으로 geometry task 통합 | low-rank / interference-aware CL | 강한 geometry prior와 작은 adaptation space를 만들 수 있는가? |
| 2025 | VGGT, CUT3R, Point3R → unified/stateful streaming | TTT layer = learnable fast-weight state | 긴 sequence에서 무엇을 memory로 유지할 것인가? |
| 2026 | TTT3R, tttLRM, ZipMap, StreamVGGT, STAC + D4RT | test-time state와 long-term adaptation 결합 | 무한히 긴 3D/4D stream에서 무엇을 기억하고 어떻게 계속 학습할 것인가? |

이 표에서 **다음 칸**, 즉 네가 노려야 할 2027 전후의 질문은 내가 보기에는 이것이다.

\[
\boxed{
\textbf{What should the model remember about how it learns?}
}
\]

---

# 11. Literature Review에서 내가 확인한 가장 중요한 빈 공간

현재 방법들을 memory 관점에서 분류하면 대략 이렇게 된다.

**CUT3R**는 recurrent hidden state를 기억한다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.html?utm_source=chatgpt.com))

**Point3R**는 3D world position에 연결된 spatial feature를 기억한다. ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2025/hash/650db8e1b0b016dc270d51c1476e91cf-Abstract-Conference.html))

**StreamVGGT/STAC**은 historical tokens 또는 compressed cache를 기억한다. ([openreview.net](https://openreview.net/forum?id=5APgTKsnx8&utm_source=chatgpt.com))

**tttLRM/ZipMap**은 과거 observation을 fast weights에 압축한다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_tttLRM_Test-Time_Training_for_Long_Context_and_Autoregressive_3D_Reconstruction_CVPR_2026_paper.html?utm_source=chatgpt.com))

**TTT3R**는 현재 observation과 state의 relation으로 update 크기를 제어한다. ([openreview.net](https://openreview.net/forum?id=aMs6FtNaY5&utm_source=chatgpt.com))

반면 CL의 InfLoRA/KeepLoRA 등은 **parameter update space의 interference와 retention**을 연구한다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html?utm_source=chatgpt.com))

그런데 내가 이번에 조사한 주요 3D/4D top-tier literature에서는 다음 문제를 명시적으로 중심에 둔 방법을 찾지 못했다.

> **“특정 3D geometric/motion context에서 성공했던 adaptation dynamics 자체를 장기 기억하고, 같은 공간·motion regime을 다시 만났을 때 그 plasticity를 검색하여 재사용하는 문제.”**

여기가 네가 파고들 가치가 있는 gap이다.

중요한 것은 **“gradient memory”라고 부르면 너무 약하다.**

나는 이것을

## **Geometry-Conditioned Plasticity Memory**

라고 정의하는 편을 추천한다.

---

# 12. 내가 제안하는 박사논문 수준의 핵심 formulation

전체 연구를 다음과 같이 잡는다.

## **Continual Test-Time Geometric Learning**

모델은 무한하거나 매우 긴 video stream

\[
I_1,I_2,\ldots,I_T
\]

을 causal하게 받는다.

시간 \(t\)에

\[
\hat{\mathcal G}_t
=
F_{\theta,W_t}(I_t)
\]

를 예측한다.

여기서 \(\mathcal G_t\)는 하나만 의미하지 않는다.

\[
\mathcal G_t
=
\{
D_t,
P_t,
C_t,
T_t
\}
\]

로 생각한다.

- \(D_t\): depth
- \(P_t\): pointmap / point cloud
- \(C_t\): camera geometry/pose
- \(T_t\): point trajectories

그리고

\[
W_t
\]

는 short-term TTT state다.

그런데 여기에 새로운 long-term memory를 둔다.

\[
\boxed{
\mathcal M_t^{P}
=
\{
(c_j,U_j,q_j,n_j)
\}_{j=1}^{M}
}
\]

여기서 \(P\)는 **Plasticity**다.

---

# 13. Memory에는 scene image가 아니라 “adaptation knowledge”가 들어간다

각 entry를 이렇게 정의한다.

\[
c_j
=
\text{geometric context}
\]

\[
U_j
=
\text{useful adaptation subspace}
\]

\[
q_j
=
\text{reliability / utility}
\]

\[
n_j
=
\text{recurrence statistics}
\]

그리고 context는 RGB feature 하나가 아니다.

\[
c_t
=
\phi
(
z_t,
D_t,
\Delta C_t,
T_t,
u_t,
p_t^{world}
)
\]

정도로 구성한다.

즉

- appearance
- depth structure
- camera motion
- world-space location
- point motion
- uncertainty

를 함께 사용한다.

이걸 나는 **geometry-conditioned retrieval**이라고 정의하겠다.

---

# 14. 가장 중요한 novelty: gradient를 저장하는 게 아니라 “plasticity basis”를 학습한다

프레임마다

\[
g_t
=
\nabla_W
\mathcal L_{\text{TTT}}
\]

가 나온다고 하자.

단순히

\[
M=\{g_1,g_2,\ldots\}
\]

를 저장하면 논문 수준이 약하다.

대신 비슷한 geometric contexts에서 반복적으로 나타난 useful updates를

\[
U_j
=
[u_{j1},\ldots,u_{jr}]
\]

라는 작은 **adaptation subspace**로 consolidate한다.

\[
r\ll d.
\]

그래서 현재 상황과 비슷한 memory를 retrieve하면

\[
U_t
=
\operatorname{Retrieve}
(c_t,\mathcal M)
\]

가 된다.

현재 gradient를

\[
g_t^{reuse}
=
U_tU_t^\top g_t
\]

와

\[
g_t^{novel}
=
g_t-g_t^{reuse}
\]

로 나눈다.

이 해석이 굉장히 중요하다.

### \(g_t^{reuse}\)

> 전에 비슷한 geometry에서 이미 유용하다고 학습한 adaptation.

### \(g_t^{novel}\)

> 지금 처음 관측된 새로운 adaptation information.

---

# 15. 그러면 Continual Learning의 stability–plasticity가 3D 의미를 갖게 된다

최종 update를

\[
\boxed{
\Delta W_t
=
-\eta
\left(
g_t^{reuse}
+
\alpha_tg_t^{novel}
\right)
}
\]

라고 하자.

여기서

\[
\alpha_t
=
f(
\text{novelty},
\text{uncertainty},
\text{memory agreement}
)
\]

이다.

익숙한 3D 영역/동작을 다시 만났다면

\[
\alpha_t\downarrow
\]

해서 검증된 adaptation을 주로 재사용한다.

새로운 공간/동작이라면

\[
\alpha_t\uparrow
\]

해서 새로운 direction을 적극적으로 배운다.

그러면 CL의 고전적인

\[
\text{Stability}
\leftrightarrow
\text{Plasticity}
\]

문제가

\[
\boxed{
\text{Geometric Familiarity}
\leftrightarrow
\text{Geometric Novelty}
}
\]

로 바뀐다.

이게 generic CL을 그냥 3D에 붙이는 것보다 훨씬 강한 research story다.

---

# 16. 그런데 나는 여기서 한 단계 더 가는 것을 추천한다

CVPR/ICLR급 contribution으로 만들려면 단순 \(U_j\) memory보다 **Adaptation Atom Dictionary**가 더 좋다.

전체 memory에

\[
B_1,\ldots,B_K
\]

개의 reusable low-rank adaptation atoms를 둔다.

현재 context가 들어오면

\[
a_t
=
R(c_t)
\]

라는 sparse routing coefficient를 만들고

\[
\boxed{
\Delta W_t
=
\sum_{k=1}^{K}
a_{t,k}B_k
+
R_t
}
\]

로 한다.

\(B_k\)는 장기간 반복적으로 유용했던 update pattern이고,

\(R_t\)는 현재 상황에서 처음 필요한 residual adaptation이다.

예를 들어 실제로는 이런 adaptation atoms가 생길 수 있다.

```text
B1 : small-parallax geometry correction
B2 : camera rotation / pose correction
B3 : depth-scale correction
B4 : dynamic correspondence correction
B5 : re-appearance after occlusion
...
```

물론 이 의미를 supervision으로 지정하는 게 아니라 training 중 자연스럽게 발견하도록 한다.

이렇게 하면 연구 질문이

> gradient를 어떻게 저장할까?

에서

> **“장기간 경험을 통해 reusable geometric learning dynamics를 발견할 수 있는가?”**

로 바뀐다.

이건 훨씬 깊다.

---

# 17. 특히 4D에서는 memory를 world-space trajectory에 연결해야 한다

이 부분은 박사논문의 두 번째 단계로 상당히 강하다.

일반적인 image-keyed memory는

\[
I_t\leftrightarrow M_j
\]

이다.

하지만 4D에서는 이것보다

\[
\boxed{
\text{3D point / trajectory}
\leftrightarrow
\text{Plasticity}
}
\]

로 가는 게 좋다.

point \(p_i\)의 trajectory가

\[
\mathcal T_i
=
\{
p_i^t
\}_{t=1}^T
\]

라면 memory를

\[
M_i
=
(
\mathcal T_i,
U_i,
q_i
)
\]

에 연결한다.

예를 들어 object가 가려졌다가 다시 등장했다면 image가 완전히 달라져도

- world position
- trajectory
- motion
- geometry

가 연결되어 있으므로 동일한 adaptation experience를 다시 가져올 수 있다.

이건 SpatialTrackerV2, St4RTrack, Dynamic Point Maps, D4RT가 만들어놓은 4D representation 흐름과 매우 자연스럽게 연결된다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2025/papers/Xiao_SpatialTrackerV2_Advancing_3D_Point_Tracking_with_Explicit_Camera_Motion_ICCV_2025_paper.pdf?utm_source=chatgpt.com))

그리고 여기부터는 사실상

> **“continual 4D world model”**

이라는 박사논문 수준의 문제가 된다.

---

# 18. 반드시 dual-memory로 생각하는 것이 좋다

나는 논문에서 memory를 두 종류로 명시적으로 분리할 것을 권한다.

\[
\boxed{
\mathcal M^{Scene}
}
\]

은

> 지금까지 무엇을 보았는가?

를 저장한다.

Point3R pointer, KV cache, recurrent geometry state 등이 여기에 해당한다.

그리고

\[
\boxed{
\mathcal M^{Plasticity}
}
\]

는

> 이전에 이런 상황을 만났을 때 **어떻게 모델을 바꿨는가?**

를 저장한다.

최종 구조는

```text
                    Streaming RGB
                         │
                         ▼
                Geometry Backbone
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Scene Memory          Context Encoder
     "What have I seen?"             │
                                     ▼
                            Plasticity Memory
                           "How did I adapt?"
                                     │
                                     ▼
                             Adaptation Prior
                                     │
                          Current TTT Residual
                                     │
                                     ▼
                            Updated Fast Weight
                                     │
             ┌─────────────┬─────────┼─────────────┐
             ▼             ▼         ▼             ▼
           Depth        Pointmap    Pose        Tracking
```

이 그림만 봐도 기존 방법과 contribution이 상당히 잘 구별된다.

---

# 19. 여기서 “좋은 adaptation”을 어떻게 판단하느냐가 핵심 연구 문제다

GT는 test stream에 없으므로 모든 gradient를 long-term memory에 넣으면 catastrophic contamination이 발생한다.

그래서 update 전후의 self-supervised geometric improvement를 사용한다.

\[
Q_t
=
\mathcal L_{geo}(W_t)
-
\mathcal L_{geo}(W_{t+1})
\]

예를 들어 \(\mathcal L_{geo}\)에

\[
\mathcal L_{geo}
=
\lambda_1 L_{reproj}
+
\lambda_2 L_{depth}
+
\lambda_3 L_{track}
+
\lambda_4 L_{pose}
\]

처럼 넣을 수 있다.

중요한 원리는 loss 개수가 아니라

\[
Q_t>0
\]

인 **실제로 geometry consistency를 향상시킨 adaptation만** long-term consolidation한다는 것이다.

그리고 uncertainty가 크면 저장하지 않는다.

\[
Q_t>\tau_Q,
\qquad
u_t<\tau_u.
\]

이것이 TTT의 error accumulation 문제와 CL의 memory contamination 문제를 동시에 다룬다.

---

# 20. 이 연구에서 “revisit”을 반드시 명시적인 problem setting으로 만들어야 한다

여기가 현재 benchmark와 차별화할 수 있는 중요한 부분이다.

일반적인 long-sequence reconstruction metric은

\[
1\rightarrow2\rightarrow3\rightarrow\cdots\rightarrow T
\]

에서 최종 reconstruction quality를 본다.

그런데 네 research question에서는 다음 sequence가 훨씬 중요하다.

\[
A
\rightarrow
B
\rightarrow
C
\rightarrow
A'
\]

즉 **과거 environment/context에 재방문**한다.

질문은

> A를 이미 경험한 모델이 A′에서 처음 본 모델보다 빨리/안정적으로 적응하는가?

이다.

이걸 새로운 evaluation axis로 정의할 수 있다.

---

# 21. 정확도만으로는 절대로 논문이 완성되지 않는다

내가 이 연구에서 반드시 측정하라고 할 것은 다음 다섯 축이다.

1. **Online reconstruction quality**  
   depth, point cloud, pose, tracking 성능.

2. **Retention**  
   과거 공간으로 돌아갔을 때 과거 adaptation이 유지되는가.

3. **Re-adaptation speed**  
   과거 context를 다시 만났을 때 몇 frame/TTT steps 안에 좋은 성능으로 회복하는가.

4. **Novelty plasticity**  
   처음 보는 environment에서도 새로운 adaptation을 배울 수 있는가.

5. **Bounded resource**  
   sequence가 길어져도 memory와 computation이 제한된 budget 안에 유지되는가.

이 다섯 개가 동시에 좋아야

\[
\boxed{
\text{Continual Streaming 3D/4D}
}
\]

라고 주장할 수 있다.

---

# 23. 어떤 baseline과 비교해야 하는지도 명확하다

첫 번째 그룹은 **geometry foundation / offline**이다: DUSt3R, VGGT.

두 번째 그룹은 **stateful streaming**이다: CUT3R, Point3R, SLAM3R, StreamVGGT, STream3R.

세 번째 그룹은 **TTT / fast-weight 3D**다: TTT3R, tttLRM, ZipMap, Scal3R.

네 번째 그룹은 **dynamic / 4D**다: SpatialTrackerV2, St4RTrack, Dynamic Point Maps, D4RT, Any4D.

다섯 번째 그룹은 method ablation 관점의 **continual adaptation**이다: InfLoRA나 KeepLoRA식 subspace control을 동일 backbone에 직접 적용한 baseline.

모든 baseline을 하나의 table에 억지로 넣을 필요는 없다. output modality가 다르므로 static streaming / 4D tracking / CL mechanism으로 evaluation table을 분리하는 것이 타당하다. 관련 방법들의 기능적 차이는 각각의 공식 발표와도 일치한다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.html?utm_source=chatgpt.com))

---

# 24. CVPR 논문으로 가져간다면 어디까지 해야 하는가

첫 논문에서는 욕심내서 처음부터 full 4D까지 하지 않는 것을 추천한다.

**CVPR 버전의 핵심 문제는:**

> **Long-horizon streaming 3D reconstruction에서 recurring geometric contexts에 대한 adaptation experience를 bounded plasticity memory에 저장하고 재활용한다.**

output은

\[
\text{Depth}
+
\text{Pointmap/Point Cloud}
+
\text{Camera Pose}
\]

정도면 충분하다.

그리고 가장 중요한 contribution은 세 개다.

**Geometry-conditioned plasticity memory**, **bounded continual consolidation**, **revisit-aware streaming evaluation**.

이 세 개가 서로 묶여 있어야 한다.

그냥

> TTT3R + gradient memory

면 약하다.

반면

> **“새로운 continual streaming geometry problem formulation + reusable adaptation subspace + long-horizon revisit benchmark”**

이면 CVPR paper story가 된다.

---

# 25. ICLR을 목표로 한다면 조금 다르게 가야 한다

ICLR에서는 3D framework 자체보다 **learning principle**을 앞에 둔다.

질문을

> **Can test-time learners acquire reusable plasticity without catastrophic interference under recurrent non-stationary streams?**

로 잡는다.

그리고 3D/4D streaming을 가장 강한 real-world validation domain으로 사용한다.

그때는

\[
g_t^{reuse}
+
g_t^{novel}
\]

분해가 왜 stability–plasticity trade-off를 개선하는지,

memory rank와 forgetting의 관계,

routing error가 adaptation stability에 미치는 영향,

memory capacity와 recurrence frequency 관계,

negative transfer 조건

등을 분석해야 한다.

그래서 같은 아이디어라도

**CVPR**
→ geometric representation / reconstruction / tracking 중심.

**ICLR**
→ continual test-time learning principle 중심.

으로 논문 framing이 달라진다.

---

# 26. 박사논문은 3단계로 확장하는 것이 가장 좋다

내가 지금 장기 연구계획을 잡는다면 다음 순서로 간다.

### 연구 1 — Continual Streaming 3D

\[
\boxed{
\text{Geometry-conditioned Plasticity Memory}
}
\]

static 또는 mostly-static stream에서

- depth
- point cloud
- camera pose

를 대상으로 한다.

핵심은 **revisit → adaptation reuse**다.

---

### 연구 2 — Continual Streaming 4D

plasticity memory를

\[
\text{3D location}
\]

에서

\[
\text{4D trajectory}
\]

로 확장한다.

그러면

- dynamic objects
- occlusion
- point reappearance
- correspondence
- camera/object motion disentanglement

까지 들어간다.

여기서 D4RT/SpatialTrackerV2/St4RTrack 계열과 직접 연결된다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Efficiently_Reconstructing_Dynamic_Scenes_One_D4RT_at_a_Time_CVPR_2026_paper.html?utm_source=chatgpt.com))

---

### 연구 3 — Lifelong Geometric World Model

마지막에는 scene-specific memory를 넘어간다.

여러 environment

\[
S_1,S_2,S_3,\ldots
\]

를 경험하면서

\[
\mathcal M^{P}_{global}
\]

을 만드는 것이다.

즉 특정 방의 geometry를 기억하는 것이 아니라

> **“이런 geometric situation에서는 어떻게 학습하는 것이 좋은가?”**

라는 transferable learning dynamics를 축적한다.

그러면 새로운 scene \(S_{new}\)에서도

\[
\text{cold-start TTT}
\]

보다

\[
\text{experience-conditioned TTT}
\]

가 빠르게 적응한다.

여기까지 가면 연구가 단순 3D reconstruction method가 아니라

\[
\boxed{
\text{Lifelong Spatial Intelligence}
}
\]

라는 박사논문 주제가 된다.

---

# 27. 이 연구에서 내가 가장 경계하는 네 가지 실패 방향

**첫째, “KV cache를 안 쓰려고 TTT를 쓴다.”**

2026에는 ZipMap, STAC 등 때문에 이것만으로는 약하다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Jin_ZipMap_Linear-Time_Stateful_3D_Reconstruction_via_Test-Time_Training_CVPR_2026_paper.html?utm_source=chatgpt.com))

**둘째, “gradient를 저장했다가 cosine similarity로 꺼낸다.”**

reviewer가 바로

> “GPM/LoRA/subspace CL을 3D에 적용한 것 아닌가?”

라고 할 가능성이 높다.

**셋째, RGB similarity로 adaptation memory를 retrieve한다.**

그러면 일반 memory bank/RAG와 구별하기 어렵다.

반드시

\[
\boxed{
\text{world-space geometry + camera + motion}
}
\]

에 memory를 ground해야 한다.

**넷째, 기존 benchmark 평균 accuracy만 개선한다.**

그러면 왜 Continual Learning이 필요한지 증명하지 못한다.

반드시

\[
A\rightarrow B\rightarrow C\rightarrow A
\]

같은 recurrence protocol을 넣어야 한다.

---

# 28. 그래서 최종적으로 내가 추천하는 연구 방향

현재 단계에서 연구의 중심 가설은 이것으로 잡는 것이 가장 좋다.

> **A streaming 3D/4D model should not only maintain a representation of the observed world, but also continually acquire a geometry-conditioned memory of how to adapt its fast weights. By consolidating successful test-time updates into reusable low-rank plasticity atoms and retrieving them upon spatial or motion-context recurrence, the model can achieve both rapid adaptation to new observations and long-term retention under bounded memory.**

한국어로 더 간단히 하면:

> **기존 streaming 3D는 “과거에 무엇을 봤는가”를 기억한다. 우리는 “과거의 이 3D/4D 상황에서 어떻게 적응했는가”까지 기억하게 한다. 새로운 상황에서는 TTT로 배우고, 익숙한 공간·motion context가 다시 나타나면 과거에 학습한 adaptation을 재사용한다.**

그리고 논문의 핵심 구조는

\[
\boxed{
\text{Scene Memory}
+
\text{Plasticity Memory}
}
\]

\[
\boxed{
\text{TTT = Short-term Learning}
}
\]

\[
\boxed{
\text{CL = Long-term Consolidation of Plasticity}
}
\]

로 잡는 것이 가장 명확하다.

---

# 29. 연구 제목도 지금 단계에서는 이렇게 잡는 편이 좋다

단일 CVPR/ICLR 논문 가칭이라면 **GeoPLAST: Geometry-Conditioned Continual Plasticity for Streaming 3D Reconstruction** 정도가 연구 내용을 가장 정확히 표현한다.

박사논문 전체 umbrella는

> **Lifelong Spatial Intelligence: Continual Test-Time Learning for Streaming 3D and 4D Perception**

정도가 좋다.

중요한 것은 명칭보다 연구 축이다.

\[
\boxed{
\text{Representation}
\rightarrow
\text{Memory}
\rightarrow
\text{Adaptation}
\rightarrow
\text{Continual Plasticity}
}
\]

2023–2026의 흐름을 보면 **representation과 scene memory는 이미 매우 빠르게 해결되고 있고, 이제 남는 문제는 장기 streaming에서 “학습 과정 자체를 어떻게 기억하고 관리할 것인가”**로 이동할 가능성이 크다. 내가 지금 CVPR/ICLR과 박사논문을 동시에 고려해 하나의 축에 베팅한다면, 바로 이 **geometry-conditioned continual plasticity**를 선택하겠다. ([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.html?utm_source=chatgpt.com))

다음 단계에서는 브레인스토밍이 아니라 실제 연구 설계로 넘어가서, **① Problem Formulation, ② 전체 architecture, ③ 각 module의 수식, ④ TTT loss, ⑤ plasticity consolidation algorithm, ⑥ CVPR용 experiment protocol과 ablation, ⑦ 가장 먼저 구현해볼 최소 prototype**까지 하나의 Method로 고정하는 게 좋다. 그 과정에서 특히 **TTT3R/CUT3R 중 어느 것을 starting backbone으로 삼을지**도 논문 novelty와 구현 난이도를 기준으로 결정할 수 있다.

2026 후반에는 관련 논문이 계속 추가될 가능성이 있으니, ICLR/CVPR뿐 아니라 ECCV/NeurIPS의 TTT·streaming 3D/4D 후속 연구를 계속 추적하는 것도 유용하다.
