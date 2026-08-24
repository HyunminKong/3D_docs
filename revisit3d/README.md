# Revisit3D

독립적인 **foundation-model 기반 reusable test-time adaptation** 연구 코드다.
tttLRM의 fast weight나 기존 `ttt_continual` bank를 재사용하지 않는다.

첫 수직 슬라이스는 다음 두 부분이다.

1. converted nuScenes pose에서 실제 location overlap을 확인하는 cross-episode
   `A → B → A'` benchmark manifest;
2. frozen VGGT/DINO feature 위에서만 동작하는 compact state `z` 기반 geometry head.

```text
frozen foundation features → StreamingGeometryHead(features, z)
                                      ↑
                          test-time update only on z
```

출력은 token-level pointmap/depth/confidence와 view-level relative pose다. 아직
memory bank는 없다. 다음 milestone은 reprojection/depth consistency online loss와
revisit-aware outer-loop 학습이며, 그 효과가 검증된 뒤에만 long-term memory를 넣는다.

`revisit3d.backbones.FrozenVGGTFeatures`는 FastVGGT의 pretrained prediction
head가 아니라 마지막 aggregator patch token만 추출한다. 따라서 pointmap/depth/
pose prediction과 TTT state 주입은 모두 Revisit3D의 새 head가 담당한다.

## Smoke test

```bash
PYTHONPATH=. python revisit3d/scripts/smoke_test.py
PYTHONPATH=. python revisit3d/scripts/meta_smoke_test.py
```

## Manifest build

```bash
PYTHONPATH=. python revisit3d/scripts/build_revisit_benchmark.py \
  --root tttLRM/data_example/nuscenes_2x2 \
  --selection tttLRM/oracle/results/sel_2x2.json \
  --out revisit3d/manifests/nuscenes_revisit.json
```

The manifest split is by connected physical-overlap component, not by individual
scene, to avoid location leakage.

`RevisitEpisodeDataset` returns separate `a`, `b`, and `a_prime` context/query
blocks with RGB, resized intrinsics, and `w2c`. It does not call any target a
test-time supervision signal.

`RevisitMetaLearner` is still memory-free. It trains an oracle matched-prior
signed residual map on an unrolled `A → B → A'` episode. A bank/retriever is
not allowed until this oracle condition beats cold TTT on held-out locations.

## First training plumbing

```bash
PYTHONPATH=. python revisit3d/scripts/train_oracle_revisit.py \
  --known-pose-bootstrap --steps 1
```

This is explicitly a bootstrap, not the final pose-estimation protocol: supplied
camera transforms are used only to validate the reprojection objective and
unrolled gradients. The deployment TTT path must replace them with predicted
poses or a permitted odometry source.

The first non-oracle pose path is already available:

```bash
PYTHONPATH=. python revisit3d/scripts/train_oracle_revisit.py --pose-source predicted --steps 1
```

## Oracle evaluation gate

```bash
PYTHONPATH=. python revisit3d/scripts/evaluate_oracle_revisit.py \
  --checkpoint revisit3d/checkpoints/oracle_predicted_pose_smoke.pt
```

It reports held-out A' reprojection for cold TTT, carried current TTT, oracle
matched residual, and a foreign residual. This is the required go/no-go before
any retrieval mechanism is implemented.
