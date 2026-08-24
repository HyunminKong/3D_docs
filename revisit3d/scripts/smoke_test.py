#!/usr/bin/env python3
"""CPU smoke test for the benchmark and compact-state TTT interface."""

import tempfile

import torch

from revisit3d.data import RevisitBenchmark, RevisitEpisodeDataset, build_manifest
from revisit3d.models import StreamingGeometryHead


def main() -> None:
    episodes = build_manifest(
        "tttLRM/data_example/nuscenes_2x2",
        selection="tttLRM/oracle/results/sel_2x2.json",
        overlap_m=2.0,
    )
    benchmark = RevisitBenchmark(episodes)
    assert episodes and all(e.min_overlap_m <= 2.0 for e in episodes)
    assert not (set(e.source_scene for e in benchmark.split("train")) &
                set(e.source_scene for e in benchmark.split("test")))
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest = f"{temp_dir}/smoke.json"
        benchmark.write(manifest)
        dataset = RevisitEpisodeDataset(manifest, "tttLRM/data_example/nuscenes_2x2", image_size=(56, 56))
        sample = dataset[0]
        assert sample["a"]["context"]["rgb"].shape == (8, 3, 56, 56)
        assert sample["a_prime"]["query"]["w2c"].shape == (4, 4, 4)

    head = StreamingGeometryHead(feature_dim=64, state_dim=8, hidden_dim=96)
    features = torch.randn(2, 3, 11, 64)
    state = head.initial_state(2, device=features.device, dtype=features.dtype)
    output = head(features, state)
    assert output["pointmap"].shape == (2, 3, 11, 3)
    assert output["depth"].shape == (2, 3, 11, 1)
    assert output["relative_pose"].shape == (2, 3, 6)

    # An online-only proxy objective; it verifies that only z is updated.
    before = [p.detach().clone() for p in head.parameters()]
    updated, losses = head.adapt(features, state, lambda pred: pred["depth"].mean(), steps=2)
    assert len(losses) == 2 and not torch.equal(updated.value, state.value)
    assert all(torch.equal(old, new) for old, new in zip(before, head.parameters()))
    print(f"benchmark={len(episodes)} {benchmark.summary()} | compact-state TTT smoke test passed")


if __name__ == "__main__":
    main()
