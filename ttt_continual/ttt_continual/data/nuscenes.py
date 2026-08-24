"""Streaming episodes from nuScenes.

An episode is one scene read in order at the full sweep rate, cut into chunks.
Each chunk supplies context frames, which the fast weight adapts on, and query
frames, which are decoded and scored. Query frames are held out of the context
so a chunk is never graded on an image it was just shown.

Two choices about *which* frames become queries are worth stating.

Some queries are drawn from the chunk's own span and some from spans already
gone past. The revisit queries are the point of the exercise: they are what
makes forgetting visible, since a chunk that has overwritten what an earlier one
learned will render those older viewpoints worse.

Poses are normalised once per episode, not per chunk. The scale a scene is
reconstructed at would otherwise drift as the car travels, and quality measured
at chunk 2 would not be comparable with quality at chunk 20.
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..models.regime import describe
from ..utils.camera import compute_rays, normalise_poses, plucker, rescale_intrinsics


@dataclass
class DataConfig:
    root: str                        # directory of converted scenes
    image_size: tuple = (224, 400)
    chunk_size: int = 4              # context frames per chunk
    n_chunks: int = 8                # chunks per episode
    stride: int = 2                  # frames skipped between context frames
    queries_current: int = 1         # held-out queries inside the current chunk
    queries_revisit: int = 1         # held-out queries from earlier chunks
    regime_window: int = 12          # frames of history the descriptor reads
    fps: float = 12.0
    max_scenes: Optional[int] = None


class StreamingEpisodes(Dataset):
    def __init__(self, cfg: DataConfig, split: str = "train", seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)
        names = sorted(os.listdir(cfg.root))
        names = [n for n in names
                 if os.path.exists(os.path.join(cfg.root, n, "opencv_cameras.json"))]
        if cfg.max_scenes:
            names = names[: cfg.max_scenes]
        # deterministic split so a scene never appears in both
        cut = int(len(names) * 0.9)
        self.scenes = names[:cut] if split == "train" else names[cut:]
        if not self.scenes:
            raise RuntimeError(f"no scenes for split={split} under {cfg.root}")
        self._meta: Dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.scenes)

    def _load_meta(self, name: str) -> dict:
        if name not in self._meta:
            path = os.path.join(self.cfg.root, name, "opencv_cameras.json")
            self._meta[name] = json.load(open(path))
        return self._meta[name]

    def _load_frame(self, frame: dict) -> tuple:
        h, w = self.cfg.image_size
        img = Image.open(frame["file_path"]).convert("RGB")
        src = (img.size[1], img.size[0])
        img = img.resize((w, h), Image.BILINEAR)
        rgb = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        intr = rescale_intrinsics(
            np.array([frame["fx"], frame["fy"], frame["cx"], frame["cy"]]), src, (h, w))
        return rgb, torch.from_numpy(intr).float()

    def __getitem__(self, index: int) -> Dict:
        cfg = self.cfg
        name = self.scenes[index]
        meta = self._load_meta(name)
        frames = meta["frames"]

        span = cfg.n_chunks * cfg.chunk_size * cfg.stride
        need = span + cfg.regime_window
        if len(frames) < need:
            raise RuntimeError(f"{name}: {len(frames)} frames, need {need}")
        start = self.rng.randint(cfg.regime_window, len(frames) - span - 1)

        picked = list(range(start, start + span, cfg.stride))
        c2w_all = np.stack([np.linalg.inv(np.asarray(frames[i]["w2c"])) for i in picked])
        c2w_all, _, _ = normalise_poses(c2w_all, "first_cam")
        pose_of = {f: torch.from_numpy(c2w_all[j]).float() for j, f in enumerate(picked)}

        # raw (unnormalised) poses feed the regime descriptor: it reads
        # frame-to-frame differences, which must be in metres to be comparable
        # between episodes
        raw_c2w = np.stack([np.linalg.inv(np.asarray(f["w2c"])) for f in frames])

        t0, t1 = picked[0], picked[-1]
        time_of = {f: (f - t0) / max(t1 - t0, 1) for f in picked}

        chunks, seen = [], []
        for c in range(cfg.n_chunks):
            block = picked[c * cfg.chunk_size:(c + 1) * cfg.chunk_size]
            ctx = block[:-cfg.queries_current] if cfg.queries_current else block
            cur_q = block[len(ctx):]
            revisit = self.rng.sample(seen, min(cfg.queries_revisit, len(seen))) if seen else []
            queries = cur_q + revisit
            if not queries:
                continue

            ctx_rgb, ctx_intr = zip(*[self._load_frame(frames[f]) for f in ctx])
            q_rgb, q_intr = zip(*[self._load_frame(frames[f]) for f in queries])
            window = raw_c2w[max(0, block[0] - cfg.regime_window):block[0] + 1]

            chunks.append({
                "context_rgb": torch.stack(ctx_rgb),
                "context_c2w": torch.stack([pose_of[f] for f in ctx]),
                "context_intr": torch.stack(ctx_intr),
                "context_times": torch.tensor([time_of[f] for f in ctx]),
                "query_rgb": torch.stack(q_rgb),
                "query_c2w": torch.stack([pose_of[f] for f in queries]),
                "query_intr": torch.stack(q_intr),
                "query_times": torch.tensor([time_of[f] for f in queries]),
                "query_is_revisit": torch.tensor([0] * len(cur_q) + [1] * len(revisit)),
                "regime": torch.from_numpy(describe(window, fps=cfg.fps)).float(),
            })
            seen.extend(ctx)

        return {"scene": name, "chunks": chunks}


def build_features(rgb: torch.Tensor, c2w: torch.Tensor, intr: torch.Tensor):
    """Pack images and their rays into the trunk's input channels."""
    b, v, _, h, w = rgb.shape
    ray_o, ray_d = compute_rays(intr, c2w, h, w)
    feats = torch.cat([plucker(ray_o, ray_d), rgb * 2.0 - 1.0], dim=2)
    return feats, ray_o, ray_d


def collate_episode(batch: List[Dict]) -> Dict:
    """Batch size one. Streaming state is per-episode, and mixing episodes in a
    batch would require carrying one fast weight per element -- possible, but it
    multiplies memory by the batch size for no gain over accumulating gradients
    across episodes instead."""
    assert len(batch) == 1, "episodes are processed one at a time"
    item = batch[0]
    out = []
    for c in item["chunks"]:
        out.append({k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in c.items()})
    return {"scene": item["scene"], "chunks": out}
