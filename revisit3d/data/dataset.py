"""Image/camera loader for an explicit cross-episode revisit manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RevisitEpisodeDataset(Dataset):
    """Load A, B and A' without conflating them into a normal video clip.

    Each item contains disjoint context and held-out query frames.  The loader
    intentionally provides RGB and cameras only: held-out GT geometry belongs
    to an evaluation adapter, never to an online TTT objective by accident.
    """

    def __init__(self, manifest: str | Path, scene_root: str | Path,
                 split: str = "train", image_size: tuple[int, int] = (224, 224)):
        records = json.loads(Path(manifest).read_text())
        self.records = [record for record in records if record["split"] == split]
        if not self.records:
            raise ValueError(f"manifest has no {split!r} episodes")
        self.scene_root = Path(scene_root)
        self.image_size = image_size
        self._metadata: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _meta(self, scene: str) -> dict:
        if scene not in self._metadata:
            path = self.scene_root / scene / "opencv_cameras.json"
            self._metadata[scene] = json.loads(path.read_text())
        return self._metadata[scene]

    def _frame(self, scene: str, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        meta = self._meta(scene)
        frame = meta["frames"][index]
        path = self.scene_root / scene / frame["file_path"]
        image = Image.open(path).convert("RGB")
        old_w, old_h = image.size
        height, width = self.image_size
        image = image.resize((width, height), Image.BILINEAR)
        rgb = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        sx, sy = width / old_w, height / old_h
        intrinsics = torch.tensor(
            [frame["fx"] * sx, frame["fy"] * sy, frame["cx"] * sx, frame["cy"] * sy], dtype=torch.float32
        )
        w2c = torch.tensor(frame["w2c"], dtype=torch.float32)
        return rgb, intrinsics, w2c

    def _load_segment(self, descriptor: dict) -> dict[str, torch.Tensor | str]:
        scene = descriptor["scene"]
        context = [self._frame(scene, index) for index in descriptor["frames"]]
        query = [self._frame(scene, index) for index in descriptor["query_frames"]]
        def pack(items):
            rgb, intr, w2c = zip(*items)
            return {"rgb": torch.stack(rgb), "intrinsics": torch.stack(intr), "w2c": torch.stack(w2c)}
        return {"scene": scene, "context": pack(context), "query": pack(query)}

    def __getitem__(self, index: int) -> dict:
        item = self.records[index]
        return {
            "episode_id": item["episode_id"],
            "split": item["split"],
            "min_overlap_m": torch.tensor(item["min_overlap_m"], dtype=torch.float32),
            "a": self._load_segment(item["a"]),
            "b": self._load_segment(item["b"]),
            "a_prime": self._load_segment(item["a_prime"]),
        }
