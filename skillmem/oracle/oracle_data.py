"""Batch construction with explicit control over which views go where.

The stock `data.dataset_scene.Dataset` picks views with k-means and normalises
poses using whatever ended up in the batch.  Both are fatal for this
experiment:

* we need to name the A segment, the B segment and the held-out targets
  ourselves, and
* `normalize_with_mean_pose` derives `scene_scale` from
  ``max |c2w[:, :3, 3]|`` over *the views present in the batch*, so a batch
  containing A alone and a batch containing A+B would be reconstructed at
  different scales and their PSNRs would not be comparable.

So the normalisation is computed once over the union of every view any
condition will use, and each condition slices from that shared frame.
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from data.dataset_scene import normalize_with_mean_pose, resize_and_crop


class SceneViews:
    """All frames of one converted DL3DV scene, in a single shared pose frame."""

    def __init__(self, cameras_json, image_size, image_size_x, frame_method="first_cam"):
        self.path = cameras_json
        self.base_dir = os.path.dirname(cameras_json)
        data = json.load(open(cameras_json))
        self.scene_name = data.get("scene_name", os.path.basename(self.base_dir))
        self.frames = data["frames"]
        self.image_size = image_size
        self.image_size_x = image_size_x
        self.frame_method = frame_method
        self._cache = {}
        self._normalised = None

    def __len__(self):
        return len(self.frames)

    def _load(self, index):
        if index not in self._cache:
            frame = self.frames[index]
            image = Image.open(os.path.join(self.base_dir, frame["file_path"]))
            fxfycxcy = [frame["fx"], frame["fy"], frame["cx"], frame["cy"]]
            image, fxfycxcy = resize_and_crop(image, (self.image_size, self.image_size_x), fxfycxcy)
            if image.mode != "RGB":
                image = image.convert("RGB")
            self._cache[index] = (
                transforms.ToTensor()(image),
                torch.tensor(fxfycxcy, dtype=torch.float32),
            )
        return self._cache[index]

    def normalise(self, universe):
        """Fix the coordinate frame using `universe`, the union of all views used.

        Must be called once, before any batch is built, and the same universe
        must be used for every condition being compared.
        """
        universe = list(universe)
        c2ws = np.stack([np.linalg.inv(np.asarray(self.frames[i]["w2c"])) for i in universe])
        normed, apos, scene_scale = normalize_with_mean_pose(c2ws, frame_method=self.frame_method)
        self._normalised = {
            "c2w": {i: torch.from_numpy(normed[j]).float() for j, i in enumerate(universe)},
            "apos": torch.from_numpy(apos).float(),
            "scene_scale": float(scene_scale),
            "universe": universe,
        }
        return self._normalised

    def batch(self, input_views, virtual_views, target_views, device=None):
        """Assemble one batch.

        `input_views` update the fast weight; `virtual_views` are the query
        cameras the Gaussians are decoded at and must be a subset of
        `input_views` (the model gathers their images from the input block);
        `target_views` are the held-out cameras the render is scored against.
        """
        if self._normalised is None:
            raise RuntimeError("call normalise(universe) before batch()")
        missing = set(virtual_views) - set(input_views)
        if missing:
            raise ValueError(f"virtual views must be a subset of input views; missing {sorted(missing)}")

        order = list(input_views) + list(target_views)
        images, fxfycxcy, c2ws = [], [], []
        for idx in order:
            image, intr = self._load(idx)
            images.append(image)
            fxfycxcy.append(intr)
            c2ws.append(self._normalised["c2w"][idx])

        pos_in_input = {v: i for i, v in enumerate(input_views)}
        virtual_idx = [pos_in_input[v] for v in virtual_views]

        batch = {
            "image": torch.stack(images).unsqueeze(0),
            "c2w": torch.stack(c2ws).unsqueeze(0),
            "fxfycxcy": torch.stack(fxfycxcy).unsqueeze(0),
            "index": torch.zeros(1, len(order), 2, dtype=torch.long),
            "scene_name": [self.scene_name],
            "virtual_c2w": torch.stack([self._normalised["c2w"][v] for v in virtual_views]).unsqueeze(0),
            "virtual_fxfycxcy": torch.stack([self._load(v)[1] for v in virtual_views]).unsqueeze(0),
            "virtual_input_indices": torch.tensor(virtual_idx, dtype=torch.long).unsqueeze(0),
            "num_input_views": torch.tensor([len(input_views)], dtype=torch.long),
            "scene_scale": torch.tensor([self._normalised["scene_scale"]], dtype=torch.float32),
            "apos": self._normalised["apos"].unsqueeze(0),
        }
        if device is not None:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
            }
        return batch


def segment_plan(num_frames, n_a, n_b, n_target_a, n_target_b, a_span=0.30, gap=0.0):
    """Lay out A, B and their held-out targets along the trajectory.

    A covers the first `a_span` of the sequence, B the remainder after `gap`.
    Targets are drawn from frames not used as input, inside the matching span,
    so that "A's region" and "B's region" are geometrically distinct.
    """
    a_end = int(num_frames * a_span)
    b_start = int(a_end + num_frames * gap)
    a_pool = list(range(0, a_end))
    b_pool = list(range(b_start, num_frames))
    if len(a_pool) < n_a + n_target_a or len(b_pool) < n_b + n_target_b:
        raise ValueError(f"scene too short: {num_frames} frames")

    a_input = [a_pool[i] for i in np.linspace(0, len(a_pool) - 1, n_a).round().astype(int)]
    b_input = [b_pool[i] for i in np.linspace(0, len(b_pool) - 1, n_b).round().astype(int)]

    def held_out(pool, used, n):
        free = [f for f in pool if f not in set(used)]
        picks = np.linspace(0, len(free) - 1, n).round().astype(int)
        return [free[i] for i in dict.fromkeys(picks)]

    return {
        "a_input": a_input,
        "b_input": b_input,
        "a_target": held_out(a_pool, a_input, n_target_a),
        "b_target": held_out(b_pool, b_input, n_target_b),
    }
