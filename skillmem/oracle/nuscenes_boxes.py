"""Attach dynamic-object boxes to converted nuScenes scenes.

The interference measurement rests on a claim that only holds for a rigid world:
segment B adds observations, so if A's region gets worse, the fast weight was
overwritten. On a street that claim leaks. Cars and pedestrians occupy different
places at A's timestamps than at B's, and a model that has seen both will smear
them; the resulting loss is a genuine conflict, but it is a conflict about
content, not about capacity, and it does not belong in the number.

So the score is restricted to static background. nuScenes annotates objects at
keyframes only -- 2 Hz against the 12 Hz the sweeps provide -- which sounds like
a problem and is not: the frames that get *scored* are the ones that need boxes,
and every input frame can still be a sweep. Targets are therefore drawn from
keyframes, and this pass records, for each of them, where the movable objects
land in the image.

Boxes are stored as axis-aligned rectangles in the original 1600x900 frame. The
harness resizes by a fixed factor with no crop, so a rectangle rescales exactly;
and since the projection of a 3D box is a hull that an axis-aligned rectangle
over-covers, the stored region already errs toward masking too much, which is
the safe direction before any dilation is added on top.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

# what can move between one pass and the next
MOVABLE_PREFIXES = ("vehicle.", "human.", "animal", "movable_object.")


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def box_corners(ann):
    """Eight corners of an annotation, in the global frame."""
    w, l, h = ann["size"]                     # nuScenes order: width, length, height
    x = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack([x, y, z])
    corners = quat_to_matrix(ann["rotation"]) @ corners
    return corners + np.asarray(ann["translation"]).reshape(3, 1)


def project(corners_global, w2c, intr):
    """Image-space axis-aligned box, or None if the object is not in front."""
    cam = w2c[:3, :3] @ corners_global + w2c[:3, 3:4]
    in_front = cam[2] > 0.1
    if in_front.sum() < 4:
        return None
    cam = cam[:, in_front]
    uv = intr @ cam
    uv = uv[:2] / uv[2]
    return [float(uv[0].min()), float(uv[1].min()), float(uv[0].max()), float(uv[1].max())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/ssd/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--channel", default="CAM_FRONT")
    args = parser.parse_args()

    base = os.path.join(args.root, args.version)
    print("reading annotations")
    categories = {c["token"]: c["name"] for c in json.load(open(f"{base}/category.json"))}
    instances = {i["token"]: categories[i["category_token"]]
                 for i in json.load(open(f"{base}/instance.json"))}
    anns_by_sample = defaultdict(list)
    for ann in json.load(open(f"{base}/sample_annotation.json")):
        name = instances[ann["instance_token"]]
        if name.startswith(MOVABLE_PREFIXES):
            anns_by_sample[ann["sample_token"]].append(ann)

    sensors = {s["token"]: s for s in json.load(open(f"{base}/sensor.json"))}
    calib = {c["token"]: c for c in json.load(open(f"{base}/calibrated_sensor.json"))}
    cam_calib = {t for t, c in calib.items()
                 if sensors[c["sensor_token"]]["channel"] == args.channel}
    # filename -> (sample_token, is_key_frame), for the one channel
    meta = {}
    for rec in json.load(open(f"{base}/sample_data.json")):
        if rec["calibrated_sensor_token"] in cam_calib:
            meta[os.path.basename(rec["filename"])] = (rec["sample_token"], rec["is_key_frame"])

    scenes = sorted(os.listdir(args.processed_dir))
    for name in scenes:
        path = os.path.join(args.processed_dir, name, "opencv_cameras.json")
        if not os.path.exists(path):
            continue
        data = json.load(open(path))
        n_key, n_box, covered = 0, 0, []
        for frame in data["frames"]:
            key = os.path.basename(frame["file_path"])
            sample_token, is_key = meta.get(key, (None, False))
            frame["is_keyframe"] = bool(is_key)
            if not is_key:
                frame["boxes"] = []
                continue
            n_key += 1
            w2c = np.asarray(frame["w2c"])
            intr = np.array([[frame["fx"], 0, frame["cx"]],
                             [0, frame["fy"], frame["cy"]],
                             [0, 0, 1]])
            boxes = []
            for ann in anns_by_sample.get(sample_token, []):
                rect = project(box_corners(ann), w2c, intr)
                if rect is None:
                    continue
                if rect[2] < 0 or rect[0] > frame["w"] or rect[3] < 0 or rect[1] > frame["h"]:
                    continue
                boxes.append([round(v, 1) for v in rect])
            frame["boxes"] = boxes
            n_box += len(boxes)
            area = np.zeros((frame["h"], frame["w"]), dtype=bool)
            for x0, y0, x1, y1 in boxes:
                area[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = True
            covered.append(area.mean())

        json.dump(data, open(path, "w"))
        cov = float(np.mean(covered)) if covered else 0.0
        print(f"{name}  keyframes {n_key:3d}  boxes {n_box:5d}  "
              f"mean image covered {cov * 100:5.1f}%")


if __name__ == "__main__":
    main()
