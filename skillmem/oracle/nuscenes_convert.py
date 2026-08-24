"""Turn nuScenes driving sequences into the format the oracle harness reads.

The DL3DV experiments answered what they could, but every unit in them was a
different place, so scene identity and motion regime were inseparable. Driving
data breaks that: the same manoeuvre -- pulling away from a stop, turning
through a junction, following a straight -- recurs across unrelated locations.
That is the structure the regime hypothesis needs and DL3DV does not have.

One camera is used, at full rate. Keyframes alone give 2 Hz over a 20 s scene,
about forty frames, too few to cut into segments; including the sweeps brings
CAM_FRONT to roughly 270 frames per scene, comparable to a DL3DV capture.

Poses compose as ego_pose x calibrated_sensor, both of which nuScenes stores per
sample_data record. The harness renormalises onto the first camera anyway, so
the global map frame these arrive in does not matter.

The metadata is awkward only in size: sample_data.json is 1.3 GB and
ego_pose.json 646 MB, so both are streamed once and indexed rather than loaded
whole per scene.
"""

import argparse
import json
import os

import numpy as np


def quaternion_to_matrix(q):
    """nuScenes stores rotations as [w, x, y, z]."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pose_matrix(record):
    m = np.eye(4)
    m[:3, :3] = quaternion_to_matrix(record["rotation"])
    m[:3, 3] = record["translation"]
    return m


def load_tables(root, version):
    base = os.path.join(root, version)
    print("reading metadata (sample_data.json is ~1.3 GB, this takes a moment)")
    scenes = json.load(open(os.path.join(base, "scene.json")))
    samples = json.load(open(os.path.join(base, "sample.json")))
    sensors = {s["token"]: s for s in json.load(open(os.path.join(base, "sensor.json")))}
    calib = {c["token"]: c for c in json.load(open(os.path.join(base, "calibrated_sensor.json")))}
    ego = {e["token"]: e for e in json.load(open(os.path.join(base, "ego_pose.json")))}
    sample_data = json.load(open(os.path.join(base, "sample_data.json")))
    return scenes, samples, sensors, calib, ego, sample_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/ssd/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--channel", default="CAM_FRONT")
    parser.add_argument("--n-scenes", type=int, default=12)
    parser.add_argument(
        "--scene-list", default="",
        help="file of scene names, one per line; overrides the first-N-by-name default",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-frames", type=int, default=200)
    args = parser.parse_args()

    scenes, samples, sensors, calib, ego, sample_data = load_tables(args.root, args.version)

    cam_calib = {t: c for t, c in calib.items()
                 if sensors[c["sensor_token"]]["channel"] == args.channel}
    sample_to_scene = {s["token"]: s["scene_token"] for s in samples}

    # sample_data holds every sensor at every timestamp; keep this channel only
    by_scene = {}
    for rec in sample_data:
        if rec["calibrated_sensor_token"] not in cam_calib:
            continue
        scene_token = sample_to_scene.get(rec["sample_token"])
        if scene_token is None:
            continue
        by_scene.setdefault(scene_token, []).append(rec)
    print(f"{args.channel}: {sum(len(v) for v in by_scene.values())} frames "
          f"over {len(by_scene)} scenes")

    ordered = sorted(scenes, key=lambda s: s["name"])
    if args.scene_list:
        wanted = {ln.strip() for ln in open(args.scene_list) if ln.strip()}
        ordered = [s for s in ordered if s["name"] in wanted]
        args.n_scenes = len(ordered)
        print(f"scene list: {len(ordered)} of {len(wanted)} names matched")
    written = 0
    for scene in ordered:
        records = by_scene.get(scene["token"], [])
        if len(records) < args.min_frames:
            continue
        records.sort(key=lambda r: r["timestamp"])

        frames = []
        for rec in records:
            cs = calib[rec["calibrated_sensor_token"]]
            c2w = pose_matrix(ego[rec["ego_pose_token"]]) @ pose_matrix(cs)
            k = np.asarray(cs["camera_intrinsic"], dtype=float)
            frames.append({
                "w": rec["width"], "h": rec["height"],
                "fx": k[0, 0], "fy": k[1, 1], "cx": k[0, 2], "cy": k[1, 2],
                "w2c": np.linalg.inv(c2w).tolist(),
                "file_path": os.path.join(args.root, rec["filename"]),
            })

        out_dir = os.path.join(args.out, scene["name"])
        os.makedirs(out_dir, exist_ok=True)
        json.dump({"scene_name": scene["name"], "frames": frames},
                  open(os.path.join(out_dir, "opencv_cameras.json"), "w"))
        span = np.linalg.norm(
            np.array(frames[-1]["w2c"])[:3, 3] - np.array(frames[0]["w2c"])[:3, 3]
        )
        print(f"{scene['name']}  {len(frames):4d} frames  ego travel {span:6.1f} m")
        written += 1
        if written >= args.n_scenes:
            break

    print(f"\nwrote {written} scenes to {args.out}")


if __name__ == "__main__":
    main()
