"""Pick scenes that fill all four cells of place x regime.

On DL3DV every scene was a different building, so "same scene" and "same kind of
motion" moved together and the pairwise penalty could not say which one it was
responding to. Driving data separates them: the car passes through the same few
neighbourhoods repeatedly, so two runs can share a street while doing entirely
different things, or perform the same manoeuvre a kilometre apart.

                     same place        different place
    same regime           A                   B
    different regime      C                   D

B and C are the cells that decide it. If borrowed adaptations transfer in B,
what makes two adaptations compatible is how the camera moved; if they transfer
in C, it is what the camera was looking at, and the project's separation from
content memory does not hold.

Both axes are measured rather than inferred. Position comes from the ego pose in
each location's shared map frame, so scenes in one neighbourhood are directly
comparable. Motion comes from the CAN bus -- speed, yaw rate, steering spread,
time stopped -- which is instrumentation, not something recovered from the
images being tested.

Scenes are chosen to spread over both axes rather than to be representative:
the pairwise design needs the corners populated, and a random draw would put
almost every pair in D.
"""

import argparse
import json
import os

import numpy as np


def frame_utime(path):
    return int(os.path.splitext(os.path.basename(path))[0].rsplit("__", 1)[-1])


def segment_indices(n, a_span=0.30, n_a=8):
    a_end = int(n * a_span)
    return [int(v) for v in np.linspace(0, a_end - 1, n_a).round()], a_end


def regime_vector(pose, steer, t0, t1):
    p = [r for r in pose if t0 <= r["utime"] <= t1]
    s = [r["value"] for r in steer if t0 <= r["utime"] <= t1]
    if len(p) < 5:
        return None
    speed = np.array([abs(r["vel"][0]) for r in p])
    yaw = np.array([abs(r["rotation_rate"][2]) for r in p])
    sv = np.array(s) if s else np.array([0.0])
    return np.array([speed.mean(), speed.std(), yaw.mean(),
                     sv.std(), float((speed < 0.5).mean())])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dirs", nargs="+", required=True)
    parser.add_argument("--canbus", default="/mnt/ssd/nuscenes/can_bus")
    parser.add_argument("--root", default="/mnt/ssd/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--n-units", type=int, default=24)
    parser.add_argument(
        "--near-m", type=float, default=30.0,
        help="metres below which two segments count as the same place; a quarter of "
             "the same-location distances sits above 300 m, which is a different "
             "street entirely, so this is set from what the cameras can actually share",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = os.path.join(args.root, args.version)
    logs = {l["token"]: l["location"] for l in json.load(open(f"{base}/log.json"))}
    location = {s["name"]: logs[s["log_token"]] for s in json.load(open(f"{base}/scene.json"))}

    units = []
    for d in args.processed_dirs:
        for name in sorted(os.listdir(d)):
            cam = os.path.join(d, name, "opencv_cameras.json")
            pose_f = os.path.join(args.canbus, f"{name}_pose.json")
            if not (os.path.exists(cam) and os.path.exists(pose_f)):
                continue
            frames = json.load(open(cam))["frames"]
            a_idx, _ = segment_indices(len(frames))
            times = [frame_utime(f["file_path"]) for f in frames]
            steer_f = os.path.join(args.canbus, f"{name}_steeranglefeedback.json")
            reg = regime_vector(json.load(open(pose_f)),
                                json.load(open(steer_f)) if os.path.exists(steer_f) else [],
                                times[a_idx[0]], times[a_idx[-1]])
            if reg is None:
                continue
            # ego positions of the A segment, in the location's map frame
            pos = np.array([np.linalg.inv(np.asarray(frames[i]["w2c"]))[:3, 3] for i in a_idx])
            units.append({"scene": name, "dir": d, "location": location.get(name, "?"),
                          "regime": reg.tolist(), "centre": pos.mean(0).tolist(),
                          "positions": pos.tolist()})

    print(f"{len(units)} candidate units")
    R = np.array([u["regime"] for u in units])
    R = (R - R.mean(0)) / (R.std(0) + 1e-9)

    def place_distance(i, j):
        """Metres between the two A segments; infinite across locations."""
        if units[i]["location"] != units[j]["location"]:
            return np.inf
        pi = np.array(units[i]["positions"])
        pj = np.array(units[j]["positions"])
        return float(np.linalg.norm(pi[:, None] - pj[None], axis=-1).min())

    n = len(units)
    place = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if i != j:
                place[i, j] = place_distance(i, j)
    regime = np.linalg.norm(R[:, None] - R[None], axis=-1)

    finite = place[np.isfinite(place)]
    print(f"same-location pairs: {int(np.isfinite(place).sum())} of {n * (n - 1)}")
    if finite.size:
        for q in (5, 25, 50):
            print(f"  {q}th percentile distance: {np.percentile(finite, q):8.1f} m")
    print(f"regime distance: median {np.median(regime[regime > 0]):.2f}")

    # greedy pick: add the unit that most improves coverage of the four corners
    near = args.near_m
    reg_split = np.median(regime[regime > 0])

    def coverage(sel):
        cells = np.zeros(4, dtype=int)
        for a in range(len(sel)):
            for b in range(len(sel)):
                if a == b:
                    continue
                i, j = sel[a], sel[b]
                same_place = place[i, j] < near
                same_reg = regime[i, j] < reg_split
                cells[(0 if same_place else 1) + (0 if same_reg else 2)] += 1
        return cells

    close = np.dstack(np.unravel_index(np.argsort(place, axis=None), place.shape))[0]
    sel = [int(close[0][0]), int(close[0][1])]
    while len(sel) < min(args.n_units, n):
        best, best_score = None, -1
        for c in range(n):
            if c in sel:
                continue
            cells = coverage(sel + [c])
            score = min(cells)  # push up the emptiest corner
            if score > best_score:
                best, best_score = c, score
        sel.append(best)

    cells = coverage(sel)
    print(f"\nselected {len(sel)} units; ordered pairs per cell")
    print(f"  A same place, same regime      {cells[0]:5d}")
    print(f"  B diff place, same regime      {cells[1]:5d}")
    print(f"  C same place, diff regime      {cells[2]:5d}")
    print(f"  D diff place, diff regime      {cells[3]:5d}")

    chosen = [units[i] for i in sel]
    json.dump({"near_m": float(near), "regime_split": float(reg_split),
               "units": [{k: u[k] for k in ("scene", "dir", "location", "regime", "centre")}
                         for u in chosen]},
              open(args.out, "w"), indent=2)
    with open(args.out + ".txt", "w") as fh:
        fh.write("\n".join(u["scene"] for u in chosen))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
