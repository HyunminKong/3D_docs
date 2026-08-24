"""Does how the car was driving predict how much the fast weight lost?

Two scenes lost twelve and thirteen decibels with barely a moving object in
frame, so whatever drove that was not other traffic. The obvious candidate is
the ego motion itself: a stretch spent stationary followed by a fast pull-away,
or a straight followed by a turn, presents the model with two segments whose
viewing geometry has almost nothing in common.

nuScenes measures that directly rather than leaving it to be inferred from
recovered poses. The CAN bus carries velocity and yaw rate at 50 Hz and steering
angle at about 95 Hz, so speed, turning and time spent stopped are instrument
readings here, not estimates -- which is a real gain over DL3DV, where every
descriptor had to be derived from the trajectory it was meant to explain.

Each scene's A and B spans are summarised separately and then contrasted, since
the hypothesis is about the *difference* between the two regimes. Correlations
partial out the A-only reconstruction quality, which tracks interference at
rho ~ 0.64 for reasons that have nothing to do with driving: a scene that
reconstructs well simply has more to lose.
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats


def frame_utime(path):
    """nuScenes encodes the capture time in the filename's last field."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem.rsplit("__", 1)[-1])


def window_stats(pose, steer, t0, t1):
    """Driving summary over one time span."""
    p = [r for r in pose if t0 <= r["utime"] <= t1]
    s = [r["value"] for r in steer if t0 <= r["utime"] <= t1]
    if len(p) < 5:
        return None
    speed = np.array([abs(r["vel"][0]) for r in p])
    yaw = np.array([abs(r["rotation_rate"][2]) for r in p])
    steer_v = np.array(s) if s else np.array([0.0])
    return {
        "speed_mean": float(speed.mean()),
        "speed_sd": float(speed.std()),
        "speed_max": float(speed.max()),
        "yaw_mean": float(yaw.mean()),
        "yaw_max": float(yaw.max()),
        "steer_sd": float(steer_v.std()),
        "steer_range": float(steer_v.max() - steer_v.min()),
        "stopped_frac": float((speed < 0.5).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canbus", default="/mnt/ssd/nuscenes/can_bus")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--results", default="oracle/results/nusc_masked.json")
    parser.add_argument("--a-span", type=float, default=0.30)
    parser.add_argument("--out", default="oracle/results/canbus_regime.json")
    args = parser.parse_args()

    measured = {r["scene"]: r for r in json.load(open(args.results))}
    rows = []
    for name, meas in sorted(measured.items()):
        cam = os.path.join(args.processed_dir, name, "opencv_cameras.json")
        pose_f = os.path.join(args.canbus, f"{name}_pose.json")
        steer_f = os.path.join(args.canbus, f"{name}_steeranglefeedback.json")
        if not (os.path.exists(cam) and os.path.exists(pose_f)):
            continue
        frames = json.load(open(cam))["frames"]
        times = [frame_utime(f["file_path"]) for f in frames]
        pose = json.load(open(pose_f))
        steer = json.load(open(steer_f)) if os.path.exists(steer_f) else []

        # same split the interference was measured under
        a_end = int(len(frames) * args.a_span)
        a = window_stats(pose, steer, times[0], times[a_end - 1])
        b = window_stats(pose, steer, times[a_end], times[-1])
        if a is None or b is None:
            continue

        row = {"scene": name, "interference": meas["interf_d0"], "a_psnr": meas["a_d0"]}
        row.update({f"a_{k}": v for k, v in a.items()})
        row.update({f"b_{k}": v for k, v in b.items()})
        # the hypothesis is about the contrast between the two regimes
        for k in a:
            row[f"d_{k}"] = abs(b[k] - a[k])
        row["speed_ratio"] = b["speed_mean"] / max(a["speed_mean"], 0.1)
        rows.append(row)

    print(f"n = {len(rows)} scenes\n")
    y = np.array([r["interference"] for r in rows])
    z = np.array([r["a_psnr"] for r in rows])

    def partial(x):
        rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
        res = lambda v: v - (stats.linregress(rz, v)[0] * rz + stats.linregress(rz, v)[1])  # noqa: E731
        return stats.spearmanr(res(rx), res(ry))

    keys = [k for k in rows[0] if k not in ("scene", "interference", "a_psnr")]
    out = []
    for k in keys:
        x = np.array([r[k] for r in rows], float)
        if np.allclose(x, x[0]):
            continue
        raw = stats.spearmanr(x, y)
        par = partial(x)
        out.append((k, raw.statistic, par.statistic, par.pvalue))
    out.sort(key=lambda t: -abs(t[2]))

    print(f"{'driving statistic':>22} {'raw rho':>9} {'partial':>9} {'p':>8}")
    for k, raw, par, p in out:
        mark = " *" if p < 0.05 else ""
        print(f"{k:>22} {raw:+9.3f} {par:+9.3f} {p:8.4f}{mark}")

    print("\nthe two scenes that collapsed without traffic:")
    for r in sorted(rows, key=lambda r: -r["interference"])[:3]:
        print(f"  {r['scene']}  interference {r['interference']:6.2f} dB   "
              f"speed A {r['a_speed_mean']:5.2f} -> B {r['b_speed_mean']:5.2f} m/s   "
              f"stopped A {r['a_stopped_frac'] * 100:4.0f}% -> B {r['b_stopped_frac'] * 100:4.0f}%   "
              f"yaw max A {r['a_yaw_max']:.3f} -> B {r['b_yaw_max']:.3f}")
    print("\nthe three that barely moved:")
    for r in sorted(rows, key=lambda r: r["interference"])[:3]:
        print(f"  {r['scene']}  interference {r['interference']:6.2f} dB   "
              f"speed A {r['a_speed_mean']:5.2f} -> B {r['b_speed_mean']:5.2f} m/s   "
              f"stopped A {r['a_stopped_frac'] * 100:4.0f}% -> B {r['b_stopped_frac'] * 100:4.0f}%   "
              f"yaw max A {r['a_yaw_max']:.3f} -> B {r['b_yaw_max']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
