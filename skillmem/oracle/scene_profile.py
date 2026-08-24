"""What distinguishes a scene that forgets a lot from one that forgets little?

E6 found interference ranging from 1.1 to 5.9 dB across scenes under an
identical protocol.  That spread is the only empirical handle on what a regime
descriptor should be built from: a descriptor is only worth designing if some
measurable property of the segment pair predicts how much adaptation gets
overwritten.

Per scene this records, under the same A/B plan E6 used:

* the interference itself (A-only minus A+B, targets fixed on A),
* geometry of the two segments, all gauge-invariant -- arc length, baseline,
  rotation traversed, how far B departs from A, how much the two segments look
  at each other -- computed in the shared normalised frame so scenes are
  comparable, and
* what the fast weight actually did: ||dW|| for each segment, the size of the
  difference, and how spread out its spectrum is.

The descriptor the plan calls for has to be computable at stream time from the
current chunk and the previous state, so nothing here uses absolute pose or
anything from the future.
"""

import argparse
import glob
import importlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.run_oracle import (  # noqa: E402
    bootstrap_distributed,
    build_config,
    per_view_psnr,
    summarise,
)


def rotation_angle(r_a, r_b):
    """Geodesic angle in degrees between two rotation matrices."""
    cos = (np.trace(r_a.T @ r_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def geometry(c2w_a, c2w_b):
    """Segment geometry in the shared normalised frame."""
    ca, cb = c2w_a[:, :3, 3], c2w_b[:, :3, 3]
    fa, fb = c2w_a[:, :3, 2], c2w_b[:, :3, 2]

    def arc(c):
        return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())

    def spread(c):
        d = np.linalg.norm(c[:, None] - c[None], axis=-1)
        return float(d.max())

    def mean_baseline(c):
        d = np.linalg.norm(c[:, None] - c[None], axis=-1)
        iu = np.triu_indices(len(c), 1)
        return float(d[iu].mean())

    def total_rotation(r):
        return float(sum(rotation_angle(r[i], r[i + 1]) for i in range(len(r) - 1)))

    cross = np.linalg.norm(ca[:, None] - cb[None], axis=-1)
    cos_ab = fa @ fb.T
    fwd_a, fwd_b = fa.mean(0), fb.mean(0)
    fwd_a /= np.linalg.norm(fwd_a)
    fwd_b /= np.linalg.norm(fwd_b)

    return {
        "arc_a": arc(ca),
        "arc_b": arc(cb),
        "arc_ratio": arc(cb) / max(arc(ca), 1e-6),
        "spread_a": spread(ca),
        "spread_b": spread(cb),
        "baseline_a": mean_baseline(ca),
        "baseline_b": mean_baseline(cb),
        "rot_a_deg": total_rotation(c2w_a[:, :3, :3]),
        "rot_b_deg": total_rotation(c2w_b[:, :3, :3]),
        # how far the second segment departs from the first
        "ab_centroid_dist": float(np.linalg.norm(ca.mean(0) - cb.mean(0))),
        "ab_min_dist": float(cross.min()),
        "ab_mean_dist": float(cross.mean()),
        "b_closest_to_a": float(cross.min(axis=1).mean()),
        # how much the two segments look the same way (covisibility proxy)
        "ab_view_angle_deg": float(np.degrees(np.arccos(np.clip(fwd_a @ fwd_b, -1, 1)))),
        "ab_mean_pair_angle_deg": float(np.degrees(np.arccos(np.clip(cos_ab, -1, 1))).mean()),
        "ab_min_pair_angle_deg": float(np.degrees(np.arccos(np.clip(cos_ab, -1, 1))).min()),
        # parallax proxy: how large is the second segment's motion relative to
        # the extent the first segment already covered
        "parallax_ratio": arc(cb) / max(spread(ca), 1e-6),
    }


def weight_stats(w_a, w_ab, w_init, ranks=(8, 32, 64)):
    from oracle.fastweight_hooks import W_KEYS

    da, dab, dd, energies, stable = [], [], [], {r: [] for r in ranks}, []
    for a, ab, i0 in zip(w_a, w_ab, w_init):
        for k in W_KEYS:
            base = i0[k].float()
            va, vab = a[k].float() - base, ab[k].float() - base
            diff = (a[k].float() - ab[k].float()).squeeze(0)
            da.append((torch.linalg.norm(va) / torch.linalg.norm(base)).item())
            dab.append((torch.linalg.norm(vab) / torch.linalg.norm(base)).item())
            dd.append((torch.linalg.norm(diff) / torch.linalg.norm(base)).item())
            s = torch.linalg.svdvals(diff)
            energy = (s**2).cumsum(0) / (s**2).sum()
            for r in ranks:
                energies[r].append(energy[min(r, len(s)) - 1].item())
            stable.append(((s**2).sum() / s[0] ** 2).item())
    out = {
        "rel_dW_a": float(np.mean(da)),
        "rel_dW_ab": float(np.mean(dab)),
        "rel_diff": float(np.mean(dd)),
        "diff_stable_rank": float(np.mean(stable)),
    }
    out.update({f"diff_energy_r{r}": float(np.mean(v)) for r, v in energies.items()})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dl3dv_ar.yaml")
    parser.add_argument("--ckpt", default="checkpoints/dl3dv_ar.pt")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--n-a", type=int, default=8)
    parser.add_argument("--n-b", type=int, default=8)
    parser.add_argument("--n-target-a", type=int, default=4)
    parser.add_argument("--n-target-b", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-weights", action="store_true", help="geometry only, no SVD")
    parser.add_argument("--out", default="oracle/results/profiles.json")
    args = parser.parse_args()

    bootstrap_distributed()
    from utils import sp_support

    sp_support.init_sp_group(sp_size=1)
    device = "cuda:0"
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    config = build_config(
        args.config,
        {
            "evaluation": True, "inference": False, "sp_size": 1,
            "model.use_anything": False, "model.act_ckpt": False,
            "model.gaussians.random_ratio": 0.0,
            "model.gaussians.usage_threshold": 0.001,
            "training.torch_compile": False, "training.target_has_input": False,
            "training.perceptual_loss_weight": 0.0, "training.depth_loss_weight": 0.0,
            "training.sample_ar": False, "training.batch_size_per_gpu": 1,
            "training.frame_method": "first_cam",
        },
    )
    module, class_name = config.model.class_name.rsplit(".", 1)
    model = importlib.import_module(module).__dict__[class_name](config).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"], strict=False)
    model.eval()

    from oracle.fastweight_hooks import FastWeightController
    from oracle.oracle_data import SceneViews, segment_plan

    ctl = FastWeightController(model)
    w_init = ctl.initial()

    scenes = sorted(glob.glob(os.path.join(args.processed_dir, "*", "opencv_cameras.json")))
    if args.limit:
        scenes = scenes[: args.limit]
    print(f"profiling {len(scenes)} scenes")

    existing = {}
    if os.path.exists(args.out):
        existing = {row["scene"]: row for row in json.load(open(args.out))}

    rows = []
    for path in scenes:
        name = os.path.basename(os.path.dirname(path))
        if name in existing:
            rows.append(existing[name])
            print(f"{name[:12]}  cached")
            continue
        try:
            scene = SceneViews(path, config.model.image_size, config.model.image_size_x)
            plan = segment_plan(
                len(scene), args.n_a, args.n_b, args.n_target_a, args.n_target_b
            )
            norm = scene.normalise(sorted(set(sum(plan.values(), []))))
            a, b, ta = plan["a_input"], plan["b_input"], plan["a_target"]

            def run(batch, n_in, n_virt, capture):
                config.training.num_input_views = n_in
                config.training.num_virtual_views = n_virt
                config.training.num_target_views = len(ta)
                ctx = ctl.capturing() if capture else ctl.injecting(None, apply_only=False)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16), ctx:
                    out = model(batch)
                psnr = per_view_psnr(out.render["render"], out.target["image"])
                snap = ctl.snapshot() if capture else None
                del out
                torch.cuda.empty_cache()
                return psnr, snap

            psnr_i, w_a = run(scene.batch(a, a, ta, device=device), len(a), len(a), True)
            psnr_ii, w_ab = run(scene.batch(a + b, a, ta, device=device), len(a + b), len(a), True)

            c2w = norm["c2w"]
            row = {
                "scene": name,
                "n_frames": len(scene),
                "psnr_a_only": summarise(psnr_i)["mean"],
                "psnr_a_plus_b": summarise(psnr_ii)["mean"],
                "interference": summarise(psnr_i)["mean"] - summarise(psnr_ii)["mean"],
            }
            row.update(geometry(
                np.stack([c2w[v].numpy() for v in a]),
                np.stack([c2w[v].numpy() for v in b]),
            ))
            if not args.skip_weights:
                row.update(weight_stats(w_a, w_ab, w_init))
            rows.append(row)
            print(f"{name[:12]}  interference {row['interference']:+.3f} dB")
            json.dump(rows, open(args.out, "w"), indent=2)
        except Exception as exc:  # keep going; a short or vertical scene is not fatal
            print(f"{name[:12]}  SKIP ({type(exc).__name__}: {exc})")

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}  ({len(rows)} scenes)")


if __name__ == "__main__":
    main()
