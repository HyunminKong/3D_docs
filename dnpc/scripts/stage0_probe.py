"""Stage 0 probe: run streaming 3DGS, log per-Gaussian statistics to parquet.

    python scripts/stage0_probe.py --dataset nrgbd --scene staircase --tag main
    python scripts/stage0_probe.py --dataset nrgbd --scene staircase --freeze-age-k 10

The trained algorithm is stock gsplat; this script only adds measurement. Use
``--no-instrument`` to confirm that fact (see scripts/check_null_effect.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from dnpc.data.nrgbd import NRGBDSequence  # noqa: E402
from dnpc.data.tum import TUMSequence  # noqa: E402
from dnpc.probe.geom_error import (  # noqa: E402
    build_gt_cloud, geometric_error, observation_geometry,
)
from dnpc.probe.trainer import ProbeConfig, StreamingProbe  # noqa: E402

NRGBD_ROOT = "/home/khm/3D_4D/FastVGGT/data/neural_rgbd_data"
TUM_ROOT = "/home/khm/3D_4D/data/tum"
OUT = "/home/khm/3D_4D/dnpc/outputs/stage0"


def build_seq(args):
    if args.dataset == "nrgbd":
        return NRGBDSequence(NRGBD_ROOT, args.scene, stride=args.frame_stride,
                             max_frames=args.max_frames)
    return TUMSequence(TUM_ROOT, args.scene, stride=args.frame_stride,
                       max_frames=args.max_frames, seed=args.seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nrgbd", "tum"], required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--replay", type=int, default=2)
    ap.add_argument("--replay-horizon", type=int, default=30)
    ap.add_argument("--scale-mult", type=float, default=2.0)
    ap.add_argument("--iters-per-frame", type=int, default=20)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--vis-stride", type=int, default=2)
    ap.add_argument("--freeze-age-k", type=int, default=0)
    ap.add_argument("--eval-stride", type=int, default=5)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--track-mod", type=int, default=8)
    ap.add_argument("--max-gaussians", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-instrument", action="store_true")
    args = ap.parse_args()

    run = f"{args.dataset}_{args.scene}_{args.tag}"
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    os.makedirs(f"{OUT}/tables", exist_ok=True)

    seq = build_seq(args)
    print(f"[{run}] {len(seq)} frames  scene_scale={seq.scene_scale():.2f} m")

    cfg = ProbeConfig(
        window=args.window, replay=args.replay, replay_horizon=args.replay_horizon,
        scale_mult=args.scale_mult, iters_per_frame=args.iters_per_frame,
        sh_degree=args.sh_degree, vis_stride=args.vis_stride,
        freeze_age_k=args.freeze_age_k, instrument=not args.no_instrument,
        checkpoint_every=args.checkpoint_every, track_mod=args.track_mod,
        max_gaussians=args.max_gaussians,
        seed=args.seed,
    )
    probe = StreamingProbe(seq, cfg)

    gt_eval = None
    if cfg.checkpoint_every and cfg.instrument:
        print("[gt] building GT cloud for checkpoint scoring ...")
        P, normals, tree = build_gt_cloud(seq)
        print(f"     {len(P):,} points")
        gt_eval = lambda m, r: geometric_error(m, r, P, normals, tree)  # noqa: E731

    t0 = time.time()
    probe.run(gt_eval=gt_eval)
    train_wall = time.time() - t0
    print(f"[{run}] streaming done in {train_wall/60:.1f} min, N={probe.N}")

    metrics, e_render, e_depth, eval_contrib = probe.evaluate(eval_stride=args.eval_stride)
    metrics.update(run=run, dataset=args.dataset, scene=args.scene, tag=args.tag,
                   n_frames=len(seq), train_wall_s=train_wall,
                   freeze_age_k=args.freeze_age_k, scene_scale=seq.scene_scale())
    print("[metrics] " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in metrics.items()))

    with open(f"{OUT}/tables/{run}_metrics.json", "w") as f:
        json.dump({"metrics": metrics, "config": cfg.to_dict(), "args": vars(args)}, f, indent=2)

    if not cfg.instrument:
        print("[skip] instrumentation disabled; no per-Gaussian log written")
        return

    print("[log] observation geometry ...")
    cam_cols = probe.cache.cam_centers[:: cfg.vis_stride]
    obs = observation_geometry(
        probe.state["vis"], probe.params["means"].detach(),
        probe.state["ray_first"], cam_cols,
    )

    print("[log] GT cloud + geometric error ...")
    if gt_eval is None:
        P, normals, tree = build_gt_cloud(seq)
        print(f"       GT cloud: {len(P):,} points")
    means = probe.params["means"].detach().cpu().numpy()
    ray = probe.state["ray_first"].cpu().numpy()
    ge = geometric_error(means, ray, P, normals, tree)

    df = pd.DataFrame({
        "uid": probe.state["uid"].cpu().numpy(),
        "gid": probe.state["gid"].cpu().numpy(),
        "dens_count": probe.state["dens_count"].cpu().numpy(),
        "birth_frame": probe.state["birth_frame"].cpu().numpy(),
        "z_first": probe.state["z_first"].cpu().numpy(),
        "err_init": probe.state["err_init"].cpu().numpy(),
        "contrib": probe.state["contrib"].cpu().numpy(),
        "eval_contrib": eval_contrib,
        "opacity": torch.sigmoid(probe.params["opacities"].detach()).cpu().numpy(),
        "scale_mean": torch.exp(probe.params["scales"].detach()).mean(-1).cpu().numpy(),
        "e_render": e_render,
        "e_depth": e_depth,
        **obs,
        **ge,
    })
    # observation counts live on the subsampled probe grid; record the stride so
    # n_obs is interpretable as "distinct probe frames", not raw frames.
    df["vis_stride"] = cfg.vis_stride
    df["run"] = run

    path = f"{OUT}/logs/{run}_gaussian_stats.parquet"
    df.to_parquet(path, index=False)
    print(f"[log] wrote {path}  ({len(df):,} rows)")

    q = df[(df.n_obs > 1) & df.gt_valid]
    print(f"\n[summary] {len(q):,} usable rows ({len(q)/len(df)*100:.0f}%)")
    for c in ["z_first", "z_mean", "n_obs", "B_acc", "B_perp", "alpha_max",
              "err_p2pl", "err_radial", "err_lateral", "e_render", "e_depth"]:
        v = q[c].values
        print(f"  {c:12s} p10={np.percentile(v,10):9.4f} p50={np.percentile(v,50):9.4f} "
              f"p90={np.percentile(v,90):9.4f}")
    lb, lz = np.log(np.clip(q.B_perp, 1e-6, None)), np.log(np.clip(q.z_mean, 1e-6, None))
    print(f"  corr(log B_perp, log z_mean) = {np.corrcoef(lb, lz)[0,1]:+.3f}   "
          f"sd(log z)={lz.std():.3f}  sd(log B_perp)={lb.std():.3f}")
    if probe.checkpoints:
        ck = pd.concat(probe.checkpoints, ignore_index=True)
        ck["run"] = run
        cpath = f"{OUT}/logs/{run}_checkpoints.parquet"
        ck.to_parquet(cpath, index=False)
        print(f"[log] wrote {cpath}  ({len(ck):,} rows, "
              f"{ck.gid.nunique():,} lineages x {ck.frame.nunique()} checkpoints)")

    print(f"  B_perp / B_acc median = {np.median(q.B_perp/np.clip(q.B_acc,1e-6,None)):.3f}")


if __name__ == "__main__":
    main()
