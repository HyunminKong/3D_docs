"""Descriptor search, redone within scenes instead of across them.

The first attempt gave one interference number per scene, from one arbitrary
A/B cut.  Across scenes that number turned out to be dominated by how well the
scene reconstructs at all (rho = +0.83 against the A-only PSNR), and once that
was partialled out no geometric property survived correction.  The variation a
regime descriptor has to key on is not between scenes -- it is between different
stretches of the same trajectory, where content and appearance are held fixed
and only the geometry of the two segments changes.

So each scene contributes several cuts, varying where A sits, how long it is and
how far B starts after it.  Analysis is then within-scene: every variable is
centred on its own scene's mean before correlating, which removes scene identity
exactly rather than approximately.
"""

import argparse
import glob
import importlib
import itertools
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
from oracle.scene_profile import geometry  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dl3dv_ar.yaml")
    parser.add_argument("--ckpt", default="checkpoints/dl3dv_ar.pt")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--n-a", type=int, default=8)
    parser.add_argument("--n-b", type=int, default=8)
    parser.add_argument("--n-target-a", type=int, default=4)
    parser.add_argument("--n-target-b", type=int, default=4)
    parser.add_argument("--a-spans", default="0.15,0.25,0.35,0.45")
    parser.add_argument("--gaps", default="0.0,0.15")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="oracle/results/multisplit.json")
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

    from oracle.oracle_data import SceneViews, segment_plan

    scenes = sorted(glob.glob(os.path.join(args.processed_dir, "*", "opencv_cameras.json")))
    if args.limit:
        scenes = scenes[: args.limit]
    spans = [float(s) for s in args.a_spans.split(",")]
    gaps = [float(g) for g in args.gaps.split(",")]
    print(f"{len(scenes)} scenes x {len(spans) * len(gaps)} splits")

    rows = []
    if os.path.exists(args.out):
        rows = json.load(open(args.out))
    done = {(r["scene"], r["a_span"], r["gap"]) for r in rows}

    for path in scenes:
        name = os.path.basename(os.path.dirname(path))
        # one instance per scene: it caches decoded frames, and the splits
        # overlap heavily, so rebuilding it per split would reload the same
        # images eight times
        scene = SceneViews(path, config.model.image_size, config.model.image_size_x)
        for a_span, gap in itertools.product(spans, gaps):
            if (name, a_span, gap) in done:
                continue
            try:
                plan = segment_plan(
                    len(scene), args.n_a, args.n_b, args.n_target_a, args.n_target_b,
                    a_span=a_span, gap=gap,
                )
                norm = scene.normalise(sorted(set(sum(plan.values(), []))))
                a, b, ta = plan["a_input"], plan["b_input"], plan["a_target"]

                def run(batch, n_in):
                    config.training.num_input_views = n_in
                    config.training.num_virtual_views = len(a)
                    config.training.num_target_views = len(ta)
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(batch)
                    psnr = per_view_psnr(out.render["render"], out.target["image"])
                    del out
                    torch.cuda.empty_cache()
                    return summarise(psnr)["mean"]

                psnr_i = run(scene.batch(a, a, ta, device=device), len(a))
                psnr_ii = run(scene.batch(a + b, a, ta, device=device), len(a + b))

                c2w = norm["c2w"]
                row = {
                    "scene": name, "a_span": a_span, "gap": gap,
                    "psnr_a_only": psnr_i, "psnr_a_plus_b": psnr_ii,
                    "interference": psnr_i - psnr_ii,
                }
                row.update(geometry(
                    np.stack([c2w[v].numpy() for v in a]),
                    np.stack([c2w[v].numpy() for v in b]),
                ))
                rows.append(row)
                print(f"{name[:12]} span={a_span} gap={gap}  {row['interference']:+.3f} dB")
            except Exception as exc:
                print(f"{name[:12]} span={a_span} gap={gap}  SKIP ({type(exc).__name__})")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}  ({len(rows)} observations)")


if __name__ == "__main__":
    main()
