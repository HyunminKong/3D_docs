"""E1 on nuScenes, scored on static background only.

Without masking the measured interference is a sum of two things: the fast
weight being overwritten, and the street simply not being the same street. Only
the first belongs in the number. Masking the annotated movable objects out of
the scored region separates them -- provided the mask is doing what it claims,
which is checked three ways rather than assumed.

Dilation is swept rather than fixed. A projected 3D box has a soft boundary, and
what moves leaves shadow and motion blur just outside it, so some growth is
needed; but growth that keeps buying reduction is growth that has started eating
static background. The signature to look for is a plateau. If the curve never
flattens, the mask is consuming signal and no single value of it is defensible.

Two further checks come out of the same run. The fraction of pixels left tells
which scenes have been masked into unreliability -- below about a fifth
remaining, PSNR over the remainder is too unstable to compare. And scenes that
lost a lot of quality when B arrived should lose a lot of *interference* when
the movers are removed, while scenes that lost little should barely move; that
contrast is stronger evidence than a null result on a static scene, because it
shows the mask removing the right thing rather than merely not breaking
anything.

A limit worth stating: boxes are drawn at the target's own timestamp, so a car
that was elsewhere during B and got smeared across otherwise static background
is not covered. That residue inflates what remains, in the safe direction.
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

from oracle.run_oracle import bootstrap_distributed, build_config  # noqa: E402


def build_mask(boxes, src_w, src_h, out_h, out_w, dilate_px):
    """Valid-pixel mask at model resolution: True where nothing movable sits.

    `resize_and_crop` rescales by target/original on each axis with no crop, so a
    rectangle carries over by the same two factors. Dilation is applied after
    scaling, in output pixels, so the sweep means the same thing on every scene.
    """
    mask = np.ones((out_h, out_w), dtype=bool)
    sx, sy = out_w / src_w, out_h / src_h
    for x0, y0, x1, y1 in boxes:
        a = int(np.floor(x0 * sx)) - dilate_px
        b = int(np.floor(y0 * sy)) - dilate_px
        c = int(np.ceil(x1 * sx)) + dilate_px
        d = int(np.ceil(y1 * sy)) + dilate_px
        a, b = max(a, 0), max(b, 0)
        c, d = min(c, out_w), min(d, out_h)
        if c > a and d > b:
            mask[b:d, a:c] = False
    return mask


def masked_psnr(render, target, masks):
    """Per-view PSNR over valid pixels only, plus the fraction that were valid."""
    out, kept = [], []
    r = render.float()[0]
    t = target.float()[0]
    for i in range(r.shape[0]):
        m = torch.as_tensor(masks[i], device=r.device)
        valid = m.sum().item() / m.numel()
        kept.append(valid)
        if valid < 1e-6:
            out.append(float("nan"))
            continue
        diff = ((r[i] - t[i]) ** 2).mean(0)          # over channels
        mse = diff[m].mean().clamp(min=1e-12)
        out.append((-10.0 * torch.log10(mse)).item())
    return out, kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dl3dv_ar.yaml")
    parser.add_argument("--ckpt", default="checkpoints/dl3dv_ar.pt")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--n-scenes", type=int, default=30)
    parser.add_argument("--n-a", type=int, default=8)
    parser.add_argument("--n-b", type=int, default=8)
    parser.add_argument("--n-target", type=int, default=4)
    parser.add_argument("--a-span", type=float, default=0.30)
    parser.add_argument("--dilations", default="0,5,10,20,40")
    parser.add_argument("--out", default="oracle/results/nusc_masked.json")
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

    from oracle.oracle_data import SceneViews

    dilations = [int(d) for d in args.dilations.split(",")]
    out_h, out_w = config.model.image_size, config.model.image_size_x
    scenes = sorted(glob.glob(os.path.join(args.processed_dir, "*", "opencv_cameras.json")))
    scenes = scenes[: args.n_scenes]
    rows = []

    for path in scenes:
        name = os.path.basename(os.path.dirname(path))
        scene = SceneViews(path, out_h, out_w)
        n = len(scene)
        frames = scene.frames
        keyframes = [i for i, f in enumerate(frames) if f.get("is_keyframe")]
        if len(keyframes) < 8:
            print(f"{name}  skip (only {len(keyframes)} keyframes)")
            continue

        a_end = int(n * args.a_span)
        a_input = [int(v) for v in np.linspace(0, a_end - 1, args.n_a).round()]
        b_input = [int(v) for v in np.linspace(a_end, n - 1, args.n_b).round()]
        # scored frames must carry annotations, so they come from keyframes only
        a_keys = [k for k in keyframes if k < a_end and k not in a_input]
        if len(a_keys) < args.n_target:
            print(f"{name}  skip (only {len(a_keys)} usable target keyframes)")
            continue
        targets = [a_keys[i] for i in
                   np.linspace(0, len(a_keys) - 1, args.n_target).round().astype(int)]
        targets = sorted(set(targets))

        scene.normalise(sorted(set(a_input + b_input + targets)))
        masks = {
            d: [build_mask(frames[t]["boxes"], frames[t]["w"], frames[t]["h"],
                           out_h, out_w, d) for t in targets]
            for d in dilations
        }

        def run(inputs):
            batch = scene.batch(inputs, a_input, targets, device=device)
            config.training.num_input_views = len(inputs)
            config.training.num_virtual_views = len(a_input)
            config.training.num_target_views = len(targets)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(batch)
            res = {}
            for d in dilations:
                psnr, kept = masked_psnr(out.render["render"], out.target["image"], masks[d])
                res[d] = (float(np.nanmean(psnr)), float(np.mean(kept)))
            del out, batch
            torch.cuda.empty_cache()
            return res

        only_a = run(a_input)
        with_b = run(a_input + b_input)
        row = {"scene": name, "n_targets": len(targets),
               "boxes_per_target": float(np.mean([len(frames[t]["boxes"]) for t in targets]))}
        for d in dilations:
            row[f"a_d{d}"] = only_a[d][0]
            row[f"ab_d{d}"] = with_b[d][0]
            row[f"interf_d{d}"] = only_a[d][0] - with_b[d][0]
            row[f"kept_d{d}"] = only_a[d][1]
        rows.append(row)
        trail = "  ".join(f"d{d}:{row[f'interf_d{d}']:+.2f}({row[f'kept_d{d}']*100:.0f}%)"
                          for d in dilations)
        print(f"{name}  boxes/tgt {row['boxes_per_target']:4.1f}   {trail}")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)

    print(f"\n{len(rows)} scenes")
    print(f"{'dilation':>9} {'interference':>13} {'vs unmasked':>12} {'pixels kept':>12}")
    base = np.mean([r["interf_d0"] for r in rows]) if rows else float("nan")
    for d in dilations:
        v = np.array([r[f"interf_d{d}"] for r in rows])
        k = np.array([r[f"kept_d{d}"] for r in rows])
        print(f"{d:>9} {v.mean():13.3f} {v.mean() - base:+12.3f} {k.mean() * 100:11.1f}%")
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
