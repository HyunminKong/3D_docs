"""Is the recurrent state the only thing carrying history in CUT3R?

Every measurement made on tttLRM rested on one architectural fact: attention ran
inside each image, so the fast weight was the sole route from one frame to the
next, and a drop in quality after streaming more frames could only be the state
being overwritten. That was verified rather than assumed -- restoring the earlier
fast weight reproduced the earlier output to 0.0000 dB.

CUT3R has to earn the same standing before any number measured on it means the
same thing. Its state is a set of tokens rather than a matrix, and it also
carries a separate pose memory, so history could in principle travel by more
than one road.

The test injects a stored state and asks whether the model retraces its steps:

    run A then probe            -> output_1, and the state as it stood after A
    run A then B then probe     -> output_2   (these differ; that is interference)
    probe alone, state set to the one saved after A  -> output_3

If output_3 matches output_1 the state is sufficient -- everything the model
knew after A is in what was saved. A residual gap is history arriving by some
other path, and it bounds how much of the interference can honestly be
attributed to the state at all.
"""

import argparse
import contextlib
import os
import sys

import numpy as np
import torch

TTT3R = "/home/khm/3D_4D/TTT3R"
sys.path.insert(0, TTT3R)
sys.path.insert(0, os.path.join(TTT3R, "src"))
# croco resolves its CUDA RoPE as `models.curope`, so its own directory has to be
# importable under that name. Without it the code silently falls back to a
# PyTorch implementation that cannot index the negative positions the pose token
# uses, and the run dies inside an embedding lookup several layers later.
sys.path.insert(0, os.path.join(TTT3R, "src/croco"))


@contextlib.contextmanager
def injected_state(model, state):
    """Force the next forward to begin from `state` instead of the first frame."""
    feat, pos, mem = state
    orig_init, orig_mem = model._init_state, model.pose_retriever.mem
    model._init_state = lambda *a, **k: (feat.clone(), pos.clone())
    model.pose_retriever.mem = torch.nn.Parameter(mem.clone(), requires_grad=False)
    try:
        yield
    finally:
        model._init_state = orig_init
        model.pose_retriever.mem = orig_mem


def build_views(images, update_flags):
    views = []
    for i, (im, upd) in enumerate(zip(images, update_flags)):
        views.append({
            "img": im["img"],
            "ray_map": torch.full(
                (im["img"].shape[0], 6, im["img"].shape[-2], im["img"].shape[-1]),
                torch.nan),
            "true_shape": torch.from_numpy(im["true_shape"]),
            "idx": i, "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(bool(upd)).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        })
    return views


def to_device(views, device):
    out = []
    for v in views:
        out.append({k: (x.to(device) if torch.is_tensor(x) else x) for k, x in v.items()})
    return out


def pts_of(res):
    """The point map a view produced, as a flat tensor."""
    for key in ("pts3d_in_other_view", "pts3d"):
        if key in res:
            return res[key].float()
    raise KeyError(f"no point map in {list(res)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="directory of images")
    ap.add_argument("--ckpt", default=os.path.join(TTT3R, "src/cut3r_512_dpt_4_64.pth"))
    ap.add_argument("--update-type", default="cut3r", choices=["cut3r", "ttt3r"])
    ap.add_argument("--n-a", type=int, default=8)
    ap.add_argument("--n-b", type=int, default=8)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.image import load_images

    device = "cuda"
    model = ARCroco3DStereo.from_pretrained(args.ckpt).to(device).eval()
    model.config.model_update_type = args.update_type

    files = sorted(f for f in os.listdir(args.scene)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    need = (args.n_a + args.n_b + 1) * args.stride
    assert len(files) >= need, f"{len(files)} images, need {need}"
    picked = [os.path.join(args.scene, files[i * args.stride])
              for i in range(args.n_a + args.n_b + 1)]
    images = load_images(picked, size=args.size, verbose=False)

    a_imgs = images[: args.n_a]
    b_imgs = images[args.n_a: args.n_a + args.n_b]
    probe = images[args.n_a + args.n_b: args.n_a + args.n_b + 1]

    def run(imgs, updates, state=None):
        views = to_device(build_views(imgs, updates), device)
        ctx = injected_state(model, state) if state else contextlib.nullcontext()
        with torch.no_grad(), ctx:
            out, states = model(views, ret_state=True)
        return out, states

    # 1. A, then the probe without writing
    out1, st1 = run(a_imgs + probe, [1] * len(a_imgs) + [0])
    o1 = pts_of(out1.ress[-1])
    after_a = st1[len(a_imgs)]          # state as it stood when the probe was read
    saved = (after_a[0].detach().clone(), after_a[1].detach().clone(),
             after_a[3].detach().clone())

    # 2. A, B, then the same probe
    out2, _ = run(a_imgs + b_imgs + probe, [1] * (len(a_imgs) + len(b_imgs)) + [0])
    o2 = pts_of(out2.ress[-1])

    # 3. the probe, starting from the state saved after A.
    #
    # A spacer goes first because the very first view of a sequence is treated
    # differently: index zero takes a learned initial pose token, every later
    # index queries the pose memory instead. Putting the probe at position zero
    # would compare two different code paths and charge the difference to the
    # state. The spacer writes nothing (update=False), so it advances the index
    # without touching what was restored.
    out3, _ = run(probe + probe, [0, 0], state=saved)
    o3 = pts_of(out3.ress[-1])

    scale = o1.abs().mean().clamp(min=1e-8)
    print(f"update rule      : {args.update_type}")
    print(f"scene            : {os.path.basename(args.scene.rstrip('/'))}"
          f"   A={args.n_a} B={args.n_b} stride={args.stride}\n")
    print(f"interference      |o2 - o1| / |o1|   = {(o2-o1).abs().mean()/scale:.6f}"
          "     (state overwritten by B)")
    print(f"plumbing          |o3 - o1| / |o1|   = {(o3-o1).abs().mean()/scale:.6f}"
          "     (0 => state is the only path)")
    ratio = float((o3 - o1).abs().mean() / (o2 - o1).abs().mean().clamp(min=1e-12))
    print(f"\nresidual as a share of the interference: {ratio:6.2%}")
    if ratio < 0.02:
        print("  -> the state accounts for essentially all of it; "
              "comparable with the tttLRM measurements")
    else:
        print("  -> a second path carries history; interference cannot be "
              "attributed to the state alone and this share must be reported")


if __name__ == "__main__":
    main()
