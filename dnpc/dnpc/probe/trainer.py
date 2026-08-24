"""Streaming 3DGS trainer with per-Gaussian instrumentation (Stage 0 probe).

Streaming protocol (RESEARCH_PLAN.md leaves this undefined; it is fixed here and
recorded in the run config because it determines whether ``B_acc`` can be read
causally):

    frame t arrives -> spawn Gaussians in under-covered pixels from depth_init
                    -> ``iters_per_frame`` optimiser steps, each sampling one
                       camera from  {t-K+1 .. t}  u  {M random past frames}

With ``replay=0`` this is a strict sliding window; with ``replay=inf`` it degrades
to online-arrival offline training, where accumulated baseline stops explaining
anything. K=10, M=2 is the configured default.

The algorithm is stock gsplat ``DefaultStrategy``. Everything this module adds is
measurement: identity tracking (:mod:`.lineage`), a visibility record, and
contribution accumulation (:mod:`.attribution`). ``--no-instrument`` disables all
of it so the null effect on training can be verified.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from gsplat import rasterization

from ..data.base import Sequence, backproject
from .attribution import attribute, error_weight_maps
from .lineage import LineageStrategy


@dataclass
class ProbeConfig:
    # streaming protocol
    window: int = 10
    replay: int = 2
    # Replay must be drawn from a *covisibility-local* past, not the whole
    # history. Sampling uniformly over all past frames lets any long-lived
    # Gaussian accumulate the full trajectory as its baseline, which collapses
    # the variance of B_acc across Gaussians -- the independent variable the
    # probe exists to measure -- and is closer to global bundle adjustment than
    # to streaming. Horizon is in stream frames back from the current window.
    replay_horizon: int = 30
    iters_per_frame: int = 20
    # representation
    sh_degree: int = 3
    init_opacity: float = 0.5
    # Spawned extent should span the spawn pixel spacing, not one pixel: with
    # scale_mult=1 and spawn_pixel_stride=2 the surface is never covered, the
    # "uncovered" spawn test stays true forever and the Gaussian count runs away.
    scale_mult: float = 2.0
    # spawning
    spawn_alpha_thresh: float = 0.5
    spawn_depth_rel_thresh: float = 0.10
    spawn_pixel_stride: int = 2
    max_spawn_per_frame: int = 30000
    # Hard budget. The age-freezing baseline needs it: frozen Gaussians cannot
    # adapt, coverage stays poor, spawning never stops, and the run diverges
    # (7.6M Gaussians / 17 GB before it died).
    max_gaussians: int = 3_000_000
    # loss
    lambda_ssim: float = 0.2
    lambda_depth: float = 0.0  # 0 = geometry must come from photometry alone
    # densification (gsplat DefaultStrategy)
    refine_start_iter: int = 500
    refine_every: int = 100
    refine_stop_frac: float = 0.9
    grow_grad2d: float = 2e-4
    prune_opa: float = 5e-3
    reset_every: int = 10 ** 9  # opacity reset is disabled: it would invalidate
    # the observation record mid-stream (and is a no-op in gsplat 1.5.3 anyway,
    # where the guard `step % reset_every == 0 & step > 0` never fires).
    # instrumentation
    instrument: bool = True
    vis_stride: int = 2
    # Checkpointed within-Gaussian tracking. Comparing *final* error against
    # *final* accumulated baseline across Gaussians is confounded: a Gaussian
    # with few observations is usually one on an occlusion boundary, a grazing
    # surface or the image border, and those independently carry high error. So
    # the between-Gaussian regression would pick up selection, not triangulation.
    # Snapshotting (B_perp, z, err) repeatedly for the same lineage identifies
    # the law from *within*-Gaussian variation instead, where that confound
    # cannot act. Lineages with gid % track_mod == 0 are followed.
    checkpoint_every: int = 0
    track_mod: int = 8
    # age-freezing baseline (RESEARCH_PLAN 6.9); 0 disables
    freeze_age_k: int = 0
    seed: int = 0

    def to_dict(self):
        return asdict(self)


def _l1(a, b):
    return (a - b).abs().mean()


def _gauss_win(ws: int, sigma: float, device):
    g = torch.arange(ws, dtype=torch.float32, device=device) - (ws - 1) / 2
    g = torch.exp(-(g**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def _ssim(x, y, win, ws: int):
    """x, y: [1, C, H, W] in [0, 1]."""
    c = x.shape[1]
    w = win.expand(c, 1, ws, ws)
    pad = ws // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    mx2, my2, mxy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = F.conv2d(x * x, w, padding=pad, groups=c) - mx2
    sy = F.conv2d(y * y, w, padding=pad, groups=c) - my2
    sxy = F.conv2d(x * y, w, padding=pad, groups=c) - mxy
    C1, C2 = 0.01**2, 0.03**2
    return (((2 * mxy + C1) * (2 * sxy + C2)) / ((mx2 + my2 + C1) * (sx + sy + C2))).mean()


class FrameCache:
    """Preload the sequence into compact CPU arrays; PNG decode would otherwise
    dominate runtime once frames start being replayed."""

    def __init__(self, seq: Sequence, device):
        self.seq, self.device = seq, device
        t0 = time.time()
        self.rgb = np.empty((len(seq), seq.H, seq.W, 3), np.uint8)
        self.dgt = np.empty((len(seq), seq.H, seq.W), np.float16)
        self.dinit = np.empty((len(seq), seq.H, seq.W), np.float16)
        self.c2w = np.empty((len(seq), 4, 4), np.float32)
        for i in range(len(seq)):
            f = seq[i]
            self.rgb[i] = (f.rgb * 255).astype(np.uint8)
            self.dgt[i] = f.depth_gt.astype(np.float16)
            self.dinit[i] = f.depth_init.astype(np.float16)
            self.c2w[i] = f.c2w
        self.K = torch.from_numpy(seq.K).to(device)
        self.viewmats = torch.from_numpy(
            np.linalg.inv(self.c2w).astype(np.float32)
        ).to(device)
        self.cam_centers = torch.from_numpy(self.c2w[:, :3, 3].copy()).to(device)
        print(f"  cached {len(seq)} frames in {time.time()-t0:.1f}s "
              f"({self.rgb.nbytes/1e9:.2f} GB rgb)")

    def gpu(self, i: int):
        rgb = torch.from_numpy(self.rgb[i]).to(self.device).float() / 255.0
        dgt = torch.from_numpy(self.dgt[i].astype(np.float32)).to(self.device)
        return rgb, dgt


class StreamingProbe:
    def __init__(self, seq: Sequence, cfg: ProbeConfig, device="cuda"):
        self.seq, self.cfg, self.device = seq, cfg, device
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.cache = FrameCache(seq, device)
        self.scene_scale = seq.scene_scale()
        self.n_probe_cols = math.ceil(len(seq) / cfg.vis_stride)

        self.params: Dict[str, torch.nn.Parameter] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.strategy = LineageStrategy(
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            refine_start_iter=cfg.refine_start_iter,
            refine_every=cfg.refine_every,
            refine_stop_iter=int(len(seq) * cfg.iters_per_frame * cfg.refine_stop_frac),
            reset_every=cfg.reset_every,
            verbose=False,
        )
        self.state: Dict[str, object] = self.strategy.initialize_state(
            scene_scale=self.scene_scale
        )
        self.step = 0
        self.ssim_win = _gauss_win(11, 1.5, device)
        self.timing = {"train_s": 0.0, "spawn_s": 0.0, "instr_s": 0.0}
        self.per_frame_ms: List[float] = []

    # ---------------------------------------------------------------- params
    @property
    def N(self) -> int:
        return 0 if not self.params else self.params["means"].shape[0]

    def _sh_dim(self):
        return (self.cfg.sh_degree + 1) ** 2 - 1

    def _make_optimizers(self):
        s = self.scene_scale
        lrs = {
            "means": 1.6e-4 * s, "scales": 5e-3, "quats": 1e-3,
            "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20,
        }
        self.optimizers = {
            k: torch.optim.Adam([self.params[k]], lr=lrs[k], eps=1e-15)
            for k in self.params
        }

    @torch.no_grad()
    def _append(self, new: Dict[str, torch.Tensor]):
        """Append freshly spawned Gaussians to params, optimizer state and stats."""
        first = not self.params
        if first:
            self.params = {k: torch.nn.Parameter(v) for k, v in new.items()}
            self._make_optimizers()
            self.strategy.init_identity(self.state, new["means"].shape[0], self.device)
            self._init_stats(new["means"].shape[0])
            return

        n_new = new["means"].shape[0]
        for k, v in new.items():
            p = self.params[k]
            opt = self.optimizers[k]
            st = opt.state.pop(p)
            newp = torch.nn.Parameter(torch.cat([p.data, v]))
            for key in ("exp_avg", "exp_avg_sq"):
                st[key] = torch.cat([st[key], torch.zeros_like(v)])
            opt.param_groups[0]["params"] = [newp]
            opt.state[newp] = st
            self.params[k] = newp

        self.strategy.append_identity(self.state, n_new, self.device)
        for k in ("grad2d", "count"):
            if isinstance(self.state.get(k), torch.Tensor):
                self.state[k] = torch.cat(
                    [self.state[k], torch.zeros(n_new, device=self.device)]
                )
        self._grow_stats(n_new)

    # ----------------------------------------------------------------- stats
    def _init_stats(self, n: int):
        d, cfg = self.device, self.cfg
        self.state["birth_frame"] = torch.full((n,), self._t, dtype=torch.int32, device=d)
        self.state["z_first"] = torch.zeros(n, device=d)
        self.state["err_init"] = torch.zeros(n, device=d)
        self.state["ray_first"] = torch.zeros(n, 3, device=d)
        self.state["contrib"] = torch.zeros(n, device=d)
        if cfg.instrument:
            self.state["vis"] = torch.zeros(n, self.n_probe_cols, dtype=torch.bool, device=d)

    def _grow_stats(self, n_new: int):
        d = self.device
        self.state["birth_frame"] = torch.cat(
            [self.state["birth_frame"], torch.full((n_new,), self._t, dtype=torch.int32, device=d)]
        )
        for k, shape in (("z_first", ()), ("err_init", ()), ("contrib", ()),
                         ("ray_first", (3,))):
            self.state[k] = torch.cat([self.state[k], torch.zeros(n_new, *shape, device=d)])
        if self.cfg.instrument:
            self.state["vis"] = torch.cat(
                [self.state["vis"], torch.zeros(n_new, self.n_probe_cols, dtype=torch.bool, device=d)]
            )

    # -------------------------------------------------------------- spawning
    @torch.no_grad()
    def _spawn(self, t: int):
        cfg = self.cfg
        rgb, dgt_full = self.cache.gpu(t)
        dinit = torch.from_numpy(self.cache.dinit[t].astype(np.float32)).to(self.device)
        H, W = self.seq.H, self.seq.W

        need = dinit > 0
        if self.N > 0:
            _, alphas, _ = self._render(t, with_depth=False)
            need = need & (alphas[..., 0] < cfg.spawn_alpha_thresh)
        if cfg.max_gaussians and self.N >= cfg.max_gaussians:
            return 0

        sub = torch.zeros_like(need)
        sub[:: cfg.spawn_pixel_stride, :: cfg.spawn_pixel_stride] = True
        need = need & sub
        idx = torch.nonzero(need, as_tuple=False)
        if idx.shape[0] == 0:
            return 0
        if idx.shape[0] > cfg.max_spawn_per_frame:
            sel = torch.randperm(idx.shape[0], device=self.device)[: cfg.max_spawn_per_frame]
            idx = idx[sel]

        v, u = idx[:, 0], idx[:, 1]
        z = dinit[v, u]
        K = self.cache.K
        x = (u.float() - K[0, 2]) / K[0, 0] * z
        y = (v.float() - K[1, 2]) / K[1, 1] * z
        cam = torch.stack([x, y, z], -1)
        c2w = torch.from_numpy(self.cache.c2w[t]).to(self.device)
        means = cam @ c2w[:3, :3].T + c2w[:3, 3]

        cc = c2w[:3, 3]
        ray = means - cc
        z_first = ray.norm(dim=-1)
        ray_first = ray / z_first.clamp_min(1e-8)[:, None]

        # one-pixel footprint at that range is the natural initial extent
        s = (z * cfg.scale_mult / float(K[0, 0])).clamp_min(1e-4)
        scales = torch.log(s)[:, None].repeat(1, 3)
        quats = torch.zeros(len(means), 4, device=self.device)
        quats[:, 0] = 1.0
        opac = torch.full((len(means),), math.log(cfg.init_opacity / (1 - cfg.init_opacity)),
                          device=self.device)
        col = rgb[v, u]
        sh0 = ((col - 0.5) / 0.2820947917738781)[:, None, :]
        shN = torch.zeros(len(means), self._sh_dim(), 3, device=self.device)

        n0 = self.N
        self._append({"means": means, "scales": scales, "quats": quats,
                      "opacities": opac, "sh0": sh0, "shN": shN})
        self.state["z_first"][n0:] = z_first
        self.state["ray_first"][n0:] = ray_first
        gt_at = dgt_full[v, u]
        self.state["err_init"][n0:] = torch.where(gt_at > 0, (z - gt_at).abs(),
                                                  torch.zeros_like(z))
        return len(means)

    # ------------------------------------------------------------- rendering
    def _render(self, t: int, with_depth=False, need_info=False, sh_degree=None):
        cfg = self.cfg
        colors = torch.cat([self.params["sh0"], self.params["shN"]], 1)
        mode = "RGB+ED" if with_depth else "RGB"
        out, alphas, info = rasterization(
            means=self.params["means"],
            quats=self.params["quats"],
            scales=torch.exp(self.params["scales"]),
            opacities=torch.sigmoid(self.params["opacities"]),
            colors=colors,
            viewmats=self.cache.viewmats[t][None],
            Ks=self.cache.K[None],
            width=self.seq.W,
            height=self.seq.H,
            sh_degree=cfg.sh_degree if sh_degree is None else sh_degree,
            packed=True,
            render_mode=mode,
        )
        rgb = out[0, ..., :3]
        depth = out[0, ..., 3] if with_depth else None
        if need_info:
            return rgb, alphas[0], depth, info
        return (rgb, alphas[0], depth) if with_depth else (rgb, alphas[0], None)

    # ---------------------------------------------------------------- freeze
    @torch.no_grad()
    def _age_freeze_mask(self, t: int) -> Optional[torch.Tensor]:
        """Baseline of RESEARCH_PLAN 6.9: freeze k frames after last observation.

        With a sliding window, "last observed" is the birth frame plus the window
        span, so this reduces to a birth-frame cutoff -- which is exactly what
        makes it such a strong and cheap baseline."""
        k = self.cfg.freeze_age_k
        if k <= 0:
            return None
        return self.state["birth_frame"] < (t - self.cfg.window - k)

    # ----------------------------------------------------------------- train
    def _sample_cameras(self, t: int, n: int) -> List[int]:
        cfg = self.cfg
        lo = max(0, t - cfg.window + 1)
        win = list(range(lo, t + 1))
        rlo = max(0, lo - cfg.replay_horizon)
        every = cfg.window // cfg.replay + 1 if cfg.replay > 0 else 0
        out = []
        for i in range(n):
            if cfg.replay > 0 and lo > rlo and i % every == 0:
                out.append(int(self.rng.integers(rlo, lo)))
            else:
                out.append(int(win[self.rng.integers(0, len(win))]))
        return out

    def _train_on_frame(self, t: int):
        cfg = self.cfg
        cams = self._sample_cameras(t, cfg.iters_per_frame)
        freeze = self._age_freeze_mask(t)
        for cam in cams:
            rgb_gt, dgt = self.cache.gpu(cam)
            rgb, alphas, depth, info = self._render(cam, with_depth=cfg.lambda_depth > 0,
                                                    need_info=True)
            loss = (1 - cfg.lambda_ssim) * _l1(rgb, rgb_gt) + cfg.lambda_ssim * (
                1 - _ssim(rgb.permute(2, 0, 1)[None], rgb_gt.permute(2, 0, 1)[None],
                          self.ssim_win, 11)
            )
            if cfg.lambda_depth > 0 and depth is not None:
                m = dgt > 0
                loss = loss + cfg.lambda_depth * (depth[m] - dgt[m]).abs().mean()

            self.strategy.step_pre_backward(self.params, self.optimizers, self.state,
                                            self.step, info)
            loss.backward()

            if freeze is not None:
                for k in self.params:
                    g = self.params[k].grad
                    if g is not None:
                        g[freeze] = 0.0

            if cfg.instrument:
                col = cam // cfg.vis_stride
                self.state["vis"][info["gaussian_ids"].long(), col] = True

            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            self.strategy.step_post_backward(self.params, self.optimizers, self.state,
                                             self.step, info, packed=True)
            self.step += 1
            if freeze is not None and freeze.shape[0] != self.N:
                freeze = self._age_freeze_mask(t)

    @torch.no_grad()
    def _accumulate_contrib(self, t: int):
        ones = torch.ones(1, self.seq.H, self.seq.W, device=self.device)
        g = attribute(self.params["means"], self.params["quats"],
                      torch.exp(self.params["scales"]),
                      torch.sigmoid(self.params["opacities"]),
                      self.cache.viewmats[t], self.cache.K,
                      self.seq.W, self.seq.H, ones)
        self.state["contrib"] += g[:, 0]

    # ----------------------------------------------------- checkpoint tracking
    @torch.no_grad()
    def _checkpoint(self, t: int):
        """Snapshot (observation geometry, geometric error) for tracked lineages."""
        from .geom_error import observation_geometry

        gid = self.state["gid"]
        keep = (gid % self.cfg.track_mod) == 0
        if not bool(keep.any()):
            return
        idx = torch.nonzero(keep, as_tuple=True)[0]
        means = self.params["means"].detach()[idx]
        ray = self.state["ray_first"][idx]
        obs = observation_geometry(
            self.state["vis"][idx], means, ray, self.cache.cam_centers[:: self.cfg.vis_stride]
        )
        ge = self.gt_eval(means.cpu().numpy(), ray.cpu().numpy())
        rec = {
            "frame": np.full(len(idx), t, np.int32),
            "gid": gid[idx].cpu().numpy(),
            "birth_frame": self.state["birth_frame"][idx].cpu().numpy(),
            "z_first": self.state["z_first"][idx].cpu().numpy(),
            "dens_count": self.state["dens_count"][idx].cpu().numpy(),
            "contrib": self.state["contrib"][idx].cpu().numpy(),
            "err_init": self.state["err_init"][idx].cpu().numpy(),
            "scale_mean": torch.exp(self.params["scales"].detach()[idx]).mean(-1).cpu().numpy(),
            "opacity": torch.sigmoid(self.params["opacities"].detach()[idx]).cpu().numpy(),
            **obs,
            **{k: v for k, v in ge.items()},
        }
        self.checkpoints.append(pd.DataFrame(rec))

    # ------------------------------------------------------------------- run
    def run(self, gt_eval=None):
        self.gt_eval = gt_eval
        self.checkpoints: List["pd.DataFrame"] = []
        T = len(self.seq)
        for t in range(T):
            self._t = t
            torch.cuda.synchronize()
            t0 = time.time()
            ts = time.time()
            self._spawn(t)
            torch.cuda.synchronize()
            self.timing["spawn_s"] += time.time() - ts
            tt = time.time()
            self._train_on_frame(t)
            torch.cuda.synchronize()
            self.timing["train_s"] += time.time() - tt
            if self.cfg.instrument:
                ti = time.time()
                self._accumulate_contrib(t)
                torch.cuda.synchronize()
                self.timing["instr_s"] += time.time() - ti
            self.per_frame_ms.append((time.time() - t0) * 1e3)
            if (self.cfg.checkpoint_every and gt_eval is not None
                    and t > 0 and t % self.cfg.checkpoint_every == 0):
                tc = time.time()
                self._checkpoint(t)
                print(f"   ckpt@{t}: {len(self.checkpoints[-1]):,} tracked rows "
                      f"({time.time()-tc:.1f}s)")
            if t % 100 == 0 or t == T - 1:
                fm = self._age_freeze_mask(t)
                fr = 0.0 if fm is None else float(fm.float().mean())
                print(f"  [{t:5d}/{T}] N={self.N:8d} step={self.step:7d} "
                      f"ms/frame={np.mean(self.per_frame_ms[-100:]):7.1f} freeze={fr:.2f} "
                      f"peakGB={torch.cuda.max_memory_allocated()/1e9:.1f}")
        return self

    # ----------------------------------------------------------- final eval
    @torch.no_grad()
    def evaluate(self, eval_stride: int = 5):
        """Final pass over the whole sequence with the *final* Gaussian set.

        Accumulates per-Gaussian photometric and depth error shares, plus the
        image metrics used for the quality/compute table. Visibility is NOT
        recorded here -- these views were never used to constrain the model."""
        n = self.N
        acc = torch.zeros(n, 4, device=self.device)
        psnrs, l1s, absrels = [], [], []
        for t in range(0, len(self.seq), eval_stride):
            rgb_gt, dgt = self.cache.gpu(t)
            rgb, _, depth = self._render(t, with_depth=True)
            rgb = rgb.clamp(0, 1)
            mse = ((rgb - rgb_gt) ** 2).mean()
            psnrs.append(float(-10 * torch.log10(mse.clamp_min(1e-12))))
            l1s.append(float((rgb - rgb_gt).abs().mean()))
            m = dgt > 0
            absrels.append(float(((depth[m] - dgt[m]).abs() / dgt[m]).mean()))
            w = error_weight_maps(rgb, rgb_gt, depth, dgt)
            acc += attribute(self.params["means"], self.params["quats"],
                             torch.exp(self.params["scales"]),
                             torch.sigmoid(self.params["opacities"]),
                             self.cache.viewmats[t], self.cache.K,
                             self.seq.W, self.seq.H, w)
        metrics = {
            "psnr": float(np.mean(psnrs)),
            "l1": float(np.mean(l1s)),
            "depth_abs_rel": float(np.mean(absrels)),
            "n_gaussians": n,
            "ms_per_frame_mean": float(np.mean(self.per_frame_ms)),
            "ms_per_frame_median": float(np.median(self.per_frame_ms)),
            "peak_gpu_gb": float(torch.cuda.max_memory_allocated() / 1e9),
            **{f"time_{k}": v for k, v in self.timing.items()},
        }
        e_render = (acc[:, 1] / acc[:, 0].clamp_min(1e-8)).cpu().numpy()
        e_depth = (acc[:, 2] / acc[:, 3].clamp_min(1e-8)).cpu().numpy()
        eval_contrib = acc[:, 0].cpu().numpy()
        return metrics, e_render, e_depth, eval_contrib
