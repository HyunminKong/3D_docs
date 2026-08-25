"""Minimal CUT3R carrier interface for local test-time plasticity.

The external recurrent model stays frozen. A small spatial code can modify
only its final image-token input to the official geometry head; zero code is
therefore an exact base-model path rather than a separately trained decoder.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RecurrentCarrierState:
    state_feat: Tensor
    state_pos: Tensor
    init_state_feat: Tensor
    mem: Tensor
    init_mem: Tensor
    previous_reset: bool


class LocalTokenResidual(nn.Module):
    """Shared low-dimensional basis for a per-patch plasticity code."""

    def __init__(self, code_dim: int = 8, token_dim: int = 768, seed: int = 3800010):
        super().__init__()
        if code_dim <= 0 or code_dim > token_dim:
            raise ValueError("code_dim must be in [1, token_dim]")
        self.code_dim = int(code_dim)
        self.token_dim = int(token_dim)
        self.projection = nn.Linear(self.code_dim, self.token_dim, bias=False)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        raw = torch.randn(self.token_dim, self.code_dim, generator=generator)
        basis, _ = torch.linalg.qr(raw, mode="reduced")
        with torch.no_grad():
            self.projection.weight.copy_(basis)

    def forward(self, code: Tensor) -> Tensor:
        if code.ndim != 3 or code.shape[-1] != self.code_dim:
            raise ValueError(f"expected [B,N,{self.code_dim}] code, got {tuple(code.shape)}")
        return self.projection(code)

    def inject(self, head_input: list[Tensor], code: Tensor | None) -> list[Tensor]:
        if code is None:
            return head_input
        final_tokens = head_input[-1]
        patch_tokens = final_tokens.shape[1] - 1
        if code.shape[:2] != (final_tokens.shape[0], patch_tokens):
            raise ValueError(
                f"code has {tuple(code.shape[:2])}, expected {(final_tokens.shape[0], patch_tokens)}"
            )
        residual = self(code).to(final_tokens.dtype)
        modified = list(head_input)
        modified[-1] = torch.cat(
            (final_tokens[:, :1], final_tokens[:, 1:] + residual), dim=1
        )
        return modified


class FrozenCUT3RCarrier(nn.Module):
    """Step-wise RGB-only interface around an unchanged official CUT3R model."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        repository: str | Path = "TTT3R",
        code_dim: int = 8,
        basis_seed: int = 3800010,
    ) -> None:
        super().__init__()
        repository = Path(repository).resolve()
        source = repository / "src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from dust3r.model import ARCroco3DStereo

        self.model = ARCroco3DStereo.from_pretrained(str(checkpoint))
        self.model.config.model_update_type = "cut3r"
        self.model.eval().requires_grad_(False)
        self.residual = LocalTokenResidual(
            code_dim=code_dim,
            token_dim=int(self.model.dec_embed_dim),
            seed=basis_seed,
        )

    @property
    def code_dim(self) -> int:
        return self.residual.code_dim

    @staticmethod
    def _flag(value: Any) -> bool:
        if torch.is_tensor(value):
            return bool(value.reshape(-1)[0].item())
        return bool(value)

    def _encode_rgb(self, view: dict, device: torch.device) -> tuple[dict, Tensor, Tensor, Tensor]:
        if self._flag(view.get("ray_mask", False)):
            raise ValueError("the paper carrier interface is RGB-only")
        gpu_view = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in view.items()
        }
        images = gpu_view["img"]
        shapes = gpu_view.get("true_shape")
        if shapes is None:
            shapes = torch.tensor(images.shape[-2:], device=device).repeat(images.shape[0], 1)
        else:
            shapes = shapes.to(device)
        image_out, image_pos, _ = self.model._encode_image(images, shapes)
        return gpu_view, image_out[-1], image_pos, shapes

    def step(
        self,
        view: dict,
        state: RecurrentCarrierState | None,
        *,
        code: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], RecurrentCarrierState, dict[str, Any]]:
        device = next(self.model.parameters()).device
        with torch.no_grad():
            gpu_view, image_feat, image_pos, shape = self._encode_rgb(view, device)
            first = state is None
            if first:
                state_feat, state_pos = self.model._init_state(image_feat, image_pos)
                mem = self.model.pose_retriever.mem.expand(image_feat.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()
                previous_reset = False
            else:
                state_feat = state.state_feat
                state_pos = state.state_pos
                init_state_feat = state.init_state_feat
                mem = state.mem
                init_mem = state.init_mem
                previous_reset = state.previous_reset

            global_image_feat = self.model._get_img_level_feat(image_feat)
            if first or previous_reset:
                pose_feat = self.model.pose_token.expand(image_feat.shape[0], -1, -1)
            else:
                pose_feat = self.model.pose_retriever.inquire(global_image_feat, mem)
            pose_pos = -torch.ones(
                image_feat.shape[0], 1, 2, device=device, dtype=image_pos.dtype
            )
            new_state_feat, decoder, *_ = self.model._recurrent_rollout(
                state_feat,
                state_pos,
                image_feat,
                image_pos,
                pose_feat,
                pose_pos,
                init_state_feat,
                img_mask=gpu_view["img_mask"],
                reset_mask=gpu_view["reset"],
                update=gpu_view.get("update"),
                # Native lighter inference requests attention even in CUT3R
                # mode. Keeping that path is required for numerical parity:
                # the external implementation otherwise switches from its
                # explicit attention kernel to SDPA.
                return_attn=True,
            )
            output_pose_feat = decoder[-1][:, :1]
            new_mem = self.model.pose_retriever.update_mem(
                mem, global_image_feat, output_pose_feat
            )
            head_input = [
                decoder[0].float(),
                decoder[self.model.dec_depth * 2 // 4][:, 1:].float(),
                decoder[self.model.dec_depth * 3 // 4][:, 1:].float(),
                decoder[self.model.dec_depth].float(),
            ]

            update = gpu_view.get("update")
            update_mask = gpu_view["img_mask"] if update is None else gpu_view["img_mask"] & update
            update_mask = update_mask[:, None, None].float()
            next_state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
            next_mem = new_mem * update_mask + mem * (1 - update_mask)
            reset = self._flag(gpu_view.get("reset", False))
            if reset:
                reset_mask = gpu_view["reset"][:, None, None].float()
                next_state_feat = init_state_feat * reset_mask + next_state_feat * (1 - reset_mask)
                next_mem = init_mem * reset_mask + next_mem * (1 - reset_mask)

        auxiliary = {
            "image_tokens": image_feat.detach(),
            "image_pos": image_pos.detach(),
            "decoder_patch_tokens": head_input[-1][:, 1:].detach(),
            # Frozen rollout cache. This avoids recomputing CUT3R recurrence
            # when several code hypotheses are read through the same official
            # DPT head; it does not change the model or recurrent state.
            "head_input": [tensor.detach() for tensor in head_input],
            "shape": shape.detach(),
        }
        prediction = self.readout(auxiliary, code=code)
        next_state = RecurrentCarrierState(
            state_feat=next_state_feat.detach(),
            state_pos=state_pos.detach(),
            init_state_feat=init_state_feat.detach(),
            mem=next_mem.detach(),
            init_mem=init_mem.detach(),
            previous_reset=reset,
        )
        return prediction, next_state, auxiliary

    def readout(
        self, auxiliary: dict[str, Any], *, code: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Apply a code through the official head using one frozen rollout cache."""
        injected = self.residual.inject(auxiliary["head_input"], code)
        if code is None or not torch.is_grad_enabled():
            with torch.no_grad():
                return self.model._downstream_head(
                    injected, auxiliary["shape"], pos=auxiliary["image_pos"]
                )
        return self.model._downstream_head(
            injected, auxiliary["shape"], pos=auxiliary["image_pos"]
        )


def patch_center_points(points: Tensor, patch_size: int = 16) -> Tensor:
    """Sample dense points at the centers represented by decoder patch tokens."""
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B,H,W,3]")
    offset = patch_size // 2
    sampled = points[:, offset::patch_size, offset::patch_size]
    return sampled.flatten(1, 2)


def symmetric_point_consistency(current: Tensor, previous: Tensor) -> Tensor:
    """Single parameter-free online loss in a shared predicted 3D frame."""
    if current.ndim != 3 or previous.ndim != 3 or current.shape[-1] != 3:
        raise ValueError("point sets must have shape [B,N,3]")
    # torch.cdist's quadratic expansion can give a non-zero self-distance for
    # large canonical coordinates. Direct differences preserve the geometry
    # needed by both the loss and identity-transport contract.
    distances = torch.linalg.vector_norm(
        current[:, :, None, :] - previous[:, None, :, :], dim=-1
    )
    return 0.5 * (distances.min(dim=-1).values.mean() + distances.min(dim=-2).values.mean())


def transport_code_3d(
    source_points: Tensor, source_code: Tensor, target_points: Tensor
) -> tuple[Tensor, Tensor]:
    """Nearest-neighbor transport of a local code in predicted canonical 3D."""
    if source_points.shape[:2] != source_code.shape[:2]:
        raise ValueError("source point/code layouts differ")
    distances = torch.linalg.vector_norm(
        target_points[:, :, None, :] - source_points[:, None, :, :], dim=-1
    )
    nearest_distance, nearest = distances.min(dim=-1)
    transported = torch.gather(
        source_code,
        1,
        nearest[..., None].expand(-1, -1, source_code.shape[-1]),
    )
    return transported, nearest_distance


def transport_code_visual(
    source_features: Tensor,
    source_code: Tensor,
    target_features: Tensor,
    *,
    temperature: float = 0.07,
) -> tuple[Tensor, Tensor]:
    """Soft cosine transport using frozen carrier patch features."""
    if source_features.shape[:2] != source_code.shape[:2]:
        raise ValueError("source feature/code layouts differ")
    if target_features.shape[0] != source_features.shape[0]:
        raise ValueError("source and target batch sizes differ")
    cosine = F.normalize(target_features.float(), dim=-1) @ F.normalize(
        source_features.float(), dim=-1
    ).transpose(-1, -2)
    weights = torch.softmax(cosine / float(temperature), dim=-1)
    return weights @ source_code, weights.max(dim=-1).values
