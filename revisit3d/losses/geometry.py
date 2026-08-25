"""Self-supervised geometric objectives for compact-state TTT.

No held-out LiDAR, depth, or pose target is accepted here.  ``w2c`` is simply
the transform used for reprojection: at deployment it must come from the
current pose estimate (or a trusted odometry source), while known transforms
are permitted only for synthetic/unit-test checks.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _intrinsics_matrix(intrinsics: Tensor, height: int, width: int,
                       image_height: int, image_width: int) -> Tensor:
    """Convert [fx,fy,cx,cy] from image pixels to a token-grid camera matrix."""
    scale_x, scale_y = width / image_width, height / image_height
    fx, fy, cx, cy = intrinsics.unbind(dim=-1)
    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)
    return torch.stack((
        torch.stack((fx * scale_x, zeros, cx * scale_x), dim=-1),
        torch.stack((zeros, fy * scale_y, cy * scale_y), dim=-1),
        torch.stack((zeros, zeros, ones), dim=-1),
    ), dim=-2)


def depth_smoothness_loss(depth: Tensor, images: Tensor) -> Tensor:
    """Edge-aware depth smoothness; both tensors are [B,V,*,H,W]."""
    if images.shape[-2:] != depth.shape[-2:]:
        batch, views = images.shape[:2]
        images = F.interpolate(images.flatten(0, 1), size=depth.shape[-2:], mode="bilinear", align_corners=True)
        images = images.reshape(batch, views, 3, *depth.shape[-2:])
    dx = depth[..., :, 1:] - depth[..., :, :-1]
    dy = depth[..., 1:, :] - depth[..., :-1, :]
    ix = images[..., :, 1:] - images[..., :, :-1]
    iy = images[..., 1:, :] - images[..., :-1, :]
    weight_x = torch.exp(-ix.abs().mean(dim=-3, keepdim=True))
    weight_y = torch.exp(-iy.abs().mean(dim=-3, keepdim=True))
    return (dx.abs() * weight_x).mean() + (dy.abs() * weight_y).mean()


def track_3d_consistency_loss(
    depth: Tensor,
    intrinsics: Tensor,
    w2c: Tensor,
    tracks: Tensor,
    visibility: Tensor,
    confidence: Tensor,
    *,
    image_size: tuple[int, int],
    return_stats: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Tie predicted depths across frozen point tracks in a common world gauge.

    ``tracks`` are fixed foundation correspondences in image-pixel coordinates,
    so a trainable depth cannot obtain a zero loss by moving projections out of
    the image.  This is deliberately a geometry-only online objective: neither
    a depth pseudo-label nor a held-out frame enters it.
    """
    if depth.ndim != 4 or intrinsics.ndim != 3 or w2c.ndim != 4 or tracks.ndim != 4:
        raise ValueError("depth [B,V,H,W], intrinsics [B,V,4], w2c [B,V,4,4], tracks [B,V,N,2] expected")
    batch, views, grid_h, grid_w = depth.shape
    if tracks.shape[:2] != (batch, views) or visibility.shape != tracks.shape[:3] or confidence.shape != tracks.shape[:3]:
        raise ValueError("track, visibility, and confidence dimensions do not match depth views")
    image_h, image_w = image_size
    points = tracks.to(depth.dtype)
    normalized = torch.stack((
        2 * points[..., 0] / max(image_w - 1, 1) - 1,
        2 * points[..., 1] / max(image_h - 1, 1) - 1,
    ), dim=-1)
    samples = F.grid_sample(depth.flatten(0, 1).unsqueeze(1), normalized.flatten(0, 1).unsqueeze(1),
                            mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled_depth = samples.squeeze(1).squeeze(1).reshape(batch, views, -1).clamp_min(1e-5)
    fx, fy, cx, cy = intrinsics.to(depth.dtype).unbind(dim=-1)
    x = (points[..., 0] - cx[..., None]) / fx[..., None] * sampled_depth
    y = (points[..., 1] - cy[..., None]) / fy[..., None] * sampled_depth
    camera_points = torch.stack((x, y, sampled_depth, torch.ones_like(sampled_depth)), dim=-1)
    c2w = torch.linalg.inv(w2c.to(depth.dtype))
    world = torch.einsum("bvij,bvnj->bvni", c2w, camera_points)[..., :3]
    residual = torch.linalg.vector_norm(world[:, 1:] - world[:, :1], dim=-1)
    in_bounds = ((points[..., 0] >= 0) & (points[..., 0] <= image_w - 1) &
                 (points[..., 1] >= 0) & (points[..., 1] <= image_h - 1)).to(depth.dtype)
    weight = (visibility.to(depth.dtype) * confidence.to(depth.dtype) * in_bounds)[:, 1:]
    loss = (weight * torch.sqrt(residual.square() + 1e-6)).sum() / weight.sum().clamp_min(1e-6)
    if return_stats:
        return loss, {"track_coverage": weight.sum() / weight.numel(),
                      "mean_track_weight": weight.mean(), "mean_3d_residual": residual.mean()}
    return loss


def track_reprojection_consistency_loss(
    depth: Tensor,
    intrinsics: Tensor,
    w2c: Tensor,
    tracks: Tensor,
    visibility: Tensor,
    confidence: Tensor,
    *,
    image_size: tuple[int, int],
    return_stats: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Symmetrically reproject frozen tracks using each view's predicted depth.

    Unlike :func:`track_3d_consistency_loss`, this objective measures a
    dimensionless image-plane residual.  Every view serves as a source, so the
    loss updates all view-local depth tokens while using no future frame, GT
    depth, or GT pose.
    """
    if depth.ndim != 4 or intrinsics.ndim != 3 or w2c.ndim != 4 or tracks.ndim != 4:
        raise ValueError("depth [B,V,H,W], intrinsics [B,V,4], w2c [B,V,4,4], tracks [B,V,N,2] expected")
    batch, views, _, _ = depth.shape
    if views < 2:
        zero = depth.sum() * 0
        return (zero, {"track_coverage": zero, "mean_reprojection_residual": zero}) if return_stats else zero
    if tracks.shape[:2] != (batch, views) or visibility.shape != tracks.shape[:3] or confidence.shape != tracks.shape[:3]:
        raise ValueError("track, visibility, and confidence dimensions do not match depth views")
    image_h, image_w = image_size
    points = tracks.to(depth.dtype)
    normalized = torch.stack((
        2 * points[..., 0] / max(image_w - 1, 1) - 1,
        2 * points[..., 1] / max(image_h - 1, 1) - 1,
    ), dim=-1)
    samples = F.grid_sample(
        depth.flatten(0, 1).unsqueeze(1), normalized.flatten(0, 1).unsqueeze(1),
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    sampled_depth = samples.squeeze(1).squeeze(1).reshape(batch, views, -1).clamp_min(1e-5)
    fx, fy, cx, cy = intrinsics.to(depth.dtype).unbind(dim=-1)
    x = (points[..., 0] - cx[..., None]) / fx[..., None] * sampled_depth
    y = (points[..., 1] - cy[..., None]) / fy[..., None] * sampled_depth
    camera_points = torch.stack((x, y, sampled_depth, torch.ones_like(sampled_depth)), dim=-1)
    c2w = torch.linalg.inv(w2c.to(depth.dtype))
    world = torch.einsum("bvij,bvnj->bvni", c2w, camera_points)
    in_bounds = (
        (points[..., 0] >= 0) & (points[..., 0] <= image_w - 1)
        & (points[..., 1] >= 0) & (points[..., 1] <= image_h - 1)
    ).to(depth.dtype)
    evidence = visibility.to(depth.dtype) * confidence.to(depth.dtype) * in_bounds
    diagonal = float((image_h ** 2 + image_w ** 2) ** 0.5)
    total = depth.new_zeros(())
    total_weight = depth.new_zeros(())
    raw_residual = depth.new_zeros(())
    pairs = 0
    for source in range(views):
        source_world = world[:, source]
        for target in range(views):
            if source == target:
                continue
            camera = torch.einsum("bij,bnj->bni", w2c[:, target].to(depth.dtype), source_world)
            z = camera[..., 2].clamp_min(1e-5)
            u = fx[:, target, None] * camera[..., 0] / z + cx[:, target, None]
            v = fy[:, target, None] * camera[..., 1] / z + cy[:, target, None]
            residual = torch.sqrt(
                (u - points[:, target, :, 0]).square()
                + (v - points[:, target, :, 1]).square() + 1e-4
            ) / diagonal
            # A source ray that lands behind the target camera is not a valid
            # reprojection constraint.  Excluding it also prevents the
            # clamped denominator from creating arbitrarily large gradients.
            positive_depth = (camera[..., 2] > 1e-5).to(depth.dtype)
            weight = evidence[:, source] * evidence[:, target] * positive_depth
            total = total + (weight * torch.sqrt(residual.square() + 1e-6)).sum()
            raw_residual = raw_residual + (weight * residual).sum()
            total_weight = total_weight + weight.sum()
            pairs += 1
    loss = total / total_weight.clamp_min(1e-6)
    if return_stats:
        return loss, {
            "track_coverage": total_weight / max(evidence.numel() * pairs / views, 1),
            "mean_reprojection_residual": raw_residual / total_weight.clamp_min(1e-6),
        }
    return loss


def relative_w2c_from_twist(twist: Tensor) -> Tensor:
    """Convert [translation, axis-angle] predictions to view-0-relative w2c.

    The reference view is fixed to identity, removing the arbitrary global
    coordinate gauge from the photometric online objective.
    """
    if twist.ndim != 3 or twist.shape[-1] != 6:
        raise ValueError("twist must be [batch, views, 6]")
    translation, omega = twist[..., :3], twist[..., 3:]
    theta = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
    wx, wy, wz = omega.unbind(dim=-1)
    zeros = torch.zeros_like(wx)
    skew = torch.stack((
        torch.stack((zeros, -wz, wy), dim=-1),
        torch.stack((wz, zeros, -wx), dim=-1),
        torch.stack((-wy, wx, zeros), dim=-1),
    ), dim=-2)
    eye = torch.eye(3, dtype=twist.dtype, device=twist.device).expand_as(skew)
    theta2 = theta.square()
    # Taylor-safe Rodrigues coefficients.
    a = torch.where(theta > 1e-4, torch.sin(theta) / theta, 1 - theta2 / 6)
    b = torch.where(theta > 1e-4, (1 - torch.cos(theta)) / theta2, 0.5 - theta2 / 24)
    rotation = eye + a[..., None] * skew + b[..., None] * (skew @ skew)
    transform = torch.zeros(*twist.shape[:-1], 4, 4, dtype=twist.dtype, device=twist.device)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    transform[..., 3, 3] = 1
    reference_inv = torch.linalg.inv(transform[:, :1])
    return transform @ reference_inv


def reprojection_loss(
    depth: Tensor,
    images: Tensor,
    intrinsics: Tensor,
    w2c: Tensor,
    *,
    reference_view: int = 0,
    return_stats: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Photometrically reproject a reference token grid into the other views.

    Args:
        depth: [B,V,Ht,Wt] positive depth in the reference camera convention.
        images: [B,V,3,H,W] RGB images in [0,1].
        intrinsics: [B,V,4] corresponding image-pixel intrinsics.
        w2c: [B,V,4,4] transforms used solely for reprojection.
    """
    if depth.ndim != 4 or images.ndim != 5 or intrinsics.ndim != 3 or w2c.ndim != 4:
        raise ValueError("expected depth [B,V,Ht,Wt], images [B,V,3,H,W], intrinsics [B,V,4], w2c [B,V,4,4]")
    batch, views, grid_h, grid_w = depth.shape
    _, _, _, image_h, image_w = images.shape
    if views < 2:
        return depth.new_zeros(())
    K = _intrinsics_matrix(intrinsics, grid_h, grid_w, image_h, image_w)
    ys, xs = torch.meshgrid(
        torch.arange(grid_h, device=depth.device, dtype=depth.dtype),
        torch.arange(grid_w, device=depth.device, dtype=depth.dtype), indexing="ij"
    )
    homogeneous = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1).reshape(1, -1, 3)
    ref_depth = depth[:, reference_view].reshape(batch, -1, 1)
    rays = torch.linalg.solve(K[:, reference_view], homogeneous.transpose(1, 2)).transpose(1, 2)
    points_ref = rays * ref_depth
    homogeneous_ref = torch.cat((points_ref, torch.ones_like(ref_depth)), dim=-1)
    c2w_ref = torch.linalg.inv(w2c[:, reference_view])
    world = torch.einsum("bij,bnj->bni", c2w_ref, homogeneous_ref)
    source_rgb = F.interpolate(images[:, reference_view], size=(grid_h, grid_w), mode="bilinear", align_corners=True)
    total, count = depth.new_zeros(()), 0
    valid_mass = depth.new_zeros(())
    total_mass = depth.new_zeros(())
    for target in range(views):
        if target == reference_view:
            continue
        cam = torch.einsum("bij,bnj->bni", w2c[:, target], world)
        z = cam[..., 2].clamp_min(1e-6)
        pixels = torch.einsum("bij,bnj->bni", K[:, target], cam[..., :3])
        u, v = pixels[..., 0] / z, pixels[..., 1] / z
        grid = torch.stack((2 * u / max(grid_w - 1, 1) - 1,
                            2 * v / max(grid_h - 1, 1) - 1), dim=-1).reshape(batch, grid_h, grid_w, 2)
        warped = F.grid_sample(images[:, target], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        valid = ((u >= 0) & (u <= grid_w - 1) & (v >= 0) & (v <= grid_h - 1) & (cam[..., 2] > 0))
        valid = valid.reshape(batch, 1, grid_h, grid_w).to(source_rgb.dtype)
        total = total + ((source_rgb - warped).abs() * valid).sum() / valid.sum().clamp_min(1)
        valid_mass = valid_mass + valid.sum()
        total_mass = total_mass + valid.numel()
        count += 1
    loss = total / max(count, 1)
    if return_stats:
        return loss, {"valid_fraction": valid_mass / total_mass.clamp_min(1)}
    return loss
