from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class GeometryContextKey(nn.Module):
    """Compact key from frozen global feature and track-motion statistics."""
    def __init__(self, feature_dim:int, key_dim:int=128):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(feature_dim+6),nn.Linear(feature_dim+6,512),nn.GELU(),nn.Linear(512,key_dim))
    def forward(self, features, tracks):
        f=features.mean((1,2)); xy=tracks['track']; flow=(xy[:,1:]-xy[:,:1]).norm(dim=-1); w=tracks['visibility'][:,1:]*tracks['confidence'][:,1:]
        stats=torch.stack((flow.mean((1,2)),flow.std((1,2)),w.mean((1,2)),w.std((1,2)),(flow*w).sum((1,2))/w.sum((1,2)).clamp_min(1e-6),tracks['confidence'].mean((1,2))),-1)
        return F.normalize(self.net(torch.cat((f,stats),-1)),dim=-1)
