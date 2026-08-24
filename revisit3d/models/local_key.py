from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
class LocalTokenKey(nn.Module):
 def __init__(self,d,key=128): super().__init__();self.p=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,key))
 def forward(self,x): return F.normalize(self.p(x),dim=-1)
 def score(self,a,b,temp=.1):
  sim=a@b.T; return .5*(temp*torch.logsumexp(sim/temp,1).mean()+temp*torch.logsumexp(sim/temp,0).mean())
