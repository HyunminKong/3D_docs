#!/usr/bin/env python3
from __future__ import annotations
import argparse,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from revisit3d.backbones import FrozenVGGTFeatures
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.models.local_key import LocalTokenKey
from revisit3d.scripts.train_oracle_revisit import to_device
def tok(ex,s):
 f=ex(s['context']['rgb'])[0];return f[torch.linspace(0,f.shape[0]-1,min(4,f.shape[0])).long()][:,::4].reshape(-1,f.shape[-1])
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default='revisit3d/manifests/nuscenes_revisit_dev.json');p.add_argument('--scene-root',default='tttLRM/data_example/nuscenes_2x2');p.add_argument('--vggt-checkpoint',default='FastVGGT/ckpt/model_tracker_fixed_e20.pt');p.add_argument('--steps',type=int,default=20);p.add_argument('--out',required=True);a=p.parse_args();ds=RevisitEpisodeDataset(a.manifest,a.scene_root,split='train',image_size=(224,224));ex=FrozenVGGTFeatures(a.vggt_checkpoint,repo_root='FastVGGT').cuda();m=LocalTokenKey(ex.feature_dim).cuda();o=torch.optim.AdamW(m.parameters(),lr=1e-3)
 for _,s in zip(range(a.steps),DataLoader(ds,batch_size=None,shuffle=True)):
  A,B,P=(to_device(s[k],'cuda') for k in ('a','b','a_prime'));sc={A['scene'],P['scene']};X=to_device(next(x for x in ds if not sc.intersection({x['a']['scene'],x['a_prime']['scene']}))['a'],'cuda');ka,kp,kb,kx=(m(tok(ex,z)) for z in (A,P,B,X));v=torch.stack((m.score(ka,kp),m.score(ka,kb),m.score(ka,kx))).unsqueeze(0);l=F.cross_entropy(v,torch.zeros(1,dtype=torch.long,device='cuda'));o.zero_grad();l.backward();o.step()
 torch.save({'key':m.state_dict()},a.out)
if __name__=='__main__':main()
