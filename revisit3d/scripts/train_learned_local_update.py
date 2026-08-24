#!/usr/bin/env python3
"""Oracle A→B→A' training of a local, track-conditioned first-order update rule."""
from __future__ import annotations
import argparse, json
from itertools import cycle
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from revisit3d.backbones import FrozenVGGTFeatures, FrozenVGGTGeometryTracker
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.losses import depth_smoothness_loss, track_3d_consistency_loss
from revisit3d.models import CompactTTTState, LocalTrackUpdateRule, build_geometry_head
from revisit3d.scripts.diagnose_track_ttt_signal import query_grid
from revisit3d.scripts.train_oracle_revisit import depth_grid, to_device

def prepare(extractor, tracker, segment, side):
    images=segment['context']['rgb']; return extractor(images), tracker(images, query_grid(images.shape[-2],images.shape[-1],side,'cuda'))
def loss(pred, images, prior, smooth=0.):
    d=depth_grid(pred); return track_3d_consistency_loss(d,prior['intrinsics'],prior['w2c'],prior['track'],prior['visibility'],prior['confidence'],image_size=images.shape[-2:])+smooth*depth_smoothness_loss(d,images)
def update(head, rule, features, prior, images, state, lr, smooth):
    z=state.value.requires_grad_(True); pred=head(features,CompactTTTState(z)); inner=loss(pred,images,prior,smooth)
    grad,=torch.autograd.grad(inner,z); a=pred['slot_assignment'].detach(); f=features.detach()
    slot=(torch.einsum('bvpk,bvpf->bkf',a,f)/a.sum((1,2)).unsqueeze(-1).clamp_min(1e-6))
    return CompactTTTState(z+rule(z,grad.detach(),slot)), inner.detach()
def main():
 p=argparse.ArgumentParser();p.add_argument('--head-checkpoint',required=True);p.add_argument('--manifest',default='revisit3d/manifests/nuscenes_revisit_dev.json');p.add_argument('--scene-root',default='tttLRM/data_example/nuscenes_2x2');p.add_argument('--vggt-checkpoint',default='FastVGGT/ckpt/model_tracker_fixed_e20.pt');p.add_argument('--steps',type=int,default=20);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--ttt-lr',type=float,default=1e-2);p.add_argument('--track-side',type=int,default=8);p.add_argument('--smoothness',type=float,default=1e-3);p.add_argument('--margin',type=float,default=1e-3);p.add_argument('--out',required=True);args=p.parse_args()
 if not torch.cuda.is_available(): raise SystemExit('CUDA required')
 ds=RevisitEpisodeDataset(args.manifest,args.scene_root,split='train',image_size=(224,224)); loader=DataLoader(ds,batch_size=None,shuffle=True)
 ex=FrozenVGGTFeatures(args.vggt_checkpoint,repo_root='FastVGGT').cuda(); tr=FrozenVGGTGeometryTracker(args.vggt_checkpoint,repo_root='FastVGGT').cuda(); ck=torch.load(args.head_checkpoint,map_location='cuda',weights_only=False)
 if ck.get('head_type')!='anchored': raise SystemExit('requires anchored checkpoint')
 head=build_geometry_head('anchored',ex.feature_dim).cuda();head.load_state_dict(ck['head']);head.eval().requires_grad_(False); rule=LocalTrackUpdateRule(ex.feature_dim).cuda();opt=torch.optim.AdamW(rule.parameters(),lr=args.lr)
 rows=[]
 # ``DataLoader`` is finite; meta-training steps may span several passes over
 # the compact development split.
 for step,s in zip(range(args.steps),cycle(loader)):
  a,b,ap=(to_device(s[k],'cuda') for k in ('a','b','a_prime')); scenes={a['scene'],ap['scene']}; fs=next(x for x in ds if not scenes.intersection({x['a']['scene'],x['a_prime']['scene']})); fa=to_device(fs['a'],'cuda')
  A,PA=prepare(ex,tr,a,args.track_side);B,PB=prepare(ex,tr,b,args.track_side);P,PP=prepare(ex,tr,ap,args.track_side);X,PX=prepare(ex,tr,fa,args.track_side);z0=head.initial_state(1,device='cuda',dtype=A.dtype)
  za,_=update(head,rule,A,PA,a['context']['rgb'],z0,args.ttt_lr,args.smoothness);zab,_=update(head,rule,B,PB,b['context']['rgb'],za,args.ttt_lr,args.smoothness);zb,_=update(head,rule,B,PB,b['context']['rgb'],z0,args.ttt_lr,args.smoothness);zf,_=update(head,rule,X,PX,fa['context']['rgb'],z0,args.ttt_lr,args.smoothness)
  cur,_=update(head,rule,P,PP,ap['context']['rgb'],zab,args.ttt_lr,args.smoothness);mat,_=update(head,rule,P,PP,ap['context']['rgb'],CompactTTTState(zab.value+za.value-z0.value),args.ttt_lr,args.smoothness);inter,_=update(head,rule,P,PP,ap['context']['rgb'],CompactTTTState(zab.value+zb.value-z0.value),args.ttt_lr,args.smoothness);foreign,_=update(head,rule,P,PP,ap['context']['rgb'],CompactTTTState(zab.value+zf.value-z0.value),args.ttt_lr,args.smoothness)
  qi=ap['query']['rgb'];Q=ex(qi);QP=tr(qi,query_grid(qi.shape[-2],qi.shape[-1],args.track_side,'cuda'));q=lambda z:loss(head(Q,z),qi,QP); vals={'current':q(cur),'matched':q(mat),'intervening':q(inter),'foreign':q(foreign)};contrast=F.relu(args.margin-(vals['intervening']-vals['matched']))+F.relu(args.margin-(vals['foreign']-vals['matched']));outer=vals['matched']+F.softplus(vals['matched']-vals['current'])+contrast
  opt.zero_grad();outer.backward();torch.nn.utils.clip_grad_norm_(rule.parameters(),1.);opt.step(); row={'step':step,'outer':float(outer.detach()),**{k:float(v.detach()) for k,v in vals.items()}};rows.append(row);print(json.dumps(row))
 out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);torch.save({'head':head.state_dict(),'head_type':'anchored','rule':rule.state_dict(),'records':rows},out);print(f'wrote {out}')
if __name__=='__main__':main()
