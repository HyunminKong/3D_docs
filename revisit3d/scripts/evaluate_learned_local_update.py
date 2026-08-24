#!/usr/bin/env python3
"""Held-out oracle evaluation for the learned local track update rule."""
from __future__ import annotations
import argparse,json
import torch
from revisit3d.backbones import FrozenVGGTFeatures,FrozenVGGTGeometryTracker
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.models import CompactTTTState,LocalTrackUpdateRule,build_geometry_head
from revisit3d.scripts.diagnose_track_ttt_signal import query_grid
from revisit3d.scripts.train_oracle_revisit import to_device
from revisit3d.scripts.train_learned_local_update import prepare,loss,update
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',default='revisit3d/manifests/nuscenes_revisit_dev.json');p.add_argument('--scene-root',default='tttLRM/data_example/nuscenes_2x2');p.add_argument('--vggt-checkpoint',default='FastVGGT/ckpt/model_tracker_fixed_e20.pt');p.add_argument('--split',choices=('val','test'),default='val');p.add_argument('--track-side',type=int,default=8);p.add_argument('--ttt-lr',type=float,default=1e-2);p.add_argument('--smoothness',type=float,default=1e-3);p.add_argument('--seed',type=int,default=20260824);p.add_argument('--out',required=True);a=p.parse_args()
 ds=RevisitEpisodeDataset(a.manifest,a.scene_root,split=a.split,image_size=(224,224));train=RevisitEpisodeDataset(a.manifest,a.scene_root,split='train',image_size=(224,224));ex=FrozenVGGTFeatures(a.vggt_checkpoint,repo_root='FastVGGT').cuda();tr=FrozenVGGTGeometryTracker(a.vggt_checkpoint,repo_root='FastVGGT').cuda();ck=torch.load(a.checkpoint,map_location='cuda',weights_only=False);head=build_geometry_head('anchored',ex.feature_dim).cuda();head.load_state_dict(ck['head']);head.eval().requires_grad_(False);rule=LocalTrackUpdateRule(ex.feature_dim).cuda();rule.load_state_dict(ck['rule']);rule.eval();gen=torch.Generator(device='cuda').manual_seed(a.seed);rows=[]
 for s in ds:
  A,B,P=(to_device(s[k],'cuda') for k in ('a','b','a_prime'));scenes={A['scene'],P['scene']};fs=next(x for x in train if not scenes.intersection({x['a']['scene'],x['a_prime']['scene']}));X=to_device(fs['a'],'cuda')
  with torch.enable_grad():
   fa,pa=prepare(ex,tr,A,a.track_side);fb,pb=prepare(ex,tr,B,a.track_side);fp,pp=prepare(ex,tr,P,a.track_side);fx,px=prepare(ex,tr,X,a.track_side);z0=head.initial_state(1,device='cuda',dtype=fa.dtype);za,_=update(head,rule,fa,pa,A['context']['rgb'],z0,a.ttt_lr,a.smoothness);zab,_=update(head,rule,fb,pb,B['context']['rgb'],za,a.ttt_lr,a.smoothness);zb,_=update(head,rule,fb,pb,B['context']['rgb'],z0,a.ttt_lr,a.smoothness);zf,_=update(head,rule,fx,px,X['context']['rgb'],z0,a.ttt_lr,a.smoothness)
   cur,_=update(head,rule,fp,pp,P['context']['rgb'],zab,a.ttt_lr,a.smoothness);mat,_=update(head,rule,fp,pp,P['context']['rgb'],CompactTTTState(zab.value+za.value-z0.value),a.ttt_lr,a.smoothness);inter,_=update(head,rule,fp,pp,P['context']['rgb'],CompactTTTState(zab.value+zb.value-z0.value),a.ttt_lr,a.smoothness);foreign,_=update(head,rule,fp,pp,P['context']['rgb'],CompactTTTState(zab.value+zf.value-z0.value),a.ttt_lr,a.smoothness);rnd=torch.randn(za.value.shape,device='cuda',generator=gen);rnd=rnd/rnd.norm(dim=-1,keepdim=True).clamp_min(1e-8)*(za.value-z0.value).norm(dim=-1,keepdim=True);random,_=update(head,rule,fp,pp,P['context']['rgb'],CompactTTTState(zab.value+rnd),a.ttt_lr,a.smoothness)
   qi=P['query']['rgb'];qf=ex(qi);qp=tr(qi,query_grid(qi.shape[-2],qi.shape[-1],a.track_side,'cuda'));q=lambda z:loss(head(qf,z),qi,qp);v={k:q(z) for k,z in {'current':cur,'matched':mat,'intervening':inter,'foreign':foreign,'random':random}.items()}
  r={'episode':s['episode_id'],**{k:float(x.detach()) for k,x in v.items()}};[r.__setitem__('matched_minus_'+k,r['matched']-r[k]) for k in ('current','intervening','foreign','random')];rows.append(r);print(json.dumps(r))
 keys=('current','matched','intervening','foreign','random','matched_minus_current','matched_minus_intervening','matched_minus_foreign','matched_minus_random');out={'checkpoint':a.checkpoint,'split':a.split,'rows':rows,'summary':{k:sum(r[k] for r in rows)/len(rows) for k in keys}};json.dump(out,open(a.out,'w'),indent=2);print(json.dumps(out['summary']))
if __name__=='__main__':main()
