#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,torch
from torch.nn import functional as F
from revisit3d.backbones import FrozenVGGTFeatures,FrozenVGGTGeometryTracker
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.models import LocalTrackUpdateRule,build_geometry_head
from revisit3d.models.local_key import LocalTokenKey
from revisit3d.scripts.train_oracle_revisit import to_device
from revisit3d.scripts.train_learned_local_update import prepare,update
def keytok(ex,m,s):
 f=ex(s['context']['rgb'])[0];return m(f[torch.linspace(0,f.shape[0]-1,4).long()][:,::4].reshape(-1,f.shape[-1]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--update-checkpoint',required=True);p.add_argument('--key-checkpoint',required=True);p.add_argument('--manifest',default='revisit3d/manifests/nuscenes_revisit_dev.json');p.add_argument('--scene-root',default='tttLRM/data_example/nuscenes_2x2');p.add_argument('--vggt-checkpoint',default='FastVGGT/ckpt/model_tracker_fixed_e20.pt');p.add_argument('--k',type=int,default=5);p.add_argument('--out',required=True);a=p.parse_args();ds=RevisitEpisodeDataset(a.manifest,a.scene_root,split='val',image_size=(224,224));ex=FrozenVGGTFeatures(a.vggt_checkpoint,repo_root='FastVGGT').cuda();tr=FrozenVGGTGeometryTracker(a.vggt_checkpoint,repo_root='FastVGGT').cuda();u=torch.load(a.update_checkpoint,map_location='cuda',weights_only=False);h=build_geometry_head('anchored',ex.feature_dim).cuda();h.load_state_dict(u['head']);h.eval().requires_grad_(False);r=LocalTrackUpdateRule(ex.feature_dim).cuda();r.load_state_dict(u['rule']);r.eval();k=LocalTokenKey(ex.feature_dim).cuda();k.load_state_dict(torch.load(a.key_checkpoint,map_location='cuda',weights_only=False)['key']);k.eval();mem=[];q=[]
 with torch.enable_grad():
  for s in ds:
   A,P=(to_device(s[x],'cuda') for x in ('a','a_prime'));fa,pa=prepare(ex,tr,A,8);fp,pp=prepare(ex,tr,P,8);z=h.initial_state(1,device='cuda',dtype=fa.dtype);za,_=update(h,r,fa,pa,A['context']['rgb'],z,.01,.001);zp,_=update(h,r,fp,pp,P['context']['rgb'],z,.01,.001);mem.append((keytok(ex,k,A).detach(),(za.value-z.value).detach().flatten()));q.append((keytok(ex,k,P).detach(),(zp.value-z.value).detach().flatten()))
 rows=[]
 for i,(qk,qu) in enumerate(q):
  score=torch.tensor([float(k.score(qk,mk)) for mk,_ in mem]);allcompat=torch.tensor([float(F.cosine_similarity(qu,mu,dim=0)) for _,mu in mem]);cand=score.topk(a.k).indices;compat=allcompat[cand];chosen=int(cand[compat.argmax()]);rank=int((score.argsort(descending=True)==i).nonzero()[0])+1;crank=int((allcompat.argsort(descending=True)==i).nonzero()[0])+1;rows.append({'key_rank':rank,'compat_rank':crank,'positive_compat':float(allcompat[i]),'mean_negative_compat':float((allcompat.sum()-allcompat[i])/(len(allcompat)-1)),'rerank_correct':chosen==i})
 upd=torch.stack([x[1] for x in mem]).float().cpu();sv=torch.linalg.svdvals(upd-upd.mean(0,keepdim=True));energy=(sv.square()/sv.square().sum()).tolist()
 out={'summary':{'key_recall_at_k':sum(x['key_rank']<=a.k for x in rows)/len(rows),'rerank_top1':sum(x['rerank_correct'] for x in rows)/len(rows),'compat_top1':sum(x['compat_rank']==1 for x in rows)/len(rows),'mean_positive_compat':sum(x['positive_compat'] for x in rows)/len(rows),'mean_negative_compat':sum(x['mean_negative_compat'] for x in rows)/len(rows),'update_centered_energy_top1':energy[0],'update_centered_energy_top3':sum(energy[:3]),'mean_update_norm':float(upd.norm(dim=1).mean()),'std_update_norm':float(upd.norm(dim=1).std())},'rows':rows};json.dump(out,open(a.out,'w'),indent=2);print(json.dumps(out['summary']))
if __name__=='__main__':main()
