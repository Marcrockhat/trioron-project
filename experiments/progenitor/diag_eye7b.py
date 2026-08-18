"""Follow-up to diag_eye7: (1) corrected standardized-cosine kNN baselines on all
test sets; (2) shift by pad+crop (no wrap seam) and soft-edged fragment;
(3) where does the class-vote offset land for shifted images (should be (2,2))."""
import os, sys, torch
import torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE,"diag_eye7.py")).read().split('if __name__')[0].replace('os.path.abspath(__file__)','os.path.abspath(%r)'%os.path.join(HERE,"diag_eye7.py")))
torch.set_num_threads(6)
lut=torch.full((100,),-1,dtype=torch.long); lut[torch.tensor(cls)]=torch.arange(len(cls)); C=25
ya=lut[ytr]; yb=lut[yte]
def knn2(Xa,ya,Xb,k):
    mu,sd=Xa.mean(0),Xa.std(0)+1e-6; Xa=F.normalize((Xa-mu)/sd,dim=1); Xb=F.normalize((Xb-mu)/sd,dim=1); out=[]
    for i in range(0,len(Xb),500):
        top=(Xb[i:i+500]@Xa.T).topk(k,dim=1).indices
        out.append(torch.stack([torch.bincount(ya[t],minlength=C).argmax() for t in top]))
    return torch.cat(out)
def shift_pad(X,d=4):   # translate content by (d,d); fill vacated border with per-image mean (no wrap seam)
    m=X.mean((2,3),keepdim=True); Y_=m.expand_as(X).clone(); Y_[:,:,d:,d:]=X[:,:,:32-d,:32-d]; return Y_
def fragment_soft(X):   # 20x20 crop, soft 4-px feathered edge into per-image mean
    r=torch.arange(32).float(); w=lambda a,b: (torch.clamp((r-a)/4,0,1)*torch.clamp((b-r)/4,0,1))
    M=(w(6,26)[:,None]*w(4,24)[None,:])[None,None]; return X*M+(1-M)*X.mean((2,3),keepdim=True)
tests={"clean":Xte,"2x-blur":Xte_blur,"shift4_roll":shift(Xte),"shift4_pad":shift_pad(Xte),"fragment20_hard":fragment(Xte),"fragment20_soft":fragment_soft(Xte)}
CB_tr=batched(cep_spectrogram,Xtr)
for name,Xq in tests.items():
    r1=(knn2(Xtr.flatten(1),ya,Xq.flatten(1),1)==yb).float().mean().item(); r5=(knn2(Xtr.flatten(1),ya,Xq.flatten(1),5)==yb).float().mean().item()
    c1=(knn2(CB_tr,ya,batched(cep_spectrogram,Xq),1)==yb).float().mean().item(); c5=(knn2(CB_tr,ya,batched(cep_spectrogram,Xq),5)==yb).float().mean().item()
    print(f"[{name:>15s}] kNN(std-cos): raw k1={r1:.4f} k5={r5:.4f} | cep(b) k1={c1:.4f} k5={c5:.4f}",flush=True)
idx=build_index(Xtr)
for name in ("shift4_pad","fragment20_soft"):
    pr=query(idx,tests[name],ya,C); print(f"[{name:>15s}] shazam: "+" ".join(f"{k}={((v==yb).float().mean().item()):.4f}" for k,v in pr.items()),flush=True)
# (3) offset diagnostic: for 200 shifted (pad) test images, argmax offset of the TRUE class's vote
S=spectrogram(tests["shift4_pad"][:200]); r,c,f,ok=peaks(S); h,ar,ac,v=hashes(r,c,f,ok)
offs=[]
for q in range(200):
    hq=h[q][v[q]]; rq=ar[q][v[q]]; cq=ac[q][v[q]]; s=idx["start"][hq]; e=idx["end"][hq]; cnt=e-s
    rep=torch.repeat_interleave(torch.arange(len(hq)),cnt)
    pos=torch.repeat_interleave(s,cnt)+(torch.arange(int(cnt.sum()))-torch.repeat_interleave(torch.cumsum(cnt,0)-cnt,cnt))
    off=(idx["ar"][pos]-rq[rep]+NPOS-1)*OFF+(idx["ac"][pos]-cq[rep]+NPOS-1); cl=ya[idx["img"][pos]]
    vc=torch.bincount(cl*OFF*OFF+off,minlength=C*OFF*OFF).view(C,-1); o=vc[yb[q]].argmax(); offs.append((o//OFF-NPOS+1, o%OFF-NPOS+1))
offs=torch.tensor(offs); print("true-class argmax offset for shift4_pad (expect (-2,-2) or (2,2) in window steps): mode counts",
      torch.unique(offs,dim=0,return_counts=True)[1].max().item(),"/200; mean",offs.float().mean(0).tolist())
S=spectrogram(Xte[:200]); r,c,f,ok=peaks(S); h,ar,ac,v=hashes(r,c,f,ok); offs=[]
for q in range(200):
    hq=h[q][v[q]]; rq=ar[q][v[q]]; cq=ac[q][v[q]]; s=idx["start"][hq]; e=idx["end"][hq]; cnt=e-s
    rep=torch.repeat_interleave(torch.arange(len(hq)),cnt)
    pos=torch.repeat_interleave(s,cnt)+(torch.arange(int(cnt.sum()))-torch.repeat_interleave(torch.cumsum(cnt,0)-cnt,cnt))
    off=(idx["ar"][pos]-rq[rep]+NPOS-1)*OFF+(idx["ac"][pos]-cq[rep]+NPOS-1); cl=ya[idx["img"][pos]]
    vc=torch.bincount(cl*OFF*OFF+off,minlength=C*OFF*OFF).view(C,-1); o=vc[yb[q]].argmax(); offs.append((o//OFF-NPOS+1, o%OFF-NPOS+1))
offs=torch.tensor(offs); print("true-class argmax offset for CLEAN (expect (0,0)): mode counts",torch.unique(offs,dim=0,return_counts=True)[1].max().item(),"/200; mean",offs.float().mean(0).tolist())
