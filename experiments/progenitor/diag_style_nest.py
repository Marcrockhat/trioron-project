"""Texture-style bands as layer 1 of a nest (Rocky, s052): k-means clusters (unsupervised, on
dense+stereo 800-d) = band router; one supervised leaf per band re-classifies within its band.
Hard route (nearest centroid) and soft route (softmax over -dist/T, mix logits). vs single leaf."""
import os,sys,math,torch
import torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
__file__=os.path.join(HERE,"diag_cluster_purity.py"); exec(open(__file__).read().split('print("k-means on train')[0])
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","8")))
fn=feats["dense + stereo pooled 25"]
Ztr=batched(fn,Xtr); Zte=batched(fn,Xte); Zbl=batched(fn,Xte_blur)
mu,sd=Ztr.mean(0),Ztr.std(0)+1e-6; Ztr=(Ztr-mu)/sd; Zte=(Zte-mu)/sd; Zbl=(Zbl-mu)/sd
U,S,V=torch.pca_lowrank(Ztr,q=64,center=False); Ptr=Ztr@V; Pte=Zte@V; Pbl=Zbl@V
def kmeans_c(Z,k,iters=30,seed=0):
    g=torch.Generator().manual_seed(seed); Cc=Z[torch.randperm(len(Z),generator=g)[:k]].clone()
    for _ in range(iters):
        a=torch.cdist(Z,Cc).argmin(1)
        for j in range(k):
            m=a==j
            if m.any(): Cc[j]=Z[m].mean(0)
    return Cc
def acc(logits,yb): return B.acc_pair(logits,yb,f2c)
# single leaf reference
sub=B.train_sub(Ztr,ytr,100,hidden=48,seed=1,epochs=8,tag="single"); L1=B.logits_of(sub,Zte); Lb=B.logits_of(sub,Zbl)
print(f"single leaf 800-d: full={acc(L1,yte)[0]:.4f} task={acc(L1,yte)[1]:.4f} | blur {acc(Lb,yte)[0]:.4f}",flush=True)
for k in (5,10,25):
    Cc=kmeans_c(Ptr,k); atr=torch.cdist(Ptr,Cc).argmin(1)
    leaves=[]
    for j in range(k):
        m=atr==j
        leaves.append(B.train_sub(Ztr[m],ytr[m],100,hidden=48,seed=1,epochs=8,tag=f"k{k} band{j} n={int(m.sum())}"))
    for name,Zq,Pq in (("clean",Zte,Pte),("blur",Zbl,Pbl)):
        D=torch.cdist(Pq,Cc); aq=D.argmin(1)
        allL=torch.stack([B.logits_of(l,Zq) for l in leaves],1)          # [N,k,100]
        hard=allL[torch.arange(len(Zq)),aq]
        T=D.std(); w=torch.softmax(-D/(0.5*T),1)                          # soft route
        soft=(w[:,:,None]*allL).sum(1)
        mean=allL.mean(1)                                                 # uniform mix (no routing) control
        fh,th=acc(hard,yte); fs,ts=acc(soft,yte); fm,tm=acc(mean,yte)
        sizes=torch.bincount(atr,minlength=k)
        print(f"k={k:2d} [{name}] hard-route full={fh:.4f} task={th:.4f} | soft full={fs:.4f} task={ts:.4f} | uniform-mix full={fm:.4f} task={tm:.4f}   (band sizes min/max {int(sizes.min())}/{int(sizes.max())}, params ~{k*45}K)",flush=True)
print("done")
