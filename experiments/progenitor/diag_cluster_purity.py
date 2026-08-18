"""Would Phasecyte templates be class-separable on the new front end? Proxy: k-means on
standardized per-image features; purity / NMI / max-purity vs the 25 probe classes.
Features: eye DoG (s051 Phasecyte input), (b), dense 2-D pooled 25, dense+stereo (800)."""
import os,sys,math,torch
HERE=os.path.dirname(os.path.abspath(__file__))
__file__=os.path.join(HERE,"diag_stereo.py"); exec(open(__file__).read().split("probes=")[0])
eye=B.Eye((16,16),rectify=False)
feats={"eye DoG signed (s051 Phasecyte input)":lambda X: eye.sense(X),"(b) 12px/5 cepstra":cep_spectrogram,
       "dense 2-D pooled 25":dense_pooled,"dense + stereo pooled 25":lambda X: torch.cat([dense_pooled(X),sync_pooled(X)],1),
       "raw pixels":lambda X: X.flatten(1)}
lut=torch.full((100,),-1,dtype=torch.long); lut[torch.tensor(cls)]=torch.arange(len(cls)); y=lut[ytr]; C=25
def kmeans(Z,k,iters=30,seed=0):
    g=torch.Generator().manual_seed(seed); Cc=Z[torch.randperm(len(Z),generator=g)[:k]].clone()
    for _ in range(iters):
        a=torch.cdist(Z,Cc).argmin(1)
        for j in range(k):
            m=a==j
            if m.any(): Cc[j]=Z[m].mean(0)
    return torch.cdist(Z,Cc).argmin(1)
def scores(a,y,k):
    J=torch.zeros(k,C).index_put_((a,y),torch.ones(len(a)),accumulate=True)
    purity=J.max(1).values.sum()/len(a); frac=J/J.sum(1,keepdim=True).clamp(min=1); best=frac.max(1).values
    n_ge34=int((best>=0.34).sum()); 
    p=J/J.sum(); pa=p.sum(1,keepdim=True); py=p.sum(0,keepdim=True)
    mi=(p*torch.log((p+1e-12)/(pa*py+1e-12))).sum(); ha=-(pa*torch.log(pa+1e-12)).sum(); hy=-(py*torch.log(py+1e-12)).sum()
    return purity.item(), (2*mi/(ha+hy)).item(), n_ge34, best.max().item()
print("k-means on train (12.5K imgs, 25 classes); purity chance ~0.04+; NMI; clusters with best-class >=34%; max best-class frac")
for name,fn in feats.items():
    Z=batched(fn,Xtr); Z=(Z-Z.mean(0))/(Z.std(0)+1e-6)
    # PCA to 64 for stable k-means on wide features
    U,S,V=torch.pca_lowrank(Z,q=64,center=False); Zp=Z@V
    for k in (25,50,100):
        a=kmeans(Zp,k); pu,nmi,n34,mx=scores(a,y,k)
        print(f"  {name:>38s} d={Z.shape[1]:4d} k={k:3d}: purity={pu:.3f} NMI={nmi:.3f} clusters>=34%={n34:3d}/{k} max={mx:.2f}",flush=True)
