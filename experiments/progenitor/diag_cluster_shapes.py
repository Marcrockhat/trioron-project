"""Do the layer-1 texture-style clusters correspond to simple geometry? Render 32x32 synthetic
circle/triangle/square/polkadots/stripes (grey, random size/pose/contrast), push through
dense+stereo, assign to the 25 CIFAR k-means clusters: per-shape cluster distribution
(concentration) + NMI(shape, cluster); and k-means on the shapes alone (NMI vs shape)."""
import os,sys,math,torch
import torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
__file__=os.path.join(HERE,"diag_cluster_purity.py"); exec(open(__file__).read().split('print("k-means on train')[0])
torch.set_num_threads(4)
PR=["circle","triangle","square","polkadots","stripes"]
def synth32(n,gen):
    S=32; yy,xx=torch.meshgrid(torch.arange(S).float(),torch.arange(S).float(),indexing="ij")
    X=torch.zeros(n,3,S,S); y=torch.randint(0,5,(n,),generator=gen)
    for i in range(n):
        k=int(y[i]); cx,cy=(torch.rand(2,generator=gen)*8+12).tolist(); r=float(torch.rand(1,generator=gen)*6+6)
        th=float(torch.rand(1,generator=gen)*math.pi); dx,dy=xx-cx,yy-cy
        u=dx*math.cos(th)+dy*math.sin(th); v=-dx*math.sin(th)+dy*math.cos(th)
        if k==0: m=(dx**2+dy**2<=r*r)
        elif k==1: m=(v>-r*0.6)&(u.abs()<(r*1.2-v)*0.6)
        elif k==2: m=(u.abs()<=r)&(v.abs()<=r)
        elif k==3: p=float(torch.rand(1,generator=gen)*6+5); m=(((u%p)-p/2)**2+((v%p)-p/2)**2)<=(p*0.28)**2
        else: p=float(torch.rand(1,generator=gen)*8+4); m=torch.sin(2*math.pi*u/p)>0
        bg=float(torch.rand(1,generator=gen)); fg=float(torch.rand(1,generator=gen))
        img=torch.where(m,torch.tensor(fg),torch.tensor(bg))+0.03*torch.randn(S,S,generator=gen)
        if float(torch.rand(1,generator=gen))<0.5: img=F.avg_pool2d(img[None,None],3,1,1)[0,0]
        X[i]=img.clamp(0,1)
    return X,y
gen=torch.Generator().manual_seed(5); Xs,ys=synth32(5000,gen)
fn=feats["dense + stereo pooled 25"]
Ztr=batched(fn,Xtr); mu,sd=Ztr.mean(0),Ztr.std(0)+1e-6; Ztr=(Ztr-mu)/sd
U,S_,V=torch.pca_lowrank(Ztr,q=64,center=False); Ptr=Ztr@V
def kmeans_c(Z,k,iters=30,seed=0):
    g=torch.Generator().manual_seed(seed); Cc=Z[torch.randperm(len(Z),generator=g)[:k]].clone()
    for _ in range(iters):
        a=torch.cdist(Z,Cc).argmin(1)
        for j in range(k):
            m=a==j
            if m.any(): Cc[j]=Z[m].mean(0)
    return Cc
Cc=kmeans_c(Ptr,25)
Zs=(batched(fn,Xs)-mu)/sd; Ps=Zs@V; a=torch.cdist(Ps,Cc).argmin(1)
def nmi(a,y,k,C):
    J=torch.zeros(k,C).index_put_((a,y),torch.ones(len(a)),accumulate=True); p=J/J.sum(); pa=p.sum(1,keepdim=True); py=p.sum(0,keepdim=True)
    mi=(p*torch.log((p+1e-12)/(pa*py+1e-12))).sum(); ha=-(pa*torch.log(pa+1e-12)).sum(); hy=-(py*torch.log(py+1e-12)).sum(); return (2*mi/(ha+hy)).item()
print(f"synthetic shapes -> 25 CIFAR style clusters: NMI(shape,cluster)={nmi(a,ys,25,5):.3f}  (CIFAR fine-class NMI was 0.111)")
J=torch.zeros(5,25).index_put_((ys,a),torch.ones(len(a)),accumulate=True)
for s in range(5):
    row=J[s]/J[s].sum(); top=row.topk(3); H=-(row*torch.log2(row.clamp(min=1e-12))).sum()
    print(f"  {PR[s]:>10s}: top clusters {[(int(i),round(float(row[i]),2)) for i in top.indices]}  entropy {H:.2f} bits (uniform 4.64)")
# how separable are shapes from each other in this space, unsupervised?
Cs=kmeans_c(Ps,5); a5=torch.cdist(Ps,Cs).argmin(1); print(f"k-means(k=5) on shapes alone: NMI(shape)={nmi(a5,ys,5,5):.3f}; purity={(torch.zeros(5,5).index_put_((a5,ys),torch.ones(len(a5)),accumulate=True).max(1).values.sum()/len(a5)):.3f}")
# and supervised: a linear probe on shapes (upper bound of what the features carry)
sub=B.train_sub(Zs[:4000],ys[:4000],5,hidden=48,seed=1,epochs=8,tag="shape probe"); pred=B.logits_of(sub,Zs[4000:]).argmax(1)
print(f"supervised leaf on dense+stereo, 5 shapes: acc={(pred==ys[4000:]).float().mean():.3f} (s051 window detector capped 0.72)")
J2=torch.zeros(5,5).index_put_((ys[4000:],pred),torch.ones(1000),accumulate=True); print("confusion rows=true", [[int(x) for x in r] for r in J2])
