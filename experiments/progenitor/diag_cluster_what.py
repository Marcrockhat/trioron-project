"""What pushes the k-means clusters on dense+stereo? sizes; NMI vs fine/superclass; per-cluster
means of image factors (luminance, contrast, HF energy, dominant orientation, saturation);
variance of each factor explained by cluster id (eta^2)."""
import os,sys,math,torch
HERE=os.path.dirname(os.path.abspath(__file__))
__file__=os.path.join(HERE,"diag_cluster_purity.py"); exec(open(__file__).read().split('print("k-means on train')[0])
Z=batched(feats["dense + stereo pooled 25"],Xtr); Z=(Z-Z.mean(0))/(Z.std(0)+1e-6); U,S,V=torch.pca_lowrank(Z,q=64,center=False); Zp=Z@V
a=kmeans(Zp,25); sizes=torch.bincount(a,minlength=25)
print("cluster sizes (sorted):",sizes.sort(descending=True).values.tolist())
def nmi(a,y,k,C):
    J=torch.zeros(k,C).index_put_((a,y),torch.ones(len(a)),accumulate=True); p=J/J.sum(); pa=p.sum(1,keepdim=True); py=p.sum(0,keepdim=True)
    mi=(p*torch.log((p+1e-12)/(pa*py+1e-12))).sum(); ha=-(pa*torch.log(pa+1e-12)).sum(); hy=-(py*torch.log(py+1e-12)).sum(); return (2*mi/(ha+hy)).item()
sup=f2c[ytr]; sup=sup-sup.min()
print(f"NMI vs fine(25)={nmi(a,y,25,25):.3f}  vs superclass(5)={nmi(a,sup,25,5):.3f}")
# image factors
Yl=Y(Xtr); lum=Yl.mean((1,2)); con=Yl.std((1,2))
hp=Yl-torch.nn.functional.avg_pool2d(Yl[:,None],5,1,2)[:,0]; hf=(hp**2).mean((1,2)).sqrt()/con.clamp(min=1e-3)   # relative HF energy
gx=Yl[:,:,1:]-Yl[:,:,:-1]; gy=Yl[:,1:,:]-Yl[:,:-1,:]; gx=gx[:,:-1,:]; gy=gy[:,:,:-1]
ori=torch.atan2((2*gx*gy).sum((1,2)),((gx**2-gy**2)).sum((1,2)))/2      # dominant orientation (structure tensor), radians
aniso=torch.sqrt(((gx**2-gy**2)).sum((1,2))**2+((2*gx*gy).sum((1,2)))**2)/((gx**2+gy**2).sum((1,2))+1e-6)   # 0 isotropic..1 oriented
mx,_=Xtr.max(1); mn,_=Xtr.min(1); sat=((mx-mn)/(mx+1e-3)).mean((1,2))
vert=(gy**2).mean((1,2))/((gx**2).mean((1,2))+1e-6)   # vertical/horizontal gradient energy ratio
factors={"luminance":lum,"contrast":con,"rel HF energy":hf,"anisotropy":aniso,"V/H gradient ratio":vert.log(),"saturation":sat,"cos2ori":torch.cos(2*ori)*aniso,"sin2ori":torch.sin(2*ori)*aniso}
print("eta^2 (fraction of factor variance explained by cluster id; fine class as reference):")
def eta2(v,a,k):
    m=torch.zeros(k).index_add_(0,a,v)/torch.bincount(a,minlength=k).clamp(min=1).float(); return (((m[a]-v.mean())**2).sum()/((v-v.mean())**2).sum()).item()
for n,v in factors.items(): print(f"  {n:>20s}: cluster {eta2(v,a,25):.3f}   fine-class {eta2(v,y,25):.3f}")
# top-5 clusters by size: composition
names={0:"aquatic mammals",1:"fish",2:"flowers",3:"food containers",4:"fruit+veg"}
J=torch.zeros(25,25).index_put_((a,y),torch.ones(len(a)),accumulate=True)
for c in sizes.argsort(descending=True)[:6].tolist():
    row=J[c]; top=row.topk(3); supd=torch.zeros(5).index_add_(0,sup[a==c],torch.ones(int((a==c).sum())))/max(1,int((a==c).sum()))
    print(f"  cluster {c:2d} n={int(sizes[c]):4d}: top fine {[(int(i),round(float(row[i]/row.sum()),2)) for i in top.indices]}  super {[round(float(x),2) for x in supd]}  lum={lum[a==c].mean():.2f} con={con[a==c].mean():.2f} hf={hf[a==c].mean():.2f} aniso={aniso[a==c].mean():.2f} V/H={vert[a==c].mean():.2f}")
