"""Speech recipe for form: cepstra (spectral SHAPE) in sliding windows.
(a) global cepstrum, (b) spatial spectrogram of cepstra (positions kept),
(c) max-pooled over positions, (d) onset window only (max DoG energy),
(e) eye DoG + (c). Same supervised leaf, 25 classes, + 2x-blur column."""
import os, sys, math, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.progenitor import cifar_eye_nest as B
torch.manual_seed(0)
Xtr,ytr=B.load_cifar100(True); Xte,yte=B.load_cifar100(False); f2c=B.fine_to_coarse()
T=5; cls=sum([(f2c==t).nonzero().squeeze(1).tolist() for t in range(T)],[])
mtr=torch.isin(ytr,torch.tensor(cls)); mte=torch.isin(yte,torch.tensor(cls))
Xtr,ytr,Xte,yte=Xtr[mtr],ytr[mtr],Xte[mte],yte[mte]
def Y(x): return 0.299*x[:,0]+0.587*x[:,1]+0.114*x[:,2]
def blur(X): return F.interpolate(F.avg_pool2d(X,2), scale_factor=2, mode="bilinear", align_corners=False)
Xte_blur=blur(Xte)

def logpolar(img, nr, nt):
    N,h,w=img.shape
    Fm=torch.fft.fftshift(torch.fft.fft2(img-img.mean((1,2),keepdim=True)),dim=(-2,-1)).abs()**2
    yy,xx=torch.meshgrid(torch.arange(h)-h//2, torch.arange(w)-w//2, indexing="ij")
    r=torch.sqrt(yy**2+xx**2).float(); th=torch.atan2(yy.float(),xx.float())%math.pi
    # log-radial bins (mel-like)
    rb=torch.clamp((torch.log1p(r)/math.log1p(h/2)*nr).long(),max=nr-1); tb=torch.clamp((th/math.pi*nt).long(),max=nt-1)
    idx=(rb*nt+tb).flatten(); mask=(r>0).flatten()
    cnt=torch.zeros(nr*nt).index_add_(0, idx[mask], torch.ones(int(mask.sum())))
    out=torch.zeros(N,nr*nt).index_add_(1, idx[mask], Fm.flatten(1)[:,mask])/cnt.clamp(min=1)
    return torch.log(out+1e-6).view(N,nr,nt)
def dct_mat(n):
    k=torch.arange(n).float().unsqueeze(1); i=torch.arange(n).float().unsqueeze(0)
    return torch.cos(math.pi*(i+0.5)*k/n)*math.sqrt(2/n)
def cepstrum(img, nr=6, nt=8, keep=(1,6)):
    """log-polar log-spectrum -> DCT along log-radius per orientation -> keep
    coefficients 1..keep (drop c0 = gain): spectral shape."""
    L=logpolar(img,nr,nt)                        # [N,nr,nt]
    D=dct_mat(nr)                                # [nr,nr]
    C=torch.einsum("kr,nrt->nkt",D,L)            # [N,nr,nt]
    return C[:,keep[0]:keep[1],:].flatten(1)     # [N,(keep-1)*nt]
def windows(y, size=12, stride=5):
    N=y.shape[0]; out=[]
    for r in range(0,32-size+1,stride):
        for c in range(0,32-size+1,stride):
            out.append(y[:,r:r+size,c:c+size])
    return out
def batched(fn,X,chunk=2000): return torch.cat([fn(X[i:i+chunk]) for i in range(0,len(X),chunk)])
def cep_global(X): y=Y(X); return torch.cat([cepstrum(y,8,8),cepstrum(y[:,8:24,8:24],6,8),cepstrum(X[:,0]-X[:,1],6,4)],1)
def cep_spectrogram(X):   # positions kept
    y=Y(X); return torch.cat([cepstrum(w,5,8,(1,5)) for w in windows(y)],1)
def cep_pooled(X):        # where doesn't matter: max + mean over windows
    y=Y(X); S=torch.stack([cepstrum(w,5,8,(1,5)) for w in windows(y)],1)   # [N,W,d]
    return torch.cat([S.amax(1),S.mean(1)],1)
def energy(w): w=w-w.mean((1,2),keepdim=True); return (w**2).mean((1,2))
def cep_onset(X):         # window with max energy contrast (the saccade target)
    y=Y(X); ws=windows(y); E=torch.stack([energy(w) for w in ws],1); best=E.argmax(1)
    S=torch.stack([cepstrum(w,5,8,(1,5)) for w in ws],1)
    return torch.cat([S[torch.arange(len(X)),best], (best//5).float().unsqueeze(1)/4, (best%5).float().unsqueeze(1)/4],1)
eye=B.Eye((16,16),rectify=False)
def eye_feats(X): return eye.sense(X)
def fit(Xa,ya,Xb,yb,tag,Xb2):
    mu,sd=Xa.mean(0),Xa.std(0)+1e-6; Xa=(Xa-mu)/sd; Xb=(Xb-mu)/sd
    sub=B.train_sub(Xa,ya,100,hidden=48,seed=1,epochs=8,tag=tag)
    fa,ta=B.acc_pair(B.logits_of(sub,Xb),yb,f2c); fb,tb=B.acc_pair(B.logits_of(sub,(Xb2-mu)/sd),yb,f2c)
    print(f"  {tag:>40s} d={Xa.shape[1]:4d}: full={fa:.4f} task={ta:.4f} | 2x-blur full={fb:.4f} task={tb:.4f}",flush=True)
probes={"(a) global cepstrum":cep_global,"(b) cepstral spectrogram, positions kept":cep_spectrogram,
        "(c) cepstra pooled over positions":cep_pooled,"(d) onset window cepstrum + where":cep_onset,
        "(e) eye DoG + (c)":lambda X: torch.cat([eye_feats(X),cep_pooled(X)],1),
        "(f) eye DoG + (b)":lambda X: torch.cat([eye_feats(X),cep_spectrogram(X)],1)}
for k,fn in probes.items(): fit(batched(fn,Xtr),ytr,batched(fn,Xte),yte,k,batched(fn,Xte_blur))
