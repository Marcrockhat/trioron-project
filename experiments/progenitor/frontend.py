"""R's fixed front end (s052 winner), lifted out of diag_eye4/diag_stereo so
probes can import it without exec-chaining CIFAR loaders.
  dense_pooled(X)   : 8px/stride-2 windows -> log-polar cepstra (4 radial x 8
                      orient, keep c1..c3 = 24/patch) -> 5x5 region pool = 600
  sync_pooled(X)    : synchronised stereo: L = horizontal 1-D spectrum of the
                      patch, R = vertical (4+4 bins), pooled 5x5 = 200
  dense_stereo(X)   : concat = 800 (0.345/0.529 on the 25-class CIFAR probe)
  colour_block(X)   : NEW s053: per-region (5x5) mean Y / RG / BY + per-region
                      Y std = 100; the class cue the luminance front end lacks
All luminance features are gain-free (spectral shape) by construction.
"""
import math, torch
import torch.nn.functional as F

def Y(x): return 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
def blur2x(X): return F.interpolate(F.avg_pool2d(X, 2), scale_factor=2, mode="bilinear", align_corners=False)

def logpolar(img, nr, nt):
    N, h, w = img.shape
    Fm = torch.fft.fftshift(torch.fft.fft2(img - img.mean((1, 2), keepdim=True)), dim=(-2, -1)).abs() ** 2
    yy, xx = torch.meshgrid(torch.arange(h) - h // 2, torch.arange(w) - w // 2, indexing="ij")
    r = torch.sqrt(yy ** 2 + xx ** 2).float(); th = torch.atan2(yy.float(), xx.float()) % math.pi
    rb = torch.clamp((torch.log1p(r) / math.log1p(h / 2) * nr).long(), max=nr - 1); tb = torch.clamp((th / math.pi * nt).long(), max=nt - 1)
    idx = (rb * nt + tb).flatten(); mask = (r > 0).flatten()
    cnt = torch.zeros(nr * nt).index_add_(0, idx[mask], torch.ones(int(mask.sum())))
    out = torch.zeros(N, nr * nt).index_add_(1, idx[mask], Fm.flatten(1)[:, mask]) / cnt.clamp(min=1)
    return torch.log(out + 1e-6).view(N, nr, nt)

def dct_mat(n):
    k = torch.arange(n).float().unsqueeze(1); i = torch.arange(n).float().unsqueeze(0)
    return torch.cos(math.pi * (i + 0.5) * k / n) * math.sqrt(2 / n)

def cepstrum(img, nr=6, nt=8, keep=(1, 6)):
    L = logpolar(img, nr, nt); C = torch.einsum("kr,nrt->nkt", dct_mat(nr), L)
    return C[:, keep[0]:keep[1], :].flatten(1)

def windows(y, size=8, stride=2):
    return [y[:, r:r + size, c:c + size] for r in range(0, 32 - size + 1, stride) for c in range(0, 32 - size + 1, stride)]

def batched(fn, X, chunk=2000): return torch.cat([fn(X[i:i + chunk]) for i in range(0, len(X), chunk)])

def spec1d(x, keep):
    L = x.shape[-1]; w = torch.hann_window(L, periodic=False); Xf = torch.fft.rfft((x - x.mean(-1, keepdim=True)) * w, dim=-1)
    return torch.log((Xf.abs() ** 2)[..., 1:keep + 1] + 1e-6)
def shape1d(P): return P - P.mean(-1, keepdim=True)
def patches(y, ws=8, st=2): return y.unfold(1, ws, st).unfold(2, ws, st)
def LR_sync(X, keep=4):
    P = patches(Y(X)); L = spec1d(P, keep).mean(-2); R = spec1d(P.transpose(-1, -2), keep).mean(-2)
    return shape1d(L), shape1d(R)
_rid = torch.tensor([[(r * 5) // 13 * 5 + (c * 5) // 13 for c in range(13)] for r in range(13)]).view(-1)
def pool25(Z):
    out = torch.zeros(len(Z), 25, Z.shape[2]); out.index_add_(1, _rid, Z)
    return (out / torch.bincount(_rid, minlength=25).float().view(1, 25, 1)).reshape(len(Z), -1)

def dense_pooled(X):
    y = Y(X); return pool25(torch.stack([cepstrum(w, 4, 8, (1, 4)) for w in windows(y, 8, 2)], 1))
def sync_pooled(X):
    L, R = LR_sync(X); return pool25(torch.cat([L, R], -1).reshape(len(X), 169, -1))
def dense_stereo(X): return torch.cat([dense_pooled(X), sync_pooled(X)], 1)

def colour_block(X):   # 5x5 regions x (Y, RG, BY mean; Y std) = 100
    y = Y(X); rg = X[:, 0] - X[:, 1]; by = X[:, 2] - 0.5 * (X[:, 0] + X[:, 1])
    m = lambda z: F.adaptive_avg_pool2d(z.unsqueeze(1), 5).squeeze(1).flatten(1)
    sd = lambda z: (F.adaptive_avg_pool2d((z ** 2).unsqueeze(1), 5).squeeze(1) - F.adaptive_avg_pool2d(z.unsqueeze(1), 5).squeeze(1) ** 2).clamp(min=0).sqrt().flatten(1)
    return torch.cat([m(y), m(rg), m(by), sd(y)], 1)
def dense_stereo_colour(X): return torch.cat([dense_stereo(X), colour_block(X)], 1)   # 900
def raw(X): return X.flatten(1)   # 3072 (over cap; reference)
