"""Motion primitive (s055, motion arc step 2): the s052 spectral machinery
re-aimed at the TIME axis.  The existing spectra keep |F|^2 only = motion-blind.
Here the CROSS-SPECTRUM between consecutive frames of each 8x8 patch keeps the
PHASE.  For a pattern translating by (dx, dy) per frame,

    F_{t+1}(kx,ky) = F_t(kx,ky) . exp(-2 pi i (kx dx + ky dy) / L)

so conj(F_t) F_{t+1} has phase  -2 pi (k . v) / L  at every 2-D bin k: the
phase rotates at a rate set by the velocity (a Reichardt / motion-energy
correlator in the Fourier domain; a moving object in a packet IS a frequency
pulse, read like the phasor / lock-in emitter).  Over the T-1 frame pairs the
unit phasors are AVERAGED: coherent motion adds, noise cancels (|mean| =
coherence).  Wagon-wheel aliasing is kept: bin k wraps when |k.v| > L/2.

Temporal adaptation (fly photoreceptor / retinal high-pass): the spectra are
DIFFERENCED in time before correlating (HP=True): D_t = F_{t+1} - F_t, then
conj(D_t) D_{t+1} = |A|^2 |e^{iw}-1|^2 e^{iw}.  A static background cancels
exactly and the phase is still exactly the per-frame rotation w (the magnitude
is a temporal band-pass).  Mean-subtraction was tried first and FAILS: at small
w it removes the signal itself.  1-D row/column spectra (the literal stereo L/R
form) were tried first and FAIL too: a body sliding along x makes column
profiles appear/disappear -- coherent but meaningless vertical phase.

Velocity read-out = a POPULATION of velocity-tuned units (MT-like): for each
candidate v on a grid, score(v) = sum_k w_k cos(theta_k + 2 pi k.v / L); the
winner is the estimate, aliasing appears as secondary peaks.

    motion_phase(X)  X [N,T,3,S,S] float -> [N, 25 regions * KBINS * 3] (cos, sin,
                     coherence per region/bin);  6 bins -> 450
    motion_energy(X) per-region log temporal-difference energy -> 25  (fires-only-when-moving)
    motion_block(X)  = motion_phase ++ motion_energy  = 475
    velocity_map(X)  per-patch (dx, dy) [N,13,13,2], coherence, energy  (population decode)
    decode_velocity(X) -> [N,2] energy-weighted global (dx, dy)
"""
import math, torch
import torch.nn.functional as F
from experiments.progenitor import frontend as FE
L = 8; ST = 2; HP = True
BINS = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2)]      # (kx, ky); kx >= 0 suffices (conjugate symmetry)
VGRID = 4.0; VSTEP = 0.25

def _win(L): w = torch.hann_window(L, periodic=False); return w[:, None] * w[None, :]

def _spectra(X):
    """X [N,T,3,S,S] -> complex [N,T',P,K] at the BINS (T' = T-1 when HP)."""
    N, T = X.shape[:2]; y = FE.Y(X.flatten(0, 1))
    P = y.unfold(1, L, ST).unfold(2, L, ST).reshape(N, T, -1, L, L)                  # [N,T,P,L,L]  (rows=y, cols=x)
    P = (P - P.mean((-1, -2), keepdim=True)) * _win(L)
    Fm = torch.fft.fft2(P)                                                           # [..., ky, kx]
    ky = torch.tensor([b[1] % L for b in BINS]); kx = torch.tensor([b[0] % L for b in BINS])
    Fb = Fm[..., ky, kx]                                                             # [N,T,P,K]
    if HP: Fb = Fb[:, 1:] - Fb[:, :-1]
    return Fb

def cross_phasors(X):
    """-> mean unit phasor Z [N,P,K] complex (|Z| = coherence), mean cross energy E [N,P,K]."""
    Fb = _spectra(X); C = Fb[:, :-1].conj() * Fb[:, 1:]; E = C.abs()
    return (C / (E + 1e-8)).mean(1), E.mean(1)

_rid = FE._rid
def _pool(Z):
    out = torch.zeros(len(Z), 25, Z.shape[2]); out.index_add_(1, _rid, Z)
    return (out / torch.bincount(_rid, minlength=25).float().view(1, 25, 1)).reshape(len(Z), -1)

def motion_phase(X):
    Z, E = cross_phasors(X)
    return _pool(torch.cat([Z.real, Z.imag, Z.abs()], -1))

def motion_energy(X):
    N, T = X.shape[:2]; y = FE.Y(X.flatten(0, 1)).view(N, T, X.shape[-2], X.shape[-1])
    d = (y[:, 1:] - y[:, :-1]).pow(2).mean(1)
    return torch.log(F.adaptive_avg_pool2d(d[:, None], 5)[:, 0].flatten(1) + 1e-5)

def motion_block(X): return torch.cat([motion_phase(X), motion_energy(X)], 1)

def _grid():
    g = torch.arange(-VGRID, VGRID + 1e-6, VSTEP); vx, vy = torch.meshgrid(g, g, indexing="xy")
    return torch.stack([vx.flatten(), vy.flatten()], 1)                              # [G,2]
def tuning(Z, E, weight="E"):
    """population response [N,P,G]: sum_k w_k Re(Z_k exp(+2 pi i k.v / L))  (phase of Z_k is -2 pi k.v/L at the true v)."""
    V = _grid(); K = torch.tensor(BINS, dtype=torch.float)                           # [K,2]
    ph = 2 * math.pi * (V @ K.t()) / L                                               # [G,K]
    tmpl = torch.polar(torch.ones_like(ph), ph)                                      # e^{+i ph}
    w = E if weight == "E" else torch.ones_like(E)
    w = w / (w.sum(-1, keepdim=True) + 1e-8)
    return torch.einsum("npk,gk->npg", (Z * w).to(torch.complex64), tmpl).real, V
def velocity_map(X):
    Z, E = cross_phasors(X); R, V = tuning(Z, E)
    best = R.argmax(-1); v = V[best]                                                 # [N,P,2]
    n = int(math.isqrt(Z.shape[1])); coh = Z.abs().mean(-1); En = E.sum(-1)
    return v.view(len(X), n, n, 2), coh.view(len(X), n, n), En.view(len(X), n, n), R.view(len(X), n, n, -1)
def decode_velocity(X):
    v, coh, E, R = velocity_map(X); w = (coh * E).flatten(1); w = w / (w.sum(1, keepdim=True) + 1e-8)
    return (v.flatten(1, 2) * w[..., None]).sum(1)

def batched(fn, X, chunk=500): return torch.cat([fn(X[i:i + chunk]) for i in range(0, len(X), chunk)])

def motion_pop(X, grid_step=1.0):
    """MT-like read-out as features: per region the energy-weighted decoded (vx, vy), mean coherence, log energy
    (25 x 4 = 100) + the GLOBAL population response on a coarse velocity grid (step 1 px, [-4,4]^2 = 81).  -> 181"""
    Z, E = cross_phasors(X); R, V = tuning(Z, E); best = R.argmax(-1); v = V[best]                 # [N,P,2]
    w = (Z.abs().mean(-1) * E.sum(-1)); reg = _pool(torch.cat([v * w[..., None], w[..., None], Z.abs().mean(-1, keepdim=True), torch.log(E.sum(-1, keepdim=True) + 1e-6)], -1))
    reg = reg.view(len(X), 25, 5); vreg = reg[:, :, :2] / (reg[:, :, 2:3] + 1e-8)
    wg = w / (w.sum(1, keepdim=True) + 1e-8); Rg = (R * wg[..., None]).sum(1)                        # [N,G]
    n = int(round(grid_step / VSTEP)); g = int(round(2 * VGRID / VSTEP)) + 1
    Rg = Rg.view(len(X), g, g)[:, ::n, ::n].flatten(1)
    return torch.cat([vreg.flatten(1), reg[:, :, 3:].flatten(1), Rg], 1)
def motion_full(X): return torch.cat([motion_block(X), motion_pop(X)], 1)   # 475 + 181 = 656
