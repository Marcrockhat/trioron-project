"""Label tap bank — labels as reference carriers.  [s034, Rocky's design]

The lock-in translation of "feed labels at a different frequency so
they cannot disturb the learning frequency": deposits inside a period
are an unordered phasor sum (no time tag), so a second frequency is
implemented the way hardware lock-ins do it — MULTIPLY BY THE
REFERENCE AT DEPOSIT TIME, THEN INTEGRATE. The label is the reference,
not a data dim: a sample arriving with label ℓ deposits its ordinary
value-phasor into the learning channel IDENTICALLY to an unlabeled
sample (non-disturbance is exact, by construction — the fundamental
tap cannot tell labeled from unlabeled), and additionally into tap ℓ
of this bank.

Demodulating tap ℓ yields a per-dim mean phasor — a full signature
template for the label, measured from however few labeled samples
ever arrived. That repairs the two s034 componential failures at the
root (run_componential.py):

  * prototype self-contamination — prototypes are now MEASURED from
    the annotation carrier, not reconstructed from self-labeled
    classes;
  * the vocabulary gap — "duck" exists as a demodulated prototype
    even though no duck CLASS exists; the organism can name a blend
    ("chicken-duck") without the percept supporting a split.

The bank is write-only from learning's side: membership, division,
the composer, the stress economy never read it. Taps cover RECEPTOR
dims only (canonical pockets — fixed width after genesis); composed
dims are excluded in this first cut (they appear/disappear with
composer spawns and have no label history).

Known limit (carried from s034): incoherent pollution (disruptor
spray) averages to a near-zero phasor, so its prototype is weak —
spray stays the reject readout's job. `member_mix` therefore reports
an explicit UNKNOWN fraction for buffer members below the margin
floor ("mostly pig, part unknown" — the "less pig" descriptor).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from trioron.core.receptor import N_QUANTA


class LabelTapBank:
    """Per-label demodulation taps over canonical receptor pockets."""

    def __init__(self, n_dims: int) -> None:
        self.n_dims = n_dims
        self.vocab: Dict[str, int] = {}
        self.taps = torch.zeros(0, n_dims, dtype=torch.complex64)
        self.counts = torch.zeros(0)
        # (class x label) counts [s034 increment, Rocky-approved]:
        # at the boundary, membership and labels align — counting
        # labels per CLASS names classes at the grain the organism
        # actually discovered (beats mode-smearing: no prototype
        # averaging is involved). Keyed by class NAME (survives
        # reindexing); division children start EMPTY (counts are
        # deposit-time annotations, not per-member tags); merges pool
        # exactly; retired names drop. Still write-only annotation.
        self.class_counts: Dict[str, Dict[str, float]] = {}

    def _row(self, label: str) -> int:
        if label not in self.vocab:
            self.vocab[label] = len(self.vocab)
            self.taps = torch.cat(
                [self.taps, torch.zeros(1, self.n_dims,
                                        dtype=torch.complex64)])
            self.counts = torch.cat([self.counts, torch.zeros(1)])
        return self.vocab[label]

    # ── deposit (the reference multiply-then-integrate) ───────────

    def deposit(self, q: torch.Tensor,
                labels: Sequence[Optional[str]]) -> int:
        """Accumulate the value-phasors of labeled rows into their
        label's tap. `q` = canonical pockets, receptor columns first
        (extra columns ignored); `labels[i] = None` -> row i deposits
        nothing here (its learning deposit already happened upstream,
        unchanged). Returns rows consumed."""
        Z = torch.exp(1j * 2 * math.pi * q[:, :self.n_dims] / N_QUANTA)
        n = 0
        for i, lab in enumerate(labels):
            if lab is None:
                continue
            r = self._row(lab)
            self.taps[r] += Z[i].to(torch.complex64)
            self.counts[r] += 1
            n += 1
        return n

    # ── demodulation readouts ─────────────────────────────────────

    def prototypes(self) -> Tuple[List[str], torch.Tensor]:
        """(names, S) — S[ℓ] = mean phasor per dim (the label's
        measured signature template)."""
        names = sorted(self.vocab, key=self.vocab.get)
        S = self.taps / self.counts.clamp_min(1).unsqueeze(1)
        return names, S

    def decompose(self, t: torch.Tensor, n_cand: int = 6) -> Dict[str, float]:
        """Nonnegative mixture weights: t ~= sum beta_ℓ S_ℓ over the
        n_cand most similar prototypes (templates are phasor MEANS, so
        a blend's template is exactly the weighted sum of its
        constituents' — complex LS as stacked real LS; active-set:
        drop negative supports, re-solve). Normalized to sum 1."""
        names, S = self.prototypes()
        if not names:
            return {}
        t = t[:self.n_dims]
        sims = (t.unsqueeze(0) * S.conj()).real.sum(-1)
        order = sims.argsort(descending=True)[:n_cand].tolist()
        active = [names[i] for i in order]
        y = torch.cat([t.real, t.imag]).to(torch.float32)
        beta: Dict[str, float] = {}
        for _ in range(len(active)):
            A = torch.stack(
                [torch.cat([S[self.vocab[l]].real, S[self.vocab[l]].imag])
                 for l in active], dim=1).to(torch.float32)
            sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
            if bool((sol >= -1e-6).all()):
                beta = {l: max(0.0, float(b)) for l, b in zip(active, sol)}
                break
            active = [l for l, b in zip(active, sol) if float(b) > 0]
            if not active:
                return {}
        tot = sum(beta.values())
        return {l: b / tot for l, b in beta.items()} if tot > 1e-6 else {}

    def name(self, t: torch.Tensor, th: float = 0.20,
             max_mods: int = 2) -> Tuple[Optional[str],
                                         List[Tuple[float, str]]]:
        """(primary label, [(beta, modifier label)]) for a class
        template — the componential readout, label-grounded."""
        beta = self.decompose(t)
        if not beta:
            return None, []
        ranked = sorted(beta.items(), key=lambda kv: -kv[1])
        primary = ranked[0][0]
        mods = [(b, l) for l, b in ranked[1:] if b >= th][:max_mods]
        return primary, mods

    def member_mix(self, Z: torch.Tensor,
                   k_unknown: Optional[float] = None) -> Dict[str, float]:
        """Composition of a buffer from its MEMBERS: each member
        phasor row matched against the label prototypes; fractions
        per winning label, plus an explicit '?' fraction for members
        whose best margin under the per-observation null falls below
        k_unknown (incoherent matter no prototype explains)."""
        names, S = self.prototypes()
        if not names or not len(Z):
            return {}
        E = (Z[:, :self.n_dims].unsqueeze(1) * S.conj().unsqueeze(0)
             ).real.sum(-1)
        sig = ((S.abs() ** 2).sum(-1) / 2).sqrt().clamp_min(1e-9)
        # margin ranking, not raw E — raw E is biased toward strong
        # prototypes (the D20 member_margin ruling, same statistic):
        # a label measured from many tight samples would absorb every
        # member of a near-collinear pair
        best = (E / sig).argmax(1)
        out: Dict[str, float] = {}
        unknown = torch.zeros(len(Z), dtype=torch.bool)
        if k_unknown is not None:
            marg = (E / sig).gather(1, best.unsqueeze(1)).squeeze(1)
            unknown = marg < k_unknown
            if bool(unknown.any()):
                out["?"] = float(unknown.float().mean())
        for i in best[~unknown].unique().tolist():
            out[names[i]] = float(((best == i) & ~unknown).float().mean())
        return out

    # ── (class x label) counts ────────────────────────────────────

    def count_members(self, class_names: Sequence[Optional[str]],
                      labels: Sequence[Optional[str]]) -> None:
        """Per-row: the class the row was just assigned to (None =
        refused / unassigned) x its label (None = unlabeled)."""
        for cname, lab in zip(class_names, labels):
            if cname is None or lab is None:
                continue
            d = self.class_counts.setdefault(cname, {})
            d[lab] = d.get(lab, 0.0) + 1.0

    def divide_counts(self, parent: str, child_a: str, child_b: str,
                      frac_a: float) -> None:
        """Children inherit the parent's counts proportionally to the
        buffer-member split (an ESTIMATOR: exact only when labels are
        uniform across the split sides — when the split separates the
        label populations, both children keep the blend's labels until
        fresh counts dominate; the alternative, restart-at-zero, left
        tail-born classes permanently unnamed — measured on the
        taxonomy: 4 final-boundary children, 0 counts, rel-cnt 0.583)."""
        src = self.class_counts.pop(parent, None)
        if not src:
            return
        for child, f in ((child_a, frac_a), (child_b, 1.0 - frac_a)):
            if f <= 0:
                continue
            d = self.class_counts.setdefault(child, {})
            for lab, n in src.items():
                d[lab] = d.get(lab, 0.0) + n * f

    def merge_counts(self, keep: str, drop: str) -> None:
        """Survivor pools the dropped class's counts (exact)."""
        src = self.class_counts.pop(drop, None)
        if src:
            d = self.class_counts.setdefault(keep, {})
            for lab, n in src.items():
                d[lab] = d.get(lab, 0.0) + n

    def retire_counts(self, name: str) -> None:
        self.class_counts.pop(name, None)

    def composition_of(self, name: str) -> Dict[str, float]:
        """Normalized label fractions for a class — its measured
        composition under the annotation it received."""
        d = self.class_counts.get(name, {})
        tot = sum(d.values())
        return {lab: n / tot for lab, n in d.items()} if tot > 0 else {}

    # ── persistence (rides MixedStreamController.state_dict) ──────

    def state_dict(self) -> dict:
        return {"n_dims": self.n_dims, "vocab": dict(self.vocab),
                "taps": self.taps.clone(), "counts": self.counts.clone(),
                "class_counts": {k: dict(v)
                                 for k, v in self.class_counts.items()}}

    def load_state_dict(self, state: dict) -> None:
        self.n_dims = state["n_dims"]
        self.vocab = dict(state["vocab"])
        self.taps = state["taps"].clone()
        self.counts = state["counts"].clone()
        self.class_counts = {k: dict(v) for k, v in
                             state.get("class_counts", {}).items()}
