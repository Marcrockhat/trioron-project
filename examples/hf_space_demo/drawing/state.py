"""Per-process state mgmt for the drawing live-learn tab.

Holds the live trioron organism, the per-class teach buffers, and the
file path the donor was loaded from. Designed for global shared state
within one Space instance — every visitor sees the cumulative effect
of teaching, with a `reset()` to bring it back to the cold-start
pretrained donor.

The shared-state pitch is "watch the demo learn over time"; the
adversarial-teaching risk is documented in the README. The reset
button is the safety valve.
"""
from __future__ import annotations
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


PRETRAIN_CLASSES = [0, 1, 2, 3, 4]
EXTEND_CLASSES = [5, 6, 7, 8, 9]
TEACH_THRESHOLD = 3  # samples in a buffer before we trigger extend()


@dataclass
class DrawingSession:
    """In-memory state. Single instance shared across all UI events
    (Gradio Blocks closure-captures it). Wrap mutations in `_lock` so
    a teach-while-predict doesn't tear the organism mid-extend."""
    donor_path: Path                      # cold-start donor (read-only)
    live_path: Path                       # current live donor (gets overwritten)
    organism: object = None               # lazily loaded
    pretrain_classes: List[int] = field(default_factory=lambda: list(PRETRAIN_CLASSES))
    learned_classes: List[int] = field(default_factory=list)
    buffers: Dict[int, List] = field(default_factory=dict)
    n_extends: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_donor(cls, donor_path: Path, live_path: Path) -> "DrawingSession":
        if not donor_path.exists():
            raise FileNotFoundError(f"donor not found: {donor_path}")
        live_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(donor_path, live_path)
        return cls(donor_path=donor_path, live_path=live_path)

    def reset(self) -> None:
        """Restore live state to the cold-start donor; clear buffers
        and learned-class history."""
        with self._lock:
            shutil.copy2(self.donor_path, self.live_path)
            self.organism = None
            self.learned_classes = []
            self.buffers = {}
            self.n_extends = 0

    def buffer_for(self, label: int) -> List:
        return self.buffers.setdefault(label, [])

    def known_classes(self) -> List[int]:
        return list(self.pretrain_classes) + list(self.learned_classes)

    def n_buffer_samples(self) -> int:
        return sum(len(v) for v in self.buffers.values())
