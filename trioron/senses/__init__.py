"""Senses — input encodings the organisms learn from: math-generated word
spectra (chord) and a frozen language model's hidden state."""
from .chord import chord_table, chords_of
from .hidden_state import HiddenStateSense

__all__ = ["chord_table", "chords_of", "HiddenStateSense"]
