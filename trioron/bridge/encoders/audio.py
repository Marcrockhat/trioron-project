"""Audio encoder backed by openai-whisper.

Model default: tiny (~75 MB on disk, 384-dim hidden state). We only
use the audio encoder tower — Whisper's decoder/text-generation path
is irrelevant for our use case (we want a fixed-dim audio
representation, not transcription text).

Install:
    pip install trioron[bridge-audio]
"""
from __future__ import annotations
from typing import List, Optional, Sequence, Union

import torch


_INSTALL_HINT = (
    "Audio encoder requires openai-whisper. Install with:\n"
    "    pip install trioron[bridge-audio]\n"
    "  or\n"
    "    pip install openai-whisper"
)


class AudioEncoder:
    """Wrap a Whisper audio encoder as a frozen Encoder.

    Inputs may be paths to audio files or pre-loaded mel spectrograms.
    Output is mean-pooled across time so the result is a single
    fixed-dim vector per clip. ``encode_dim`` is the encoder hidden
    dim (384 for tiny, 512 for base, 768 for small, 1024 for medium).
    """

    def __init__(
        self,
        model_name: str = "tiny",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        try:
            import whisper
        except ImportError as e:  # pragma: no cover
            raise ImportError(_INSTALL_HINT) from e
        self.model_name = model_name
        self.device = device or "cpu"
        self.normalize = normalize
        self._whisper = whisper
        model = whisper.load_model(model_name, device=self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        self._model = model
        # Whisper's encoder hidden dim is exposed via dims.n_audio_state.
        self.encode_dim: int = int(model.dims.n_audio_state)

    def __call__(self, batch) -> torch.Tensor:
        """Accepts a single audio file path or a list of paths.
        Returns (B, encode_dim)."""
        single = isinstance(batch, str)
        paths = [batch] if single else list(batch)
        outs = []
        with torch.no_grad():
            for p in paths:
                audio = self._whisper.load_audio(p)
                audio = self._whisper.pad_or_trim(audio)
                mel = self._whisper.log_mel_spectrogram(audio).to(self.device)
                # encoder forward: (1, n_mels, n_frames) → (1, n_frames, n_state)
                feat = self._model.encoder(mel.unsqueeze(0))
                pooled = feat.mean(dim=1)               # (1, n_state)
                if self.normalize:
                    pooled = pooled / pooled.norm(
                        dim=-1, keepdim=True,
                    ).clamp_min(1e-12)
                outs.append(pooled.cpu())
        return torch.cat(outs, dim=0)


__all__ = ["AudioEncoder"]
