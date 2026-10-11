"""ToneHub's own guitar tone encoder ("the ears"): a small conv net on log-mel.

Input: mono 48 kHz audio of any length (trained on 4 s). Front end (LogMel): RMS-normalize,
STFT 2048 / hop 512 (Hann), power -> 96 mel bands 30 Hz-20 kHz (triangular, HTK mel scale),
10*log10(. + 1e-10), minus the clip's mean over all bands and frames, divided by 20.
Network (ToneNet): 4 conv stages on (mel, time) -> per-frame projection of (channels x mel) ->
mean + std pooling over time -> 256-d trunk -> 128-d L2-normalized embedding, and a 7-knob head
(sigmoid, tonebench.amp.KNOBS order) for predict-then-search.

Trained by bench/train_encoder.py (contrastive NT-Xent + knob regression + tone-distance
regression) on renders from tonebench.augment. Checkpoints: bench/models/encoder/.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import SR

MODELS = Path(__file__).resolve().parents[1] / "models" / "encoder"
N_FFT, HOP, N_MELS, FMIN, FMAX = 2048, 512, 96, 30.0, 20000.0
EMB = 128


def mel_matrix(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    pts = 700 * (10 ** (np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2) / 2595) - 1)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(freqs)), np.float32)
    for i in range(n_mels):
        lo, c, hi = pts[i : i + 3]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return fb


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.register_buffer("mel", torch.from_numpy(mel_matrix()), persistent=False)

    def forward(self, x):  # (B, samples) -> (B, 1, mels, frames)
        x = x / x.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=1e-6)
        s = torch.stft(x, N_FFT, HOP, window=self.window, center=True, return_complex=True)
        p = s.real ** 2 + s.imag ** 2
        m = 10 * torch.log10(torch.matmul(self.mel, p) + 1e-10)
        m = m - m.mean(dim=(1, 2), keepdim=True)
        return (m / 20.0)[:, None]


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU())


class ToneNet(nn.Module):
    """Log-mel (B, 1, 96, T) -> (embedding (B, 128) L2-normalized, knobs (B, 7) in 0..1)."""

    def __init__(self, chans=(24, 48, 96, 128), hidden=256, emb=EMB, n_knobs=7):
        super().__init__()
        layers, cin = [], 1
        for i, c in enumerate(chans):
            layers += [_block(cin, c), nn.AvgPool2d((2, 2) if i < 3 else (2, 1))]
            cin = c
        self.convs = nn.Sequential(*layers)
        fdim = chans[-1] * (N_MELS // 16)
        self.frame = nn.Sequential(nn.Conv1d(fdim, hidden, 1), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(2 * hidden + N_MELS, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.emb = nn.Linear(hidden, emb)
        self.knobs = nn.Linear(hidden, n_knobs)

    def forward(self, m):
        ltas = m.mean(dim=3)[:, 0]                      # long-term mel spectrum (B, 96)
        h = self.convs(m)                               # (B, C, F', T')
        h = self.frame(h.flatten(1, 2))                 # (B, hidden, T')
        h = torch.cat([h.mean(dim=2), h.std(dim=2), ltas], dim=1)
        h = self.trunk(h)
        return F.normalize(self.emb(h), dim=-1), torch.sigmoid(self.knobs(h))


class Encoder(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.front = LogMel()
        self.net = ToneNet(**kw)

    def forward(self, x):
        return self.net(self.front(x))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- bench API

@lru_cache(maxsize=2)
def load(path=None, device="cpu"):
    import os
    path = Path(path or os.environ.get("TONEBENCH_ENCODER", MODELS / "best.pt"))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Encoder(**ck.get("model_kw", {}))
    m.load_state_dict(ck["model"])
    torch.set_num_threads(4)
    return m.eval().to(device)


def run(batch, path=None):
    """(embeddings (n, 128), knob predictions (n, 7)) as numpy for a list of mono 48 kHz signals."""
    m = load(path)
    n = min(len(b) for b in batch)
    x = torch.as_tensor(np.stack([np.asarray(b[:n]) for b in batch]), dtype=torch.float32)
    with torch.no_grad():
        e, k = m(x)
    return e.numpy(), k.numpy()


def embed(batch, path=None):
    return run(batch, path)[0]
