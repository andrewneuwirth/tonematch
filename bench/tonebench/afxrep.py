"""AFx-Rep audio-effect embedding (Steinmetz et al., ST-ITO, ISMIR 2024).

Model code adapted from github.com/csteinmetz1/st-ito, st_ito/models/panns.py (Apache-2.0), which
itself adapts PANNs Cnn14 (github.com/qiuqiangkong/audioset_tagging_cnn, MIT). The torchlibrosa
front end is replaced by the same computation in plain torch, using the STFT and mel matrices
stored in the checkpoint. Weights: huggingface.co/csteinmetz1/afx-rep (Apache-2.0), see
docs/LICENSES.md. Expects data/models/afx-rep/afx-rep-encoder.pt (encoder-only state dict cut
from afx-rep.ckpt; see README).

Only the mono path is used: a mono input gives the 512-dim "mid" embedding.
Memory: about 330 MB of weights. Threads are capped at 4.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = Path(__file__).resolve().parents[2] / "data" / "models" / "afx-rep" / "afx-rep-encoder.pt"
N_FFT, HOP = 2048, 1024
torch.set_num_threads(4)


class _ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x, pool):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool)


class _Stft(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_real = nn.Conv1d(1, N_FFT // 2 + 1, N_FFT, HOP, bias=False)
        self.conv_imag = nn.Conv1d(1, N_FFT // 2 + 1, N_FFT, HOP, bias=False)


class _Spec(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = _Stft()


class _Mel(nn.Module):
    def __init__(self):
        super().__init__()
        self.melW = nn.Parameter(torch.zeros(N_FFT // 2 + 1, 128))


class Cnn14(nn.Module):
    """AFx-Rep encoder (embed_dim 512, 48 kHz, 128 mels, min-max input norm, batchnorm)."""

    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = _Spec()
        self.logmel_extractor = _Mel()
        self.bn0 = nn.BatchNorm2d(128)  # in the checkpoint, unused with minmax input norm
        chans = [1, 64, 128, 256, 512, 1024, 2048]
        for i in range(6):
            setattr(self, f"conv_block{i + 1}", _ConvBlock(chans[i], chans[i + 1]))
        self.fc_mid = nn.Linear(2048, 512)
        self.fc_side = nn.Linear(2048, 512)

    def forward(self, x):  # x: (batch, samples), mono
        x = F.pad(x[:, None, :], (N_FFT // 2, N_FFT // 2), mode="reflect")
        st = self.spectrogram_extractor.stft
        power = st.conv_real(x) ** 2 + st.conv_imag(x) ** 2           # (B, 1025, T)
        mel = power.transpose(1, 2) @ self.logmel_extractor.melW       # (B, T, 128)
        x = 10 * torch.log10(mel.clamp(min=1e-10))[:, None]             # (B, 1, T, 128)
        x = (x.clamp(-80, 40) + 80) / 120 * 2 - 1
        for i in range(6):
            x = getattr(self, f"conv_block{i + 1}")(x, (2, 2) if i < 5 else (1, 1))
        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)
        return self.fc_mid(x)


@lru_cache(maxsize=1)
def model(device=None):
    if device is None:
        device = "cpu"  # CPU is fast enough (about 13 ms per 4 s clip) and keeps memory predictable
    m = Cnn14()
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return m.eval().to(device)


def embed(batch) -> np.ndarray:
    """L2-normalized 512-dim embeddings for a list of mono 48 kHz signals of equal length."""
    m = model()
    dev = next(m.parameters()).device
    x = torch.as_tensor(np.stack(batch), dtype=torch.float32)
    x = x / x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # peak normalize, as ST-ITO does
    with torch.no_grad():
        e = m(x.to(dev))
    return F.normalize(e, dim=-1).cpu().numpy()
