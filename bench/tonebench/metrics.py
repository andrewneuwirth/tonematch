"""Tone distances.

- ltas_bands: long-term average spectrum in 1/3-octave bands (dB, level-normalized). Independent of
  what notes were played, so matchers use it to compare a render of the user's DI with the song.
- mrstft_db: multi-resolution STFT log-magnitude distance between two time-aligned signals. Used for
  scoring: estimate and truth are rendered from the SAME held-out DI, so any difference is tone.
"""

import numpy as np

from . import SR

_CENTERS = 1000 * 2 ** (np.arange(-14, 11) / 3)  # ~40 Hz .. ~10 kHz


def ltas_bands(x, sr=SR, n_fft=4096):
    hop = n_fft // 2
    win = np.hanning(n_fft)
    n = 1 + (len(x) - n_fft) // hop
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop][:n] * win
    power = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    out = []
    for fc in _CENTERS:
        m = (freqs >= fc / 2 ** (1 / 6)) & (freqs < fc * 2 ** (1 / 6))
        out.append(10 * np.log10(np.mean(power[m]) + 1e-20))
    out = np.array(out)
    return out - np.mean(out)


def ltas_db(a, b):
    """RMS difference of the two long-term spectra, in dB."""
    return float(np.sqrt(np.mean((ltas_bands(a) - ltas_bands(b)) ** 2)))


def _logmag(x, n_fft):
    hop = n_fft // 4
    win = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * win
    return 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) + 1e-12)


def mrstft_db(a, b, sizes=(512, 1024, 2048), range_db=80.0):
    """Mean absolute log-magnitude difference over three STFT sizes, in dB. 0 = identical.

    Both spectrograms are floored at range_db below their joint peak, so differences in the noise
    floor (a mic'd amp's hiss vs. a model's digital silence) can't dominate the score.
    """
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    out = []
    for s in sizes:
        la, lb = _logmag(a, s), _logmag(b, s)
        floor = max(la.max(), lb.max()) - range_db
        out.append(np.mean(np.abs(np.maximum(la, floor) - np.maximum(lb, floor))))
    return float(np.mean(out))
