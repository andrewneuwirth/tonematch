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


def band_stats(x, sr=SR, n_fft=2048):
    """Spectral statistics: per 1/3-octave band, the long-term mean (level-normalized) and the
    standard deviation over time of the band's log power. The std part reacts to distortion and
    compression (a driven amp flattens how much each band moves), not only to EQ."""
    hop = n_fft // 2
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    bands = np.stack([power[:, (freqs >= fc / 2 ** (1 / 6)) & (freqs < fc * 2 ** (1 / 6))].mean(axis=1)
                      for fc in _CENTERS[3:]], axis=1)  # from ~80 Hz: 2048-FFT bins too coarse below
    logp = 10 * np.log10(bands + 1e-20)
    loud = logp.max(axis=1) > logp.max() - 50  # skip near-silent frames
    logp = logp[loud] if loud.sum() > 4 else logp
    mean = 10 * np.log10(np.mean(10 ** (logp / 10), axis=0))
    return np.concatenate([mean - mean.mean(), logp.std(axis=0)])


def _frame_err(a, b, n_fft, range_db=80.0):
    la, lb = _logmag(a, n_fft), _logmag(b, n_fft)
    floor = max(la.max(), lb.max()) - range_db
    return np.mean(np.abs(np.maximum(la, floor) - np.maximum(lb, floor)), axis=1)


def mrstft_trimmed_db(a, b, keep=0.8, sizes=(512, 1024, 2048)):
    """mrstft_db, but averaging only the best `keep` fraction of frames at each resolution, so a
    few wrong notes or misaligned stretches in a play-along don't dominate the fit."""
    n = min(len(a), len(b))
    out = []
    for s in sizes:
        e = np.sort(_frame_err(a[:n], b[:n], s))
        out.append(np.mean(e[: max(1, int(keep * len(e)))]))
    return float(np.mean(out))


def _mel_fb(sr=SR, n_fft=2048, n_mels=64, fmin=40.0, fmax=16000.0):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    pts = 700 * (10 ** (np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2) / 2595) - 1)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, c, hi = pts[i : i + 3]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return fb


_MEL = _mel_fb()


def mel_db(a, b, n_fft=2048, range_db=80.0):
    """Log-mel (64 bands, 40 Hz-16 kHz) distance between two time-aligned signals, in dB, each
    level-normalized and floored like mrstft_db. A cross-check on mrstft_db, whose linear-frequency
    bins put most of the weight above 5 kHz; mel bands weight the spectrum closer to how we hear it."""
    n = min(len(a), len(b))
    out = []
    for x in (a[:n], b[:n]):
        x = x / (np.sqrt(np.mean(x**2)) + 1e-12)
        frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[:: n_fft // 4] * np.hanning(n_fft)
        out.append(10 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) ** 2 @ _MEL.T + 1e-12))
    la, lb = out
    floor = max(la.max(), lb.max()) - range_db
    return float(np.mean(np.abs(np.maximum(la, floor) - np.maximum(lb, floor))))


def direct_mask(x, sr=SR, look=0.25, drop_db=6.0):
    """Dereverb-lite: True where x is near its recent peak level (attacks and held notes, where the
    direct sound dominates), False in decaying stretches that are mostly reverb tail. A sample is
    kept if its 10 ms level is within drop_db of the max level over the previous `look` seconds."""
    a = np.exp(-1 / (0.010 * sr))
    from scipy import signal
    e = 10 * np.log10(signal.lfilter([1 - a], [1, -a], x**2) + 1e-12)
    step = sr // 1000  # 1 ms
    ed = e[::step]
    w = int(look * 1000)
    m = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.full(w - 1, -np.inf), ed]), w).max(axis=1)
    keep = ed >= m - drop_db
    return np.repeat(keep, step)[: len(x)]


def mrstft_masked_db(a, b, mask, keep=0.8, min_frac=0.5, sizes=(512, 1024, 2048)):
    """mrstft_trimmed_db over only the frames where at least min_frac of the samples are in mask."""
    n = min(len(a), len(b))
    out = []
    for s in sizes:
        e = _frame_err(a[:n], b[:n], s)
        fm = np.lib.stride_tricks.sliding_window_view(mask[:n].astype(float), s)[:: s // 4].mean(axis=1)[: len(e)]
        e = e[fm >= min_frac] if (fm >= min_frac).sum() >= 8 else e
        e = np.sort(e)
        out.append(np.mean(e[: max(1, int(keep * len(e)))]))
    return float(np.mean(out))
