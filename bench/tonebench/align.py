"""Line up a play-along DI with the song it was played along to.

The DI and the target sound nothing alike (clean vs. amp + production), so we align *onsets*:
render the DI through the amp at noon, take a spectral-flux onset envelope of both, find one
global lag by cross-correlation, then refine the lag per 0.5 s chunk (+-40 ms) and splice the DI
into place with short crossfades. Piecewise cross-correlation rather than full DTW: a player's
timing drifts slowly, and splicing keeps pitch and waveform intact.
"""

import numpy as np

from . import SR, amp

HOP = 128  # 2.7 ms


def onset_env(x, n_fft=1024):
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::HOP] * np.hanning(n_fft)
    mag = np.log1p(100 * np.abs(np.fft.rfft(frames, axis=1))[:, 4:300])  # ~190 Hz .. 14 kHz
    flux = np.maximum(np.diff(mag, axis=0), 0).sum(axis=1)
    flux = np.concatenate([[0], flux])
    return (flux - flux.mean()) / (flux.std() + 1e-9)


def _best_lag(a, b, lags):
    """Lag L (frames) maximizing sum a[t] * b[t - L]."""
    best, arg = -np.inf, 0
    for L in lags:
        if L >= 0:
            c = np.dot(a[L:], b[: len(b) - L]) / max(1, len(a) - L)
        else:
            c = np.dot(a[: len(a) + L], b[-L:]) / max(1, len(a) + L)
        if c > best:
            best, arg = c, L
    return arg


def align(target, di, max_global=0.25, max_local=0.04, chunk=0.5, fade=0.01):
    """Return (aligned_di, lags_seconds): di warped so that aligned_di[t] lines up with target[t]."""
    et = onset_env(target)
    ed = onset_env(amp.render(amp.AmpParams(), di))
    g = _best_lag(et, ed, range(-int(max_global * SR / HOP), int(max_global * SR / HOP) + 1))
    n, L, f = len(target), int(chunk * SR), int(fade * SR)
    loc = int(max_local * SR / HOP)
    ctx = int(0.25 * SR / HOP)
    lags = []
    for t0 in range(0, n, L):
        a0, a1 = max(0, t0 // HOP - ctx), min(len(et), (t0 + L) // HOP + ctx)
        seg = np.zeros_like(et)
        seg[a0:a1] = et[a0:a1]
        lags.append(_best_lag(seg, ed, range(g - loc, g + loc + 1)))
    lags = np.array(lags)
    if len(lags) >= 3:  # median-smooth so one confused chunk (a wrong note) doesn't jump
        lags = np.array([int(np.median(lags[max(0, i - 1) : i + 2])) for i in range(len(lags))])
    out = np.zeros(n)
    for k, t0 in enumerate(range(0, n, L)):
        src0 = t0 - lags[k] * HOP
        m = min(L + f, n - t0)
        piece = np.zeros(m)
        lo, hi = max(src0, 0), min(src0 + m, len(di))
        if hi > lo:
            piece[lo - src0 : hi - src0] = di[lo:hi]
        w = np.ones(m)
        if k > 0:
            w[: min(f, m)] = np.linspace(0, 1, min(f, m))
        if t0 + L < n:
            w[-f:] = np.linspace(1, 0, f)
        out[t0 : t0 + m] += piece * w
    return out, lags * HOP / SR
