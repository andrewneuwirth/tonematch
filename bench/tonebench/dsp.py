"""Small DSP helpers: RBJ cookbook biquads (as SOS rows), level normalization."""

import numpy as np
from scipy import signal

from . import SR


def _norm(b, a):
    b = np.asarray(b, float) / a[0]
    a = np.asarray(a, float) / a[0]
    return np.concatenate([b, a])


def peak(f0, gain_db, q, sr=SR):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / sr
    al = np.sin(w) / (2 * q)
    return _norm([1 + al * A, -2 * np.cos(w), 1 - al * A], [1 + al / A, -2 * np.cos(w), 1 - al / A])


def low_shelf(f0, gain_db, s=0.8, sr=SR):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / sr
    al = np.sin(w) / 2 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2)
    c, sq = np.cos(w), 2 * np.sqrt(A) * al
    return _norm(
        [A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sq)],
        [(A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq],
    )


def high_shelf(f0, gain_db, s=0.8, sr=SR):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / sr
    al = np.sin(w) / 2 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2)
    c, sq = np.cos(w), 2 * np.sqrt(A) * al
    return _norm(
        [A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq)],
        [(A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq],
    )


def butter(kind, fc, order, sr=SR):
    return signal.butter(order, fc, btype=kind, fs=sr, output="sos")


def apply(x, *sections):
    """Run x through a cascade of SOS rows / SOS arrays."""
    sos = np.vstack([np.atleast_2d(s) for s in sections])
    return signal.sosfilt(sos, x)


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)


def normalize(x, target_db=-20.0):
    return x * 10 ** ((target_db - rms_db(x)) / 20)
