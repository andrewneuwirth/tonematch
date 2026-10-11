"""Guitar-TECHS loading (CC BY 4.0, zenodo.org/records/14963133).

Each musical excerpt has a clean DI file and a time-matched mic'd-amp file of the same performance.
Files are dual mono, 48 kHz.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from . import SR

DATA = Path(__file__).resolve().parents[2] / "data" / "guitar-techs"
MUSIC = DATA / "P3_music" / "P3_music" / "audio"


def _mono(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    assert sr == SR, f"{path}: {sr} Hz"
    return x[:, 0]


def music_ids():
    return sorted(p.stem.split("_")[-1] for p in (MUSIC / "directinput").glob("directinput_*.wav"))


@lru_cache(maxsize=None)
def di(take: str) -> np.ndarray:
    return _mono(MUSIC / "directinput" / f"directinput_{take}.wav")


@lru_cache(maxsize=None)
def micamp(take: str) -> np.ndarray:
    """The mic'd amp recording, shifted so it lines up sample-for-sample with di(take)."""
    d, m = di(take), _mono(MUSIC / "micamp" / f"micamp_{take}.wav")
    lag = _lag(d, m)
    m = m[lag:] if lag >= 0 else np.pad(m, (-lag, 0))
    return np.pad(m, (0, max(0, len(d) - len(m))))[: len(d)]


def _lag(d, m, max_ms=60):
    """Latency of m behind d: cross-correlate the busiest 10 s, high-passed so pickup thump doesn't dominate."""
    hp = signal.butter(2, 200, "highpass", fs=SR, output="sos")
    n = 10 * SR
    start = _busiest(d, n)
    a = signal.sosfilt(hp, d[start : start + n])
    b = signal.sosfilt(hp, m[start : start + n])
    c = signal.correlate(b, a, mode="full", method="fft")
    mid = len(a) - 1
    k = int(max_ms / 1000 * SR)
    return int(np.argmax(np.abs(c[mid - k : mid + k + 1])) - k)


def _busiest(x, n, hop=SR // 2):
    if len(x) <= n:
        return 0
    starts = np.arange(0, len(x) - n, hop)
    return int(starts[np.argmax([np.mean(x[s : s + n] ** 2) for s in starts])])


def segment(x, seconds, rng, min_db=-45.0, tries=50):
    """A random segment with actual playing in it (RMS above min_db). Returns (start, segment)."""
    n = int(seconds * SR)
    for _ in range(tries):
        s = int(rng.integers(0, max(1, len(x) - n)))
        seg = x[s : s + n]
        if 20 * np.log10(np.sqrt(np.mean(seg**2)) + 1e-12) > min_db:
            return s, seg
    s = _busiest(x, n)
    return s, x[s : s + n]
