"""Random studio production applied to synthetic targets: mix EQ, bus compression, reverb, lossy codec.

A real record never contains the bare amp tone, so a matcher has to see through this. The amp
settings are still the answer; production is a nuisance the matcher should not copy into the knobs.
"""

import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal

from . import SR, dsp


def _compress(x, rng):
    thresh_db = rng.uniform(-30, -18)
    ratio = rng.uniform(2, 6)
    # RMS envelope with ~10 ms attack/release via a one-pole smoother on x^2.
    a = np.exp(-1 / (0.010 * SR))
    env = signal.lfilter([1 - a], [1, -a], x**2)
    level_db = 10 * np.log10(env + 1e-12)
    over = np.maximum(level_db - thresh_db, 0)
    gain_db = -over * (1 - 1 / ratio)
    return x * 10 ** (gain_db / 20)


def _reverb(x, rng):
    rt60 = rng.uniform(0.3, 1.4)
    mix = rng.uniform(0.05, 0.25)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * 10 ** (-3 * t / rt60)
    ir = dsp.apply(ir, dsp.butter("lowpass", rng.uniform(3000, 8000), 2))
    wet = signal.fftconvolve(x, ir)[: len(x)]
    return (1 - mix) * x + mix * dsp.normalize(wet, dsp.rms_db(x))


def _codec(x, rng):
    """Round-trip through MP3 with ffmpeg. Returns x unchanged if ffmpeg is missing."""
    kbps = int(rng.choice([128, 160, 192, 256]))
    with tempfile.TemporaryDirectory() as d:
        src, mp3, out = (os.path.join(d, n) for n in ("a.wav", "a.mp3", "b.wav"))
        sf.write(src, x / max(1.0, np.max(np.abs(x)) * 1.01), SR)
        try:
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-b:a", f"{kbps}k", mp3], check=True)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", mp3, "-ar", str(SR), "-ac", "1", out], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return x
        y, _ = sf.read(out)
    # MP3 adds encoder delay; realign by cross-correlation on the first second.
    k = SR
    lag = np.argmax(signal.correlate(y[: 2 * k], x[:k], mode="valid"))
    y = y[lag : lag + len(x)]
    return np.pad(y, (0, len(x) - len(y)))


def produce(x: np.ndarray, rng: np.random.Generator, level: str = "full") -> tuple[np.ndarray, dict]:
    """level: 'none' (bare amp), 'mix' (EQ + compression), 'full' (+ reverb + MP3)."""
    info = {"level": level}
    if level == "none":
        return x, info
    x = dsp.apply(
        x,
        dsp.butter("highpass", rng.uniform(60, 140), 2),
        dsp.peak(rng.uniform(200, 5000), rng.uniform(-4, 4), rng.uniform(0.7, 2.0)),
        dsp.high_shelf(rng.uniform(5000, 9000), rng.uniform(-3, 3)),
    )
    x = _compress(x, rng)
    if level == "full":
        x = _reverb(x, rng)
        x = _codec(x, rng)
    return dsp.normalize(x, -20.0), info
