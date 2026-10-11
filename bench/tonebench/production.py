"""Random studio production applied to synthetic targets: mix EQ, bus compression, reverb, lossy codec.

A real record never contains the bare amp tone, so a matcher has to see through this. The amp
settings are still the answer; production is a nuisance the matcher should not copy into the knobs.

Stems: produce_stems()/room_stems() process a full mix and, alongside it, the guitar stem's share of
that mix ("oracle separation"): every gain (compressor, AGC, limiter, normalization, speaker
clipping) is computed from the mix and applied to the stem, linear filters and reverbs are applied
to both, the codec runs on each, and room noise goes into the mix only. With no stems the outputs
are identical to the original single-signal versions.
"""

import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal

from . import SR, dsp


def _compress(x, rng, stems=()):
    thresh_db = rng.uniform(-30, -18)
    ratio = rng.uniform(2, 6)
    # RMS envelope with ~10 ms attack/release via a one-pole smoother on x^2.
    a = np.exp(-1 / (0.010 * SR))
    env = signal.lfilter([1 - a], [1, -a], x**2)
    level_db = 10 * np.log10(env + 1e-12)
    over = np.maximum(level_db - thresh_db, 0)
    g = 10 ** (-over * (1 - 1 / ratio) / 20)
    return x * g, [st * g for st in stems]


def _reverb(x, rng, stems=()):
    rt60 = rng.uniform(0.3, 1.4)
    mix = rng.uniform(0.05, 0.25)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * 10 ** (-3 * t / rt60)
    ir = dsp.apply(ir, dsp.butter("lowpass", rng.uniform(3000, 8000), 2))
    wet = signal.fftconvolve(x, ir)[: len(x)]
    k = 10 ** ((dsp.rms_db(x) - dsp.rms_db(wet)) / 20)  # = dsp.normalize(wet, rms_db(x)) / wet
    rev = lambda y: (1 - mix) * y + mix * k * signal.fftconvolve(y, ir)[: len(y)]
    return (1 - mix) * x + mix * k * wet, [rev(st) for st in stems]


def _codec(x, rng, stems=()):
    """Round-trip through MP3 with ffmpeg (each stem separately). Unchanged if ffmpeg is missing."""
    kbps = int(rng.choice([128, 160, 192, 256]))
    scale = max(1.0, np.max(np.abs(x)) * 1.01)  # stems use the mix's clip guard and encoder delay
    y, lag = _mp3(x, kbps, scale)
    return y, [_mp3(st, kbps, scale, lag)[0] for st in stems]


def _mp3(x, kbps, scale, lag=None):
    with tempfile.TemporaryDirectory() as d:
        src, mp3, out = (os.path.join(d, n) for n in ("a.wav", "a.mp3", "b.wav"))
        sf.write(src, x / scale, SR)
        try:
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-b:a", f"{kbps}k", mp3], check=True)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", mp3, "-ar", str(SR), "-ac", "1", out], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return x, 0
        y, _ = sf.read(out)
    # MP3 adds encoder delay; realign by cross-correlation on the first second.
    if lag is None:
        k = SR
        lag = int(np.argmax(signal.correlate(y[: 2 * k], x[:k], mode="valid")))
    y = y[lag : lag + len(x)]
    return np.pad(y, (0, len(x) - len(y))), lag


def _pink(n, rng):
    """Pink noise (1/f power), shaped in the frequency domain."""
    spec = np.fft.rfft(rng.standard_normal(n))
    spec /= np.sqrt(np.maximum(np.fft.rfftfreq(n, 1 / SR), 20.0))
    return np.fft.irfft(spec, n)


def _env_db(x, seconds):
    a = np.exp(-1 / (seconds * SR))
    return 10 * np.log10(signal.lfilter([1 - a], [1, -a], x**2) + 1e-12)


def room(x: np.ndarray, rng: np.random.Generator, calibrate: str = "none") -> tuple[np.ndarray, dict]:
    """Listen mode: the song plays from a small speaker and an iPhone mic hears it across a room.

    speaker (band-limited, resonant, uneven, slightly clipping) -> room (early reflections + diffuse
    tail) -> background noise (pink + hum) -> phone mic response -> AGC + fast limiter.
    calibrate: "eq" / "eq_denoise" to clean the recording up afterwards (see room_stems()).
    """
    x, _, info = room_stems(x, (), rng, calibrate)
    return x, info


def room_chain(rng, n):
    """Draw one listen-mode chain: speaker, room, noise (for an n-sample recording), mic, AGC."""
    c = {}
    hp = rng.uniform(150, 300)
    c["speaker"] = (
        dsp.butter("highpass", hp, 4),
        dsp.peak(rng.uniform(1.1, 1.6) * hp, rng.uniform(3, 8), 1.5),  # box resonance
        dsp.peak(rng.uniform(1000, 2500), rng.uniform(-5, 5), 1.2),
        dsp.peak(rng.uniform(2500, 6000), rng.uniform(-5, 5), 1.5),
        dsp.butter("lowpass", rng.uniform(7000, 12000), 4),
    )
    c["drive"] = rng.uniform(1.0, 3.0)
    rt60 = rng.uniform(0.25, 0.7)
    drr_db = rng.uniform(-3, 8)  # direct-to-reverberant ratio: phone 0.5..3 m from the speaker
    m = int(rt60 * SR)
    t = np.arange(m) / SR
    tail = dsp.apply(rng.standard_normal(m) * 10 ** (-3 * t / rt60), dsp.butter("lowpass", rng.uniform(2500, 6000), 1))
    ramp = int(0.02 * SR)
    tail[:ramp] *= np.linspace(0, 1, ramp)
    early = np.zeros(int(0.03 * SR))
    for _ in range(6):
        early[int(rng.uniform(0.002, 0.025) * SR)] += rng.uniform(-0.6, 0.6)
    c.update(tail=tail, early=early, drr_db=drr_db, snr=rng.uniform(15, 35), hum_hz=rng.choice([50, 60]))
    c["pink"] = _pink(n, rng)
    c["mic"] = (
        dsp.butter("highpass", rng.uniform(80, 150), 2),
        dsp.peak(rng.uniform(2500, 6000), rng.uniform(1, 4), 1.0),
        dsp.butter("lowpass", rng.uniform(12000, 16000), 2),
    )
    c["agc_s"] = rng.uniform(0.2, 0.6)
    c["info"] = {"speaker_hp": round(hp), "rt60": round(rt60, 2), "drr_db": round(drr_db, 1), "snr_db": round(c["snr"], 1)}
    return c


def room_apply(x, stems, c, pink=None):
    """Play x (and the stems' shares of it) through chain c. pink: noise to use instead of c's own."""
    stems = list(stems)
    x, stems = dsp.apply(x, *c["speaker"]), [dsp.apply(st, *c["speaker"]) for st in stems]
    drive = c["drive"]
    g = 10 ** ((-12.0 - dsp.rms_db(x)) / 20)
    xn = x * g
    y = np.tanh(drive * xn) / drive
    ratio = np.divide(y, xn, out=np.ones_like(xn), where=np.abs(xn) > 1e-12)  # stem share of the clipping
    x, stems = y, [st * g * ratio for st in stems]

    tail, early = c["tail"], c["early"]
    wet = signal.fftconvolve(x, tail)[: len(x)]
    k = 10 ** ((dsp.rms_db(x) - c["drr_db"] - dsp.rms_db(wet)) / 20)
    sp = lambda v, w: v + signal.fftconvolve(v, early)[: len(v)] + k * w
    x, stems = sp(x, wet), [sp(st, signal.fftconvolve(st, tail)[: len(st)]) for st in stems]

    hum = np.sin(2 * np.pi * c["hum_hz"] * np.arange(len(x)) / SR)
    noise = dsp.normalize(c["pink"] if pink is None else pink, 0.0) + 0.3 * hum
    x = x + dsp.normalize(noise, dsp.rms_db(x) - c["snr"])  # noise is not part of any stem

    x, stems = dsp.apply(x, *c["mic"]), [dsp.apply(st, *c["mic"]) for st in stems]
    # AGC rides the level toward -20 dB (+-12 dB) with a slow envelope, then a fast limiter.
    g = 10 ** (np.clip(-20.0 - _env_db(x, c["agc_s"]), -12, 12) / 20)
    x, stems = x * g, [st * g for st in stems]
    g = 10 ** (-0.8 * np.maximum(_env_db(x, 0.002) + 12, 0) / 20)
    x, stems = x * g, [st * g for st in stems]
    g = 10 ** ((-20.0 - dsp.rms_db(x)) / 20)
    return x * g, [st * g for st in stems]


CALIB = ("none", "eq", "eq_denoise")


def room_stems(x, stems, rng, calibrate="none"):
    """room() on a mix plus the stems' shares of it (see module docstring). Returns (x, stems, info).

    calibrate: "none"; "eq" = compensate with calibrate_eq(); "eq_denoise" = that, then denoise()
    (the noise suppression is computed on the mix and applied to the stems too)."""
    if calibrate is True:
        calibrate = "eq"
    c = room_chain(rng, len(x))
    x, stems = room_apply(x, stems, c)
    info = dict(c["info"])
    if calibrate and calibrate != "none":  # drawn after the chain: calibrated cases share the recording
        gains = calibrate_eq(c, np.random.default_rng(int(rng.integers(2**63))))
        x, stems = _eq(x, gains), [_eq(st, gains) for st in stems]
        if calibrate == "eq_denoise":
            x, stems = denoise(x, stems)
        g = 10 ** ((-20.0 - dsp.rms_db(x)) / 20)
        x, stems = x * g, [st * g for st in stems]
        info["calib"] = calibrate
    return x, stems, info


CAL_SECONDS = 3.0
_CAL_F = np.geomspace(40, 20000, 400)


def calibrate_eq(c, rng, max_db=12.0, smooth_oct=1 / 6):
    """One-time listen-mode calibration, done the way the app could do it.

    The phone plays CAL_SECONDS of known pink noise through the same speaker and records it across
    the room (same chain: clipping, reverb, its own background noise, mic, AGC). The chain's
    magnitude response is estimated as sqrt(recorded PSD / played PSD) (Welch), smoothed to
    smooth_oct octaves, normalized to 0 dB on average over 200 Hz-5 kHz, and inverted, clamped to
    +-max_db. Returns compensation gains in dB on the _CAL_F grid. It never sees the true filters.
    """
    n = int(CAL_SECONDS * SR)
    probe = dsp.normalize(_pink(n, rng), -20.0)
    rec, _ = room_apply(probe, (), c, pink=_pink(n, rng))
    skip = int(0.5 * SR)  # let the AGC settle
    f, pxx = signal.welch(probe[skip:], SR, nperseg=8192)
    _, pyy = signal.welch(rec[skip:], SR, nperseg=8192)
    h_db = 10 * np.log10((pyy + 1e-20) / (pxx + 1e-20))
    df = f[1] - f[0]  # each band spans smooth_oct, and at least +-1.5 bins at the low end
    sm = np.array([h_db[(f >= min(fc * 2 ** (-smooth_oct / 2), fc - 1.5 * df))
                        & (f <= max(fc * 2 ** (smooth_oct / 2), fc + 1.5 * df))].mean() for fc in _CAL_F])
    sm -= sm[(_CAL_F >= 200) & (_CAL_F <= 5000)].mean()
    return np.clip(-sm, -max_db, max_db)


def denoise(x, stems=(), n_fft=2048, pct=10, over=2.0, floor_db=-20.0):
    """Spectral subtraction with a minimum-statistics noise floor (no clean reference needed).

    Noise power per bin = the pct-th percentile over time (smoothed across frequency); the
    Wiener-like gain sqrt(max(1 - over * noise / power, floor)) is smoothed over 3 frames and
    applied to x and, unchanged, to each stem."""
    hop = n_fft // 4
    _, _, X = signal.stft(x, SR, nperseg=n_fft, noverlap=n_fft - hop)
    p = np.abs(X) ** 2
    noise = np.percentile(p, pct, axis=1, keepdims=True)
    noise = signal.convolve(noise[:, 0], np.ones(5) / 5, mode="same")[:, None]
    g = np.sqrt(np.maximum(1 - over * noise / (p + 1e-20), 10 ** (floor_db / 10)))
    g = signal.convolve(g, np.ones((1, 3)) / 3, mode="same")

    def run(v):
        _, _, V = signal.stft(v, SR, nperseg=n_fft, noverlap=n_fft - hop)
        return signal.istft(V * g, SR, nperseg=n_fft, noverlap=n_fft - hop)[1][: len(v)]

    return run(x), [run(st) for st in stems]


def _eq(x, gains_db):
    """Zero-phase EQ by gains on the _CAL_F grid (held flat outside it)."""
    nfft = 1 << int(np.ceil(np.log2(len(x) + SR // 2)))
    f = np.fft.rfftfreq(nfft, 1 / SR)
    g = 10 ** (np.interp(np.log(np.maximum(f, 1.0)), np.log(_CAL_F), gains_db) / 20)
    return np.fft.irfft(np.fft.rfft(x, nfft) * g, nfft)[: len(x)]


def produce(x: np.ndarray, rng: np.random.Generator, level: str = "full") -> tuple[np.ndarray, dict]:
    """level: 'none' (bare amp), 'mix' (EQ + compression), 'full' (+ reverb + MP3),
    'room' (full, then heard through a small speaker and a phone mic; see room())."""
    x, _, info = produce_stems(x, (), rng, level)
    return x, info


def produce_stems(x, stems, rng, level="full"):
    """produce() on a mix plus the stems' shares of it (see module docstring). Returns (x, stems, info)."""
    stems = list(stems)
    info = {"level": level}
    if level == "room":
        x, stems, _ = produce_stems(x, stems, rng, "full")
        x, stems, extra = room_stems(x, stems, rng)
        return x, stems, {**info, **extra}
    if level == "none":
        return x, stems, info
    eq = (
        dsp.butter("highpass", rng.uniform(60, 140), 2),
        dsp.peak(rng.uniform(200, 5000), rng.uniform(-4, 4), rng.uniform(0.7, 2.0)),
        dsp.high_shelf(rng.uniform(5000, 9000), rng.uniform(-3, 3)),
    )
    x, stems = dsp.apply(x, *eq), [dsp.apply(st, *eq) for st in stems]
    x, stems = _compress(x, rng, stems)
    if level == "full":
        x, stems = _reverb(x, rng, stems)
        x, stems = _codec(x, rng, stems)
    g = 10 ** ((-20.0 - dsp.rms_db(x)) / 20)
    return x * g, [st * g for st in stems], info
