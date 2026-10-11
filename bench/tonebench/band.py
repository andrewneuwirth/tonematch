"""Synthetic backing band for full-band cases: drums, bass, and sometimes keys and a vocal-like lead.

Everything is synthesized here (no third-party audio, so no license question). It is not meant to
sound like a record, only to put realistic energy where a band puts it: kick and bass under the
guitar, snare and vocal formants in the guitar's midrange, hats and cymbal noise above it. That is
what a separator has to undo and what leaks into a matcher's view of the guitar.

backing(n, rng) returns (stems, info): a dict of level-set mono stems at 48 kHz, to be added to a
guitar stem normalized to -20 dB RMS.
"""

import numpy as np

from . import SR, dsp

A4 = 440.0


def _hz(midi):
    return A4 * 2 ** ((np.asarray(midi, float) - 69) / 12)


def _env(n, attack, decay):
    t = np.arange(n) / SR
    e = np.exp(-t / decay)
    a = max(1, int(attack * SR))
    e[:a] *= np.linspace(0, 1, a)
    return e


def _place(out, sig, t0):
    i = int(t0 * SR)
    if i >= len(out):
        return
    m = min(len(sig), len(out) - i)
    out[i : i + m] += sig[:m]


def _saw(f, n, rng):
    """Band-limited-enough sawtooth (additive, harmonics under 16 kHz) with per-note random phase."""
    t = np.arange(n) / SR
    out = np.zeros(n)
    for k in range(1, int(16000 / f) + 1):
        out += np.sin(2 * np.pi * k * f * t + rng.uniform(0, 2 * np.pi)) / k
    return out


def drums(n, rng, bpm):
    out = np.zeros(n)
    eighth = 30 / bpm
    kick_d, snare_d = rng.uniform(0.15, 0.35), rng.uniform(0.10, 0.22)
    hat_d = rng.uniform(0.02, 0.06)
    for i in range(int(n / SR / eighth) + 1):
        t = i * eighth + rng.normal(0, 0.004)
        if t < 0:
            continue
        beat8 = i % 8
        if beat8 in (0, 4) or (beat8 in (3, 5, 7) and rng.random() < 0.25):  # kick on 1 and 3, some pickups
            m = int(kick_d * 5 * SR)
            tt = np.arange(m) / SR
            f = 45 + 80 * np.exp(-tt / 0.03)
            k = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(m, 0.001, kick_d)
            _place(out, k * rng.uniform(0.8, 1.0), t)
        if beat8 in (2, 6):  # snare on 2 and 4
            m = int(snare_d * 5 * SR)
            body = np.sin(2 * np.pi * rng.uniform(170, 220) * np.arange(m) / SR) * _env(m, 0.001, 0.05)
            noise = dsp.apply(rng.standard_normal(m), dsp.butter("bandpass", [1500, 9000], 2)) * _env(m, 0.001, snare_d)
            _place(out, (0.6 * body + 0.7 * noise) * rng.uniform(0.8, 1.0), t)
        m = int(hat_d * 6 * SR)  # hats every eighth, accented on the beat
        hat = dsp.apply(rng.standard_normal(m), dsp.butter("highpass", 7000, 4)) * _env(m, 0.0005, hat_d)
        _place(out, hat * (0.35 if beat8 % 2 == 0 else 0.22) * rng.uniform(0.7, 1.0), t)
    if rng.random() < 0.5:  # a crash at the top
        m = int(1.5 * SR)
        crash = dsp.apply(rng.standard_normal(m), dsp.butter("highpass", 4000, 2)) * _env(m, 0.001, 0.5)
        _place(out, 0.4 * crash, 0.0)
    return out


def _progression(rng, key):
    """Four chords (root midi note, minor?) from a few common rock/pop progressions."""
    degrees = [[(0, 0), (7, 0), (9, 1), (5, 0)], [(0, 0), (5, 0), (7, 0), (5, 0)],
               [(9, 1), (5, 0), (0, 0), (7, 0)], [(0, 1), (10, 0), (8, 0), (7, 0)]]
    prog = degrees[int(rng.integers(len(degrees)))]
    return [(key + d, minor) for d, minor in prog]


def bass(n, rng, bpm, chords):
    out = np.zeros(n)
    eighth = 30 / bpm
    bar = 16 * eighth / 2  # 4/4: 8 eighths per bar
    cutoff = rng.uniform(500, 1400)
    for i in range(int(n / SR / eighth) + 1):
        if rng.random() < 0.15:
            continue  # rests
        t = i * eighth + rng.normal(0, 0.005)
        root, _ = chords[int(t // bar) % len(chords)]
        note = root - 24 + (12 if (i % 8 == 7 and rng.random() < 0.3) else 0)
        m = int(eighth * SR * 0.95)
        f = float(_hz(note))
        tone = 0.5 * _saw(f, m, rng) + np.sin(2 * np.pi * f * np.arange(m) / SR)
        rel = np.ones(m)
        rel[-int(0.01 * SR):] = np.linspace(1, 0, int(0.01 * SR))
        _place(out, tone * _env(m, 0.003, rng.uniform(0.3, 0.8)) * rel, max(t, 0))
    return dsp.apply(out, dsp.butter("lowpass", cutoff, 2))


def keys(n, rng, bpm, chords):
    """Sustained triads, one per bar: detuned saws through a soft low-pass (a pad / organ-ish keys part)."""
    out = np.zeros(n)
    bar = 4 * 60 / bpm
    for b in range(int(n / SR / bar) + 1):
        root, minor = chords[b % len(chords)]
        m = int(bar * SR)
        chord = np.zeros(m)
        for iv in (0, 3 if minor else 4, 7):
            f = float(_hz(root + iv))
            for det in (-0.004, 0.004):
                chord += _saw(f * (1 + det), m, rng)
        chord *= _env(m, 0.03, 2.5)
        chord[-int(0.02 * SR):] *= np.linspace(1, 0, int(0.02 * SR))
        _place(out, chord, b * bar)
    return dsp.apply(out, dsp.butter("lowpass", rng.uniform(1800, 3500), 2), dsp.butter("highpass", 150, 2))


_VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (530, 1840, 2480), (300, 870, 2240), (570, 840, 2410)]


def vocal(n, rng, bpm, chords, key):
    """A crude sung lead: a glottal-ish saw with vibrato through three vowel formants, in phrases."""
    out = np.zeros(n)
    beat = 60 / bpm
    scale = np.array([0, 2, 4, 5, 7, 9, 11, 12])
    t = rng.uniform(0, 1.0)
    base = key + int(rng.choice([0, 12]))  # male or higher register
    while t < n / SR:
        phrase_end = t + rng.uniform(1.5, 3.5)
        while t < phrase_end and t < n / SR:
            d = beat * float(rng.choice([0.5, 1, 1, 2]))
            m = int(d * SR)
            f0 = float(_hz(base + scale[int(rng.integers(len(scale)))]))
            tt = np.arange(m) / SR
            f = f0 * (1 + 0.012 * np.sin(2 * np.pi * 5.5 * tt) * np.minimum(tt / 0.3, 1))
            ph = 2 * np.pi * np.cumsum(f) / SR
            src = sum(np.sin(k * ph) / k for k in range(1, int(8000 / f0)))
            F = _VOWELS[int(rng.integers(len(_VOWELS)))]
            v = sum(dsp.apply(src, dsp.peak(fc, 18, 6)) for fc in F) - 2.5 * src  # boost formants, cut the rest
            v = dsp.apply(v, dsp.butter("highpass", 120, 2))
            v *= _env(m, 0.04, d * 2)
            v[-int(0.03 * SR):] *= np.linspace(1, 0, int(0.03 * SR))
            _place(out, v, t)
            t += d
        t += rng.uniform(0.3, 1.5)  # breath between phrases
    return out


def backing(n, rng):
    """Random band arrangement. Stem levels are dB re a guitar stem at -20 dB RMS."""
    bpm = float(rng.uniform(80, 160))
    key = int(rng.integers(40, 52))  # E2..D#3 roots; chords sit in the guitar's register
    chords = _progression(rng, key + 12)
    stems = {"drums": (drums(n, rng, bpm), rng.uniform(-3, 2)),
             "bass": (bass(n, rng, bpm, chords), rng.uniform(-6, -1))}
    if rng.random() < 0.6:
        stems["keys"] = (keys(n, rng, bpm, chords), rng.uniform(-12, -6))
    if rng.random() < 0.5:
        stems["vocal"] = (vocal(n, rng, bpm, chords, key + 12), rng.uniform(-6, 0))
    out = {k: dsp.normalize(s, -20.0 + lv) for k, (s, lv) in stems.items()}
    info = {"bpm": round(bpm), "parts": {k: round(lv, 1) for k, (_, lv) in stems.items()}}
    return out, info
