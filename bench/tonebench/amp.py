"""Our own gray-box guitar amp + cab model, with knobs in 0..1 like a real amp.

Chain: input high-pass -> tight pre-filter -> preamp gain -> asymmetric soft clip -> tone stack
(bass/mid/treble) -> power-amp soft clip -> presence -> cab (resonance, high-pass, low-pass).

It is deliberately simple. Its job is to produce known-answer test tones that behave like an amp
(gain changes the distortion, the tone stack sits after the clipper, the cab removes fizz), not to
sound like any particular real amp. No oversampling: target and estimate alias the same way.
"""

from dataclasses import dataclass, asdict, fields

import numpy as np

from . import dsp

KNOBS = ("gain", "bass", "mid", "treble", "presence", "cab_tone", "cab_res")


@dataclass
class AmpParams:
    gain: float = 0.5      # preamp drive: 0 -> +0 dB, 1 -> +48 dB
    bass: float = 0.5      # low shelf 110 Hz, +-12 dB
    mid: float = 0.5       # peak 650 Hz, +-12 dB
    treble: float = 0.5    # high shelf 2.8 kHz, +-12 dB
    presence: float = 0.5  # high shelf 4.5 kHz after the power amp, +-8 dB
    cab_tone: float = 0.5  # cab low-pass corner 2.5 kHz (dark) .. 7.5 kHz (bright)
    cab_res: float = 0.5   # cab low resonance at 110 Hz, 0 .. +9 dB

    def vector(self):
        return np.array([getattr(self, k) for k in KNOBS])

    @classmethod
    def from_vector(cls, v):
        v = np.clip(np.asarray(v, float), 0.0, 1.0)
        return cls(**dict(zip(KNOBS, v)))

    def as_dict(self):
        return {k: round(float(v), 4) for k, v in asdict(self).items()}


def _bip(x):  # 0..1 knob -> -1..+1
    return 2.0 * x - 1.0


def render(p: AmpParams, di: np.ndarray) -> np.ndarray:
    x = dsp.apply(di, dsp.butter("highpass", 40, 2))
    # Tighten the lows before the clipper, more so at high gain (as many high-gain preamps do).
    x = dsp.apply(x, dsp.low_shelf(250, -4.0 - 6.0 * p.gain))
    x = dsp.normalize(x, -24.0) * 10 ** (48.0 * p.gain / 20)
    bias = 0.15
    x = np.tanh(x + bias) - np.tanh(bias)
    x = dsp.apply(
        x,
        dsp.low_shelf(110, 12 * _bip(p.bass)),
        dsp.peak(650, 12 * _bip(p.mid), 0.7),
        dsp.high_shelf(2800, 12 * _bip(p.treble)),
    )
    x = np.tanh(1.5 * dsp.normalize(x, -12.0)) / 1.5
    x = dsp.apply(
        x,
        dsp.high_shelf(4500, 8 * _bip(p.presence)),
        dsp.peak(110, 9 * p.cab_res, 1.4),
        dsp.butter("highpass", 75, 2),
        dsp.butter("lowpass", 2500 * 3 ** p.cab_tone, 4),
    )
    return dsp.normalize(x, -20.0)


def random_params(rng: np.random.Generator) -> AmpParams:
    return AmpParams.from_vector(rng.uniform(0.05, 0.95, len(KNOBS)))


assert tuple(f.name for f in fields(AmpParams)) == KNOBS
