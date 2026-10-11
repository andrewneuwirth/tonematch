"""Toy sketch: express a gear-neutral tone-stack curve as knob positions on a specific device.

Design doc: docs/design/gear-aware-output.md. numpy only, runs in well under a second.

What it shows
1. A device profile = knob list (names, dial range, taper) + a gray-box model mapping knob
   rotation -> frequency response. Two toy devices:
   - "fmv": the passive Fender/Marshall/Vox tone stack, using the closed-form transfer function
     of Yeh & Smith, "Discretization of the '59 Fender Bassman Tone Stack", DAFx-06 (Eq. 1),
     with the '59 Bassman component values printed in that paper's Fig. 1. Implemented here
     from the paper's equations, not from any other codebase.
   - "shelf": independent low-shelf / mid-peak / high-shelf, like a digital amp's EQ.
     Corner frequencies and ranges are made up (they match bench/tonebench/amp.py).
2. Fit: grid search over dial positions, then coordinate refinement. Cost is RMS dB error after
   removing the mean level (the volume knob absorbs broadband level).
3. Sensitivity: how much the cost rises if a knob is off by 1 dial number, so the UI can say
   which knobs matter.
4. Reachability: if the best fit leaves a residual, fit a 7-band graphic EQ pedal
   (100 Hz .. 6.4 kHz, +-15 dB, the Boss GE-7 band layout) to the residual and suggest it.

Not modeled (on purpose): distortion, gain staging, cab, mic. Only the linear tone-stack part.

Run:  cd bench && uv run python gearmap/gearmap.py
"""

from dataclasses import dataclass

import numpy as np

FREQS = np.geomspace(80.0, 8000.0, 64)  # analysis band for guitar tone
TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------- tapers
def taper(rotation, kind):
    """Knob rotation 0..1 -> fraction of pot resistance 0..1.

    'B' (linear) or 'A10' / 'A30': audio taper with 10 % / 30 % resistance at mid rotation,
    modeled as an exponential (a**r - 1)/(a - 1). With fraction p at r = 0.5, sqrt(a) = 1/p - 1.
    Real audio pots are often two linear segments; the exponential is an approximation.
    """
    r = np.clip(rotation, 0.0, 1.0)
    if kind == "B":
        return r
    p = {"A10": 0.10, "A30": 0.30}[kind]
    a = (1.0 / p - 1.0) ** 2
    return (a**r - 1.0) / (a - 1.0)


# ---------------------------------------------------------------- device models
def fmv_response_db(t, m, l, comps, f=FREQS):
    """|H(j 2 pi f)| in dB for the FMV stack. t, m, l are pot fractions (after taper).

    Coefficients copied term by term from Yeh & Smith DAFx-06, Eq. 1.
    """
    C1, C2, C3, R1, R2, R3, R4 = comps
    b1 = t * C1 * R1 + m * C3 * R3 + l * (C1 * R2 + C2 * R2) + (C1 * R3 + C2 * R3)
    b2 = (t * (C1 * C2 * R1 * R4 + C1 * C3 * R1 * R4)
          - m**2 * (C1 * C3 * R3**2 + C2 * C3 * R3**2)
          + m * (C1 * C3 * R1 * R3 + C1 * C3 * R3**2 + C2 * C3 * R3**2)
          + l * (C1 * C2 * R1 * R2 + C1 * C2 * R2 * R4 + C1 * C3 * R2 * R4)
          + l * m * (C1 * C3 * R2 * R3 + C2 * C3 * R2 * R3)
          + (C1 * C2 * R1 * R3 + C1 * C2 * R3 * R4 + C1 * C3 * R3 * R4))
    k = C1 * C2 * C3
    b3 = (l * m * (k * R1 * R2 * R3 + k * R2 * R3 * R4)
          - m**2 * (k * R1 * R3**2 + k * R3**2 * R4)
          + m * (k * R1 * R3**2 + k * R3**2 * R4)
          + t * k * R1 * R3 * R4 - t * m * k * R1 * R3 * R4
          + t * l * k * R1 * R2 * R4)
    a0 = 1.0
    a1 = (C1 * R1 + C1 * R3 + C2 * R3 + C2 * R4 + C3 * R4) + m * C3 * R3 + l * (C1 * R2 + C2 * R2)
    a2 = (m * (C1 * C3 * R1 * R3 - C2 * C3 * R3 * R4 + C1 * C3 * R3**2 + C2 * C3 * R3**2)
          + l * m * (C1 * C3 * R2 * R3 + C2 * C3 * R2 * R3)
          - m**2 * (C1 * C3 * R3**2 + C2 * C3 * R3**2)
          + l * (C1 * C2 * R2 * R4 + C1 * C2 * R1 * R2 + C1 * C3 * R2 * R4 + C2 * C3 * R2 * R4)
          + (C1 * C2 * R1 * R4 + C1 * C3 * R1 * R4 + C1 * C2 * R3 * R4
             + C1 * C2 * R1 * R3 + C1 * C3 * R3 * R4 + C2 * C3 * R3 * R4))
    a3 = (l * m * (k * R1 * R2 * R3 + k * R2 * R3 * R4)
          - m**2 * (k * R1 * R3**2 + k * R3**2 * R4)
          + m * (k * R3**2 * R4 + k * R1 * R3**2 - k * R1 * R3 * R4)
          + l * k * R1 * R2 * R4 + k * R1 * R3 * R4)
    s = 1j * TWO_PI * np.asarray(f)[..., :]
    # Broadcast knob arrays (shape [...]) against frequencies (shape [F]).
    ex = lambda c: np.asarray(c)[..., None]
    num = ex(b1) * s + ex(b2) * s**2 + ex(b3) * s**3
    den = ex(a0) + ex(a1) * s + ex(a2) * s**2 + ex(a3) * s**3
    return 20.0 * np.log10(np.abs(num / den) + 1e-12)


def _shelf_low(f, fc, g):  # first-order-ish shelf in dB, smooth in log f
    return g / (1.0 + (f / fc) ** 2)


def _shelf_high(f, fc, g):
    return g / (1.0 + (fc / f) ** 2)


def _peak(f, fc, g, q):
    x = q * (f / fc - fc / f)
    return g / (1.0 + x**2)


def shelf_response_db(bass, mid, treble, f=FREQS):
    ex = lambda c: np.asarray(c)[..., None]
    return (_shelf_low(f, 110.0, 12 * (2 * ex(bass) - 1))
            + _peak(f, 650.0, 12 * (2 * ex(mid) - 1), 0.7)
            + _shelf_high(f, 2800.0, 12 * (2 * ex(treble) - 1)))


# ---------------------------------------------------------------- device profiles
@dataclass
class Knob:
    name: str
    taper: str          # 'B', 'A10', 'A30'
    dial_min: float = 0.0
    dial_max: float = 10.0

    def to_dial(self, rot):
        return self.dial_min + rot * (self.dial_max - self.dial_min)


@dataclass
class Device:
    name: str
    knobs: tuple
    model: callable     # (pot fractions per knob) -> dB response

    def response(self, rotations):
        fr = [taper(rotations[..., i], k.taper) for i, k in enumerate(self.knobs)]
        return self.model(*fr)


BASSMAN_59 = (0.25e-9, 20e-9, 20e-9, 250e3, 1e6, 25e3, 56e3)  # C1 C2 C3 R1 R2 R3 R4, DAFx-06 Fig. 1

# Paper: treble and middle are linear pots, bass is log taper. Order: treble, middle, bass.
FMV = Device("Toy FMV stack ('59 Bassman values)",
             (Knob("Treble", "B"), Knob("Middle", "B"), Knob("Bass", "A10")),
             lambda t, m, l: fmv_response_db(t, m, l, BASSMAN_59))
SHELF = Device("Toy digital amp EQ (independent shelves)",
               (Knob("Treble", "B"), Knob("Middle", "B"), Knob("Bass", "B")),
               lambda t, m, l: shelf_response_db(l, m, t))


# ---------------------------------------------------------------- fitting
def cost(resp, target):
    d = resp - target
    d = d - d.mean(axis=-1, keepdims=True)  # level is the volume knob's job
    return np.sqrt((d**2).mean(axis=-1))


def fit(device, target, grid=21, steps=3):
    g = np.linspace(0.0, 1.0, grid)
    mesh = np.stack(np.meshgrid(*([g] * len(device.knobs)), indexing="ij"), -1).reshape(-1, len(device.knobs))
    c = cost(device.response(mesh), target)
    best = mesh[np.argmin(c)].copy()
    h = 1.0 / (grid - 1)
    for _ in range(steps):  # coordinate refinement at shrinking steps
        h /= 4.0
        for i in range(len(best)):
            cand = np.repeat(best[None], 9, 0)
            cand[:, i] = np.clip(best[i] + h * np.arange(-4, 5), 0, 1)
            best = cand[np.argmin(cost(device.response(cand), target))]
    return best, float(cost(device.response(best), target))


def sensitivity(device, rot, target):
    """Cost increase (dB RMS) if each knob is off by one dial number in the worse direction."""
    base = cost(device.response(rot), target)
    out = []
    for i, k in enumerate(device.knobs):
        step = 1.0 / (k.dial_max - k.dial_min)
        worst = 0.0
        for sgn in (-1, 1):
            r = rot.copy()
            r[i] = np.clip(r[i] + sgn * step, 0, 1)
            worst = max(worst, float(cost(device.response(r), target) - base))
        out.append(worst)
    return out


# ---------------------------------------------------------------- reachability: graphic EQ pedal
GEQ_BANDS = np.array([100, 200, 400, 800, 1600, 3200, 6400.0])  # GE-7 band layout, +-15 dB


def geq_suggest(residual, q=1.2, limit=15.0):
    """Least-squares band gains for a 7-band graphic EQ that best explain `residual` (dB).

    Uses the shape of each band at 1 dB as a linear basis (good for moderate gains). Q is a
    guess; the real pedal's band shapes would come from its device profile. The top band is a
    shelf on the GE-7 per several retailer listings; treated as a peak here for simplicity.
    """
    basis = np.stack([_peak(FREQS, fc, 1.0, q) for fc in GEQ_BANDS], -1)
    r = residual - residual.mean()
    g, *_ = np.linalg.lstsq(np.c_[basis, np.ones_like(FREQS)], r, rcond=None)
    gains = np.clip(np.round(g[:-1] * 2) / 2, -limit, limit)
    after = r - basis @ gains
    return gains, float(np.sqrt(((after - after.mean()) ** 2).mean()))


# ---------------------------------------------------------------- report
def describe(device, rot):
    return ", ".join(f"{k.name} {round(k.to_dial(x) * 2) / 2:g}" for k, x in zip(device.knobs, rot))


def translate(device, target, label, explain_threshold_db=1.0):
    rot, c = fit(device, target)
    sens = sensitivity(device, rot, target)
    print(f"\n== {label}\n   on {device.name}: {describe(device, rot)}   (residual {c:.2f} dB RMS)")
    print("   knob importance (+dB error if off by 1):",
          ", ".join(f"{k.name} {s:.2f}" for k, s in zip(device.knobs, sens)))
    if c > explain_threshold_db:
        resid = target - device.response(rot)
        resid = resid - resid.mean()
        i = np.argmax(np.abs(resid))
        print(f"   cannot reach: largest miss {resid[i]:+.1f} dB at {FREQS[i]:.0f} Hz")
        gains, after = geq_suggest(resid)
        bands = ", ".join(f"{int(f) if f < 1000 else f'{f/1000:g}k'} {g:+g}" for f, g in zip(GEQ_BANDS, gains) if abs(g) >= 1)
        print(f"   suggest graphic EQ after the amp: {bands}  -> residual {after:.2f} dB RMS")
    return rot, c


if __name__ == "__main__":
    # 1) Round trip: target made by the FMV stack itself at known dial positions.
    true_rot = np.array([0.7, 0.3, 0.4])
    rot, _ = translate(FMV, FMV.response(true_rot), f"Round trip, true: {describe(FMV, true_rot)}")
    print("   (different knobs can give near-identical curves; score the curve, not the knobs)")

    # 2) Cross-device: a scooped digital-EQ tone, expressed on the passive stack and back.
    scooped = SHELF.response(np.array([0.75, 0.2, 0.7]))
    translate(FMV, scooped, "Scooped tone from the digital EQ -> passive FMV stack")

    # 3) Unreachable: a strong mid boost. A passive FMV stack can only cut, so it misses.
    mid_push = SHELF.response(np.array([0.5, 0.95, 0.4]))
    translate(FMV, mid_push, "Mid-boosted tone -> passive FMV stack (expected to fall short)")
    translate(SHELF, mid_push, "Same tone -> digital EQ (reachable)")
