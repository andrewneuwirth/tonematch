"""Benchmark cases.

Every case gives a matcher two things, like a real user: the song (target) and the user's own DI
(match_di). It is scored on a third, held-out passage (eval_di), rendered with the estimated
settings and compared with the reference (eval_ref), so the score is about tone and not about
which notes were played.

Variants of match_di:
- "different": the user plays something else (another take). The original setup.
- "playalong": the user plays the SAME part along with the song: a bit early or late overall,
  drifting timing (+-25 ms), and a few wrong notes.
- "playalong_exact": the exact DI that made the target (an upper bound for play-along).

Targets, eval passages and seeds are identical across variants and across --level full / room,
so results can be compared case by case.

Band (--band) puts the guitar in a full mix (tonebench.band: drums, bass, sometimes keys and a
vocal-like lead), runs production / room on the mix, then gives the matcher a guitar-only target:
- "none":   no band, the guitar alone (the original cases, a best case).
- "oracle": perfect separation: the guitar's own share of the produced mix (production.produce_stems).
- "demucs": the produced mix separated by Demucs htdemucs_6s (tonebench.separate; bench only).
- "solo_demucs": no band, but the guitar-only target still goes through Demucs (separation as a
  cleanup step: does it remove listen-mode noise and room?).
- "spleeter": the produced mix separated by Spleeter 4stems, the "other" stem (tonebench.separate).
- "spleeter_hf": the same, with the mix passed through above 11 kHz (Spleeter's stem stops there).
- "mix": no separation: the matcher gets the whole produced mix (what the app matches on today).
Real cases follow their guitar-only convention: no studio production at --level full (the mic'd amp
is just summed with the band), room() at --level room.

calib (--calib, room level only): clean the listen-mode target up before matching. "eq": a one-time
calibration (production.calibrate_eq: known pink noise played through the same speaker/room/mic,
inverse EQ); "eq_denoise": that plus spectral-subtraction noise suppression (production.denoise).
"""

from dataclasses import dataclass, field

import numpy as np

from . import SR, amp, band as bandsynth, data, dsp, production


@dataclass
class Case:
    id: str
    track: str                 # "synthetic" or "real"
    target: np.ndarray
    match_di: np.ndarray
    eval_di: np.ndarray
    eval_ref: np.ndarray
    true: amp.AmpParams | None = None
    info: dict = field(default_factory=dict)
    mix: np.ndarray | None = None  # full-band cases: the produced mix the separator sees


SEG_MATCH = 6.0
SEG_EVAL = 6.0
VARIANTS = ("different", "playalong", "playalong_exact")
BANDS = ("none", "oracle", "demucs", "solo_demucs", "spleeter", "spleeter_hf", "mix")
SEPARATED = {"demucs": "demucs", "solo_demucs": "demucs", "spleeter": "spleeter", "spleeter_hf": "spleeter_hf"}


def perform(src, start, n, rng, other, max_offset=0.06, max_jitter=0.025, fade=0.01):
    """Re-play src[start:start+n] the way someone playing along would.

    Overall offset up to +-max_offset s, timing that drifts in a random walk (bounded by
    +-max_jitter s, changing every 0.8-1.6 s), and 2-4 wrong notes (0.25-0.6 s stretches
    replaced by playing from `other`). No pitch change: chunks are re-cut and crossfaded.
    """
    f = int(fade * SR)
    off = rng.uniform(-max_offset, max_offset) * SR
    out = np.zeros(n)
    t, jit = 0, 0.0
    while t < n:
        L = int(rng.uniform(0.8, 1.6) * SR)
        jit = float(np.clip(jit + rng.uniform(-0.012, 0.012), -max_jitter, max_jitter))
        i = int(start + t + off + jit * SR)
        piece = np.zeros(L + f)
        lo, hi = max(i, 0), min(i + L + f, len(src))
        if hi > lo:
            piece[lo - i : hi - i] = src[lo:hi]
        w = np.ones(L + f)
        if t > 0:
            w[:f] = np.linspace(0, 1, f)
        w[-f:] = np.linspace(1, 0, f)
        end = min(n, t + L + f)
        out[t:end] += (piece * w)[: end - t]
        t += L
    for _ in range(int(rng.integers(2, 5))):
        L = int(rng.uniform(0.25, 0.6) * SR)
        p = int(rng.integers(0, n - L))
        q = int(rng.integers(0, len(other) - L))
        wrong = other[q : q + L] * 10 ** ((dsp.rms_db(out[p : p + L]) - dsp.rms_db(other[q : q + L])) / 20)
        w = np.ones(L)
        w[:f], w[-f:] = np.linspace(0, 1, f), np.linspace(1, 0, f)
        out[p : p + L] = out[p : p + L] * (1 - w) + wrong * w
    return out


def _match_di(variant, src, start, other_mdi, other_src, prng):
    n = int(SEG_MATCH * SR)
    if variant == "different":
        return other_mdi
    if variant == "playalong_exact":
        return src[start : start + n].copy()
    if variant == "playalong":
        return perform(src, start, n, prng, other_src)
    raise ValueError(variant)


def _band_target(guitar, level, prod_rng, room_rng, band_rng, studio=True, calib="none"):
    """Mix guitar (normalized to -20 dB) with a synthetic band, produce, return (oracle guitar, mix, info)."""
    g = dsp.normalize(guitar, -20.0)
    stems, binfo = bandsynth.backing(len(g), band_rng)
    mix = g + sum(stems.values())
    info = {"band": binfo}
    if studio:
        mix, (g,), pinfo = production.produce_stems(mix, [g], prod_rng, "full" if level == "room" else level)
        info.update(pinfo)
    else:
        mix, g = dsp.normalize(mix, -20.0), g * 10 ** ((-20.0 - dsp.rms_db(mix)) / 20)
    if level == "room":
        mix, (g,), extra = production.room_stems(mix, [g], room_rng, calib)
        info.update(level="room", **extra)
    return g, mix, info


def synthetic(n, seed=0, level="full", variant="different", band="none", calib="none"):
    rng = np.random.default_rng(seed)
    takes = data.music_ids()
    out = []
    for i in range(n):
        a, b, c = rng.choice(takes, 3, replace=False)
        true = amp.random_params(rng)
        s, seg = data.segment(data.di(a), SEG_MATCH, rng)
        mix = None
        if band in ("none", "solo_demucs"):
            target, prod = production.produce(amp.render(true, seg), rng, "full" if level == "room" else level)
            if level == "room":  # separate stream, so room cases pair with full-level cases
                target, extra = production.room(target, np.random.default_rng([seed, i, 1]), calib)
                prod = {**prod, "level": "room", **extra}
        else:  # same production draws as band="none" (they come first from rng), plus the band
            target, mix, prod = _band_target(amp.render(true, seg), level, rng, np.random.default_rng([seed, i, 1]),
                                             np.random.default_rng([seed, i, 3]), calib=calib)
            if band == "mix":
                target = mix
        _, mdi = data.segment(data.di(b), SEG_MATCH, rng)
        _, edi = data.segment(data.di(c), SEG_EVAL, rng)
        mdi = _match_di(variant, data.di(a), s, mdi, data.di(b), np.random.default_rng([seed, i, 2]))
        if band == "solo_demucs":
            mix = target
        out.append(Case(f"syn-{level}-{i:02d}", "synthetic", target, mdi, edi, amp.render(true, edi), true,
                        {"takes": [a, b, c], "variant": variant, **prod}, mix))
    return out


def real(seed=0, per_take=1, level="full", variant="different", band="none", calib="none"):
    """Guitar-TECHS excerpts: match the mic'd amp, score on another passage of the same take.

    per_take > 1 adds more cases per excerpt, drawn with seeds seed+1000, seed+2000, ...; the
    first round is identical to per_take=1. level="room" plays the mic'd-amp target through
    production.room() (no studio production: these are bare amp recordings).
    """
    out = []
    for k in range(per_take):
        rng = np.random.default_rng(seed + 1000 * k)
        takes = data.music_ids()
        for j, a in enumerate(takes):
            d, m = data.di(a), data.micamp(a)
            half = len(d) // 2  # target from one half, scoring from the other, so they never overlap
            s, _ = data.segment(d[:half], SEG_MATCH, rng)
            target = m[s : s + int(SEG_MATCH * 48000)]
            e, _ = data.segment(d[half:], SEG_EVAL, rng)
            e += half
            n = int(SEG_EVAL * 48000)
            b = rng.choice([t for t in takes if t != a])
            _, mdi = data.segment(data.di(b), SEG_MATCH, rng)
            info = {"takes": [a, b], "variant": variant}
            mix = None
            if band not in ("none", "solo_demucs"):
                target, mix, extra = _band_target(target, level, None, np.random.default_rng([seed, k, j, 1]),
                                                  np.random.default_rng([seed, k, j, 3]), studio=False, calib=calib)
                if band == "mix":
                    target = mix
                info.update(extra)
            elif level == "room":
                target, extra = production.room(dsp.normalize(target, -20.0), np.random.default_rng([seed, k, j, 1]),
                                                calib)
                info.update(level="room", **extra)
            mdi = _match_di(variant, d, s, mdi, data.di(b), np.random.default_rng([seed, k, j, 2]))
            cid = f"real-{a}" + (f"-{k}" if k else "")
            if band == "solo_demucs":
                mix = target
            out.append(Case(cid, "real", target, mdi, d[e : e + n], m[e : e + n], None, info, mix))
    return out
