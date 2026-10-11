"""Benchmark cases.

Every case gives a matcher two things, like a real user: the song (target) and the user's own DI
playing something else (match_di). It is scored on a third, held-out passage (eval_di), rendered
with the estimated settings and compared with the reference (eval_ref), so the score is about
tone and not about which notes were played.
"""

from dataclasses import dataclass, field

import numpy as np

from . import amp, data, production


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


SEG_MATCH = 6.0
SEG_EVAL = 6.0


def synthetic(n, seed=0, level="full"):
    rng = np.random.default_rng(seed)
    takes = data.music_ids()
    out = []
    for i in range(n):
        a, b, c = rng.choice(takes, 3, replace=False)
        true = amp.random_params(rng)
        _, seg = data.segment(data.di(a), SEG_MATCH, rng)
        target, prod = production.produce(amp.render(true, seg), rng, level)
        _, mdi = data.segment(data.di(b), SEG_MATCH, rng)
        _, edi = data.segment(data.di(c), SEG_EVAL, rng)
        out.append(Case(f"syn-{level}-{i:02d}", "synthetic", target, mdi, edi, amp.render(true, edi), true,
                        {"takes": [a, b, c], **prod}))
    return out


def real(seed=0):
    """One case per Guitar-TECHS excerpt: match the mic'd amp, score on another passage of the same take."""
    rng = np.random.default_rng(seed)
    takes = data.music_ids()
    out = []
    for a in takes:
        d, m = data.di(a), data.micamp(a)
        half = len(d) // 2  # target from one half, scoring from the other, so they never overlap
        s, _ = data.segment(d[:half], SEG_MATCH, rng)
        target = m[s : s + int(SEG_MATCH * 48000)]
        e, _ = data.segment(d[half:], SEG_EVAL, rng)
        e += half
        n = int(SEG_EVAL * 48000)
        b = rng.choice([t for t in takes if t != a])
        _, mdi = data.segment(data.di(b), SEG_MATCH, rng)
        out.append(Case(f"real-{a}", "real", target, mdi, d[e : e + n], m[e : e + n], None, {"takes": [a, b]}))
    return out
