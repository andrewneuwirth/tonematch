"""Baseline matchers. Each takes (target, match_di, rng) and returns AmpParams.

These are the floor every real method has to beat:
- default: every knob at noon. "Doing nothing."
- random: random search, keeping the settings whose render of the user's DI has the closest
  long-term spectrum to the target.
- random_powell: random search, then Powell's method (derivative-free local search) from the best.

Experiments (2026-10-11):
- cmaes_ltas: CMA-ES on the same long-term-spectrum cost (does a better optimizer help?).
- aligned: for play-along cases. Aligns the user's DI to the song, then CMA-ES on a time-aligned
  multi-resolution STFT cost (trimmed, so wrong notes don't dominate).
- aligned_dry: aligned, but the cost ignores reverb-tail frames of the target (dereverb-lite).
- cmaes_afx: CMA-ES maximizing AFx-Rep embedding similarity (the ST-ITO recipe). Needs torch:
  `uv run --group afx ...`.
- cmaes_afx_spec: AFx-Rep + spectral statistics + a pull toward noon, each cost term scaled to 1 at
  the default setting (see _afx_spec_cost).
"""

import warnings

import numpy as np
from scipy import optimize

from . import amp, metrics

FIT_SECONDS = 4.0


def _cost_fn(target, match_di):
    di = match_di[: int(FIT_SECONDS * 48000)]
    tb = metrics.ltas_bands(target)

    def cost(v):
        y = amp.render(amp.AmpParams.from_vector(v), di)
        return float(np.sqrt(np.mean((metrics.ltas_bands(y) - tb) ** 2)))

    return cost


def default(target, match_di, rng):
    return amp.AmpParams()


def random_search(target, match_di, rng, n=120):
    cost = _cost_fn(target, match_di)
    cands = rng.uniform(0, 1, (n, len(amp.KNOBS)))
    costs = [cost(v) for v in cands]
    return amp.AmpParams.from_vector(cands[int(np.argmin(costs))])


def random_powell(target, match_di, rng, n_random=60, max_evals=150):
    cost = _cost_fn(target, match_di)
    cands = rng.uniform(0, 1, (n_random, len(amp.KNOBS)))
    best = cands[int(np.argmin([cost(v) for v in cands]))]
    res = optimize.minimize(cost, best, method="Powell", bounds=[(0, 1)] * len(amp.KNOBS),
                            options={"maxfev": max_evals, "xtol": 1e-3, "ftol": 1e-4})
    return amp.AmpParams.from_vector(res.x)


def _cmaes(cost_batch, rng, popsize=12, iters=25, sigma0=0.3, x0=None):
    """Bounded CMA-ES from noon (or x0). cost_batch maps a list of vectors to a list of costs.
    Returns the best vector seen. ST-ITO used popsize 64, 25 steps, sigma0 0.3; we use a smaller
    population to keep runtime sane (12 x 25 = 300 evaluations)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cma
    k = len(amp.KNOBS)
    es = cma.CMAEvolutionStrategy(np.full(k, 0.5) if x0 is None else x0, sigma0,
                                  {"bounds": [0, 1], "popsize": popsize, "maxiter": iters, "verbose": -9,
                                   "seed": int(rng.integers(1, 2**31))})
    best, best_c = np.full(k, 0.5), np.inf
    while not es.stop():
        xs = es.ask()
        cs = cost_batch(xs)
        es.tell(xs, cs)
        i = int(np.argmin(cs))
        if cs[i] < best_c:
            best, best_c = np.asarray(xs[i]), cs[i]
    return best


def cmaes_ltas(target, match_di, rng):
    cost = _cost_fn(target, match_di)
    return amp.AmpParams.from_vector(_cmaes(lambda xs: [cost(v) for v in xs], rng))


def aligned(target, match_di, rng):
    """Play-along matcher: time-align the DI to the song, then fit on an aligned MR-STFT cost."""
    from . import align
    di, _ = align.align(target, match_di)
    n = min(len(di), len(target))
    di, tg = di[:n], target[:n]

    def cost_batch(xs):
        return [metrics.mrstft_trimmed_db(amp.render(amp.AmpParams.from_vector(v), di), tg) for v in xs]

    return amp.AmpParams.from_vector(_cmaes(cost_batch, rng, sigma0=0.25))


def aligned_dry(target, match_di, rng):
    """aligned, plus dereverb-lite: the cost skips frames where the target is in a decaying
    (reverb-dominated) stretch (metrics.direct_mask). For listen mode / reverberant targets."""
    from . import align
    di, _ = align.align(target, match_di)
    n = min(len(di), len(target))
    di, tg = di[:n], target[:n]
    mask = metrics.direct_mask(tg)

    def cost_batch(xs):
        return [metrics.mrstft_masked_db(amp.render(amp.AmpParams.from_vector(v), di), tg, mask) for v in xs]

    return amp.AmpParams.from_vector(_cmaes(cost_batch, rng, sigma0=0.25))


def _afx_cost(target, match_di):
    from . import afxrep
    et = afxrep.embed([target])[0]

    def cost_batch(xs):
        ys = [amp.render(amp.AmpParams.from_vector(v), match_di) for v in xs]
        return 1.0 - afxrep.embed(ys) @ et, ys

    return cost_batch


def cmaes_afx(target, match_di, rng):
    cb = _afx_cost(target, match_di)
    return amp.AmpParams.from_vector(_cmaes(lambda xs: list(cb(xs)[0]), rng))


W_SPEC, W_PRIOR = 0.25, 0.25


def _afx_spec_cost(target, match_di):
    """AFx-Rep distance + spectral-statistics distance + pull toward noon.

    Adapted from the literature review: AFx-Rep alone misses EQ detail (Chen 2024) and fails on
    guitar (ST-ITO), while the long-term spectrum alone overfits mix EQ and pushes knobs to the
    rails (this bench). Each distance is divided by its value at the default setting, so "1" means
    "no better than noon". The prior W_PRIOR * mean((v - 0.5)^2) / 0.25 (= W_PRIOR with every knob
    at a rail) keeps knobs off the rails unless the evidence is strong (the identifiability problem
    in section 3 of the review).
    """
    cb = _afx_cost(target, match_di)
    st = metrics.band_stats(target)
    spec = lambda y: float(np.sqrt(np.mean((metrics.band_stats(y) - st) ** 2)))
    d0, ys0 = cb([np.full(len(amp.KNOBS), 0.5)])
    a0, s0 = max(d0[0], 1e-4), max(spec(ys0[0]), 1e-4)

    def cost_batch(xs):
        d, ys = cb(xs)
        return [d[i] / a0 + W_SPEC * spec(ys[i]) / s0 + W_PRIOR * np.mean((np.asarray(v) - 0.5) ** 2) / 0.25
                for i, v in enumerate(xs)]

    return cost_batch


def cmaes_afx_spec(target, match_di, rng):
    return amp.AmpParams.from_vector(_cmaes(_afx_spec_cost(target, match_di), rng))


MATCHERS = {"default": default, "random": random_search, "random_powell": random_powell,
            "cmaes_ltas": cmaes_ltas, "aligned": aligned, "aligned_dry": aligned_dry, "cmaes_afx": cmaes_afx, "cmaes_afx_spec": cmaes_afx_spec}
# Run by default only when no --matchers are given; the rest are opt-in (play-along / torch).
DEFAULT_MATCHERS = ["default", "random", "random_powell"]


# ---- ToneHub's own encoder (tonebench.encoder, trained by train_encoder.py) -------------------

def _ours_spec_cost(target, match_di, center=None):
    """cmaes_afx_spec's cost with our encoder in place of AFx-Rep: our embedding distance + spectral
    statistics + a pull toward `center` (noon by default), each scaled to 1 at the default setting."""
    from . import encoder
    et = encoder.embed([target])[0]
    st = metrics.band_stats(target)
    spec = lambda y: float(np.sqrt(np.mean((metrics.band_stats(y) - st) ** 2)))
    c = np.full(len(amp.KNOBS), 0.5) if center is None else np.asarray(center)

    def raw(xs):
        ys = [amp.render(amp.AmpParams.from_vector(v), match_di) for v in xs]
        return 1.0 - encoder.embed(ys) @ et, ys

    d0, ys0 = raw([np.full(len(amp.KNOBS), 0.5)])
    a0, s0 = max(d0[0], 1e-4), max(spec(ys0[0]), 1e-4)

    def cost_batch(xs):
        d, ys = raw(xs)
        return [d[i] / a0 + W_SPEC * spec(ys[i]) / s0 + W_PRIOR * np.mean((np.asarray(v) - c) ** 2) / 0.25
                for i, v in enumerate(xs)]

    return cost_batch


def cmaes_ours(target, match_di, rng):
    """CMA-ES from noon on our encoder distance + the cmaes_afx_spec spectral and prior terms."""
    return amp.AmpParams.from_vector(_cmaes(_ours_spec_cost(target, match_di), rng))


def pred_then_search(target, match_di, rng):
    """Predict-then-search: our knob head reads the target, CMA-ES starts there (sigma 0.2) with the
    cmaes_ours cost, its prior pulling toward the prediction instead of noon."""
    from . import encoder
    k0 = np.clip(encoder.run([target])[1][0], 0.02, 0.98)
    return amp.AmpParams.from_vector(_cmaes(_ours_spec_cost(target, match_di, center=k0), rng, sigma0=0.2, x0=k0))


def pred_only(target, match_di, rng):
    """Our knob head alone (no search)."""
    from . import encoder
    return amp.AmpParams.from_vector(encoder.run([target])[1][0])


MATCHERS.update({"cmaes_ours": cmaes_ours, "pred_then_search": pred_then_search, "pred_only": pred_only})
