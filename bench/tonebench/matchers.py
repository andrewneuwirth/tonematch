"""Baseline matchers. Each takes (target, match_di, rng) and returns AmpParams.

These are the floor every real method has to beat:
- default: every knob at noon. "Doing nothing."
- random: random search, keeping the settings whose render of the user's DI has the closest
  long-term spectrum to the target.
- random_powell: random search, then Powell's method (derivative-free local search) from the best.
"""

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


MATCHERS = {"default": default, "random": random_search, "random_powell": random_powell}
