"""Paired, per-case comparison of saved runs against a baseline run (same seed and cases).

    uv run python paired.py results/run-A.json results/run-B.json [results/run-C.json ...]

For each later run, matcher and track: median tone_db of the baseline (A) and the run, the median of
the per-case difference run minus A with a 95% bootstrap CI (negative = the run is better), the
mean difference, and wins / losses (cases where the run is lower / higher). mel_db alongside.
Cases are matched by id, so the runs must share --seed, --synthetic and --real-per-take.
"""

import json
import sys
from pathlib import Path

import numpy as np

from tonebench import stats


def label(run, path):
    a = run["args"]
    return f"{a.get('level')}/{a.get('band', 'none')}  ({Path(path).name})"


def compare(base, other, key):
    rows = []
    b = {(r["matcher"], r["case"]): r for r in base["results"]}
    matchers = list(dict.fromkeys(r["matcher"] for r in other["results"]))
    for m in matchers:
        for track in ("synthetic", "real"):
            pairs = [(b[(m, r["case"])][key], r[key]) for r in other["results"]
                     if r["matcher"] == m and r["track"] == track and (m, r["case"]) in b]
            if not pairs:
                continue
            x0, x1 = np.array(pairs).T
            d = x1 - x0
            rows.append((m, track, len(d), np.median(x0), np.median(x1), stats.median_ci(d), d.mean(),
                         int((d < 0).sum()), int((d > 0).sum())))
    return rows


def main():
    paths = sys.argv[1:]
    base = json.load(open(paths[0]))
    print(f"baseline: {label(base, paths[0])}")
    for p in paths[1:]:
        run = json.load(open(p))
        print(f"\n{label(run, p)} minus baseline")
        for key in ("tone_db", "mel_db"):
            print(f"  {key:<8}{'matcher':<18}{'track':<10}{'n':>3}{'base':>7}{'run':>7}  "
                  f"{'paired median [95% CI]':<25}{'mean':>7}  better/worse")
            for m, t, n, a, b_, (med, lo, hi), mean, w, l in compare(base, run, key):
                print(f"  {'':<8}{m:<18}{t:<10}{n:>3}{a:7.2f}{b_:7.2f}  "
                      f"{f'{med:+.2f} [{lo:+.2f}, {hi:+.2f}]':<25}{mean:+7.2f}  {w}/{l}")


if __name__ == "__main__":
    main()
