"""Medians with simple percentile-bootstrap confidence intervals."""

import numpy as np


def median_ci(x, n_boot=5000, ci=0.95, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, len(x), (n_boot, len(x)))], axis=1)
    lo, hi = np.quantile(meds, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(np.median(x)), float(lo), float(hi)


def summarize(results, matchers, baseline="default", key="tone_db"):
    """Rows of (matcher, track, n, score median+CI, paired median diff vs baseline + CI, win rate, ltas, knob).
    key picks the score (tone_db by default; mel_db for the cross-check)."""
    rows = []
    for name in matchers:
        for track in ("synthetic", "real"):
            rs = [r for r in results if r["matcher"] == name and r["track"] == track]
            if not rs:
                continue
            row = {"matcher": name, "track": track, "n": len(rs), "tone": median_ci([r[key] for r in rs]),
                   "ltas": float(np.median([r["ltas_db"] for r in rs]))}
            if "knob_err" in rs[0]:
                row["knob"] = float(np.median([r["knob_err"] for r in rs]))
            base = {r["case"]: r[key] for r in results if r["matcher"] == baseline and r["track"] == track}
            if name != baseline and base:
                d = [r[key] - base[r["case"]] for r in rs if r["case"] in base]
                row["diff"] = median_ci(d)
                row["win"] = float(np.mean(np.array(d) < 0))
            rows.append(row)
    return rows


def print_table(rows, baseline="default", key="tone_db"):
    h = (f"{'matcher':<16}{'track':<10}{'n':>3}  {key + ' [95% CI]':<22}{'vs ' + baseline + ' [95% CI]':<26}"
         f"{'wins':>5}{'ltas_db':>9}{'knob_err':>9}")
    print(h)
    print("-" * len(h))
    for r in rows:
        t = r["tone"]
        d = f"{r['diff'][0]:+.2f} [{r['diff'][1]:+.2f}, {r['diff'][2]:+.2f}]" if "diff" in r else ""
        w = f"{r['win']:.0%}" if "win" in r else ""
        k = f"{r['knob']:.3f}" if "knob" in r else ""
        print(f"{r['matcher']:<16}{r['track']:<10}{r['n']:>3}  {t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]".ljust(53)
              + f"{d:<26}{w:>5}{r['ltas']:9.2f}{k:>9}")
