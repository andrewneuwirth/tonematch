"""Run the tone-match benchmark.

    uv run python run.py                       # all matchers, 12 synthetic + 12 real cases
    uv run python run.py --synthetic 24 --level mix --matchers default random

Scores (lower is better):
- tone_db:   multi-resolution STFT log-magnitude error on a held-out passage (the main score)
- ltas_db:   long-term spectrum error on the same passage
- knob_err:  mean absolute knob error, 0..1 scale (synthetic only; settings are known)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tonebench import amp, cases, metrics
from tonebench.matchers import MATCHERS


def evaluate(case, est):
    y = amp.render(est, case.eval_di)
    row = {"tone_db": metrics.mrstft_db(y, case.eval_ref), "ltas_db": metrics.ltas_db(y, case.eval_ref)}
    if case.true is not None:
        row["knob_err"] = float(np.mean(np.abs(est.vector() - case.true.vector())))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=12)
    ap.add_argument("--real", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--level", default="full", choices=["none", "mix", "full"])
    ap.add_argument("--matchers", nargs="*", default=list(MATCHERS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    all_cases = cases.synthetic(args.synthetic, args.seed, args.level) + (cases.real(args.seed) if args.real else [])
    print(f"{len(all_cases)} cases built in {time.time() - t0:.1f}s")

    results = []
    for name in args.matchers:
        fn = MATCHERS[name]
        t1 = time.time()
        for c in all_cases:
            est = fn(c.target, c.match_di, np.random.default_rng(args.seed))
            results.append({"matcher": name, "case": c.id, "track": c.track, **evaluate(c, est),
                            "est": est.as_dict(), "true": c.true.as_dict() if c.true else None})
        print(f"  {name}: {time.time() - t1:.1f}s")

    print()
    header = f"{'matcher':<15}{'track':<11}{'tone_db':>9}{'ltas_db':>9}{'knob_err':>10}   (medians, lower is better)"
    print(header)
    print("-" * len(header))
    for name in args.matchers:
        for track in ("synthetic", "real"):
            rows = [r for r in results if r["matcher"] == name and r["track"] == track]
            if not rows:
                continue
            med = lambda k: np.median([r[k] for r in rows]) if k in rows[0] else float("nan")
            print(f"{name:<15}{track:<11}{med('tone_db'):9.2f}{med('ltas_db'):9.2f}{med('knob_err'):10.3f}")

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    path = out / f"run-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"args": vars(args), "results": results}, indent=1))
    print(f"\nsaved {path.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
