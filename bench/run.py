"""Run the tone-match benchmark.

    uv run python run.py                                   # baselines, 24 synthetic + 24 real cases
    uv run python run.py --variant playalong --matchers default aligned
    uv run python run.py --level room --matchers default random
    uv run --group afx python run.py --matchers default cmaes_afx cmaes_afx_spec   # needs torch
    uv run --group afx --group sep python run.py --band demucs --matchers default cmaes_afx   # full band
    uv run python run.py --band spleeter_hf --matchers default pred_only   # Spleeter, in its own env (separate.py)
    uv run --group afx python run.py --level room --calib eq --matchers default cmaes_afx      # listen mode

Scores (lower is better):
- tone_db:   multi-resolution STFT log-magnitude error on a held-out passage (the main score)
- ltas_db:   long-term spectrum error on the same passage
- mel_db:    log-mel distance on the same passage (cross-check: mel bands, not linear STFT bins)
- knob_err:  mean absolute knob error, 0..1 scale (synthetic only; settings are known)
Medians come with a 95% percentile-bootstrap interval; "vs default" is the median per-case
difference from the default matcher on the same cases (negative = better), "wins" the share of
cases where the matcher beats default.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tonebench import amp, cases, metrics, stats
from tonebench.matchers import DEFAULT_MATCHERS, MATCHERS


def evaluate(case, est):
    y = amp.render(est, case.eval_di)
    row = {"tone_db": metrics.mrstft_db(y, case.eval_ref), "ltas_db": metrics.ltas_db(y, case.eval_ref),
           "mel_db": metrics.mel_db(y, case.eval_ref)}
    if case.true is not None:
        row["knob_err"] = float(np.mean(np.abs(est.vector() - case.true.vector())))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=24)
    ap.add_argument("--real", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--real-per-take", type=int, default=2, help="cases per Guitar-TECHS excerpt (12 excerpts)")
    ap.add_argument("--level", default="full", choices=["none", "mix", "full", "room"])
    ap.add_argument("--variant", default="different", choices=cases.VARIANTS)
    ap.add_argument("--band", default="none", choices=cases.BANDS,
                    help="guitar alone, or in a full mix: oracle / Demucs / Spleeter separation, or the raw mix")
    ap.add_argument("--calib", default="none", choices=["none", "eq", "eq_denoise"],
                    help="listen mode (--level room): clean the target with a one-time calibration first")
    ap.add_argument("--matchers", nargs="*", default=DEFAULT_MATCHERS, choices=list(MATCHERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if "default" not in args.matchers:
        args.matchers = ["default"] + args.matchers  # every table is relative to default

    t0 = time.time()
    all_cases = cases.synthetic(args.synthetic, args.seed, args.level, args.variant, args.band, args.calib)
    if args.real:
        all_cases += cases.real(args.seed, args.real_per_take, args.level, args.variant, args.band, args.calib)
    if args.band in cases.SEPARATED:
        from tonebench import separate
        separate.apply(all_cases, cases.SEPARATED[args.band])
    print(f"{len(all_cases)} cases ({args.level}, {args.variant}, band={args.band}, calib={args.calib}) built in {time.time() - t0:.1f}s")

    results = []
    for name in args.matchers:
        fn = MATCHERS[name]
        t1 = time.time()
        for c in all_cases:
            est = fn(c.target, c.match_di, np.random.default_rng(args.seed))
            results.append({"matcher": name, "case": c.id, "track": c.track, **evaluate(c, est),
                            "est": est.as_dict(), "true": c.true.as_dict() if c.true else None})
        print(f"  {name}: {time.time() - t1:.1f}s", flush=True)

    print()
    rows = stats.summarize(results, args.matchers)
    stats.print_table(rows)

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    tag = f"-{args.tag}" if args.tag else ""
    path = out / f"run-{time.strftime('%Y%m%d-%H%M%S')}{tag}.json"
    path.write_text(json.dumps({"args": vars(args), "results": results, "summary": rows}, indent=1))
    print(f"\nsaved {path.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
