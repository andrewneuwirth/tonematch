"""Print the summary table of one or more saved runs.

    uv run python report.py results/run-*.json          # tone_db tables
    uv run python report.py --mel results/run-*.json    # also the mel_db cross-check (re-scored if missing)
"""

import json
import sys

import numpy as np

from tonebench import stats


def add_mel(run):
    """Re-render each saved estimate on its case's held-out passage and score mel_db. Cases are
    deterministic given the run's args; the band setting doesn't change eval passages."""
    from tonebench import amp, cases, metrics
    a = run["args"]
    cs = cases.synthetic(a["synthetic"], a["seed"], a["level"], a.get("variant", "different"))
    if a.get("real", True):
        cs += cases.real(a["seed"], a.get("real_per_take", 1), a["level"], a.get("variant", "different"))
    by_id = {c.id: c for c in cs}
    for r in run["results"]:
        c = by_id[r["case"]]
        y = amp.render(amp.AmpParams(**r["est"]), c.eval_di)
        r["mel_db"] = metrics.mel_db(y, c.eval_ref)


args = sys.argv[1:]
mel = "--mel" in args
for path in [p for p in args if p != "--mel"]:
    run = json.load(open(path))
    a = run["args"]
    print(f"\n{path}  (level={a.get('level')}, variant={a.get('variant', 'different')}, "
          f"band={a.get('band', 'none')}, calib={a.get('calib', 'none')}, seed={a.get('seed')})")
    matchers = list(dict.fromkeys(r["matcher"] for r in run["results"]))
    stats.print_table(stats.summarize(run["results"], matchers))
    if mel:
        if "mel_db" not in run["results"][0]:
            add_mel(run)
            json.dump(run, open(path, "w"), indent=1)
        print()
        stats.print_table(stats.summarize(run["results"], matchers, key="mel_db"), key="mel_db")
