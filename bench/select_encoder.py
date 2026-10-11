"""Pick the encoder checkpoint for the bench with a bigger proxy bench (never touches P3_music).

    uv run --group afx python select_encoder.py [--n 48]

Every models/encoder/ckpt-*.pt (and latest.pt) is scored on n bench-like cases drawn from the val
split of every training folder (train_encoder.proxy_cases): median tone_db of the knob-head
prediction and of CMA-ES on the embedding distance. The checkpoint with the lowest mean of the two
is copied to best.pt; results go to models/encoder/selection.csv.
"""

import argparse
import csv
import shutil

import numpy as np
import torch

import train_encoder as T
from tonebench import encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cases = T.proxy_cases(a.n, seed=777)
    paths = sorted(T.OUT.glob("ckpt-*.pt")) + [T.OUT / "latest.pt"]
    rows = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = encoder.Encoder().to(dev)
        m.load_state_dict(ck["model"])
        r = T.proxy_bench(m, cases, dev)
        r.update(ckpt=p.name, step=ck["step"], score=(r["pred"] + r["search"]) / 2)
        rows.append(r)
        print(r, flush=True)
    best = min(rows, key=lambda r: r["score"])
    shutil.copy(T.OUT / best["ckpt"], T.OUT / "best.pt")
    with open(T.OUT / "selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["ckpt", "step", "noon", "pred", "search", "score"])
        w.writeheader()
        w.writerows(rows)
    print("best:", best)


if __name__ == "__main__":
    main()
