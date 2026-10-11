"""Listening examples for the overnight report: results/overnight/audio/.

    uv run --group afx --group sep python overnight_audio.py --ours cmaes_ours

For a few situations, picks the case where our matcher's tone_db is closest to its median (a
typical case, not a cherry-picked one), rebuilds it exactly as run.py did (same seed), and writes
6 s, 48 kHz, 16-bit WAVs, all at -20 dB RMS:
  1-target     what the matcher heard (the song excerpt, after production / room / separation)
  2-reference  the true tone on the held-out eval passage (our amp with the true knobs, or the real
               mic'd amp)
  3-default, 4-cmaes_afx, 5-ours   each estimate rendered on that same eval passage
Compare 3-5 against 2: they play the same notes, so only the tone differs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from tonebench import SR, amp, cases, dsp

HERE = Path(__file__).parent
OUT = HERE / "results" / "overnight" / "audio"
PICKS = [("full-different", "real"), ("full-different", "synthetic"), ("room-different", "real"),
         ("room-solo-demucs", "synthetic"), ("full-band-demucs", "real"), ("full-band-demucs", "synthetic")]


def build(args, ids):
    cs = cases.synthetic(args["synthetic"], args["seed"], args["level"], args["variant"], args.get("band", "none"),
                         args.get("calib", "none"))
    cs += cases.real(args["seed"], args["real_per_take"], args["level"], args["variant"], args.get("band", "none"),
                     args.get("calib", "none"))
    cs = [c for c in cs if c.id in ids]
    if args.get("band") in ("demucs", "solo_demucs"):
        from tonebench import separate
        separate.apply(cs)
    return {c.id: c for c in cs}


def write(path, x):
    x = dsp.normalize(np.asarray(x, float)[: 6 * SR], -20.0)
    x = x / max(1.0, np.max(np.abs(x)) / 0.98)
    sf.write(path, x, SR, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="cmaes_ours")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for k, (sit, track) in enumerate(PICKS, 1):
        fs = sorted((HERE / "results").glob(f"run-*-overnight-{sit}.json"))
        if not fs:
            continue
        d = json.loads(fs[-1].read_text())
        res = [r for r in d["results"] if r["track"] == track]
        ours = [r for r in res if r["matcher"] == a.ours]
        med = np.median([r["tone_db"] for r in ours])
        pick = min(ours, key=lambda r: abs(r["tone_db"] - med))["case"]
        c = build(d["args"], {pick})[pick]
        est = {r["matcher"]: r for r in res if r["case"] == pick}
        stem = OUT / f"{k:02d}-{sit}-{track}-{pick}"
        write(f"{stem}-1-target.wav", c.target)
        write(f"{stem}-2-reference.wav", c.eval_ref)
        for i, m in ((3, "default"), (4, "cmaes_afx"), (5, a.ours)):
            p = amp.AmpParams(**est[m]["est"])
            write(f"{stem}-{i}-{'ours-' + m if m == a.ours else m}.wav", amp.render(p, c.eval_di))
        print(stem.name, {m: round(est[m]["tone_db"], 2) for m in ("default", "cmaes_afx", a.ours)}, flush=True)


if __name__ == "__main__":
    main()
