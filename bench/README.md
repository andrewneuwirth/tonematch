# tonebench

Measures how well a tone matcher recovers a guitar tone. Python, run with uv.

```
cd bench
uv run python run.py                                   # 12 synthetic + 12 real cases, all matchers
uv run python run.py --synthetic 24 --level mix        # more cases, lighter production
uv run python run.py --matchers default random         # only some matchers
```

Needs Guitar-TECHS `P3_music` in `../data/guitar-techs/` (see `docs/LICENSES.md`).

## How a case works
A matcher gets what a real user has: the **song** (target) and the user's **own DI playing something else**. It returns knob settings. Scoring renders a **third, held-out passage** with those settings and compares it to the reference, so the score measures tone, not which notes were played.

| Track | Target | Reference for scoring | Knob error? |
|---|---|---|---|
| synthetic | DI through our amp model with random known settings, then random production (`--level none / mix / full`: EQ, compression, reverb, MP3) | the true settings on the held-out DI | yes |
| real | Guitar-TECHS mic'd amp recording of one half of a take | the mic'd amp on a passage from the other half | no (settings unknown) |

## Scores (lower is better)
- `tone_db`: multi-resolution STFT log-magnitude error on the held-out passage, floored 80 dB below peak so noise-floor differences don't count. **Main score.**
- `ltas_db`: long-term spectrum (1/3-octave) error.
- `knob_err`: mean absolute knob error on a 0..1 scale (synthetic only).

## Baselines (2026-10-11, seed 0, 12 + 12 cases, full production)

| matcher | track | tone_db | ltas_db | knob_err |
|---|---|---|---|---|
| default (all knobs at noon) | synthetic | **1.49** | 4.03 | **0.240** |
| default | real | **3.75** | 6.34 | |
| random search, long-term spectrum cost | synthetic | 2.14 | 4.05 | 0.287 |
| random search | real | 5.24 | 6.04 | |
| random + Powell refinement | synthetic | 1.96 | 4.38 | 0.333 |
| random + Powell | real | 5.21 | 6.25 | |

**Finding:** fitting the long-term spectrum is *worse than not fitting at all*. It does so on every track, even without production (bare amp: knob error 0.27 vs 0.23 for default). The long-term spectrum can't tell gain from treble from cab brightness, and it absorbs the mix EQ and reverb into the knobs. It pushes knobs to wrong extremes, the same failure seen in BellyDSP's matcher (12-25/100 with knobs at their limits). Any real method must beat `default`, not just random.

## Known limits
- Our amp model is simple and fixed; it can't reproduce the real amp's mic and room, which caps the real track.
- No oversampling in the model; target and estimate alias identically, so it doesn't bias synthetic scores.
- 12 cases per track is small; differences under ~0.3 dB are noise until we add more cases and seeds.
