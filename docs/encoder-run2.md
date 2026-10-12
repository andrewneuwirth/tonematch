# Encoder run 2 vs run 1 (2026-10-12)

Run 2: `train_encoder.py --hours 6.5` with GuitarSet and P1_scales added (452 files, 476 min). Stopped
early at step 14062 (proxy plateau). Shipped checkpoint: step 12236 (best mean of pred/search on a
48-case proxy among run 2 checkpoints).

48-case proxy (`proxy_cases(48, seed=777)`, mean of pred and search tone_db, lower is better):
run 1 best (step 19530) 0.431, run 2 step 12236 0.485. Run 1 led here, but the real bench below
disagrees, and the real bench is what Match uses.

Real bench, seed 0, median tone_db (synthetic / real), lower is better:

| matcher | room/different run 1 | room/different run 2 | room solo_demucs run 1 | room solo_demucs run 2 |
|---|---|---|---|---|
| pred_only | 0.93 / 2.60 | **0.62 / 1.97** | 1.51 / 3.57 | **0.90 / 2.67** |
| pred_then_search (the app) | 0.90 / 2.35 | **0.74 / 2.17** | 1.12 / 2.89 | **0.91** / 2.92 |
| cmaes_ours | 0.84 / 2.11 | **0.78 / 2.02** | 1.17 / **2.61** | **0.95** / 2.87 |

Clean recordings (full/different, the Use a file case), run 1 → run 2: pred_then_search
0.94 / 1.69 → **0.56 / 1.56**; pred_only 1.29 / 2.13 → **0.36 / 1.63**. On clean real audio the search
still helps a little (1.56 vs 1.63), so the file path's CMA-ES search stays.

Run 2 wins most cells, by the most on the knob prediction (pred_only). It ships as
`app/GtrTone/ToneMatch/ToneEncoder.mlpackage`. Run 1's weights stay in `models/encoder-run1-backup/`.

Note: with run 2, the search after the prediction makes the bench worse in most cases (paired, per
case: search better in only 7/24 real room cases and 4/24 real separated-band cases; median +0.08 and
+0.40 dB). The app's Listen path is not the same thing, though: its greedy ±1-step refine renders on
ToneHub's NAM captures, not the bench amp, and exists to correct for the encoder's knobs describing the
bench amp. The bench can't measure that, so the app's matcher is unchanged. Deciding this needs an
on-device A/B (predicted vs refined, judged by ear or by a play-along capture).

Results: `bench/results/run-20261012-092224-run2-room-different.json`,
`bench/results/run-20261012-093059-run2-room-solo-demucs.json`,
`bench/results/run-20261012-094254-run2-full-different.json` (bench/results is gitignored; local only).

## Listen refine on the app's amps (2026-10-12)

The question above, answered in the app's own domain: does Listen's greedy ±1-step refine
(`ToneMatchScorer.refine`) after the run 2 prediction make the result better or worse on ToneHub's
NAM captures? **Better, on average. Refine stays.**

Harness: `app/GtrTone/ToneMatch/RefineEval.swift` (DEBUG only, headless: the Mac binary run with
`-refineeval` exits from `GtrToneApp.init` before any window). It uses the app's own code: the NAM
core, `ToneHubRenderer`, Core ML `ToneEncoderRunner`, `ToneMatchScorer`, `matchListen`.

- Targets with known answers: Guitar-TECHS `P3_music` DI (12 files, held out of encoder training),
  loudest 10 s, through each of the 10 built-in amps at random known settings (Gain 1.5-8.5,
  Bass / Mid / Treble 2-8, half steps; random stand-in cab for the amp-only captures). 6 per amp,
  each matched clean and through a phone-in-a-room chain (a Swift take on the bench's `room_chain`:
  speaker, room DRR -3...8 dB, pink noise + hum at SNR 15-35 dB, phone mic). 120 paired cases.
- Each case runs `matchListen` as is ("app": 3 s budget, amp pick, refine) and the same steps with no
  deadline ("full": refine runs until a pass changes nothing).
- Scored without the encoder: a held-out DI (a different P3 file, loudest 6 s) rendered at the
  candidate on the picked amp vs the same DI through the true amp and settings, measured log-mel
  distance (`ToneDistance.melDB`, dB, lower is better); and mean absolute knob error (Gain, Bass,
  Mid, Treble) vs the truth.

Paired, refined minus predicted (negative = refine better); medians with bootstrap 95% CI:

| cases | mel pred → refined (median) | paired median | refine better / worse | knob error paired median |
|---|---|---|---|---|
| all 120, app | 6.22 → 4.99 | **-0.58 [-1.43, -0.32]** | **79 / 41** | -0.25 [-0.44, -0.06] |
| all 120, full | 6.22 → 5.00 | **-1.23 [-1.59, -0.93]** | **83 / 37** | -0.25 [-0.38, +0.19] |
| clean 60, app | 6.37 → 4.99 | -1.11 [-1.78, -0.39] | 43 / 17 | -0.31 [-0.56, -0.12] |
| room 60, app | 6.01 → 4.99 | -0.42 [-1.30, +0.08] | 36 / 24 | -0.25 [-0.38, +0.06] |
| room 60, full | 6.01 → 5.15 | -0.95 [-1.58, -0.20] | 39 / 21 | +0.06 [-0.25, +0.50] |

For scale: every dial at noon (the gain ladder) scores 6.58 median, and the true settings on the
picked amp 5.25 (the picked amp is the true one in only 19/120 cases).

By the amp the match picked (app, mel): the help is on the cab-baked captures, where the encoder's
bench-amp knobs are furthest off.

| picked amp | n | pred → refined | paired median | better / worse |
|---|---|---|---|---|
| Lead 100 | 43 | 6.72 → 4.94 | -1.97 | 41 / 2 |
| Brit Crunch (brit800) | 17 | 7.53 → 4.64 | -3.76 | 16 / 1 |
| gain ladder (amp-only + stand-in cab) | 45 | 5.46 → 5.35 | +0.04 | 22 / 23 |
| Class A Chime | 13 | 2.46 → 3.70 | **+1.44** | **0 / 13** |
| Modern Metal | 2 | 7.62 → 13.14 | +5.51 | 0 / 2 |

By true amp (app), refine helps on 6 of 10 (Citrus Drive -1.87, 12/0; Brit Vintage -2.73, 10/2;
Studio Clean -2.26, 9/3; Brit Crunch -1.99; Texas Clean -0.86; Dark Stack -0.40, 10/2), is neutral
on Lead 100 / Class A Chime / Modern Metal (6/6 each), and slightly worse on Roadhouse (+0.15, 4/8).

Reading it:
- Unlike the bench, on the app's amps the refine pays: the embedding distance it descends tracks
  the measured sound well enough to move the knobs toward the target. It does so partly by
  compensating for the picked amp being a different capture (knob error barely moves while the sound
  gets closer), which is exactly the job the doc comment gives it.
- One clear exception: when Class A Chime is picked, refine made the sound worse every time (13/13,
  +1.44 dB). The prediction there was already the best of any amp (2.46 dB). Worth a follow-up
  (e.g. keep the prediction on that capture, or a margin before accepting a step), not a reason to
  drop refine everywhere.
- On the gain ladder refine is a wash (22/23).

Caveats: the Mac Debug build ran niced in the background, so `matchListen` took 3.3 s median and
refine got 24 renders (full: 26); an iPhone gets fewer renders in the 3 s budget, landing between
"pred" and "app". The room chain is synthetic. Rows: `bench/results/refineeval-20261012-seed7.jsonl`
(gitignored; local only). Re-run: build Debug for macOS, then
`ToneHub.app/Contents/MacOS/ToneHub -refineeval -refineeval-di <P3_music/audio/directinput>
-refineeval-out <file> -refineeval-per-amp 6 -refineeval-seed 7`.

Decision: refine helps (79/120, CI excludes 0), so Listen keeps it. The "This needs an on-device
A/B" note above is answered for the measurable part; an ear test is still useful for Class A Chime.

## Amp pick and refine (2026-10-12)

Follow-up to the section above: the amp pick found the true amp in only 19 of 120 cases, and the
refine made Class A Chime worse 13 of 13 times. Both trace back to what Listen renders and how it
compares the renders.

### Diagnosis: an oracle run

`-refineeval-oracle` (same harness, seed and 120 cases) renders the prediction on **every** amp
option (the gain ladder and the 7 cab-baked captures) and runs the refine to convergence on each, once
with the app's pluck-synth reference (`ReferenceDI.samples()`, "synth") and once with a real guitar
DI from the bundled preview phrases (1.5 s "Chords" + 1.5 s "Lead line", "guitar"). Each is scored on
the held-out DI (log-mel dB). Rows: `bench/results/refineeval-20261012-seed7-oracle.jsonl` (local).

Refine on each amp, held-out log-mel, prediction → refined (median), better / worse of 120:

| amp option | prediction | synth refine | better / worse | guitar refine | better / worse |
|---|---|---|---|---|---|
| Class A Chime | 4.68 | 6.96 (+2.78) | **10 / 110** | 4.60 (+0.00) | 42 / 41 |
| Texas Clean | 4.67 | 5.90 (+0.91) | 41 / 79 | 4.45 (-0.10) | 65 / 40 |
| gain ladder | 6.14 | 5.30 | 76 / 44 | 4.93 | 76 / 24 |
| Brit Crunch | 7.05 | 5.63 | 70 / 50 | 5.05 | 102 / 14 |
| Brit Vintage | 8.13 | 5.45 | 91 / 29 | 5.47 | 115 / 4 |
| Citrus Drive | 6.77 | 4.77 | 85 / 35 | 4.83 | 86 / 23 |
| Lead 100 | 6.74 | 4.87 | 101 / 19 | 4.70 | 87 / 22 |
| Modern Metal | 7.87 | 5.16 | 98 / 22 | 5.46 | 90 / 22 |

**Class A Chime: the cause is the synthetic reference DI**, not the knob mapping or the capture. On
Chime the refine always went the same way (treble down to 0-3, bass and mid up), even when Chime was
the true amp at known settings: it was bending the knobs to make the Karplus-Strong pluck's spectrum
look like a real guitar's. The clean cab-baked captures pass that spectrum through with the least
compression, so they show it most (Texas Clean too, 79/120 worse). On a real guitar DI the same refine
is neutral on Chime and helps more often on every amp. A distance-scale effect exists but is minor:
the calibrated estimate is clamped at 0 dB, and 14 of 120 synth refines on Chime ended at 0.

**Amp pick: the metric, not only the candidates.**
- Candidates: the true option (counting the three amp-only captures as "the ladder") was among the
  ladder + 2 style-gain neighbours in 79/120 cases. The encoder's gain reads high (8-10 for most targets,
  true gain 1.5-8.5), so the neighbours are usually the high-gain captures.
- Metric: across all 8 options, the encoder distance at the prediction ranks the amps only loosely
  against the held-out sound (per-case Spearman, median 0.36 synth, 0.50 guitar). After a refine the
  ranking is worse (0.18 / 0.15): each amp's refine fits the embedding, so refined distances are
  closer together. So comparing at refined settings does not pick better, and widening the
  candidates does not help the final sound (oracle, refined, held-out median: 2 neighbours 4.95, all
  7 captures 5.27 with the synth; 4.54 vs 4.49 with the guitar DI), at 5 more renders.
- At the encoder's prediction Class A Chime sounds closest of all amps in 52/120 cases whatever the
  true amp is, and it is rarely a candidate. Adding it (+1 render) did not beat the plain neighbours
  in the app run (same median, more spread).
- Clamp: with the guitar DI, 4-37 % of options read exactly 0 dB at the prediction, so they tie and
  the first candidate (the ladder) wins by list order. Ranking now uses the unclamped calibration line.
- Family hint margin: not exercised (the harness passes no researched rig).
- Headroom: picking by the held-out sound itself (oracle) would reach 3.73 dB at the prediction and
  3.24 dB refined (median), vs 4.95 for today's pick. That best amp is the true option in 44/120
  cases at the prediction and 64/120 refined, so pick accuracy is only loosely tied to the goal.

### What changed (`ToneMatcher.swift`)

1. Listen's amp pick and refine render a real guitar DI, `ReferenceDI.guitar()` (the bundled
   preview phrases, synth fallback). The score, its noon baseline and re-scores still render the synth
   (`ReferenceDI.samples()`, through `scoreReference`): one more render per match. The file path is
   unchanged (synth, CMA-ES; not measured here).
2. The pick and the refine compare the unclamped calibrated distance (`rankDB`,
   `ToneEncoderRunner.calibratedDB`); everything shown stays clamped at 0.
3. The refine times a step by the recent real renders (the seed is cached by the pick, so the old
   per-step estimate was ~0) and leaves room for the score's render.
4. Candidates unchanged (ladder + 2 neighbours by gain): wider sets cost renders and didn't pay.

### Before / after (app path, 3 s budget)

Same harness, seed 7, 120 cases; before (main) and after run side by side on the same Mac (Debug,
under the same background load). Held-out log-mel dB, median; paired after minus before, bootstrap
95 % CI (negative = after closer):

| cases | before | after | paired median | after better / worse |
|---|---|---|---|---|
| all 120, app | 5.04 | 4.90 | **-0.22 [-0.40, +0.00]** (mean -0.36) | 69 / 45 |
| clean 60, app | 5.05 | 5.07 | -0.03 [-0.56, +0.25] | 30 / 25 |
| room 60, app | 4.99 | 4.88 | **-0.27 [-0.52, -0.07]** | 39 / 20 |
| all 120, full (no deadline) | 5.00 | 4.38 | **-0.63 [-0.98, -0.36]** | 84 / 36 |
| room 60, full | 5.15 | 4.67 | -0.66 [-1.01, -0.31] | 44 / 16 |

- Refine vs prediction (app): better / worse 82 / 38 before, **93 / 23** after. When Class A Chime is
  picked: 0 / 13 before (section above), 5 / 2 after (-0.08 dB). On the gain ladder: 65 / 18, -0.41
  dB (was a wash).
- Pick: true amp 19 → 22 of 120; the ladder is picked 87 times (was ~45).
- Timing (Mac Debug, under load): match 3.30 s → **3.19 s** median (p90 3.63 → 3.51 s). Neither
  ends inside 3 s: the noon baseline render runs after the deadline in both, as before.
- Renders per match: amp pick 3 (ladder + 2, unchanged), refine median 23 → 9, score +1 (new),
  baseline 1. The refine count falls because the ladder is picked more and its amp-only captures
  render ~2.5x slower than the cab-baked ones (refine renders on the ladder: 10 → 8; on a
  cab-baked capture: 26 → 26). An iPhone fits fewer renders; the change adds one per match.
- **The shown estimate moves.** Scored on the synth after searching on the guitar DI, the score no
  longer grades the knobs on the DI they were tuned on: median estimate 0.90 → 2.08 dB (score 86 →
  71), verdicts close / near / rough 94 / 13 / 13 → 57 / 21 / 42, while the measured sound got closer.
  Scoring on the guitar DI instead would read 0 dB (score 99) for most matches. Either way the
  estimate's link to the measured sound is loose (see the Spearman numbers); worth a look before
  release.

Rows: `bench/results/refineeval-20261012-seed7-amppick-before.jsonl`,
`refineeval-20261012-seed7-amppick-after.jsonl` (local). Re-run the oracle with
`-refineeval-oracle` (add `-refineeval-shard k/n` and `-refineeval-conditions clean|room` to split it
over processes; ~1 h on 8).
