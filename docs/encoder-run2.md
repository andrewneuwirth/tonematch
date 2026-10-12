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
