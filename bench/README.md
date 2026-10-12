# tonebench

Measures how well a tone matcher recovers a guitar tone. Python, run with uv.

```
cd bench
uv run python run.py                                          # baselines: 24 synthetic + 24 real cases
uv run --group afx python run.py --matchers default cmaes_afx cmaes_afx_spec          # AFx-Rep (torch)
uv run --group afx python run.py --variant playalong --matchers default aligned       # play-along
uv run --group afx python run.py --level room --calib eq --matchers default cmaes_afx # iPhone listen mode
uv run --group afx --group sep python run.py --band demucs --matchers default cmaes_afx  # full band
uv run python report.py --mel results/run-*.json                                      # re-print tables
```

Needs Guitar-TECHS `P3_music` in `../data/guitar-techs/` and the AFx-Rep encoder in
`../data/models/afx-rep/` (see `docs/LICENSES.md`). The `sep` group (Demucs) is bench-only: its
weights are for research/personal use and must never ship. Memory: run one benchmark at a time
(torch is capped at 4 threads; an AFx-Rep run takes about 4 GB).

## How a case works
A matcher gets what a real user has: the **song** (target) and the user's **own DI**. It returns knob
settings. Scoring renders a **third, held-out passage** with those settings and compares it to the
reference, so the score measures tone, not which notes were played.

| Track | Target | Reference for scoring | Knob error? |
|---|---|---|---|
| synthetic | DI through our amp model with random known settings, then production | the true settings on the held-out DI | yes |
| real | Guitar-TECHS mic'd amp recording of one half of a take (a clean amp) | the mic'd amp on a passage from the other half | no (settings unknown) |

Options. Targets, DIs and eval passages are the same under every option, so runs can be compared case by case:
- `--variant`: what the user's DI is. `different` = another take (the original setup).
  `playalong` = the same part played along with the song: +-60 ms offset, +-25 ms drifting timing,
  2-4 wrong notes. `playalong_exact` = the exact DI that made the target (upper bound).
- `--level`: production on synthetic targets. `none`; `mix` (EQ + bus compression); `full`
  (+ reverb + MP3); `room` = full, then iPhone **listen mode**: small speaker (band-limited,
  resonant, slightly clipping) -> room (early reflections + tail, DRR -3..8 dB) -> pink noise + hum
  (SNR 15-35 dB) -> phone mic -> AGC + limiter. Real targets get no studio production; `room` plays
  them through the same listen-mode chain.
- `--band`: `none` = guitar alone. `oracle` = the guitar inside a synthetic band mix (drums, bass,
  sometimes keys and a vocal-like lead; `tonebench/band.py`, our own synthesis), produced as a mix,
  then perfectly separated (the guitar's own share of the produced mix). `demucs` = the same mix
  separated by Demucs htdemucs_6s. `solo_demucs` = no band, but the target still goes through Demucs.
  `spleeter` = the same mix through Spleeter 4stems, the "other" stem (run in its own env as a
  subprocess, `spleeter_worker.py`; set `TONEBENCH_SPLEETER_DIR`, default `../.scratch/spleeter`).
  `spleeter_hf` = that stem below 11 kHz plus the mix above it. `mix` = the same mix, unseparated
  (the no-separation baseline for the band modes). `paired.py` compares runs case by case.
- `--calib` (room only): `eq` = one-time calibration. The phone plays known pink noise through the same
  speaker/room/mic (with its own noise and AGC) and estimates the chain's response from that
  recording (1/6 octave, inverse clamped to +-12 dB); the target is then inverse-EQ'd. `eq_denoise` =
  the same, plus spectral-subtraction noise suppression.

## Matchers
- `default`: every knob at noon. **The bar to beat.**
- `random`, `random_powell`, `cmaes_ltas`: fit the long-term spectrum (random search, + Powell, CMA-ES).
- `cmaes_afx`: CMA-ES (12 candidates x 25 iterations, starting from noon) maximizing AFx-Rep embedding
  similarity between the target and the user's DI rendered through the candidate settings (the ST-ITO recipe).
- `cmaes_afx_spec`: AFx-Rep distance + 0.25 x spectral-statistics distance + 0.25 x pull toward noon,
  each term scaled to 1 at the default setting.
- `aligned` (play-along only): onset-align the DI to the song, then CMA-ES on a time-aligned,
  trimmed multi-resolution STFT cost. `aligned_dry`: the same, but skipping reverb-tail frames (dereverb-lite).

## Scores (lower is better)
- `tone_db`: multi-resolution STFT log-magnitude error on the held-out passage, floored 80 dB below
  peak. **Main score.** Its linear-frequency bins put most of the weight above ~5 kHz (fizz).
- `mel_db`: log-mel (64 bands) error on the same passage. A cross-check that weights frequencies
  closer to how we hear them. The tables show it as "median diff vs default (win rate)".
- `ltas_db`: long-term 1/3-octave spectrum error. `knob_err`: mean absolute knob error, 0..1 (synthetic).

Medians come with 95% percentile-bootstrap intervals. "vs default" is the median per-case
difference from `default` on the same cases (negative = better). "wins" is the share of cases
where the matcher beats default.

## Results (2026-10-11, seed 0, 24 synthetic + 24 real cases per row)

`default` scores the same in every setting because it ignores the target: tone_db 1.69 [1.29, 2.17]
synthetic, 3.68 [3.03, 4.19] real; knob_err 0.218.

### 1. Studio production (`--level full`), different take vs play-along

| setting | matcher | track | tone_db [95% CI] | vs default [95% CI] | wins | mel vs default (wins) | knob_err |
|---|---|---|---|---|---|---|---|
| different | random | synthetic | 2.35 [2.07, 3.19] | +0.67 [-0.92, +1.54] | 38% | +1.79 (38%) | 0.273 |
| different | random | real | 4.77 [3.40, 5.72] | +1.37 [+0.37, +1.95] | 25% | +2.77 (21%) |  |
| different | random_powell | synthetic | 2.06 [1.75, 2.88] | +0.52 [-0.22, +1.06] | 33% | +1.59 (33%) | 0.327 |
| different | random_powell | real | 5.21 [3.88, 5.80] | +1.31 [+0.80, +2.21] | 12% | +3.28 (17%) |  |
| different | cmaes_ltas | synthetic | 2.19 [1.96, 2.58] | +0.59 [-0.49, +1.12] | 38% | +1.42 (33%) | 0.300 |
| different | cmaes_ltas | real | 4.77 [2.34, 6.92] | +1.57 [-0.19, +2.88] | 33% | +2.47 (29%) |  |
| different | **cmaes_afx** | synthetic | 0.98 [0.57, 1.34] | **-0.94 [-1.51, -0.39]** | 83% | -2.03 (83%) | 0.199 |
| different | **cmaes_afx** | real | 1.39 [1.26, 1.60] | **-2.07 [-2.48, -1.62]** | 100% | -1.76 (92%) |  |
| different | **cmaes_afx_spec** | synthetic | 0.96 [0.66, 1.25] | **-0.66 [-1.49, -0.30]** | 88% | -1.64 (88%) | **0.165** |
| different | **cmaes_afx_spec** | real | 2.03 [1.31, 2.43] | **-1.38 [-1.65, -1.17]** | 96% | **-2.27 (92%)** |  |
| playalong | random | synthetic | 1.83 [1.05, 2.40] | +0.13 [-1.21, +0.55] | 50% | +0.21 (50%) | 0.255 |
| playalong | random | real | 2.21 [1.83, 2.83] | -1.20 [-1.31, -1.02] | 83% | -2.43 (83%) |  |
| playalong | cmaes_afx | synthetic | 0.84 [0.43, 1.03] | -1.04 [-1.79, -0.57] | 96% | -2.49 (92%) | 0.192 |
| playalong | cmaes_afx | real | 1.43 [1.22, 1.80] | -2.00 [-2.35, -1.65] | 100% | -2.09 (96%) |  |
| playalong | **aligned** | synthetic | **0.38 [0.26, 0.48]** | **-1.34 [-1.78, -0.93]** | 100% | **-3.61 (100%)** | 0.163 |
| playalong | **aligned** | real | 1.16 [1.02, 1.32] | **-2.46 [-2.89, -1.69]** | 100% | -1.06 (71%) |  |
| playalong_exact | aligned | synthetic | 0.30 [0.25, 0.52] | -1.22 [-1.88, -0.93] | 100% | -3.65 (100%) | 0.156 |
| playalong_exact | aligned | real | 1.16 [0.99, 1.34] | -2.12 [-2.77, -1.68] | 100% | -1.33 (71%) |  |

### 2. iPhone listen mode (`--level room`), with and without cleanup

| setting | matcher | track | tone_db [95% CI] | vs default [95% CI] | wins | mel vs default (wins) | knob_err |
|---|---|---|---|---|---|---|---|
| different | random | synthetic | 6.08 [4.95, 7.67] | +4.89 [+2.71, +5.99] | 17% | +10.28 (21%) | 0.336 |
| different | random | real | 10.79 [10.14, 11.59] | +7.70 [+6.98, +8.08] | 0% | +11.95 (0%) |  |
| different | cmaes_afx | synthetic | 2.39 [1.89, 2.83] | +0.32 [-0.18, +1.48] | 38% | +0.67 (38%) | 0.341 |
| different | cmaes_afx | real | 4.25 [3.19, 4.96] | +0.30 [-0.25, +1.06] | 42% | +0.43 (46%) |  |
| different | cmaes_afx_spec | synthetic | 2.06 [1.65, 2.58] | +0.58 [-0.20, +1.24] | 38% | +0.79 (42%) | 0.247 |
| different | cmaes_afx_spec | real | 4.78 [3.59, 5.60] | +0.90 [+0.47, +1.65] | 21% | +1.45 (29%) |  |
| different + eq | cmaes_afx | synthetic | 2.48 [1.77, 3.88] | +0.48 [-0.18, +2.10] | 38% | +0.58 (42%) | 0.302 |
| different + eq | cmaes_afx | real | 3.78 [2.31, 6.73] | +0.54 [-0.79, +2.77] | 46% | +1.38 (42%) |  |
| different + eq | cmaes_afx_spec | synthetic | 3.51 [2.52, 4.73] | +1.88 [+0.73, +3.64] | 25% | +3.90 (25%) | 0.252 |
| different + eq | cmaes_afx_spec | real | 7.48 [5.36, 9.59] | +2.89 [+2.16, +6.22] | 4% | +4.23 (17%) |  |
| different + eq_denoise | cmaes_afx | synthetic | 2.29 [1.82, 3.35] | +0.44 [-0.46, +1.94] | 38% | +1.47 (42%) | 0.319 |
| different + eq_denoise | cmaes_afx | real | 3.47 [1.96, 5.29] | -0.52 [-1.36, +1.80] | 54% | +1.25 (33%) |  |
| different + eq_denoise | cmaes_afx_spec | synthetic | 3.45 [2.74, 4.29] | +1.89 [+0.86, +3.15] | 17% | +4.22 (21%) | 0.232 |
| different + eq_denoise | cmaes_afx_spec | real | 8.18 [6.37, 10.06] | +4.97 [+2.44, +7.24] | 4% | +8.11 (17%) |  |
| different + Demucs cleanup | **cmaes_afx** | synthetic | 1.52 [1.00, 1.85] | -0.66 [-0.85, +0.53] | 58% | -1.42 (54%) | 0.318 |
| different + Demucs cleanup | **cmaes_afx** | real | 2.36 [1.49, 3.70] | **-1.02 [-1.38, -0.66]** | 79% | -1.18 (62%) |  |
| different + Demucs cleanup | cmaes_afx_spec | synthetic | 1.59 [0.97, 2.44] | -0.32 [-0.92, +0.43] | 58% | -0.60 (58%) | 0.221 |
| different + Demucs cleanup | cmaes_afx_spec | real | 4.15 [2.79, 5.04] | +0.13 [-0.91, +1.31] | 50% | +0.63 (46%) |  |
| playalong | cmaes_afx | synthetic | 2.11 [1.64, 2.93] | +0.50 [-0.45, +1.06] | 33% | +1.27 (33%) | 0.338 |
| playalong | cmaes_afx | real | 4.46 [3.90, 4.77] | +0.73 [+0.23, +1.44] | 25% | +1.51 (29%) |  |
| playalong | cmaes_afx_spec | synthetic | 2.21 [1.64, 2.71] | +0.66 [-0.36, +1.28] | 33% | +0.72 (38%) | 0.266 |
| playalong | cmaes_afx_spec | real | 4.94 [4.41, 6.40] | +1.56 [+0.49, +1.98] | 8% | +2.78 (12%) |  |
| playalong | aligned | synthetic | 3.63 [2.32, 6.43] | +1.96 [+0.86, +4.24] | 25% | +3.74 (29%) | 0.356 |
| playalong | aligned | real | 8.28 [6.56, 10.91] | +5.03 [+3.26, +7.27] | 0% | +6.97 (4%) |  |
| playalong | aligned_dry | synthetic | 3.27 [2.40, 6.97] | +2.18 [+0.42, +4.73] | 21% | +2.87 (25%) | 0.361 |
| playalong | aligned_dry | real | 7.65 [6.58, 10.86] | +4.71 [+2.56, +7.27] | 4% | +6.93 (4%) |  |
| playalong + eq | cmaes_afx | synthetic | 1.73 [1.27, 2.78] | -0.07 [-0.84, +0.90] | 50% | -0.48 (62%) | 0.278 |
| playalong + eq | cmaes_afx | real | 3.46 [2.30, 8.13] | -0.28 [-1.09, +4.64] | 54% | +0.41 (46%) |  |
| playalong + eq | cmaes_afx_spec | synthetic | 4.87 [3.00, 6.50] | +3.20 [+1.87, +5.67] | 21% | +6.71 (21%) | 0.241 |
| playalong + eq | cmaes_afx_spec | real | 9.15 [5.98, 12.01] | +6.55 [+1.79, +8.08] | 0% | +9.02 (8%) |  |
| playalong + eq | aligned | synthetic | 6.43 [5.29, 9.60] | +5.43 [+2.65, +7.33] | 12% | +7.60 (17%) | 0.399 |
| playalong + eq | aligned | real | 10.78 [8.16, 14.57] | +7.96 [+4.74, +10.33] | 0% | +11.69 (4%) |  |
| playalong + eq | aligned_dry | synthetic | 6.17 [4.55, 8.08] | +4.81 [+1.49, +6.06] | 17% | +6.88 (21%) | 0.374 |
| playalong + eq | aligned_dry | real | 9.78 [7.41, 13.58] | +7.12 [+4.04, +10.28] | 4% | +11.30 (8%) |  |
| playalong + eq_denoise | cmaes_afx | synthetic | 1.92 [1.48, 3.07] | +0.51 [-0.78, +1.54] | 46% | +0.27 (46%) | 0.302 |
| playalong + eq_denoise | cmaes_afx | real | 3.89 [3.03, 7.71] | +0.47 [-0.60, +4.92] | 46% | +2.12 (42%) |  |
| playalong + eq_denoise | cmaes_afx_spec | synthetic | 3.61 [2.73, 5.27] | +1.95 [+1.42, +3.73] | 17% | +4.52 (21%) | 0.247 |
| playalong + eq_denoise | cmaes_afx_spec | real | 8.10 [6.37, 10.73] | +5.05 [+2.08, +7.96] | 0% | +7.93 (8%) |  |
| playalong + eq_denoise | aligned | synthetic | 5.20 [4.13, 7.29] | +3.74 [+1.79, +5.73] | 17% | +5.95 (21%) | 0.374 |
| playalong + eq_denoise | aligned | real | 8.73 [6.21, 12.18] | +5.79 [+2.07, +8.73] | 4% | +9.90 (8%) |  |
| playalong + eq_denoise | aligned_dry | synthetic | 4.54 [3.61, 6.15] | +2.74 [+0.88, +5.08] | 21% | +4.34 (21%) | 0.358 |
| playalong + eq_denoise | aligned_dry | real | 8.26 [6.38, 12.38] | +5.57 [+2.30, +8.39] | 4% | +9.29 (12%) |  |

### 3. Full band: what separation costs (`--variant different`)

| setting | matcher | track | tone_db [95% CI] | vs default [95% CI] | wins | mel vs default (wins) | knob_err |
|---|---|---|---|---|---|---|---|
| full, guitar only | cmaes_afx | synthetic | 0.98 [0.57, 1.34] | -0.94 [-1.51, -0.39] | 83% | -2.03 (83%) | 0.199 |
| full, guitar only | cmaes_afx | real | 1.39 [1.26, 1.60] | -2.07 [-2.48, -1.62] | 100% | -1.76 (92%) |  |
| full, guitar only | cmaes_afx_spec | synthetic | 0.96 [0.66, 1.25] | -0.66 [-1.49, -0.30] | 88% | -1.64 (88%) | 0.165 |
| full, guitar only | cmaes_afx_spec | real | 2.03 [1.31, 2.43] | -1.38 [-1.65, -1.17] | 96% | -2.27 (92%) |  |
| full, band + oracle sep | cmaes_afx | synthetic | 1.42 [0.85, 1.72] | -0.74 [-1.54, +0.17] | 67% | -1.75 (62%) | 0.218 |
| full, band + oracle sep | cmaes_afx_spec | synthetic | 0.82 [0.60, 1.32] | -1.00 [-1.67, -0.41] | 79% | -2.50 (71%) | 0.166 |
| full, band + Demucs | cmaes_afx | synthetic | 1.39 [1.02, 1.93] | -0.36 [-1.03, +0.13] | 58% | -0.48 (54%) | 0.317 |
| full, band + Demucs | cmaes_afx | real | 1.66 [1.50, 1.77] | -1.72 [-2.41, -1.37] | 100% | -1.17 (71%) |  |
| full, band + Demucs | cmaes_afx_spec | synthetic | 1.40 [0.96, 1.75] | -0.75 [-1.10, -0.23] | 75% | -1.78 (71%) | 0.221 |
| full, band + Demucs | cmaes_afx_spec | real | 2.43 [1.89, 2.76] | -1.21 [-1.31, -1.04] | 96% | -1.59 (79%) |  |
| room, guitar only | cmaes_afx | synthetic | 2.39 [1.89, 2.83] | +0.32 [-0.18, +1.48] | 38% | +0.67 (38%) | 0.341 |
| room, guitar only | cmaes_afx | real | 4.25 [3.19, 4.96] | +0.30 [-0.25, +1.06] | 42% | +0.43 (46%) |  |
| room, guitar only | cmaes_afx_spec | synthetic | 2.06 [1.65, 2.58] | +0.58 [-0.20, +1.24] | 38% | +0.79 (42%) | 0.247 |
| room, guitar only | cmaes_afx_spec | real | 4.78 [3.59, 5.60] | +0.90 [+0.47, +1.65] | 21% | +1.45 (29%) |  |
| room, band + oracle sep | cmaes_afx | synthetic | 2.89 [2.19, 4.55] | +0.75 [-0.45, +2.57] | 42% | +1.68 (42%) | 0.366 |
| room, band + oracle sep | cmaes_afx | real | 5.27 [3.61, 6.09] | +1.34 [+0.32, +2.42] | 29% | +1.14 (29%) |  |
| room, band + oracle sep | cmaes_afx_spec | synthetic | 2.70 [2.27, 3.58] | +0.83 [-0.29, +2.25] | 42% | +1.72 (42%) | 0.299 |
| room, band + oracle sep | cmaes_afx_spec | real | 5.72 [4.41, 6.72] | +2.44 [+0.83, +3.14] | 17% | +3.24 (17%) |  |
| room, band + Demucs | cmaes_afx | synthetic | 1.76 [0.99, 2.32] | -0.03 [-1.05, +0.48] | 50% | +0.57 (46%) | 0.334 |
| room, band + Demucs | cmaes_afx | real | 2.62 [1.79, 3.38] | -1.20 [-1.42, -0.23] | 71% | -0.37 (62%) |  |
| room, band + Demucs | cmaes_afx_spec | synthetic | 1.69 [1.12, 2.64] | -0.23 [-0.79, +0.45] | 54% | -0.10 (50%) | 0.234 |
| room, band + Demucs | cmaes_afx_spec | real | 3.46 [2.94, 5.16] | -0.57 [-0.94, +1.16] | 54% | +0.37 (42%) |  |

Real cases at full level get no studio production, so there "band + oracle" equals "guitar only"
(rows omitted). On one synthetic band mix, Demucs's guitar stem reached 11.6 dB SDR; real songs will be harder.

## What we learned
1. **Fitting the long-term spectrum is worse than doing nothing.** Random search, Powell and CMA-ES
   on the long-term spectrum all lose to default on every track. A better optimizer on the wrong
   objective doesn't help.
2. **AFx-Rep + CMA-ES beats default** in studio conditions: -0.9 dB on synthetic (83% of cases) and
   -2.1 dB on real (100%). Adding spectral statistics and a pull toward noon (`cmaes_afx_spec`) gives
   the best knob error (0.165 vs 0.218) and the best real-track mel score. It is also the most robust
   to separation: still -0.75 dB (synthetic) and -1.2 dB (real) with Demucs on a full band.
3. **Play-along helps, and the alignment works.** `aligned` is the best matcher in studio conditions:
   -1.3 dB synthetic, -2.5 dB real, 100% wins. It does as well with drifting timing and wrong notes
   as with the exact DI. Caveat: on real cases its mel-score gain is smaller than cmaes_afx's
   (71% vs 96% wins).
4. **Listen mode breaks every matcher.** At `--level room` nothing beats default, and aligned
   play-along is worst (+5 dB on real). The matchers fit the speaker, room, noise and AGC into the
   knobs. One-time EQ calibration doesn't fix this: cmaes_afx at best ties default, and the
   spectral-statistics term and aligned get worse, because the inverse EQ also boosts noise. Adding
   a denoiser doesn't help either. Why: above ~5 kHz the room noise is 17-38 dB louder than the guitar
   (the cab leaves little energy there), and below ~200 Hz the speaker removes 15-24 dB, more than
   a +-12 dB EQ can restore. The dereverb-lite matcher (`aligned_dry`) helps only marginally.
5. **A learned separator is the listen-mode fix that works.** Running Demucs on the room recording,
   even with no band, makes cmaes_afx beat default again: -1.0 dB real (79% wins), and -0.7 dB synthetic
   (CI crosses 0). With a band, room + Demucs beats room + perfect separation by about 1-2.5 dB. The
   separator strips out noise and room sound that the matcher would otherwise copy into the knobs.

## Recommendation
- Ship `cmaes_afx_spec` (or `cmaes_afx`) as the matcher for studio recordings now, and `aligned`
  when the user plays along with a clean source. In listen mode, run a guitar separator / cleanup
  model first, and fall back toward default when its output is weak.
- Next training step: **our own guitar-tone encoder** (contrastive, Open-Amp / Chen 2024 style).
  Train it with positives that share the amp chain but differ in content *and* in degradation:
  studio production, band leakage and separation artifacts, and the full listen-mode chain
  (speaker, room, noise, AGC). AFx-Rep shows the objective is right in clean conditions, but it is
  not invariant to listen mode. Second: **train our own guitar separator / cleanup model** (the
  Demucs weights can't ship), with listen-mode degradations in its training mixes. The Demucs result
  shows that this step carries most of the listen-mode gain.

## Known limits
- One seed; 24 cases per track. Real cases are 12 excerpts x 2 passages (not independent), and the
  Guitar-TECHS amp is clean, so much of the real-track win comes from finding gain near 0.
- tone_db weights high frequencies heavily. mel_db agrees on every main conclusion above but
  shrinks some real-track margins.
- Our amp model is simple and fixed; it can't reproduce the real amp's mic and room, which caps the real track.
- The synthetic band is crude synthesis; real mixes will be harder to separate.
- No oversampling in the model; target and estimate alias identically, so it doesn't bias synthetic scores.
