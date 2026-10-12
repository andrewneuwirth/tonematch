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

Run 2 wins most cells, by the most on the knob prediction (pred_only). It ships as
`app/GtrTone/ToneMatch/ToneEncoder.mlpackage`. Run 1's weights stay in `models/encoder-run1-backup/`.

Note: with run 2, pred_only beats pred_then_search on real recordings in both settings, so the short
search after the prediction may no longer help. Worth a follow-up bench before changing the app.

Results: `bench/results/run-20261012-092224-run2-room-different.json`,
`bench/results/run-20261012-093059-run2-room-solo-demucs.json`.
