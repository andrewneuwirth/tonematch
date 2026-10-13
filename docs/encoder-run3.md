# Encoder run 3 vs run 2 (2026-10-12)

Run 3: `train_encoder.py --hours 6.5 --p-band 0.55 --p-raw-mix 0.5` (resumed once after a full disk at
step 5116). Run 2 always separated band targets (oracle or a simulated separator); the app never
separates, so run 3 leaves half of the band targets as the raw mix and uses a band more often.
Finished at step 15723 (6.5 h budget). Shipped: **ckpt-015723**.

Real bench, seed 0, `--variant different`, median tone_db (synthetic / real), lower is better.
`mix` = the band mix with no separation (what the app hears), `none` = the guitar alone.

| setup | matcher | run 2 (12236) | run 3 (14394) | run 3 (15723) |
|---|---|---|---|---|
| room, mix | pred_only | 1.25 / 2.76 | 0.76 / 1.95 | **0.72 / 1.94** |
| room, mix | pred_then_search | 1.29 / 2.22 | 0.93 / 1.89 | **0.92 / 1.72** |
| room, none | pred_only | 0.62 / 1.97 | 0.66 / 1.94 | **0.62 / 1.91** |
| room, none | pred_then_search | 0.74 / 2.17 | 0.78 / 2.00 | 0.83 / **1.96** |
| full, mix | pred_only | 0.81 / 2.12 | 0.71 / 1.88 | 0.72 / **1.84** |
| full, mix | pred_then_search | 1.01 / 1.98 | 0.80 / **1.55** | **0.77** / 1.60 |
| full, none | pred_only | **0.36 / 1.63** | 0.46 / 1.89 | 0.44 / 1.81 |
| full, none | pred_then_search | 0.56 / 1.56 | 0.59 / 1.66 | **0.56 / 1.54** |

Run 3 (15723) wins every band case by a wide margin (Listen through a room with a band: real
2.76 → 1.94 for the prediction, 2.22 → 1.72 with the search) and ties or wins the guitar-only cases
with the search, which both app paths use. Only the bare prediction on a clean guitar-only file is
worse (1.63 → 1.81), a rare input for a song match.

Run 2's weights are in `bench/models/encoder-run2-backup/`, run 1's in `encoder-run1-backup/`.
Results: `bench/results/run-*-run3-*` (gitignored, local only).
