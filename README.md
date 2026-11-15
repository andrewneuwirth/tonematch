<div align="center">

# ToneMatch

**Match a guitar tone from a song's real audio.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](bench/pyproject.toml)
[![Swift](https://img.shields.io/badge/on--device-Swift%20%C2%B7%20Core%20ML-F05138.svg)](swift/ToneMatch/)
[![Model](https://img.shields.io/badge/encoder-3.4%20MB%20Core%20ML-555.svg)](bench/models/encoder/)
[![Platform](https://img.shields.io/badge/runs%20on-iPhone%20%C2%B7%20Apple%20silicon-lightgrey.svg)](#quick-start)

[Project page](https://newitools.app/lab/tonematch/) · [Results](#results) · [Quick start](#quick-start)

<img src="docs/images/hero.png" width="860" alt="An amp panel with seven matched knobs, the song and guitar spectra overlaid, and a 1.54 dB tone error">

<sub>Illustration from the project page. The 1.54 dB tone error is the measured median for a studio file, guitar only; the knob positions are illustrative.</sub>

</div>

Give it a recording of the song and your own dry guitar (DI), and it returns amp knob settings that make your guitar
sound like the record. This is the matcher behind the ToneHub guitar app.

| 58% | 24 / 24 | 3.4 MB | 7 |
|:---:|:---:|:---:|:---:|
| less tone error than knobs at noon (studio file, guitar only) | held-out real cases improved | Core ML encoder, runs on an iPhone | amp knobs predicted |

## How it works

```mermaid
flowchart LR
    Song["The song<br/>studio file or phone mic"] --> Enc["1 · Tone encoder<br/>log-mel conv net"]
    Enc -->|first guess| Search["2 · CMA-ES search<br/>render DI at candidates,<br/>compare with the song"]
    DI["Your DI<br/>dry guitar"] --> Search
    Song --> Search
    Search --> Knobs["7 knob settings"]
```

1. **A tone encoder.** A small conv net on log-mel audio that predicts 7 amp knobs (gain, bass, mid, treble,
   presence, cab tone, cab resonance) straight from the song, plus a 128-d tone embedding. It's 3.4 MB as Core ML
   and runs on an iPhone.
2. **CMA-ES search.** Starting from the encoder's guess, it renders your DI through the amp at candidate settings and
   refines the knobs to minimize the distance to the song.

The repo also includes **tonebench**, the benchmark used to develop it.

## Results

Run 3 encoder (`bench/models/encoder/best.pt`, step 15,723). Median tone error is in dB (multi-resolution STFT
log-magnitude error on a **held-out passage**, so it measures tone rather than which notes were played). Lower is
better. The **baseline** sets every knob to noon. "Wins" counts held-out cases that beat the baseline. Each cell is
24 real cases.

Real cases are Guitar-TECHS mic'd-amp recordings, matched from a different take of the player's DI.

```mermaid
xychart-beta
    title "Studio file, guitar only: median tone error (dB, lower is better)"
    x-axis ["Noon", "Encoder", "Enc+search"]
    y-axis "dB" 0 --> 4
    bar [3.68, 1.81, 1.54]
```

```mermaid
xychart-beta
    title "Enc+search (bars) vs noon (line), dB"
    x-axis ["Studio gtr", "Studio mix", "Room gtr", "Room mix"]
    y-axis "dB" 0 --> 4
    bar [1.54, 1.60, 1.96, 1.72]
    line [3.68, 3.68, 3.68, 3.68]
```

Studio = studio file, Room = phone mic in a room; gtr = guitar only, mix = full band mix.

| Condition | Baseline | Encoder only | Encoder + search | Wins (encoder + search) |
|---|---:|---:|---:|---:|
| Studio file, guitar only | 3.68 | 1.81 | **1.54** | 24 / 24 |
| Studio file, full band mix | 3.68 | 1.84 | **1.60** | 24 / 24 |
| Phone mic in a room, guitar only | 3.68 | **1.91** | 1.96 | 21 / 24 |
| Phone mic in a room, full band mix | 3.68 | 1.94 | **1.72** | 22 / 24 |

The encoder's prediction alone beats the baseline on 24 / 24 real cases in every condition. In the hardest setting
(a band playing through a phone speaker into a room, heard by a phone mic), encoder + search cuts tone error by 53%.

The test set is small (24 real cases), and the medians come with bootstrap intervals in the run output.
[`docs/encoder-run3.md`](docs/encoder-run3.md) compares training runs.

## What the benchmark tests

A matcher only gets what a real user has: the song and their own DI. It returns knob settings, and scoring renders a
third, held-out passage with them.

| Option | What it varies |
|---|---|
| **synthetic** track | Targets made with known settings through the bench's own gray-box amp (`tonebench/amp.py`), so knob error can be measured too. |
| **real** track | Guitar-TECHS mic'd-amp recordings, scored against the real amp on a different passage. |
| `--level` | Production on the target: `none`, `mix` (EQ + bus compression), `full` (+ reverb + MP3), `room` (iPhone listen mode: small speaker, room reflections, noise, phone mic, AGC). |
| `--band` | The guitar alone, inside a synthetic band mix, or separated back out (oracle, Demucs, Spleeter). |
| `--variant` | A different DI take, or a play-along take with timing drift and wrong notes. |

See [`bench/README.md`](bench/README.md) for every option and matcher.

## Quick start

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Data (not included): download Guitar-TECHS from zenodo.org/records/14963133
#    and put it in data/guitar-techs/ at the repo root (P3_music is the benchmark set).
cd bench

# 2. Baselines (numpy only)
uv run python run.py

# 3. The released encoder, with and without search (needs torch)
uv run --group afx python run.py --matchers default pred_only pred_then_search
uv run --group afx python run.py --level room --band mix --matchers default pred_only pred_then_search

# Re-print tables from saved results
uv run python report.py results/run-*.json
```

```mermaid
flowchart LR
    T["train_encoder.py<br/>Apple Silicon GPU"] --> Sel["select_encoder.py<br/>pick a checkpoint"] --> X["export_coreml.py<br/>Core ML"]
```

Train your own encoder with `train_encoder.py` (Apple Silicon GPU, about 6.5 h for run 3; also needs GuitarSet in
`data/guitarset/`), pick a checkpoint with `select_encoder.py`, and export it to Core ML with `export_coreml.py`.

## Layout

| Path | What |
|---|---|
| `bench/tonebench/` | Benchmark: cases, amp model, production and room simulation, band synthesis, separation, metrics, matchers, encoder |
| `bench/run.py`, `report.py`, `paired.py` | Run the benchmark, print tables, compare runs case by case |
| `bench/train_encoder.py`, `select_encoder.py`, `export_coreml.py` | Train, select, export the encoder |
| `bench/models/encoder/` | Released weights: `best.pt` (PyTorch), `ToneEncoder.mlpackage` (Core ML), `mel_matrix.npy` (front end), `train_log.csv` |
| `bench/gearmap/` | Sketch: map a gear-neutral tone-stack curve onto a specific amp's knobs |
| `swift/ToneMatch/` | The on-device Swift code from the app: log-mel front end, Core ML runner, CMA-ES, and the matcher. Reference only: `ToneMatcher.swift` depends on the app's audio engine and doesn't build on its own |
| `docs/` | Notes comparing encoder training runs |

## License

Apache License 2.0. See [`LICENSE`](LICENSE). Third-party credits, including the datasets the released weights were
trained on, are in [`NOTICE`](NOTICE).
