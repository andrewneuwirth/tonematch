"""Training split for the tone encoder: Guitar-TECHS DI (+ time-aligned mic'd amp) outside P3_music.

P3_music is the benchmark (tonebench.cases uses every one of its takes), so it is never used for
training. Every other Guitar-TECHS folder that is present under data/guitar-techs/ goes into the
training pool. Each file is split by time: the first 85% is "train", the last 15% "val".

prepare() converts the files once into float32 48 kHz mono .npy arrays in data/cache/encoder/
(44.1 kHz files are resampled) and writes manifest.json. Workers memory-map them.

    uv run python -m tonebench.trainset      # (re)build the cache from whatever is downloaded
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from . import SR, data

CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "encoder"
HELD_OUT = ("P3_music",)  # the benchmark's takes
VAL_FRAC = 0.15


def _load(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x[:, 0]
    if sr != SR:
        g = np.gcd(sr, SR)
        x = signal.resample_poly(x, SR // g, sr // g)
    return x


def _files():
    out = []
    for d in sorted(data.DATA.glob("*/*/audio/directinput")):
        folder = d.parents[2].name
        if folder in HELD_OUT or folder.startswith("__"):
            continue
        for f in sorted(d.glob("directinput_*.wav")):
            if f.name.startswith("._"):
                continue
            m = d.parent / "micamp" / f.name.replace("directinput_", "micamp_")
            out.append((folder, f, m if m.exists() else None))
    return out


def prepare(verbose=True):
    CACHE.mkdir(parents=True, exist_ok=True)
    old = manifest() if (CACHE / "manifest.json").exists() else {"files": []}
    done = {e["name"]: e for e in old["files"]}
    entries = []
    for folder, f, m in _files():
        name = f"{folder}__{f.stem.replace('directinput_', '')}"
        if name in done and (CACHE / done[name]["di"]).exists():
            entries.append(done[name])
            continue
        try:
            d = _load(f)
        except Exception as e:  # half-unzipped file etc.
            print("skip", f, e)
            continue
        e = {"name": name, "folder": folder, "di": f"{name}-di.npy", "n": len(d), "split": int(len(d) * (1 - VAL_FRAC))}
        np.save(CACHE / e["di"], d.astype(np.float32))
        if m is not None:
            try:
                mm = _load(m)
                lag = data._lag(d, mm)
                mm = mm[lag:] if lag >= 0 else np.pad(mm, (-lag, 0))
                mm = np.pad(mm, (0, max(0, len(d) - len(mm))))[: len(d)]
                np.save(CACHE / f"{name}-mic.npy", mm.astype(np.float32))
                e["mic"] = f"{name}-mic.npy"
                e["lag"] = lag
            except Exception as ex:
                print("no micamp", m, ex)
        entries.append(e)
        if verbose:
            print(f"{name}: {len(d) / SR:.0f} s, micamp={'mic' in e}", flush=True)
    man = {"files": entries, "held_out": list(HELD_OUT), "val_frac": VAL_FRAC}
    (CACHE / "manifest.json").write_text(json.dumps(man, indent=1))
    tot = sum(e["n"] for e in entries) / SR / 60
    if verbose:
        print(f"{len(entries)} files, {tot:.1f} min total")
    return man


def manifest():
    return json.loads((CACHE / "manifest.json").read_text())


if __name__ == "__main__":
    prepare()
