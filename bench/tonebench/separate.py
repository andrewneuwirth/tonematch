"""Guitar separation for full-band cases: Demucs htdemucs_6s (github.com/adefossez/demucs) or Spleeter.

BENCH ONLY. The Demucs code is MIT, but the author says the pretrained weights are for research /
personal use (trained on non-commercial MUSDB); see docs/LICENSES.md. Never ship them. Needs the
`sep` dependency group: `uv run --group afx --group sep python run.py --band demucs ...`.

Results are cached in data/cache/sep/ keyed by a hash of the mix, so a run only separates once.
The model is loaded on CPU with 4 threads and freed before matching starts.

Spleeter (github.com/deezer/spleeter, code and 4stems weights MIT; docs/review/guitar-isolation.md):
the 4stems "other" stem. Spleeter pins TF 2.12 and old numpy, so it runs in its own env as a
subprocess (bench/spleeter_worker.py); the bench env stays clean. Env: TONEBENCH_SPLEETER_DIR
(default <repo>/.scratch/spleeter) holds sp-env/ and pretrained_models/4stems. Cached like Demucs.
- "spleeter":    the "other" stem as is (nothing above 11 kHz: 4stems uses mask_extension "zeros").
- "spleeter_hf": the stem below 11 kHz plus the mix above it (zero-phase Butterworth crossover; the
  two filtfilt halves sum to exactly 1, so an unchanged input passes through unchanged).
"""

import gc
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy import signal

from . import SR

CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "sep"
MODEL = "htdemucs_6s"


def _key(mix):
    return hashlib.sha1(np.ascontiguousarray(mix, dtype=np.float32).tobytes()).hexdigest()[:16]


def guitar(mixes):
    """Separated guitar stem (mono, 48 kHz, same length) for each mix in the list."""
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = [CACHE / f"{MODEL}-{_key(m)}.npy" for m in mixes]
    todo = [i for i, p in enumerate(paths) if not p.exists()]
    if todo:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        torch.set_num_threads(4)
        model = get_model(MODEL).eval()
        gi = model.sources.index("guitar")
        sr = model.samplerate
        g = np.gcd(sr, SR)
        for i in todo:
            x = signal.resample_poly(mixes[i], sr // g, SR // g)
            peak = max(np.max(np.abs(x)), 1e-8)
            wav = torch.as_tensor(np.stack([x, x]) / peak, dtype=torch.float32)[None]  # (1, 2, T), dual mono
            with torch.no_grad():
                est = apply_model(model, wav, shifts=1, split=True, overlap=0.25, progress=False)[0, gi]
            y = signal.resample_poly(est.mean(0).numpy().astype(np.float64) * peak, SR // g, sr // g)
            y = np.pad(y, (0, max(0, len(mixes[i]) - len(y))))[: len(mixes[i])]
            np.save(paths[i], y)
        del model
        gc.collect()
    return [np.load(p) for p in paths]


SPLEETER_DIR = Path(os.environ.get("TONEBENCH_SPLEETER_DIR",
                                   Path(__file__).resolve().parents[2] / ".scratch" / "spleeter"))
SPLEETER_SR = 44_100
HF_CUT = 11_000.0


def spleeter_other(mixes):
    """Spleeter 4stems "other" stem (mono, 48 kHz, same length) for each mix, via the worker."""
    CACHE.mkdir(parents=True, exist_ok=True)
    keys = [_key(m) for m in mixes]
    paths = [CACHE / f"spleeter4-other-{k}.npy" for k in keys]
    todo = [i for i, p in enumerate(paths) if not p.exists()]
    if todo:
        py = SPLEETER_DIR / "sp-env" / "bin" / "python"
        if not py.exists():
            raise FileNotFoundError(f"Spleeter env not found at {py} (set TONEBENCH_SPLEETER_DIR)")
        g = np.gcd(SPLEETER_SR, SR)
        tmp = Path(tempfile.mkdtemp(prefix="tonebench-spleeter-"))
        try:
            (tmp / "in").mkdir()
            for i in todo:
                np.save(tmp / "in" / f"{keys[i]}.npy",
                        signal.resample_poly(mixes[i], SPLEETER_SR // g, SR // g).astype(np.float32))
            worker = Path(__file__).resolve().parents[1] / "spleeter_worker.py"
            env = {**os.environ, "MODEL_PATH": str(SPLEETER_DIR / "pretrained_models"),
                   "TF_CPP_MIN_LOG_LEVEL": "2"}
            subprocess.run([str(py), "-I", str(worker), str(tmp / "in"), str(tmp / "out")],
                           check=True, env=env, cwd=tmp)
            for i in todo:
                y = np.load(tmp / "out" / f"{keys[i]}.npy").astype(np.float64)
                y = signal.resample_poly(y, SR // g, SPLEETER_SR // g)
                np.save(paths[i], np.pad(y, (0, max(0, len(mixes[i]) - len(y))))[: len(mixes[i])])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return [np.load(p) for p in paths]


def hf_restore(stem, mix, cut=HF_CUT):
    """stem below `cut` + mix above it. filtfilt'd Butterworth LP + HP with the same cutoff sum to 1."""
    lp = signal.butter(8, cut, "lowpass", fs=SR, output="sos")
    hp = signal.butter(8, cut, "highpass", fs=SR, output="sos")
    return signal.sosfiltfilt(lp, stem) + signal.sosfiltfilt(hp, mix)


def apply(cases, method="demucs"):
    """Replace each case's target with the separated guitar estimate from its mix.

    method: "demucs" (htdemucs_6s guitar), "spleeter" (4stems other), "spleeter_hf" (other + mix > 11 kHz).
    """
    mixes = [c.mix for c in cases]
    if method == "demucs":
        ests = guitar(mixes)
    elif method == "spleeter":
        ests = spleeter_other(mixes)
    elif method == "spleeter_hf":
        ests = [hf_restore(y, m) for y, m in zip(spleeter_other(mixes), mixes)]
    else:
        raise ValueError(method)
    for c, y in zip(cases, ests):
        c.target = y
    return cases
