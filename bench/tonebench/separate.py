"""Guitar separation for full-band cases with Demucs htdemucs_6s (github.com/adefossez/demucs).

BENCH ONLY. The Demucs code is MIT, but the author says the pretrained weights are for research /
personal use (trained on non-commercial MUSDB); see docs/LICENSES.md. Never ship them. Needs the
`sep` dependency group: `uv run --group afx --group sep python run.py --band demucs ...`.

Results are cached in data/cache/sep/ keyed by a hash of the mix, so a run only separates once.
The model is loaded on CPU with 4 threads and freed before matching starts.
"""

import gc
import hashlib
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


def apply(cases):
    """Replace each full-band case's target with Demucs's guitar estimate from its mix."""
    ests = guitar([c.mix for c in cases])
    for c, y in zip(cases, ests):
        c.target = y
    return cases
