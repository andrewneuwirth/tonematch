"""Spleeter 4stems worker for tonebench.separate (runs in Spleeter's own env, never the bench env).

    <spleeter-env>/bin/python -I spleeter_worker.py <in_dir> <out_dir>

Reads every <key>.npy in in_dir (mono float32 at 44.1 kHz), writes <key>.npy to out_dir: the
"other" stem, mono (mean of the two channels), same length. Spleeter's TF / numpy pins conflict
with the bench env, so tonebench.separate calls this as a subprocess with MODEL_PATH set to the
folder holding pretrained_models/4stems (Deezer release v1.4.0, MIT; see
docs/review/guitar-isolation.md). Default 4stems config: mask_extension "zeros", so the stem has
no content above 11 kHz.
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

from spleeter.separator import Separator  # noqa: E402


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    sep = Separator("spleeter:4stems", multiprocess=False)
    files = sorted(src.glob("*.npy"))
    for i, f in enumerate(files):
        x = np.load(f).astype(np.float32)
        st = sep.separate(np.stack([x, x], 1))  # (T, 2), dual mono
        y = st["other"].mean(1).astype(np.float32)
        y = np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]
        np.save(dst / f.name, y)
        print(f"spleeter {i + 1}/{len(files)} {f.stem}", flush=True)


if __name__ == "__main__":
    main()
