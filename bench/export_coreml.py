"""Export the tone encoder to Core ML (not added to the app).

    uv run --group afx --group export python export_coreml.py [--ckpt models/encoder/best.pt]

Writes models/encoder/ToneEncoder.mlpackage: the network after the front end.
  input  "logmel": float32 (1, 1, 96, T), T = 1 + samples // 512 frames (T in 32..1024; 6 s -> 563),
         computed as in tonebench.encoder.LogMel: mono 48 kHz, RMS-normalize, STFT n_fft 2048 /
         hop 512 / periodic Hann / centered (reflect pad 1024), power -> 96 triangular HTK-mel bands
         30 Hz-20 kHz (mel_matrix.npy saved next to the model), 10*log10(x + 1e-10), minus the mean
         over all bands and frames, divided by 20.
  outputs "embedding" (1, 128) L2-normalized, "knobs" (1, 7) in 0..1: gain, bass, mid, treble,
         presence, cab_tone, cab_res.
Checks the Core ML output against PyTorch on a test clip and prints the max difference.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from tonebench import encoder

OUT = encoder.MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT / "best.pt"))
    a = ap.parse_args()
    import coremltools as ct

    m = encoder.load(a.ckpt)
    net = m.net.eval()
    x = torch.randn(1, 6 * 48000) * 0.1
    mel = m.front(x)
    traced = torch.jit.trace(net, mel)
    T = ct.RangeDim(lower_bound=32, upper_bound=1024, default=mel.shape[-1])
    ml = ct.convert(traced, inputs=[ct.TensorType(name="logmel", shape=(1, 1, encoder.N_MELS, T), dtype=np.float32)],
                    outputs=[ct.TensorType(name="embedding"), ct.TensorType(name="knobs")],
                    minimum_deployment_target=ct.target.iOS17, convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT32)
    ml.short_description = "ToneHub guitar tone encoder: log-mel -> 128-d tone embedding + 7 knob estimates"
    ml.input_description["logmel"] = "(1,1,96,T) normalized log-mel, see bench/export_coreml.py"
    path = OUT / "ToneEncoder.mlpackage"
    ml.save(str(path))
    np.save(OUT / "mel_matrix.npy", encoder.mel_matrix())
    with torch.no_grad():
        e, k = net(mel)
    out = ml.predict({"logmel": mel.numpy()})
    print("saved", path)
    print("max |diff| embedding", float(np.abs(out["embedding"] - e.numpy()).max()),
          "knobs", float(np.abs(out["knobs"] - k.numpy()).max()))


if __name__ == "__main__":
    main()
