"""On-the-fly training data for the tone encoder (tonebench.encoder).

Every item is a GROUP of G tones that share one "candidate" DI clip:
- view A ("target"): the tone on a random DI clip, then a random degradation: studio production
  (none / mix / full), sometimes a full band + approximate separation, sometimes listen mode
  (production.room). This is what the app hears from a song.
- view B ("candidate"): the same tone on the group's shared DI clip, usually clean (what CMA-ES
  renders from the user's DI), sometimes lightly degraded.
Positives are A_i / B_i (same tone, different DI, different processing); every other view in the
batch is a negative. Because the B views of a group share their DI, the true tone distance between
any two of them can be measured with the bench's own score (tone_db, mrstft on the same DI); it is
returned so the encoder can also be trained to put tones at distances that track tone_db.

Tones:
- gray-box: tonebench.amp with random knobs (tone 0), a near neighbour (knobs +-0.08), a medium
  neighbour (+-0.2) and an unrelated setting, so each group carries hard negatives. 30% of groups
  use a "foreign" amp (random internals: filter corners, bias, power-amp drive, cab) so the encoder
  does not only know our one amp. Knob labels are the knob vector.
- real: 15% of groups put a Guitar-TECHS mic'd amp (training folders only) in slot 0: views are its
  own recording at two different times; the other three tones are gray-box renders of the same DI
  clip as its B view, so tone_db between a real amp and our amp is known too. No knob label.

Separation is approximated cheaply (no Demucs in the loop): a noisy, smeared ratio mask computed from
the true guitar and band stems is applied to the mix, plus extra band bleed and a top-end roll-off.
The backing band comes from a pre-rendered bank (band_bank()), since synthesis is slow.
"""

import os
import time
import zlib

import numpy as np
from scipy import signal

from . import SR, amp, band as bandsynth, dsp, production, trainset

CLIP = 4.0
G = 4
N = int(CLIP * SR)
BANK = trainset.CACHE / "band_bank.npy"
BANK_SECONDS = 12.0


# ---------------------------------------------------------------- band bank

def _bank_one(seed):
    stems, _ = bandsynth.backing(int(BANK_SECONDS * SR), np.random.default_rng([seed, 77]))
    return sum(stems.values()).astype(np.float16)


def build_band_bank(n=240, workers=4):
    """n backing tracks of BANK_SECONDS (summed stems, level re a -20 dB guitar), float16."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_bank_one, range(n), chunksize=4))
    np.save(BANK, np.stack(out))
    return BANK


# ---------------------------------------------------------------- foreign amp

def random_variant(rng):
    """Random amp internals (kept the same for a tone and its neighbours)."""
    return {"hp": rng.uniform(30, 120), "pre_f": rng.uniform(150, 500), "pre_db": rng.uniform(0, 10),
            "gain_db": rng.uniform(30, 60), "bias": rng.uniform(0.0, 0.35), "f_b": rng.uniform(70, 160),
            "f_m": rng.uniform(400, 1000), "q_m": rng.uniform(0.5, 1.2), "f_t": rng.uniform(1800, 4000),
            "drive": rng.uniform(0.8, 3.0), "f_p": rng.uniform(3000, 6000), "f_r": rng.uniform(80, 140),
            "cab_hp": rng.uniform(60, 110), "cab_lo": rng.uniform(1800, 3500), "cab_span": rng.uniform(2.0, 4.0),
            "cab_order": int(rng.choice([2, 4])), "cab_peak_f": rng.uniform(1500, 4000), "cab_peak_db": rng.uniform(-6, 6)}


def render(p, di, var=None):
    """amp.render, or a foreign amp with internals var (same knobs, same structure)."""
    if var is None:
        return amp.render(p, di)
    b = lambda v: 2.0 * v - 1.0
    x = dsp.apply(di, dsp.butter("highpass", var["hp"], 2))
    x = dsp.apply(x, dsp.low_shelf(var["pre_f"], -var["pre_db"] - 6.0 * p.gain))
    x = dsp.normalize(x, -24.0) * 10 ** (var["gain_db"] * p.gain / 20)
    x = np.tanh(x + var["bias"]) - np.tanh(var["bias"])
    x = dsp.apply(x, dsp.low_shelf(var["f_b"], 12 * b(p.bass)), dsp.peak(var["f_m"], 12 * b(p.mid), var["q_m"]),
                  dsp.high_shelf(var["f_t"], 12 * b(p.treble)))
    d = var["drive"]
    x = np.tanh(d * dsp.normalize(x, -12.0)) / d
    x = dsp.apply(x, dsp.high_shelf(var["f_p"], 8 * b(p.presence)), dsp.peak(var["f_r"], 9 * p.cab_res, 1.4),
                  dsp.peak(var["cab_peak_f"], var["cab_peak_db"], 1.0), dsp.butter("highpass", var["cab_hp"], 2),
                  dsp.butter("lowpass", min(var["cab_lo"] * var["cab_span"] ** p.cab_tone, 20000), var["cab_order"]))
    return dsp.normalize(x, -20.0)


# ---------------------------------------------------------------- degradations

def _fast_codec(x, rng, stems=()):
    """Cheap stand-in for MP3: steep low-pass at 15-19 kHz."""
    sos = dsp.butter("lowpass", rng.uniform(15000, 19000), 8)
    return dsp.apply(x, sos), [dsp.apply(s, sos) for s in stems]


def produce_stems(x, stems, rng, level, mp3_p=0.3):
    """production.produce_stems, but the MP3 round trip (slow: ffmpeg) runs only mp3_p of the time."""
    if level != "full" or rng.random() < mp3_p:
        return production.produce_stems(x, stems, rng, level)
    real = production._codec
    production._codec = _fast_codec
    try:
        return production.produce_stems(x, stems, rng, level)
    finally:
        production._codec = real


def pseudo_separate(mix, g, rng):
    """Approximate separator output: a noisy, smeared ratio mask from the true stems applied to the
    mix, a random top-end roll-off, plus some extra band bleed."""
    n_fft, hop = 2048, 512
    _, _, M = signal.stft(mix, SR, nperseg=n_fft, noverlap=n_fft - hop)
    _, _, Gs = signal.stft(g, SR, nperseg=n_fft, noverlap=n_fft - hop)
    O = M - Gs
    p = rng.uniform(1.0, 2.5)
    a, o = np.abs(Gs) ** p, np.abs(O) ** p
    mask = a / (a + o + 1e-12)
    lg = np.log(mask + 1e-4) - np.log(1 - mask + 1e-4)
    lg = lg * rng.uniform(0.5, 1.5) + rng.uniform(-0.5, 0.5)
    lg = lg + rng.normal(0, rng.uniform(0.2, 1.2), lg.shape)
    kf, kt = int(rng.integers(1, 6)), int(rng.integers(1, 4))
    lg = signal.convolve(lg, np.ones((kf, kt)) / (kf * kt), mode="same")
    mask = 1 / (1 + np.exp(-lg))
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    if rng.random() < 0.6:  # separators often dull the top end
        fc = rng.uniform(5000, 16000)
        mask *= (1 / (1 + (f / fc) ** rng.uniform(2, 8)))[:, None]
    y = signal.istft(M * mask, SR, nperseg=n_fft, noverlap=n_fft - hop)[1][: len(mix)]
    y = np.pad(y, (0, len(mix) - len(y)))
    if rng.random() < 0.5:
        y = y + (mix - g) * 10 ** (rng.uniform(-30, -12) / 20)
    return y


def degrade(x, rng, bank, cfg=None):
    """Random target degradation. Returns (signal at -20 dB, info dict)."""
    cfg = cfg or {}
    level = cfg.get("level") or rng.choice(["none", "mix", "full"], p=[0.2, 0.3, 0.5])
    use_band = cfg.get("band", rng.random() < 0.4)
    use_room = cfg.get("room", rng.random() < 0.35)
    info = {"level": str(level), "band": bool(use_band), "room": bool(use_room)}
    g = dsp.normalize(x, -20.0)
    if use_band and bank is not None:
        k = int(rng.integers(len(bank)))
        s = int(rng.integers(0, bank.shape[1] - len(g)))
        others = bank[k, s : s + len(g)].astype(np.float64) * 10 ** (rng.uniform(-6, 3) / 20)
        mix = g + others
        mix, (g2,), _ = produce_stems(mix, [g], rng, level)
        if use_room:
            calib = "eq" if rng.random() < 0.2 else "none"
            mix, (g2,), _ = production.room_stems(mix, [g2], rng, calib)
        oracle = rng.random() < 0.25
        y = g2 if oracle else pseudo_separate(mix, g2, rng)
        info["oracle"] = bool(oracle)
    else:
        y, _, _ = produce_stems(g, [], rng, level)
        if use_room:
            calib = "eq" if rng.random() < 0.2 else "none"
            y, _, _ = production.room_stems(y, [], rng, calib)
    return dsp.normalize(y, -20.0), info


# ---------------------------------------------------------------- tone distances

def _logmags(x, sizes=(512, 1024, 2048)):
    x = dsp.normalize(x, -20.0)
    out = []
    for s in sizes:
        fr = np.lib.stride_tricks.sliding_window_view(x, s)[:: s // 4] * np.hanning(s)
        out.append(20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)) + 1e-12))
    return out


def tone_db_matrix(signals, range_db=80.0):
    """Pairwise metrics.mrstft_db between level-normalized signals (same DI, so this is tone)."""
    L = [_logmags(s) for s in signals]
    k = len(signals)
    D = np.zeros((k, k), np.float32)
    for i in range(k):
        for j in range(i + 1, k):
            v = []
            for a, b in zip(L[i], L[j]):
                fl = max(a.max(), b.max()) - range_db
                v.append(np.mean(np.abs(np.maximum(a, fl) - np.maximum(b, fl))))
            D[i, j] = D[j, i] = np.mean(v)
    return D


# ---------------------------------------------------------------- dataset

class Pool:
    """Memory-mapped DI / mic'd-amp arrays for one split."""

    def __init__(self, split):
        man = trainset.manifest()
        self.files = []
        for e in man["files"]:
            di = np.load(trainset.CACHE / e["di"], mmap_mode="r")
            mic = np.load(trainset.CACHE / e["mic"], mmap_mode="r") if "mic" in e else None
            lo, hi = (0, e["split"]) if split == "train" else (e["split"], e["n"])
            if hi - lo > 2 * N:
                self.files.append((e["name"], di, mic, lo, hi))
        w = np.array([f[4] - f[3] for f in self.files], float)
        self.w = w / w.sum()

    def clip(self, rng, need_mic=False, min_db=-45.0, tries=30):
        """(file index, start, di segment) with actual playing in it."""
        idx = [i for i, f in enumerate(self.files) if f[2] is not None] if need_mic else range(len(self.files))
        w = self.w[list(idx)] / self.w[list(idx)].sum()
        i = int(rng.choice(list(idx), p=w))
        _, di, _, lo, hi = self.files[i]
        for _ in range(tries):
            s = int(rng.integers(lo, hi - N))
            seg = np.asarray(di[s : s + N], np.float64)
            if dsp.rms_db(seg) > min_db:
                return i, s, seg
        return i, s, seg

    def mic(self, i, s):
        return np.asarray(self.files[i][2][s : s + N], np.float64)


def make_group(pool, bank, rng, p_real=0.15, p_foreign=0.3, p_b_degraded=0.2):
    """One group: x (2G, N) float32 [A views then B views], knobs (G, 7), kmask (G,), tone_db (G, G),
    ids (G,) tone identities (equal ids are never used as negatives)."""
    real = rng.random() < p_real and any(f[2] is not None for f in pool.files)
    var = random_variant(rng) if rng.random() < p_foreign else None
    t0 = rng.uniform(0.02, 0.98, len(amp.KNOBS))
    tones = [t0, np.clip(t0 + rng.normal(0, 0.08, 7), 0, 1), np.clip(t0 + rng.normal(0, 0.2, 7), 0, 1),
             rng.uniform(0.02, 0.98, len(amp.KNOBS))]
    if real:
        tones[1] = np.full(7, 0.5)  # noon is a useful reference next to a real amp
    ids = [int(rng.integers(1, 2**62)) for _ in range(G)]
    kmask = np.ones(G, np.float32) * (0.5 if var is not None else 1.0)

    gi, gs, gdi = pool.clip(rng, need_mic=real)
    A, B = [], []
    if real:
        name = pool.files[gi][0]
        ids[0] = zlib.crc32(name.encode())
        kmask[0] = 0.0
        B.append(pool.mic(gi, gs))
        # A view: same recording, another time (at least 1 clip away)
        for _ in range(20):
            _, di, mic, lo, hi = pool.files[gi]
            s = int(rng.integers(lo, hi - N))
            if abs(s - gs) > N and dsp.rms_db(np.asarray(di[s : s + N])) > -45:
                break
        A.append(pool.mic(gi, s))
        start = 1
    else:
        start = 0
    for t in tones[start:]:
        p = amp.AmpParams.from_vector(t)
        _, _, adi = pool.clip(rng)
        A.append(render(p, adi, var))
        B.append(render(p, gdi, var))
    D = tone_db_matrix(B)
    Aout = [degrade(a, rng, bank)[0] for a in A]
    Bout = []
    for b in B:
        if rng.random() < p_b_degraded:  # light: mix-level production only
            b = degrade(b, rng, bank, {"level": "mix", "band": False, "room": False})[0]
        Bout.append(dsp.normalize(b, -20.0))
    x = np.stack(Aout + Bout).astype(np.float32)
    return {"x": x, "knobs": np.stack(tones).astype(np.float32), "kmask": kmask, "tone_db": D,
            "ids": np.array(ids, np.int64)}


try:
    import torch
    from torch.utils.data import IterableDataset, get_worker_info

    class ToneGroups(IterableDataset):
        """Endless stream of make_group() items. Seeded per worker from (seed, worker id, time)."""

        def __init__(self, split="train", seed=None):
            self.split, self.seed = split, seed

        def __iter__(self):
            wi = get_worker_info()
            wid = wi.id if wi else 0
            torch.set_num_threads(1)
            seed = self.seed if self.seed is not None else int(time.time() * 1000) % 2**31
            rng = np.random.default_rng([seed, wid, os.getpid()])
            pool = Pool(self.split)
            bank = np.load(BANK, mmap_mode="r") if BANK.exists() else None
            while True:
                yield make_group(pool, bank, rng)
except ImportError:  # numpy-only use (bank building, tests)
    pass


def fixed_groups(split, n, seed):
    """A deterministic list of n groups (validation)."""
    rng = np.random.default_rng(seed)
    pool = Pool(split)
    bank = np.load(BANK, mmap_mode="r") if BANK.exists() else None
    return [make_group(pool, bank, rng) for _ in range(n)]


if __name__ == "__main__":
    t = time.time()
    build_band_bank()
    print(f"band bank {BANK} in {time.time() - t:.0f}s")
