"""Train ToneHub's tone encoder (tonebench.encoder) on the M-series GPU (MPS).

    uv run --group afx python train_encoder.py --hours 6.5          # fresh run
    uv run --group afx python train_encoder.py --hours 6.5 --resume # continue after a stop

Data: tonebench.augment.ToneGroups (on-the-fly renders, <= 4 CPU worker processes), training split
of tonebench.trainset (Guitar-TECHS without P3_music). If new folders are added to the cache
(python -m tonebench.trainset) the loader picks them up at the next checkpoint.

Loss = NT-Xent over all views (temperature 0.1; views of the same tone identity are never
negatives) + W_KNOB * L1 knob error (gray-box tones) + W_METRIC * Huber(a * (1 - cos(A_i, B_j)) + b,
tone_db_ij) within each group (embedding distance tracks the bench's tone score).

Validation every --val-every steps on a fixed set from the val split (last 15% of each training file):
loss terms, A->B top-1 retrieval, knob MAE. Every checkpoint (30 min) also runs a proxy bench:
16 val-split cases matched by CMA-ES on our distance, scored with the bench's tone_db. best.pt is
the checkpoint with the lowest proxy tone_db. Learning rate follows a cosine over the time budget.
Logs: models/encoder/train_log.csv (+ val rows), checkpoints in models/encoder/.
"""

import argparse
import csv
import math
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tonebench import amp, augment, dsp, encoder, metrics, production, trainset

OUT = encoder.MODELS
W_KNOB, W_METRIC, TEMP = 2.0, 1.0, 0.1


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS: bytes


class Head(torch.nn.Module):
    """Affine map from embedding distance to tone_db (two scalars, trained with the encoder)."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(3.0))
        self.b = torch.nn.Parameter(torch.tensor(0.0))


def losses(model, head, batch, dev):
    x = batch["x"].to(dev)                      # (Bg, 2G, N)
    Bg, V, N = x.shape
    G = V // 2
    e, k = model(x.reshape(Bg * V, N))
    e = e.reshape(Bg, V, -1)
    k = k.reshape(Bg, V, -1)
    zA, zB = e[:, :G].reshape(Bg * G, -1), e[:, G:].reshape(Bg * G, -1)
    M = Bg * G
    z = torch.cat([zA, zB])
    ids = batch["ids"].to(dev).reshape(-1)
    ids2 = torch.cat([ids, ids])
    logits = z @ z.T / TEMP
    same = ids2[:, None] == ids2[None, :]
    pos = torch.arange(2 * M, device=dev)
    pos = torch.where(pos < M, pos + M, pos - M)
    mask = same.clone()
    mask[torch.arange(2 * M), pos] = False             # keep the positive
    logits = logits.masked_fill(mask, float("-inf"))   # self and same-identity views are not negatives
    nt = F.cross_entropy(logits, pos)
    with torch.no_grad():
        top1 = (logits[:M].argmax(dim=1) == pos[:M]).float().mean()

    knobs = batch["knobs"].to(dev)                     # (Bg, G, 7)
    km = batch["kmask"].to(dev)                        # (Bg, G)
    kerr = (k[:, :G] - knobs).abs().mean(-1)           # A views (degraded): what the app predicts from
    kerr_b = (k[:, G:] - knobs).abs().mean(-1)
    kl = ((0.7 * kerr + 0.3 * kerr_b) * km).sum() / km.sum().clamp(min=1)
    kmae = (kerr * (km == 1)).sum() / (km == 1).sum().clamp(min=1)

    eA, eB = e[:, :G], e[:, G:]
    d = 1 - torch.einsum("gid,gjd->gij", eA, eB)       # (Bg, G, G): target i vs candidate j
    td = batch["tone_db"].to(dev)
    pred = F.softplus(head.a) * d + head.b
    ml = F.smooth_l1_loss(pred, td)
    total = nt + W_KNOB * kl + W_METRIC * ml
    return total, {"nt": nt.item(), "knob": kl.item(), "knob_mae": kmae.item(), "metric": ml.item(), "top1": top1.item()}, d.detach(), td


def collate(groups):
    return {k: torch.as_tensor(np.stack([g[k] for g in groups])) for k in groups[0]}


@torch.no_grad()
def validate(model, head, val, dev, bs=16):
    model.eval()
    rows, ds, ts = [], [], []
    for i in range(0, len(val), bs):
        b = collate(val[i : i + bs])
        _, r, d, td = losses(model, head, b, dev)
        rows.append(r)
        ds.append(d.cpu().numpy().ravel())
        ts.append(td.cpu().numpy().ravel())
    # retrieval over the whole val set: each target A_i against every candidate B_j
    es = []
    for i in range(0, len(val), bs):
        b = collate(val[i : i + bs])
        x = b["x"].to(dev)
        Bg, V, N = x.shape
        e, _ = model(x.reshape(Bg * V, N))
        es.append(e.reshape(Bg, V, -1).cpu())
    e = torch.cat(es)
    G = e.shape[1] // 2
    A, B = e[:, :G].reshape(-1, e.shape[-1]), e[:, G:].reshape(-1, e.shape[-1])
    top1_all = float(((A @ B.T).argmax(1) == torch.arange(len(A))).float().mean())
    model.train()
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    d, t = np.concatenate(ds), np.concatenate(ts)
    from scipy.stats import spearmanr
    out["metric_rho"] = float(spearmanr(d, t)[0])
    out["top1_all"] = top1_all
    return out


# ---------------------------------------------------------------- proxy bench

def proxy_cases(n=16, seed=4242):
    """Bench-like cases on the val split: target = degraded render of DI clip a, match on clip b,
    score on clip c (all different), like tonebench.cases.synthetic."""
    rng = np.random.default_rng(seed)
    pool = augment.Pool("val")
    bank = np.load(augment.BANK, mmap_mode="r")
    cfgs = [{"level": "full", "band": False, "room": False}, {"level": "full", "band": False, "room": True},
            {"level": "full", "band": True, "room": False}, {"level": "full", "band": True, "room": True}]
    out = []
    for i in range(n):
        true = amp.random_params(rng)
        a, b, c = (pool.clip(rng)[2] for _ in range(3))
        tgt, _ = augment.degrade(amp.render(true, a), rng, bank, cfgs[i % 4])
        out.append((tgt, b, c, amp.render(true, c), true))
    return out


def proxy_bench(model, cases, dev, iters=15):
    """Median tone_db of: noon, knob-head prediction, CMA-ES on embedding distance."""
    import cma
    model.eval()

    def emb(sigs):
        with torch.no_grad():
            e, k = model(torch.as_tensor(np.stack(sigs), dtype=torch.float32).to(dev))
        return e.cpu().numpy(), k.cpu().numpy()

    res = {"noon": [], "pred": [], "search": []}
    for tgt, mdi, edi, ref, true in cases:
        et, kt = emb([tgt])
        score = lambda v: metrics.mrstft_db(amp.render(amp.AmpParams.from_vector(v), edi), ref)
        res["noon"].append(score(np.full(7, 0.5)))
        res["pred"].append(score(kt[0]))
        es = cma.CMAEvolutionStrategy(np.full(7, 0.5), 0.3, {"bounds": [0, 1], "popsize": 12, "maxiter": iters,
                                                               "verbose": -9, "seed": 1})
        best, bc = None, np.inf
        while not es.stop():
            xs = es.ask()
            e, _ = emb([amp.render(amp.AmpParams.from_vector(v), mdi) for v in xs])
            cs = list(1 - e @ et[0])
            es.tell(xs, cs)
            j = int(np.argmin(cs))
            if cs[j] < bc:
                best, bc = xs[j], cs[j]
        res["search"].append(score(best))
    model.train()
    return {k: float(np.median(v)) for k, v in res.items()}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.5)
    ap.add_argument("--groups", type=int, default=16, help="groups per batch (x4 tones x2 views)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--ckpt-min", type=float, default=30.0)
    ap.add_argument("--patience", type=int, default=4, help="checkpoints without proxy improvement before stopping")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    model = encoder.Encoder().to(dev)
    head = Head().to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=1e-4)
    step, elapsed, best, bad = 0, 0.0, np.inf, 0
    if args.resume and (OUT / "latest.pt").exists():
        ck = torch.load(OUT / "latest.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        step, elapsed, best, bad = ck["step"], ck["elapsed"], ck["best"], ck.get("bad", 0)
        print(f"resumed at step {step}, {elapsed / 3600:.2f} h, best proxy {best:.3f}")
    print(f"{encoder.n_params(model)} params on {dev}", flush=True)

    vpath = OUT / "val.pt"
    if vpath.exists():
        val = torch.load(vpath, weights_only=False)
    else:
        val = augment.fixed_groups("val", 64, seed=123)
        torch.save(val, vpath)
    pcases = proxy_cases()

    def loader():
        ds = augment.ToneGroups("train")
        return iter(torch.utils.data.DataLoader(ds, batch_size=args.groups, num_workers=args.workers,
                                                collate_fn=collate, prefetch_factor=4, persistent_workers=False))

    man_mtime = os.path.getmtime(trainset.CACHE / "manifest.json")
    it = loader()
    budget = args.hours * 3600
    log = open(OUT / "train_log.csv", "a", newline="")
    w = csv.writer(log)
    if step == 0:
        w.writerow(["kind", "step", "hours", "lr", "loss", "nt", "knob", "knob_mae", "metric", "top1", "top1_all",
                    "metric_rho", "proxy_noon", "proxy_pred", "proxy_search", "rss_gb", "files"])
    t_start, t_ck = time.time() - elapsed, time.time()
    acc, n_acc, t_data = {}, 0, 0.0

    def checkpoint(name, extra=None):
        ck = {"model": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(), "step": step,
              "elapsed": time.time() - t_start, "best": best, "bad": bad, "model_kw": {}, **(extra or {})}
        torch.save(ck, OUT / f"{name}.pt.tmp")
        os.replace(OUT / f"{name}.pt.tmp", OUT / f"{name}.pt")

    while True:
        el = time.time() - t_start
        if el > budget:
            break
        frac = el / budget
        lr = args.lr * min(1.0, (step + 1) / 300) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        t0 = time.time()
        batch = next(it)
        t_data += time.time() - t0
        loss, r, _, _ = losses(model, head, batch, dev)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        step += 1
        r["loss"] = loss.item()
        for k, v in r.items():
            acc[k] = acc.get(k, 0.0) + v
        n_acc += 1
        if step % 50 == 0:
            a = {k: v / n_acc for k, v in acc.items()}
            w.writerow(["train", step, f"{el / 3600:.4f}", f"{lr:.2e}", *(f"{a[k]:.4f}" for k in ("loss", "nt", "knob", "knob_mae", "metric", "top1")),
                        "", "", "", "", "", f"{rss_gb():.2f}", ""])
            log.flush()
            print(f"step {step} {el / 3600:.2f}h lr {lr:.1e} loss {a['loss']:.3f} nt {a['nt']:.3f} knob_mae {a['knob_mae']:.3f} "
                  f"metric {a['metric']:.3f} top1 {a['top1']:.2f} data-wait {t_data / n_acc:.2f}s rss {rss_gb():.1f}GB", flush=True)
            acc, n_acc, t_data = {}, 0, 0.0
        if step % args.val_every == 0:
            v = validate(model, head, val, dev)
            w.writerow(["val", step, f"{el / 3600:.4f}", f"{lr:.2e}", "", f"{v['nt']:.4f}", f"{v['knob']:.4f}",
                        f"{v['knob_mae']:.4f}", f"{v['metric']:.4f}", f"{v['top1']:.4f}", f"{v['top1_all']:.4f}",
                        f"{v['metric_rho']:.4f}", "", "", "", f"{rss_gb():.2f}", ""])
            log.flush()
            print("VAL", step, {k: round(x, 4) for k, x in v.items()}, flush=True)
        if time.time() - t_ck > args.ckpt_min * 60:
            t_ck = time.time()
            p = proxy_bench(model, pcases, dev)
            nfiles = len(trainset.manifest()["files"])
            w.writerow(["proxy", step, f"{el / 3600:.4f}", f"{lr:.2e}", "", "", "", "", "", "", "", "",
                        f"{p['noon']:.4f}", f"{p['pred']:.4f}", f"{p['search']:.4f}", f"{rss_gb():.2f}", nfiles])
            log.flush()
            print("PROXY", step, p, flush=True)
            score = p["search"]
            if score < best:
                best, bad = score, 0
                checkpoint("best", {"proxy": p})
            else:
                bad += 1
            checkpoint("latest")
            checkpoint(f"ckpt-{step:06d}")
            if bad >= args.patience and frac > 0.5:
                print("proxy stopped improving; stopping", flush=True)
                break
            mt = os.path.getmtime(trainset.CACHE / "manifest.json")
            if mt != man_mtime:  # new training folders arrived
                man_mtime = mt
                print("manifest changed: restarting loader with", nfiles, "files", flush=True)
                del it
                it = loader()
    p = proxy_bench(model, pcases, dev)
    print("FINAL PROXY", p, flush=True)
    if p["search"] < best:
        best = p["search"]
        checkpoint("best", {"proxy": p})
    checkpoint("latest")
    print("done", step, flush=True)


if __name__ == "__main__":
    main()
