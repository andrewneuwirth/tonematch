"""Tables and charts for the overnight encoder report (bench/results/overnight/).

    uv run --group report python overnight_report.py

Reads the newest results/run-*-overnight-<situation>.json for each situation and
models/encoder/train_log.csv; writes results/overnight/tables.md, loss_curves.png, tone_bars.png.
Closeness score = 100 * exp(-tone_db / 6): >= 75 close, 50-75 recognizable, < 50 off.
"""

import csv
import json
from pathlib import Path

import numpy as np

from tonebench import stats

HERE = Path(__file__).parent
RES = HERE / "results"
OUT = RES / "overnight"
SITUATIONS = [("full-different", "Studio record, you play something else"),
              ("full-playalong", "Studio record, you play along"),
              ("room-different", "Phone mic in a room, you play something else"),
              ("room-playalong", "Phone mic in a room, you play along"),
              ("room-solo-demucs", "Phone mic in a room, Demucs cleanup of the recording"),
              ("full-band-demucs", "Full band, separated with Demucs (studio)"),
              ("room-band-demucs", "Full band through a phone mic, separated with Demucs")]
GROUPS = {"studio": ["full-different", "full-playalong"], "phone-mic room": ["room-different", "room-playalong", "room-solo-demucs"],
          "full-band": ["full-band-demucs", "room-band-demucs"]}
MATCHERS = ["default", "cmaes_afx", "cmaes_ours", "pred_then_search", "pred_only", "cmaes_ours_pure"]
OURS = ["cmaes_ours", "pred_then_search", "cmaes_ours_pure"]


def closeness(db):
    return 100 * np.exp(-db / 6)


def label(s):
    return "close" if s >= 75 else "recognizable" if s >= 50 else "off"


def load():
    out = {}
    for sit, _ in SITUATIONS:
        fs = sorted(RES.glob(f"run-*-overnight-{sit}.json"))
        if fs:
            out[sit] = json.loads(fs[-1].read_text())["results"]
    return out


def paired(res, m, base, track):
    a = {r["case"]: r["tone_db"] for r in res if r["matcher"] == m and r["track"] == track}
    b = {r["case"]: r["tone_db"] for r in res if r["matcher"] == base and r["track"] == track}
    d = [a[c] - b[c] for c in a if c in b]
    if not d:
        return None
    med, lo, hi = stats.median_ci(d)
    return med, lo, hi, float(np.mean(np.array(d) < 0)), len(d)


def verdict(p):
    if p is None:
        return "n/a"
    med, lo, hi, _, _ = p
    if hi < 0:
        return "yes"
    if lo > 0:
        return "no (worse)"
    return "tie (not significant)" if med <= 0.05 else "no (n.s. worse)"


def tables(allres):
    lines, head = [], {}
    for sit, title in SITUATIONS:
        res = allres.get(sit)
        if not res:
            continue
        lines += [f"### {title} (`{sit}`)", "",
                  "| matcher | track | n | tone_db median [95% CI] | mel_db median | closeness | vs default: diff [CI], wins | vs cmaes_afx: diff [CI], wins |",
                  "|---|---|---|---|---|---|---|---|"]
        for track in ("synthetic", "real"):
            for m in MATCHERS:
                v = [r["tone_db"] for r in res if r["matcher"] == m and r["track"] == track]
                if not v:
                    continue
                med, lo, hi = stats.median_ci(v)
                s = closeness(med)
                cells = []
                for base in ("default", "cmaes_afx"):
                    p = paired(res, m, base, track) if m != base else None
                    cells.append(f"{p[0]:+.2f} [{p[1]:+.2f}, {p[2]:+.2f}], {p[3]:.0%}" if p else "")
                mel = np.median([r.get("mel_db", np.nan) for r in res if r["matcher"] == m and r["track"] == track])
                lines.append(f"| {m} | {track} | {len(v)} | {med:.2f} [{lo:.2f}, {hi:.2f}] | {mel:.2f} | {s:.0f} ({label(s)}) | {cells[0]} | {cells[1]} |")
            for m in OURS:
                head[(sit, track, m)] = paired(res, m, "cmaes_afx", track)
        lines.append("")
    return lines, head


def charts(allres):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # loss curves
    rows = list(csv.DictReader(open(HERE / "models" / "encoder" / "train_log.csv")))
    tr = [r for r in rows if r["kind"] == "train"]
    va = [r for r in rows if r["kind"] == "val"]
    px = [r for r in rows if r["kind"] == "proxy"]
    f = lambda rs, k: (np.array([float(r["hours"]) for r in rs]), np.array([float(r[k]) for r in rs]))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k, c in (("loss", "#1f6f8b"), ("nt", "#99a8b2")):
        h, y = f(tr, k)
        ax[0].plot(h, y, color=c, lw=1, label=f"train {k}")
    h, y = f(va, "nt")
    ax[0].plot(h, y, "o-", color="#e6a157", ms=3, label="val nt")
    ax[0].set(xlabel="hours", title="Contrastive loss", ylim=(0, None))
    ax[0].legend(frameon=False)
    h, y = f(va, "top1_all")
    ax[1].plot(h, y, "o-", color="#1f6f8b", ms=3, label="val retrieval top-1 (256 tones)")
    h, y = f(va, "knob_mae")
    ax[1].plot(h, y, "o-", color="#e6a157", ms=3, label="val knob MAE (degraded targets)")
    h, y = f(va, "metric_rho")
    ax[1].plot(h, y, "o-", color="#6a9c78", ms=3, label="val rank corr. distance vs tone_db")
    ax[1].set(xlabel="hours", title="Validation", ylim=(0, 1))
    ax[1].legend(frameon=False, fontsize=8)
    if px:
        for k, c in (("proxy_noon", "#99a8b2"), ("proxy_pred", "#e6a157"), ("proxy_search", "#1f6f8b")):
            h, y = f(px, k)
            ax[2].plot(h, y, "o-", color=c, ms=3, label=k.replace("proxy_", ""))
        ax[2].set(xlabel="hours", title="Proxy bench (val takes): tone_db median", ylabel="dB")
        ax[2].legend(frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "loss_curves.png", dpi=130)

    # bar chart
    sits = [s for s, _ in SITUATIONS if s in allres]
    colors = {"default": "#99a8b2", "cmaes_afx": "#e6a157", "cmaes_ours": "#1f6f8b", "pred_then_search": "#6a9c78", "cmaes_ours_pure": "#7b5ea7"}
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)
    for a, track in zip(ax, ("synthetic", "real")):
        w = 0.8 / len(colors)
        for j, (m, c) in enumerate(colors.items()):
            meds, errs = [], [[], []]
            for s in sits:
                v = [r["tone_db"] for r in allres[s] if r["matcher"] == m and r["track"] == track]
                med, lo, hi = stats.median_ci(v) if v else (np.nan, np.nan, np.nan)
                meds.append(med)
                errs[0].append(med - lo)
                errs[1].append(hi - med)
            a.bar(np.arange(len(sits)) + (j - (len(colors) - 1) / 2) * w, meds, w, yerr=errs, color=c, label=m, capsize=2)
        a.set_xticks(np.arange(len(sits)), sits, rotation=20, fontsize=8)
        a.set(title=f"{track} cases: tone_db median, 95% CI (lower is better)")
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("tone_db")
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "tone_bars.png", dpi=130)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allres = load()
    lines, head = tables(allres)
    hl = ["| situation | track | cmaes_ours vs cmaes_afx | pred_then_search vs cmaes_afx | cmaes_ours_pure vs cmaes_afx |", "|---|---|---|---|---|"]
    for sit, _ in SITUATIONS:
        for track in ("synthetic", "real"):
            ps = [head.get((sit, track, m)) for m in OURS]
            if any(ps):
                hl.append(f"| {sit} | {track} | " + " | ".join(
                    f"{verdict(p)} ({p[0]:+.2f} dB, wins {p[3]:.0%})" if p else "n/a" for p in ps) + " |")
    (OUT / "tables.md").write_text("\n".join(["## Verdicts vs cmaes_afx (paired median diff, 95% CI)", ""] + hl + ["", "## Tables", ""] + lines))
    charts(allres)
    print((OUT / "tables.md").read_text())


if __name__ == "__main__":
    main()
