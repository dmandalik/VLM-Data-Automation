"""
analyze.py -- the cheap half: load a cached features.json, fuse the signals, find
t1/t2, and write a diagnostic figure + results.json. No torch/GPU needed, so the
boundary logic can be tuned in a tight local loop.

    python -m src.analyze features.json [--truth 445,510] [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml

from .boundary import find_boundaries
from .fuse import assemble, discussion_score


def fmt(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


def run(features_path: str, cfg: dict, truth=None, out_dir: str = "outputs") -> dict:
    with open(features_path) as fh:
        data = json.load(fh)
    rows = data["rows"]
    t, sig = assemble(rows)
    score, extras = discussion_score(t, sig, cfg)
    res = find_boundaries(t, extras, cfg)

    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(data.get("meta", {}).get("video")
                                             or features_path))[0]
    out = {"meta": data.get("meta", {}),
           "t1": res["t1"], "t2": res["t2"],
           "t1_str": fmt(res["t1"]), "t2_str": fmt(res["t2"]),
           "gap_s": res["gap_s"], "confidence": res["confidence"]}
    if truth:
        out["truth"] = {"t1": truth[0], "t2": truth[1]}
        out["abs_err_s"] = {"t1": abs(res["t1"] - truth[0]),
                            "t2": abs(res["t2"] - truth[1])}
    with open(os.path.join(out_dir, f"{name}.results.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    _plot(t, score, extras, res, truth, os.path.join(out_dir, f"{name}.diagnostic.png"))
    return out


def _plot(t, score, extras, res, truth, path, seams=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tm = t / 60.0
    fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    ax[0].plot(tm, extras["sit_sm"]); ax[0].set_ylabel("sit_fraction"); ax[0].set_ylim(0, 1)
    mx = np.nanmax(extras["motion_sm"]) or 1.0
    ax[1].plot(tm, extras["motion_sm"] / mx); ax[1].set_ylabel("motion (norm)")
    ax[2].plot(tm, score); ax[2].set_ylabel("discussion score"); ax[2].set_xlabel("minutes")
    ax[1].axvline(res["peak_t"] / 60, c="tab:orange", ls=":", lw=1.5, label="walk-over")
    for a in ax:
        for sm in (seams or []):
            a.axvline(sm / 60, c="0.6", ls=":", lw=0.8, alpha=.6)
        if res.get("t0", 0) and res["t0"] > t[0]:
            a.axvline(res["t0"] / 60, c="tab:blue", lw=2, label="t0 (demo start)")
        a.axvline(res["t1"] / 60, c="tab:red", lw=2, label="t1 (pred)")
        a.axvline(res["t2"] / 60, c="tab:green", lw=2, label="t2 (pred)")
        if truth:
            a.axvline(truth[0] / 60, c="tab:red", ls="--", alpha=.6, label="t1 (truth)")
            a.axvline(truth[1] / 60, c="tab:green", ls="--", alpha=.6, label="t2 (truth)")
    ax[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"t1={fmt(res['t1'])}  t2={fmt(res['t2'])}  "
                 f"gap={res['gap_s']:.0f}s  conf={res['confidence']:.2f}")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--truth", default=None, help="eyeballed t1,t2 in seconds, e.g. 445,510")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    truth = [float(x) for x in a.truth.split(",")] if a.truth else None
    res = run(a.features, cfg, truth, a.out)
    print(f"t1 = {res['t1_str']}  ({res['t1']:.0f}s)")
    print(f"t2 = {res['t2_str']}  ({res['t2']:.0f}s)   gap = {res['gap_s']:.0f}s")
    print(f"confidence = {res['confidence']:.2f}")
    if truth:
        print(f"abs error: t1 {res['abs_err_s']['t1']:.0f}s, t2 {res['abs_err_s']['t2']:.0f}s")


if __name__ == "__main__":
    main()
