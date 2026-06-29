"""
fuse.py -- smoothing + fusion into a per-frame discussion-likelihood score
(handoff Stage E).

The discussion-score is HIGH during the seated, low-motion debrief and LOW during
the active demo. It is built from the two room-invariant signals validated on
video:
  + posture  (sit_fraction: HIGH when seated)
  + motion   (LOW in the discussion tail -> enters as -motion)
with a small optional spatial term kept only for ablation (density/packing
default to zero weight -- they failed on the real frames).
"""
from __future__ import annotations

import numpy as np

SIGNAL_KEYS = ["sit_fraction", "motion", "hull_area_frac",
               "frac_tightly_packed", "density_peak_per_person", "person_count"]


def interp_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    good = ~np.isnan(x)
    if good.sum() == 0:
        return np.zeros(len(x))
    if good.all():
        return x
    return np.interp(np.arange(len(x)), np.flatnonzero(good), x[good])


def median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    n, half = len(x), k // 2
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(x[max(0, i - half):min(n, i + half + 1)])
    return out


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    n, half = len(x), win // 2
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = (c[hi] - c[lo]) / (hi - lo)
    return out


def smooth(x: np.ndarray, median_k: int, ma_win: int) -> np.ndarray:
    return moving_average(median_filter(interp_nan(x), median_k), ma_win)


def zscore(x: np.ndarray) -> np.ndarray:
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 1e-9 else np.zeros_like(x)


def assemble(rows: list) -> tuple:
    t = np.array([r["t"] for r in rows], float)
    sig = {k: np.array([r.get(k, np.nan) for r in rows], float) for k in SIGNAL_KEYS}
    return t, sig


def discussion_score(t: np.ndarray, sig: dict, cfg: dict) -> tuple:
    sm = cfg["smoothing"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    ma = max(1, int(round(sm["ma_window_s"] / dt)))
    med = sm["median_k"]

    sit = smooth(sig["sit_fraction"], med, ma)
    mot = smooth(sig["motion"], med, ma)
    w = cfg["fusion"]["weights"]

    score = w.get("posture", 0.0) * zscore(sit) + w.get("motion", 0.0) * zscore(-mot)
    if w.get("dispersion", 0.0):
        score = score + w["dispersion"] * zscore(smooth(sig["hull_area_frac"], med, ma))

    has_count = not bool(np.all(np.isnan(sig["person_count"])))
    extras = {"sit_sm": sit, "motion_sm": mot, "ma_samples": ma, "dt": dt,
              "has_count": has_count,
              "count_sm": smooth(sig["person_count"], med, ma) if has_count else None}
    return score, extras
