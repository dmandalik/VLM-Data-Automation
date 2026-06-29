"""
boundary.py -- walk-over-anchored boundary extraction (handoff Stage F).

Validated on real video: the demo->discussion cut is marked by the WALK-OVER
MOTION SPIKE (people standing and crossing to their seats), not by posture or
motion levels (both fail to separate the states over a full session -- see
FINDINGS_phase2.md).

The cut is chosen from candidate motion peaks by three soft criteria, so a stray
burst inside the demo (e.g. active resuscitation) doesn't win:
  1. prominence of the spike,
  2. whether motion DROPS into a sustained quiet tail afterward (the discussion),
  3. soft length priors -- demos run ~11-15 min, discussions ~25-30 min.
None is a hard gate; a single clear spike still wins even if the lengths are
atypical (confidence just reflects the mismatch).
"""
from __future__ import annotations

import numpy as np


def _edges(mot, peak, pre_base, post_base, spike_frac):
    """Indices where motion rises off the demo baseline (t1) and falls into the
    tail (t2), around a chosen peak."""
    n = len(mot)
    i = peak
    while i > 0 and mot[i] > pre_base + spike_frac * (mot[peak] - pre_base):
        i -= 1
    j = peak
    while j < n - 1 and mot[j] > post_base + spike_frac * (mot[peak] - post_base):
        j += 1
    return i, j


def locate_walkover(mot, dt, edge_exclude_s=30.0, spike_frac=0.30):
    """Global-max walk-over locator (used by the fine-refinement pass, where the
    window already contains a single spike)."""
    n = len(mot)
    edge = min(int(round(edge_exclude_s / dt)), max(0, n // 2 - 1))
    peak = edge + int(np.argmax(mot[edge:max(edge + 1, n - edge)]))
    pre_base = float(np.median(mot[:peak])) if peak > 0 else float(mot[peak])
    post_base = float(np.median(mot[peak + 1:])) if peak < n - 1 else float(mot[peak])
    i, j = _edges(mot, peak, pre_base, post_base, spike_frac)
    return {"t1_idx": i, "t2_idx": j, "peak_idx": peak,
            "prominence": float(mot[peak] / max(pre_base, post_base, 1e-9)),
            "pre_base": pre_base, "post_base": post_base}


def _trapezoid(x, lo, hi, margin):
    """1.0 inside [lo,hi], linear falloff to 0 over `margin` outside."""
    if lo <= x <= hi:
        return 1.0
    d = (lo - x) if x < lo else (x - hi)
    return float(max(0.0, 1.0 - d / max(margin, 1e-9)))


def find_boundaries(t, extras, cfg):
    b = cfg["boundary"]
    dt = extras["dt"]
    mot = extras["motion_sm"]      # smoothed, NaN-free
    sit = extras["sit_sm"]
    n = len(mot)
    edge = min(int(round(b.get("edge_exclude_s", 30.0) / dt)), max(0, n // 2 - 1))
    min_prom = b.get("min_prominence", 1.8)
    spike_frac = b.get("spike_frac", 0.30)

    # candidate walk-over peaks: prominent relative to the signal's dynamic range,
    # so micro-peaks in near-static motion can't qualify (avoids divide-by-~0).
    med = float(np.median(mot))
    mx = float(np.nanmax(mot)) if n else 0.0
    floor = max(0.05 * mx, 1e-9)
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(mot, prominence=max((min_prom - 1.0) * med, 0.15 * (mx - med), 1e-6),
                              distance=max(1, int(round(b.get("min_separation_s", 30.0) / dt))))
        peaks = [int(p) for p in peaks if edge <= p < n - edge]
    except Exception:
        peaks = []
    if not peaks:
        peaks = [edge + int(np.argmax(mot[edge:max(edge + 1, n - edge)]))]

    # soft length priors (seconds); 0/inf defaults make them no-ops
    demo_lo = b.get("expected_demo_min_s", 0.0)
    demo_hi = b.get("expected_demo_max_s", 1e12)
    disc_lo = b.get("expected_discussion_min_s", 0.0)
    disc_hi = b.get("expected_discussion_max_s", 1e12)
    margin = b.get("length_prior_margin_s", 300.0)

    W = max(1, int(round(b.get("base_window_s", 300.0) / dt)))         # local level window
    skip = max(1, int(round(b.get("spike_halfwidth_s", 30.0) / dt)))   # skip the spike itself

    def _med(a, c):
        seg = mot[max(0, a):max(0, c)]
        return float(np.median(seg)) if len(seg) else float(np.median(mot))

    cands = []
    for p in peaks:
        pre_base = _med(p - W, p - skip)        # demo level just before the spike
        post_base = _med(p + skip, p + W)       # discussion level once it settles
        prom = float(mot[p]) / max(pre_base, post_base, floor)
        prom_score = float(np.clip((prom - 1.0) / (2 * min_prom - 1.0), 0, 1))
        drop_score = float(np.clip((pre_base - post_base) / max(pre_base, floor) / 0.25, 0, 1))
        demo_plaus = _trapezoid(float(t[p] - t[0]), demo_lo, demo_hi, margin)
        disc_plaus = _trapezoid(float(t[-1] - t[p]), disc_lo, disc_hi, margin)
        len_prior = 0.5 + 0.25 * demo_plaus + 0.25 * disc_plaus
        cands.append(dict(p=p, score=(0.5 * prom_score + 0.5 * drop_score) * len_prior,
                          prom=prom, pre=pre_base, post=post_base, prom_score=prom_score,
                          drop_score=drop_score, len_plaus=0.5 * demo_plaus + 0.5 * disc_plaus))

    # The walk-over is a genuine motion SPIKE, so require real prominence -- a mere
    # high->low level step (busy setup -> calmer demo) is NOT a cut even if motion
    # "drops" across it. Among real spikes, drop + length priors break the tie.
    eligible = [c for c in cands if c["prom"] >= min_prom] or cands
    best = max(eligible, key=lambda c: c["score"])

    p = best["p"]
    i, j = _edges(mot, p, best["pre"], best["post"], spike_frac)
    # confidence leans on prominence (the reliable cut cue here); drop is corroboration
    conf = float(np.clip(0.65 * best["prom_score"] + 0.35 * best["drop_score"], 0, 1))
    return {"t1": float(t[i]), "t2": float(t[j]), "t1_idx": int(i), "t2_idx": int(j),
            "peak_idx": int(p), "peak_t": float(t[p]), "prominence": best["prom"],
            "confidence": conf, "length_plausibility": float(best["len_plaus"]),
            "reaches_hi": bool(best["prom"] >= min_prom),
            "gap_s": float(t[j] - t[i]), "n_candidates": len(peaks)}


def find_demo_start(t, extras, cfg, t1_idx):
    """Soft setup-trim. If the recording opens with a quiet 'setup' lead-in (a few
    people, not much happening) before a crowd floods in, return that flood time as
    the demo start t0, so the Demo clip skips the dead intro. Fires only when the
    quiet->busy jump is clear; otherwise returns t0 = 0 (nothing trimmed).

    Activity = person_count when the detector ran, else motion (a flood still shows
    up as a motion surge). Everything is soft + gated -- never a mandatory cut.
    """
    b = cfg["boundary"]
    dt = extras["dt"]
    no_trim = {"t0": float(t[0]), "t0_idx": 0, "trimmed": False,
               "influx_conf": 0.0, "signal": None}
    if not b.get("trim_setup", True) or t1_idx is None or t1_idx < 6:
        return no_trim

    has_count = bool(extras.get("has_count"))
    P = np.asarray((extras["count_sm"] if has_count else extras["motion_sm"])[:t1_idx], float)
    signal = "person_count" if has_count else "motion"
    n = len(P)
    if n < 6:
        return no_trim

    probe = min(n // 2, max(1, int(round(b.get("setup_probe_s", 45.0) / dt))))
    body = float(np.median(P[n // 2:]))            # demo body activity
    start = float(np.median(P[:probe]))            # opening activity
    if body <= 1e-9:
        return no_trim
    rel_step = (body - start) / body
    if rel_step < b.get("min_influx_step", 0.30):  # opening isn't notably quieter -> no setup
        return no_trim

    mid = start + 0.5 * (body - start)
    floor = start + 0.25 * (body - start)
    sustain = max(1, int(round(b.get("influx_sustain_s", 20.0) / dt)))
    t0_idx = 0
    for i in range(n):                             # first sustained rise = the flood
        if P[i] >= mid and float(np.mean(P[i:i + sustain] >= floor)) > 0.7:
            t0_idx = i
            break
    if t0_idx <= 0 or (t[t0_idx] - t[0]) < b.get("min_setup_s", 20.0):
        return no_trim                             # too short a setup to bother
    return {"t0": float(t[t0_idx]), "t0_idx": int(t0_idx), "trimmed": True,
            "influx_conf": float(np.clip(rel_step / 0.6, 0, 1)), "signal": signal}
