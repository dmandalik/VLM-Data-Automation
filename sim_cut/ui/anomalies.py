"""
anomalies.py -- flag sessions that need a human eye, each with a short reason and
a severity. The UI turns these into badges + an alert banner with choices.
"""
from __future__ import annotations


def detect(session: dict, result: dict, cfg: dict) -> list:
    """Return [{key, severity, msg}, ...]. severity: 'warn' | 'info'."""
    a = cfg.get("anomaly", {})
    dur = session.get("duration_s", 0.0)
    flags = []

    def add(key, sev, msg):
        flags.append({"key": key, "severity": sev, "msg": msg})

    # --- segment sequencing (your case) ---------------------------------------
    for i, g in enumerate(session.get("seg_gaps", []) or []):
        if g is None:
            continue
        if g < -a.get("overlap_s", 5.0):
            add("seg_overlap", "warn", f"segments {i}/{i+1} overlap by {abs(g):.0f}s")
        elif g > a.get("seg_gap_warn_s", 20.0):
            add("seg_gap", "warn", f"{g:.0f}s gap between segments {i}/{i+1} "
                                   "(missing footage?)")

    # --- duration -------------------------------------------------------------
    if dur and dur < a.get("min_session_s", 360.0):
        add("short", "warn", f"unusually short ({dur/60:.1f} min)")

    if not result:
        add("unprocessed", "info", "not processed yet")
        return flags

    # --- detection quality ----------------------------------------------------
    if not result.get("reaches_hi", True) or result.get("confidence", 1.0) < a.get("min_conf", 0.30):
        add("no_cut", "warn", "no confident demo->discussion cut found")
    if result.get("median_motion") is not None and \
            result["median_motion"] < a.get("min_motion", 0.0):
        add("low_action", "warn", "little movement throughout (empty room?)")
    if result.get("length_plausibility", 1.0) < a.get("min_len_plaus", 0.15):
        add("odd_length", "info", "demo/discussion length outside the usual range")
    if result.get("n_candidates", 1) >= a.get("ambiguous_n", 4) and \
            result.get("confidence", 1.0) < a.get("ambiguous_conf", 0.6):
        add("ambiguous", "info", "several candidate cuts -- please verify")

    # --- cut position ---------------------------------------------------------
    t1, t2 = result.get("t1"), result.get("t2")
    if t1 is not None and t1 < a.get("edge_s", 60.0):
        add("early_cut", "warn", "cut is very near the start")
    if t2 is not None and dur and (dur - t2) < a.get("edge_s", 60.0):
        add("late_cut", "warn", "cut near the end -- discussion may be cut off")
    return flags
