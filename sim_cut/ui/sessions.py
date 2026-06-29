"""
sessions.py -- scan a folder of clips, order oldest->newest, and chain
consecutive segments into sessions by timestamp contiguity.

Start time is parsed from the LRV_<date>_<time> filename (falling back to file
mtime); duration from ffprobe. Segments whose next-start lands within a small
window of the previous end (the ~1 s camera-cut gap) chain into one session; a
larger gap starts a new session. Per-segment gaps are recorded so the anomaly
pass can flag non-sequential / missing / overlapping segments.
"""
from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timedelta

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mpg", ".mts")
_TS = re.compile(r"(\d{8})[_-](\d{6})")


def parse_start(path: str) -> datetime:
    m = _TS.search(os.path.basename(path))
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return datetime.fromtimestamp(os.path.getmtime(path))


def duration_s(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip() or 0.0)
    except Exception:
        return 0.0


def scan_folder(folder: str) -> list:
    items = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and name.lower().endswith(VIDEO_EXT) and not name.startswith("."):
            items.append({"path": p, "name": name, "start": parse_start(p),
                          "duration": duration_s(p)})
    items.sort(key=lambda x: x["start"])
    return items


def group_sessions(items: list, chain_gap_s: float = 90.0,
                   overlap_tol_s: float = 5.0) -> list:
    """Chain consecutive segments into sessions. Returns a list of sessions, each
    {segments: [...], duration_s, start, seg_gaps: [gap after each segment]}."""
    sessions = []
    cur = []
    for it in items:
        if not cur:
            cur = [dict(it, gap_before=None)]
            continue
        prev = cur[-1]
        prev_end = prev["start"] + timedelta(seconds=prev["duration"])
        gap = (it["start"] - prev_end).total_seconds()
        if -overlap_tol_s <= gap <= chain_gap_s:
            cur.append(dict(it, gap_before=gap))
        else:
            sessions.append(_finalize(cur))
            cur = [dict(it, gap_before=None)]
    if cur:
        sessions.append(_finalize(cur))
    return sessions


def single_session(items: list) -> list:
    """Treat the WHOLE folder as one session: every clip, oldest->newest, glued.
    Per-segment gaps are still recorded so the anomaly pass can flag clips that
    don't follow on sequentially. `items` must already be sorted by start time."""
    if not items:
        return []
    segs = [dict(items[0], gap_before=None)]
    for prev, it in zip(items, items[1:]):
        prev_end = prev["start"] + timedelta(seconds=prev["duration"])
        segs.append(dict(it, gap_before=(it["start"] - prev_end).total_seconds()))
    return [_finalize(segs)]


def _finalize(segs: list) -> dict:
    return {
        "segments": segs,
        "start": segs[0]["start"],
        "duration_s": sum(s["duration"] for s in segs),
        "seg_gaps": [s["gap_before"] for s in segs[1:]],
        "name": f"session {segs[0]['start'].strftime('%H:%M')} "
                f"({len(segs)} seg{'s' if len(segs) > 1 else ''})",
    }
