"""
session.py -- glue per-segment feature tables into one global timeline.

The recording camera cuts out every ~30 min, so one session is several segment
files, and the demo->discussion cut can fall anywhere (including mid-segment, or
in any segment). We extract features PER SEGMENT -- so motion is never
differenced across a segment boundary (that would be a false walk-over spike) --
then concatenate the tables on a global clock and null the motion at each seam.

The raw video is only ever concatenated for the final output clips (see
cut.run_session_split); analysis works entirely on the glued feature table, so a
55 GB session never gets built into one file.
"""
from __future__ import annotations

import json


def _load(obj):
    return json.load(open(obj)) if isinstance(obj, str) else obj


def order_segments(objs, order=None):
    """Order segment feature-objects. Default: by video basename (the
    LRV_<date>_<time>_... names sort chronologically). Pass `order` (a list of
    basenames) to override."""
    if order:
        rank = {name: i for i, name in enumerate(order)}
        return sorted(objs, key=lambda o: rank.get(o["meta"].get("video", ""), 1e9))
    return sorted(objs, key=lambda o: o["meta"].get("video", ""))


def glue_features(feature_objs, order=None):
    """Concatenate per-segment feature tables into one global-timeline table.

    Returns {meta, rows (global t, with a 'seg' index), segments [{idx, video,
    duration_s, offset_s}]}.
    """
    objs = order_segments([_load(o) for o in feature_objs], order)
    glued, segments, offset = [], [], 0.0
    for idx, o in enumerate(objs):
        meta, rows = o["meta"], o["rows"]
        dur = float(meta.get("duration_s") or (rows[-1]["t"] + 1.0 if rows else 0.0))
        for j, r in enumerate(rows):
            gr = dict(r)
            gr["t"] = float(r["t"]) + offset
            gr["seg"] = idx
            if j == 0:
                gr["motion"] = float("nan")        # no frame-diff across the seam
            glued.append(gr)
        segments.append({"idx": idx, "video": meta.get("video"),
                         "duration_s": dur, "offset_s": offset})
        offset += dur
    return {"meta": {"video": "SESSION", "fps": objs[0]["meta"].get("fps", 1.0) if objs else 1.0,
                     "duration_s": offset, "n_segments": len(segments)},
            "rows": glued, "segments": segments}
