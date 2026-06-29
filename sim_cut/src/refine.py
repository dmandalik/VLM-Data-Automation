"""
refine.py -- coarse->fine boundary refinement (handoff Stage G).

The cut is a MOTION event, so refinement re-samples only motion (no detector /
torch) in a narrow window around the coarse [t1, t2], at higher fps, via OpenCV.
This pins the walk-over edges and snaps them to local motion minima -- closing
the ~1 s coarse resolution toward the <=2 s target.
"""
from __future__ import annotations

import numpy as np

from .boundary import locate_walkover
from .cut import snap_to_motion_min
from .fuse import smooth
from .motion import frame_diff, to_small_gray


def sample_motion_window(video_path: str, start_s: float, end_s: float, fps: float,
                         downscale: int = 320):
    """(timestamps, motion) sampled sequentially from start_s..end_s at ~fps."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / max(fps, 0.1))))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_s) * 1000.0)
    ts, mots, prev, k = [], [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if pos > end_s:
            break
        if k % step == 0:
            g = to_small_gray(frame, downscale)
            ts.append(pos)
            mots.append(frame_diff(prev, g))
            prev = g
        k += 1
    cap.release()
    return np.array(ts, float), np.array(mots, float)


def refine_boundary(video_path: str, t1: float, t2: float, cfg: dict) -> dict:
    pad = cfg["sampling"].get("refine_pad_s", 8.0)
    fps = cfg["sampling"].get("fine_fps", 6.0)
    ts, mot = sample_motion_window(video_path, max(0.0, t1 - pad), t2 + pad, fps,
                                   cfg["motion"].get("downscale_long_edge", 320))
    if len(ts) < 5:
        return {"t1": t1, "t2": t2, "refined": False}

    dt = float(np.median(np.diff(ts)))
    mot_s = smooth(mot, max(1, cfg["smoothing"]["median_k"]), max(1, int(round(2.0 / dt))))
    w = locate_walkover(mot_s, dt, edge_exclude_s=1.0,
                        spike_frac=cfg["boundary"].get("spike_frac", 0.30))
    rt1, rt2 = float(ts[w["t1_idx"]]), float(ts[w["t2_idx"]])

    if cfg["cut"].get("snap_to_motion_min", True):
        snap_w = cfg["cut"].get("snap_window_s", 4.0)
        rt1 = snap_to_motion_min(rt1, ts, mot_s, snap_w)
        rt2 = snap_to_motion_min(rt2, ts, mot_s, snap_w)
    return {"t1": rt1, "t2": rt2, "refined": True, "fps": fps}
