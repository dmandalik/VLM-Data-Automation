"""
extract.py -- Stages A-D driver: per-frame features + motion over a whole video,
with caching.

This is the GPU-friendly half of the pipeline (detection is the expensive step).
Run it once where the model lives (Colab / a GPU box), then iterate the cheap
analysis half (smoothing/fusion/boundary) on the cached table -- tuning the
boundary logic never re-runs the model.

Cache key = (file path, size, mtime, fps, detector, conf), so re-runs skip decode
and detection automatically.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from .detector import build_detector
from .features import extract_features, filter_people
from .frames import FrameSource
from .motion import frame_diff, stream_motion_cv2, to_small_gray


def file_signature(path: str, fps: float, extra: str = "") -> str:
    st = os.stat(path)
    h = hashlib.sha1()
    h.update(os.path.abspath(path).encode())
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    h.update(f"{fps}:{extra}".encode())
    return h.hexdigest()[:16]


def extract_video(video_path: str, cfg: dict, cache_dir: str = "cache",
                  use_cache: bool = True, progress: bool = True,
                  start_s: float = 0.0, end_s: Optional[float] = None,
                  motion_only: bool = False) -> dict:
    """Return {'meta': {...}, 'rows': [ {t, motion, ...features...}, ... ]}.

    Full mode computes the spatial/posture/motion signals (needs the detector).
    motion_only=True skips the detector entirely (no torch) and computes just the
    motion signal via OpenCV -- enough to find the cut, since it's a motion event;
    the posture/spatial features are only diagnostics.
    """
    samp, det_cfg, fp = cfg["sampling"], cfg["detector"], cfg["features"]
    fps = samp["coarse_fps"]
    motion_le = cfg["motion"].get("downscale_long_edge", 320)
    windowed = (start_s > 0.0) or (end_s is not None)

    tag = "motiononly" if motion_only else f'{det_cfg["backend"]}:{det_cfg["conf"]}'
    sig = file_signature(video_path, fps, extra=f'{tag}:{start_s}:{end_s}')
    cache_path = os.path.join(cache_dir, f"{os.path.basename(video_path)}.{sig}.json")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as fh:
            return json.load(fh)

    if motion_only:
        import shutil
        if shutil.which("ffmpeg"):                       # fast path: ffmpeg pipe + hwaccel
            from .motion import stream_motion_ffmpeg
            ts, mots, duration_s = stream_motion_ffmpeg(video_path, fps, motion_le)
        else:                                            # fallback: decode in Python
            ts, mots, duration_s = stream_motion_cv2(video_path, fps, motion_le, start_s, end_s)
        rows = [{"t": float(a), "motion": float(c)} for a, c in zip(ts, mots)]
        detector_name = "none"
    else:
        det = build_detector(det_cfg)
        fs = FrameSource(video_path, fps=fps, start_s=start_s, end_s=end_s,
                         downscale_long_edge=samp.get("downscale_long_edge", 0))
        drop = det_cfg.get("drop_manikin", True)
        it = fs
        if progress:
            try:
                from tqdm import tqdm
                span = (end_s or fs.duration_s or 0) - start_s
                it = tqdm(fs, total=int(span * fps) or None, desc="extract")
            except Exception:
                pass
        rows, prev_small = [], None
        for t, frame in it:
            H, W = frame.shape[:2]
            people = det.detect(frame)
            if drop:
                people = filter_people(people, H, W)
            feats = extract_features(people, H, W, eps_pack=fp["eps_pack"],
                                     tall_aspect=fp["tall_aspect"], kp_vis=fp["kp_vis"])
            small = to_small_gray(frame, motion_le)
            feats["t"] = float(t)
            feats["motion"] = frame_diff(prev_small, small)
            prev_small = small
            rows.append(feats)
        duration_s = fs.duration_s
        detector_name = det_cfg["backend"]

    out = {
        "meta": {
            "video": os.path.basename(video_path), "fps": fps,
            "detector": detector_name, "conf": det_cfg["conf"],
            "duration_s": duration_s, "n_frames": len(rows),
            "windowed": windowed, "start_s": start_s, "end_s": end_s,
            "motion_only": motion_only,
        },
        "rows": rows,
    }
    if not windowed:                      # only cache full-video passes
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(out, fh)
    return out
