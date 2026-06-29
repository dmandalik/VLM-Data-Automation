"""
motion.py -- cheap frame-difference motion signal (handoff Stage D).

Mean absolute difference between consecutive sampled frames, computed on a small
grayscale image. Detector-independent and free. The discussion is the global
motion-minimum tail, so this is a strong cross-check on the posture signal.
"""
from __future__ import annotations

import subprocess

import cv2
import numpy as np


def to_small_gray(frame_bgr: np.ndarray, long_edge: int = 320) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = long_edge / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)
    return gray


def frame_diff(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Mean |Δpixel| between two small gray frames; NaN if no/!=-shape previous."""
    if prev_gray is None or prev_gray.shape != gray.shape:
        return float("nan")
    return float(np.mean(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16))))


def stream_motion_cv2(video_path, fps, downscale=320, start_s=0.0, end_s=None):
    """(timestamps, motion, duration_s) sampled at ~fps via OpenCV -- no PyAV and
    no detector. Backs the detector-free 'motion only' local path: since the cut
    is a motion event, this alone is enough to find it."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_s = (frame_count / src_fps) if (frame_count and src_fps) else None
    step = max(1, int(round(src_fps / max(fps, 0.1))))
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    ts, mots, prev, k = [], [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if end_s is not None and pos > end_s:
            break
        if k % step == 0:
            g = to_small_gray(frame, downscale)
            ts.append(pos)
            mots.append(frame_diff(prev, g))
            prev = g
        k += 1
    cap.release()
    if duration_s is None and ts:
        duration_s = ts[-1] + 1.0 / max(fps, 0.1)
    return np.array(ts, float), np.array(mots, float), float(duration_s or 0.0)


def _probe_wh(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
            capture_output=True, text=True, timeout=30).stdout.strip()
        w, h = out.split("x")[:2]
        return int(w), int(h)
    except Exception:
        return 0, 0


def _readn(f, n):
    chunks, got = [], 0
    while got < n:
        c = f.read(n - got)
        if not c:
            break
        chunks.append(c)
        got += len(c)
    return b"".join(chunks)


def stream_motion_ffmpeg(video_path, fps, long_edge=320):
    """(timestamps, motion, duration_s) via an ffmpeg pipe. ffmpeg decodes once in
    C -- with macOS videotoolbox hardware acceleration when available -- and emits
    only the downscaled grayscale frames at `fps`, so we never decode every frame
    in Python. Much faster than stream_motion_cv2 on long clips."""
    w, h = _probe_wh(video_path)
    if not w:
        raise RuntimeError(f"ffprobe could not read {video_path}")
    sc = long_edge / max(w, h) if max(w, h) > long_edge else 1.0
    ow = max(2, (int(round(w * sc)) // 2) * 2)
    oh = max(2, (int(round(h * sc)) // 2) * 2)
    cmd = ["ffmpeg", "-v", "error", "-i", video_path, "-an",
           "-vf", f"fps={fps},scale={ow}:{oh}", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    ts, mots, prev, i, fsz = [], [], None, 0, ow * oh
    while True:
        buf = _readn(proc.stdout, fsz)
        if len(buf) < fsz:
            break
        g = np.frombuffer(buf, np.uint8).reshape(oh, ow)
        mots.append(float("nan") if prev is None else
                    float(np.mean(np.abs(g.astype(np.int16) - prev.astype(np.int16)))))
        ts.append(i / fps)
        prev = g
        i += 1
    proc.stdout.close()
    proc.wait()
    if not ts:
        return np.array([], float), np.array([], float), 0.0
    return np.array(ts, float), np.array(mots, float), float(ts[-1] + 1.0 / fps)
