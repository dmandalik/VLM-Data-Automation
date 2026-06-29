"""
cut.py -- snap the cut to a quiet frame and split with ffmpeg (handoff Stage G).

demo.mp4 = [0, t1], discussion.mp4 = [t2, end]; the [t1, t2] walk-over is dropped.
Stream-copy (`mode: copy`) is instant and snaps to keyframes; re-encode is
frame-accurate but slow. Never deletes the source.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np


def snap_to_motion_min(t_target: float, t_arr: np.ndarray, motion: np.ndarray,
                       window_s: float) -> float:
    """Nudge a cut point to the lowest-motion frame within +/- window_s, so the
    cut never lands mid-stride."""
    idx = np.flatnonzero(np.abs(t_arr - t_target) <= window_s)
    if len(idx) == 0:
        return float(t_target)
    good = idx[~np.isnan(motion[idx])]
    if len(good) == 0:
        return float(t_target)
    return float(t_arr[good[int(np.argmin(motion[good]))]])


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def split_commands(video: str, t0: float, t1: float, t2: float, outdir: str, mode: str = "copy"):
    """Return [(out_path, ffmpeg_argv), ...] for Demo [t0,t1] and Discussion [t2,end]
    (t0 > 0 skips a trimmed setup lead-in; t0 = 0 means from the start)."""
    base = os.path.splitext(os.path.basename(video))[0]
    demo = os.path.join(outdir, f"{base}_Demo.mp4")
    disc = os.path.join(outdir, f"{base}_Discussion.mp4")
    enc = ["-c", "copy"] if mode == "copy" else ["-c:v", "libx264", "-c:a", "aac"]
    if t0 and t0 > 0:
        c_demo = ["ffmpeg", "-y", "-ss", f"{t0:.3f}", "-i", video,
                  "-t", f"{max(0.0, t1 - t0):.3f}", *enc, demo]
    else:
        c_demo = ["ffmpeg", "-y", "-i", video, "-t", f"{t1:.3f}", *enc, demo]
    c_disc = ["ffmpeg", "-y", "-ss", f"{t2:.3f}", "-i", video, *enc, disc]
    return [(demo, c_demo), (disc, c_disc)]


def run_split(video: str, t0: float, t1: float, t2: float, outdir: str, mode: str = "copy"):
    os.makedirs(outdir, exist_ok=True)
    outs = []
    for path, cmd in split_commands(video, t0, t1, t2, outdir, mode):
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        outs.append(path)
    return outs


# --------------------------------------------------------------------------- #
# Multi-segment (session) split: the cut may span several ~30-min segments.
# --------------------------------------------------------------------------- #
def map_global_to_segment(t_global: float, segments: list):
    """(segment idx, local time) for a global timestamp."""
    for s in segments:
        if s["offset_s"] <= t_global < s["offset_s"] + s["duration_s"]:
            return s["idx"], t_global - s["offset_s"]
    last = segments[-1]
    return last["idx"], float(np.clip(t_global - last["offset_s"], 0.0, last["duration_s"]))


def plan_session_split(segments: list, t0: float, t1: float, t2: float):
    """Piece lists for Demo [t0,t1] and Discussion [t2,end] across segments.
    Each piece = (seg_idx, start_local, end_local) where end_local None = to end."""
    demo, disc = [], []
    for s in segments:
        a, b = s["offset_s"], s["offset_s"] + s["duration_s"]
        if a < t1 and b > t0:                        # this segment is (partly) Demo [t0,t1]
            demo.append((s["idx"], max(t0, a) - a, None if t1 >= b else (t1 - a)))
        if b > t2:                                   # this segment is (partly) Discussion [t2,end]
            disc.append((s["idx"], 0.0 if t2 <= a else (t2 - a), None))
    return demo, disc


def _piece_file(label, idx, s0, e0, video, outdir, mode):
    if s0 == 0.0 and e0 is None:
        return video                                 # whole segment, no trim
    tmp = os.path.join(outdir, f"._{label}_{idx}.mp4")
    cmd = ["ffmpeg", "-y", "-ss", f"{s0:.3f}", "-i", video]
    if e0 is not None:
        cmd += ["-t", f"{max(0.0, e0 - s0):.3f}"]
    cmd += (["-c", "copy"] if mode == "copy" else ["-c:v", "libx264", "-c:a", "aac"]) + [tmp]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp


def run_session_split(segments, video_paths, t0, t1, t2, outdir, mode="copy", name="SESSION"):
    """Write Demo/Discussion clips that may each span several segments, via
    trim + concat (stream-copy). `video_paths[idx]` is the file for segment idx."""
    os.makedirs(outdir, exist_ok=True)
    demo, disc = plan_session_split(segments, t0, t1, t2)
    outs = {}
    for label, pieces in (("Demo", demo), ("Discussion", disc)):
        if not pieces:
            continue
        files = [_piece_file(label, idx, s0, e0, video_paths[idx], outdir, mode)
                 for (idx, s0, e0) in pieces]
        temps = [f for f in files if os.path.basename(f).startswith("._")]
        out = os.path.join(outdir, f"{name}_{label}.mp4")
        if len(files) == 1:
            subprocess.run(["ffmpeg", "-y", "-i", files[0], "-c", "copy", out],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            listfile = os.path.join(outdir, f"._{label}_list.txt")
            with open(listfile, "w") as fh:
                for f in files:
                    fh.write(f"file '{os.path.abspath(f)}'\n")
            subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
                            "-c", "copy", out],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            temps.append(listfile)
        for tf in temps:                          # drop intermediate trims / concat list
            try:
                os.remove(tf)
            except OSError:
                pass
        outs[label] = out
    return outs


# --------------------------------------------------------------------------- #
# Generic span export: cut ANY [start, end] (global, may span glued clips) to a
# named output. Powers the manual segment editor (add/crop arbitrary clips).
# --------------------------------------------------------------------------- #
def plan_span(segments: list, start: float, end: float):
    """Pieces (seg_idx, local_start, local_end|None) covering global [start, end]."""
    pieces = []
    for s in segments:
        a, b = s["offset_s"], s["offset_s"] + s["duration_s"]
        if a < end and b > start:
            pieces.append((s["idx"], max(start, a) - a, None if end >= b else (end - a)))
    return pieces


def export_span(segments, video_paths, start, end, outdir, out_path, mode="copy"):
    """Write global [start, end] (stitched across clips) to out_path. Returns the
    path, or None if the span is empty."""
    os.makedirs(outdir, exist_ok=True)
    pieces = plan_span(segments, float(start), float(end))
    if not pieces:
        return None
    tag = os.path.splitext(os.path.basename(out_path))[0]
    files, temps = [], []
    for k, (idx, s0, e0) in enumerate(pieces):
        if s0 == 0.0 and e0 is None:
            files.append(video_paths[idx])                 # whole source clip, no trim
        else:
            tmp = os.path.join(outdir, f"._{tag}_{k}.mp4")
            cmd = ["ffmpeg", "-y", "-ss", f"{s0:.3f}", "-i", video_paths[idx]]
            if e0 is not None:
                cmd += ["-t", f"{max(0.0, e0 - s0):.3f}"]
            cmd += (["-c", "copy"] if mode == "copy" else ["-c:v", "libx264", "-c:a", "aac"]) + [tmp]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            files.append(tmp)
            temps.append(tmp)
    if len(files) == 1 and not temps:                      # one full source clip
        subprocess.run(["ffmpeg", "-y", "-i", files[0], "-c", "copy", out_path],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif len(files) == 1:                                  # one trimmed piece -> it is the output
        os.replace(files[0], out_path)
        temps = []
    else:
        listfile = os.path.join(outdir, f"._{tag}_list.txt")
        with open(listfile, "w") as fh:
            for f in files:
                fh.write(f"file '{os.path.abspath(f)}'\n")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
                        "-c", "copy", out_path],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temps.append(listfile)
    for tf in temps:
        try:
            os.remove(tf)
        except OSError:
            pass
    return out_path
