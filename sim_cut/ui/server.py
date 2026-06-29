"""
server.py -- FastAPI backend for the review app.

Opens a folder, groups clips into sessions (oldest->newest, gap-chained), runs the
motion-only pipeline per session in a background worker, and serves: session
data, range-enabled video streaming, frame thumbnails, edits, and confirm->split.
Paths are whitelisted to the opened folder's segments.
"""
from __future__ import annotations

import asyncio
import os
import re
import json
import threading

import numpy as np
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from src.boundary import find_boundaries, find_demo_start
from src.cut import export_span, have_ffmpeg
from src.fuse import assemble, discussion_score
from src.extract import extract_video
from src.session import glue_features
from . import anomalies as anom
from . import sessions as sess_mod

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = yaml.safe_load(open(os.path.join(PKG, "config.yaml")))
CACHE = os.path.join(PKG, "cache")
OUT = os.path.join(PKG, "outputs")
STATIC = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI()
S = {"folder": None, "sessions": [], "allowed": set(), "worker": None}


def _fmt(s):
    s = max(0, int(s or 0))
    return f"{s // 60}:{s % 60:02d}"


def _downsample(arr, n=400):
    arr = np.asarray(arr, float)
    if len(arr) == 0:
        return []
    mx = float(np.nanmax(arr)) or 1.0
    idx = np.linspace(0, len(arr) - 1, min(n, len(arr))).astype(int)
    return [round(float(arr[i] / mx), 3) for i in idx]


def _safe_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def _folder_name():
    return os.path.basename(os.path.normpath(S["folder"])) if S["folder"] else "session"


def _output_dir():
    return (os.path.normpath(S["folder"]) + "_output") if S["folder"] else os.path.join(OUT, "session")


# --------------------------------------------------------------------------- #
# Background processing
# --------------------------------------------------------------------------- #
def process_one(sess):
    sess["status"] = "processing"
    try:
        objs = [extract_video(seg["path"], CFG, cache_dir=CACHE, motion_only=True,
                              progress=False) for seg in sess["segments"]]
        order = [os.path.basename(s["path"]) for s in sess["segments"]]
        if len(objs) == 1:
            data = objs[0]
            seglist = [{"idx": 0, "path": sess["segments"][0]["path"], "offset_s": 0.0,
                        "duration_s": float(objs[0]["meta"].get("duration_s") or 0.0)}]
        else:
            data = glue_features(objs, order=order)
            b2p = {os.path.basename(s["path"]): s["path"] for s in sess["segments"]}
            seglist = [{"idx": sm["idx"], "path": b2p.get(sm["video"]),
                        "offset_s": sm["offset_s"], "duration_s": sm["duration_s"]}
                       for sm in data["segments"]]
        t, sig = assemble(data["rows"])
        score, extras = discussion_score(t, sig, CFG)
        res = find_boundaries(t, extras, CFG)
        t0i = find_demo_start(t, extras, CFG, res.get("t1_idx"))
        mot = extras["motion_sm"]
        dur = float(data["meta"].get("duration_s") or (t[-1] if len(t) else 0.0))
        result = {"t0": t0i["t0"], "t1": res["t1"], "t2": res["t2"],
                  "confidence": res["confidence"],
                  "length_plausibility": res.get("length_plausibility"),
                  "n_candidates": res.get("n_candidates", 1),
                  "reaches_hi": res.get("reaches_hi", True),
                  "trimmed_setup": t0i["trimmed"], "influx_conf": t0i["influx_conf"],
                  "median_motion": float(np.median(mot)) if len(mot) else 0.0,
                  "peak_t": res.get("peak_t"), "duration_s": dur}
        sess.update(segments_resolved=seglist, result=result, duration_s=dur,
                    spark=_downsample(mot, 400),
                    edit={"segments": [
                        {"label": "Demo", "start": result["t0"], "end": result["t1"]},
                        {"label": "Discussion", "start": result["t2"], "end": dur}]},
                    anomalies=anom.detect(sess, result, CFG), status="ready")
    except Exception as e:  # noqa: BLE001
        sess.update(status="error", error=str(e),
                    anomalies=[{"key": "error", "severity": "warn",
                                "msg": f"processing failed: {e}"}])


_pick = threading.Lock()
_frame_lock = threading.Lock()


def _worker():
    while True:
        with _pick:
            nxt = next((s for s in S["sessions"] if s["status"] == "queued"), None)
            if nxt is None:
                return
            nxt["status"] = "processing"
        process_one(nxt)


def _start_worker():
    n = max(1, min(3, (os.cpu_count() or 2) - 1))      # a few ffmpeg decodes in parallel
    workers = [t for t in S.get("workers", []) if t.is_alive()]
    while len(workers) < n:
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        workers.append(t)
    S["workers"] = workers


# --------------------------------------------------------------------------- #
# Public shaping
# --------------------------------------------------------------------------- #
def _summary(s):
    r = s.get("result")
    return {"id": s["id"], "name": s["name"], "start": s["start"].strftime("%Y-%m-%d %H:%M"),
            "duration": _fmt(s["duration_s"]), "n_segments": len(s["segments"]),
            "status": s["status"], "anomalies": s.get("anomalies", []),
            "confidence": (round(r["confidence"], 2) if r else None)}


def _detail(s):
    d = _summary(s)
    r = s.get("result")
    e = s.get("edit")
    d.update(duration_s=s["duration_s"], spark=s.get("spark", []),
             segments=[{"idx": sg["idx"], "offset_s": sg["offset_s"],
                        "duration_s": sg["duration_s"],
                        "url": f"/api/video?path={_q(sg['path'])}"}
                       for sg in s.get("segments_resolved", [])],
             result=r, edit=e, outputs=s.get("outputs"))
    return d


def _q(path):
    from urllib.parse import quote
    return quote(path)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC, "index.html")) as fh:
        return fh.read()


@app.post("/api/open")
async def open_folder(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "bad request body"}, status_code=400)
    folder = os.path.expanduser((body.get("folder") or "").strip())
    if not folder or not os.path.isdir(folder):
        return JSONResponse({"error": f"not a folder: {folder or '(empty)'}"}, status_code=400)
    try:                                          # scan off the event loop (ffprobe per file)
        items = await asyncio.to_thread(sess_mod.scan_folder, folder)
    except Exception as e:
        return JSONResponse({"error": f"could not scan folder: {e}"}, status_code=500)
    if not items:
        return JSONResponse({"error": "no video files found in that folder"}, status_code=400)
    grouped = sess_mod.single_session(items)          # whole folder = one glued session
    S["folder"] = folder
    S["sessions"] = []
    S["allowed"] = set()
    for i, g in enumerate(grouped):
        g.update(id=i, status="queued")
        S["sessions"].append(g)
        for seg in g["segments"]:
            S["allowed"].add(os.path.abspath(seg["path"]))
    _start_worker()
    return {"folder": folder, "sessions": [_summary(s) for s in S["sessions"]]}


@app.get("/api/sessions")
def list_sessions():
    return {"folder": S["folder"], "sessions": [_summary(s) for s in S["sessions"]]}


@app.get("/api/session/{sid}")
def get_session(sid: int):
    if sid < 0 or sid >= len(S["sessions"]):
        return JSONResponse({"error": "no such session"}, status_code=404)
    return _detail(S["sessions"][sid])


@app.post("/api/edit/{sid}")
async def edit(sid: int, req: Request):
    body = await req.json()
    s = S["sessions"][sid]
    segs = body.get("segments")
    if isinstance(segs, list):
        clean = []
        for x in segs:
            try:
                a, b = float(x["start"]), float(x["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if b < a:
                a, b = b, a
            clean.append({"label": (str(x.get("label") or "clip").strip() or "clip"),
                          "start": a, "end": b})
        s["edit"] = {"segments": clean}
    return {"ok": True, "edit": s.get("edit")}


def _safe_label(label, i, used):
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", (label or f"clip{i+1}")).strip("_") or f"clip{i+1}"
    lab, k = base, 2
    while lab in used:
        lab, k = f"{base}_{k}", k + 1
    used.add(lab)
    return lab


@app.post("/api/confirm/{sid}")
def confirm(sid: int):
    if not have_ffmpeg():
        return JSONResponse({"error": "ffmpeg not found on PATH (brew install ffmpeg)"},
                            status_code=400)
    s = S["sessions"][sid]
    segs = s.get("segments_resolved", [])
    usegs = (s.get("edit") or {}).get("segments", [])
    if not segs:
        return JSONResponse({"error": "session not processed yet"}, status_code=400)
    if not usegs:
        return JSONResponse({"error": "no segments to export"}, status_code=400)
    name = _folder_name()
    outdir = _output_dir()                          # <input folder>_output
    seg_objs = [{"idx": sg["idx"], "offset_s": sg["offset_s"],
                 "duration_s": sg["duration_s"], "video": os.path.basename(sg["path"])}
                for sg in segs]
    vids = {sg["idx"]: sg["path"] for sg in segs}
    outputs, used = {}, set()
    for i, u in enumerate(usegs):
        lab = _safe_label(u.get("label"), i, used)
        out_path = os.path.join(outdir, f"{name}_{lab}.mp4")
        try:
            r = export_span(seg_objs, vids, u["start"], u["end"], outdir, out_path,
                            CFG["cut"].get("mode", "copy"))
        except Exception as ex:  # noqa: BLE001
            return JSONResponse({"error": f"export '{lab}' failed: {ex}"}, status_code=500)
        if r:
            outputs[lab] = os.path.basename(r)
    s["status"] = "confirmed"
    s["outputs"] = outputs
    _save_manifest()
    return {"ok": True, "outputs": outputs, "dir": outdir}


@app.post("/api/skip/{sid}")
def skip(sid: int):
    S["sessions"][sid]["status"] = "skipped"
    _save_manifest()
    return {"ok": True}


@app.api_route("/api/video", methods=["GET", "HEAD"])
def video(path: str):
    # Starlette's FileResponse handles Range + HEAD correctly, which macOS
    # WKWebView's media loader requires -- a hand-rolled streamer renders gray.
    path = os.path.abspath(path)
    if path not in S["allowed"] or not os.path.isfile(path):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return FileResponse(path)


@app.get("/api/frame")
def frame(path: str, t: float = 0.0, w: int = 200):
    import cv2
    path = os.path.abspath(path)
    if path not in S["allowed"]:
        return JSONResponse({"error": "forbidden"}, status_code=403)
    with _frame_lock:                              # reuse one capture -> fast hover-scrub
        if S.get("_cap_path") != path:
            if S.get("_cap") is not None:
                S["_cap"].release()
            S["_cap"], S["_cap_path"] = cv2.VideoCapture(path), path
        cap = S["_cap"]
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, fr = cap.read()
    if not ok:
        return Response(b"", media_type="image/jpeg")
    fh, fw = fr.shape[:2]
    long = max(120, min(int(w), 1280))
    scale = long / max(fh, fw)
    if scale < 1:
        fr = cv2.resize(fr, (int(fw * scale), int(fh * scale)))
    ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=3600"})


def _save_manifest():
    od = _output_dir()
    os.makedirs(od, exist_ok=True)
    rows = []
    for s in S["sessions"]:
        rows.append({"name": s["name"], "status": s["status"],
                     "segments": (s.get("edit") or {}).get("segments"),
                     "confidence": (s.get("result") or {}).get("confidence"),
                     "outputs": s.get("outputs")})
    with open(os.path.join(od, "manifest.json"), "w") as fh:
        json.dump({"folder": S["folder"], "sessions": rows}, fh, indent=2)
