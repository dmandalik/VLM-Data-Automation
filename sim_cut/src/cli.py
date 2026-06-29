"""
cli.py -- local end-to-end: glue one-or-more segment feature tables -> find the
demo->discussion cut on the global timeline -> (optional) refine on the video(s)
-> (optional) ffmpeg split. No torch needed (coarse extraction runs in Colab).

Fully local, single video (extracts, finds the cut, splits):
    python -m src.cli --video session.mp4 --split

Fully local, multi-segment session (camera cuts every ~30 min -> several files):
    python -m src.cli --video seg1.mp4 seg2.mp4 seg3.mp4 --split

Using features.json already extracted elsewhere (e.g. on Colab):
    python -m src.cli --features features.json --video session.mp4 --split

Notes: extracting locally needs `ultralytics` (+torch) and `av`; add `--device mps`
for Apple-GPU acceleration. Segments are ordered by filename (override --order).
"""
from __future__ import annotations

import argparse
import json
import os

import yaml

from .analyze import _plot, fmt
from .boundary import find_boundaries, find_demo_start
from .cut import (have_ffmpeg, map_global_to_segment, run_session_split, run_split)
from .fuse import assemble, discussion_score
from .refine import refine_boundary
from .session import glue_features


def _match_videos(segments, video_args):
    by_base = {os.path.basename(v): v for v in (video_args or [])}
    return {s["idx"]: (by_base.get(s.get("video") or "")
                       or by_base.get(os.path.basename(s.get("video") or "")))
            for s in segments}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", nargs="*", default=[],
                    help="pre-extracted features.json file(s); omit to extract from --video")
    ap.add_argument("--video", nargs="*", default=[],
                    help="video file(s); for a session, the segment files (in order)")
    ap.add_argument("--order", nargs="*", default=None,
                    help="explicit segment order as video basenames (default: sort by name)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default=None, help="detector device override: cpu | mps | cuda")
    ap.add_argument("--motion-only", action="store_true", dest="motion_only",
                    help="extract motion only via OpenCV -- no detector / torch (cut is motion-based)")
    ap.add_argument("--keep-setup", action="store_true", dest="keep_setup",
                    help="do not trim a quiet setup lead-in (Demo always starts at 0)")
    ap.add_argument("--split", action="store_true", help="write demo/discussion clips (needs video + ffmpeg)")
    ap.add_argument("--truth", default=None, help="eyeballed t1,t2 in seconds")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.device is not None:
        cfg["detector"]["device"] = a.device

    if a.features:
        objs = [json.load(open(p)) for p in a.features]
    elif a.video:
        from .extract import extract_video                 # lazy: only now do we need torch/av
        objs = []
        for vp in a.video:
            if not os.path.isfile(vp):
                hint = "  (that's a directory -- pass the .mp4 file, not the folder)" if os.path.isdir(vp) else ""
                ap.error(f"--video must point to a video file; got: {vp}{hint}")
            mode = "motion-only" if a.motion_only else "detector"
            print(f"[extract] {os.path.basename(vp)} ... (1 fps, {mode}; cached after first run)")
            objs.append(extract_video(vp, cfg, cache_dir=cfg.get("cache", {}).get("dir", "cache"),
                                      motion_only=a.motion_only))
    else:
        ap.error("provide --features (pre-extracted) or --video (to extract locally)")

    if len(objs) == 1:
        data, segments = objs[0], None
    else:
        data = glue_features(objs, order=a.order)
        segments = data["segments"]
        print(f"[glue]    {len(segments)} segments -> {data['meta']['duration_s']/60:.1f} min total")

    t, sig = assemble(data["rows"])
    score, extras = discussion_score(t, sig, cfg)
    res = find_boundaries(t, extras, cfg)
    t1, t2, conf = res["t1"], res["t2"], res["confidence"]
    print(f"[coarse]  t1={fmt(t1)}  t2={fmt(t2)}  conf={conf:.2f}  "
          f"len-plausibility={res.get('length_plausibility', float('nan')):.2f}  "
          f"({res.get('n_candidates', 1)} candidate spike(s))")

    if a.keep_setup:
        cfg["boundary"]["trim_setup"] = False
    t0info = find_demo_start(t, extras, cfg, res.get("t1_idx"))
    t0 = t0info["t0"]
    if t0info["trimmed"]:
        print(f"[setup]   demo starts at {fmt(t0)} -- trimming {t0:.0f}s of quiet setup "
              f"(via {t0info['signal']}, influx conf {t0info['influx_conf']:.2f})")

    seams = [s["offset_s"] for s in segments][1:] if segments else None
    refined = False

    if segments:
        vids = _match_videos(segments, a.video)
        s1, l1 = map_global_to_segment(t1, segments)
        s2, l2 = map_global_to_segment(t2, segments)
        if a.video and s1 == s2 and vids.get(s1) and os.path.isfile(vids[s1]):
            rf = refine_boundary(vids[s1], l1, l2, cfg)
            if rf["refined"]:
                off = segments[s1]["offset_s"]
                t1, t2, refined = off + rf["t1"], off + rf["t2"], True
                print(f"[refine]  t1={fmt(t1)}  t2={fmt(t2)}  (segment {s1}, fine {rf['fps']} fps)")
        elif a.video:
            print("[refine]  walk-over straddles a seam or segment video missing -> keeping coarse")
    elif a.video and os.path.isfile(a.video[0]):
        rf = refine_boundary(a.video[0], t1, t2, cfg)
        if rf["refined"]:
            t1, t2, refined = rf["t1"], rf["t2"], True
            print(f"[refine]  t1={fmt(t1)}  t2={fmt(t2)}  (fine {rf['fps']} fps)")

    name = os.path.splitext(os.path.basename(data.get("meta", {}).get("video") or a.features[0]))[0]
    outdir = os.path.join(a.out, name)            # one well-labeled folder per video/session
    os.makedirs(outdir, exist_ok=True)
    truth = [float(x) for x in a.truth.split(",")] if a.truth else None
    out = {"meta": data.get("meta", {}), "t0": t0, "t1": t1, "t2": t2,
           "t0_str": fmt(t0), "t1_str": fmt(t1), "t2_str": fmt(t2),
           "demo_s": t1 - t0, "gap_s": t2 - t1, "confidence": conf,
           "length_plausibility": res.get("length_plausibility"),
           "trimmed_setup": t0info["trimmed"], "influx_conf": t0info["influx_conf"],
           "refined": refined}
    if segments:
        out["segments"] = segments
    if truth:
        out["abs_err_s"] = {"t1": abs(t1 - truth[0]), "t2": abs(t2 - truth[1])}
    json.dump(out, open(os.path.join(outdir, f"{name}_results.json"), "w"), indent=2)
    _plot(t, score, extras, dict(res, t0=t0, t1=t1, t2=t2), truth,
          os.path.join(outdir, f"{name}_diagnostic.png"), seams=seams)

    if a.split:
        if not a.video:
            print("[split]   need --video to split")
        elif not have_ffmpeg():
            print("[split]   ffmpeg not found on PATH  (brew install ffmpeg)")
        elif segments:
            vids = _match_videos(segments, a.video)
            if any(not (vids.get(s["idx"]) and os.path.isfile(vids[s["idx"]])) for s in segments):
                print("[split]   need a video for every segment to split a session")
            else:
                print("[split]   writing clips (trim + concat, stream-copy)...")
                for label, path in run_session_split(segments, vids, t0, t1, t2, outdir,
                                                     cfg["cut"].get("mode", "copy"), name=name).items():
                    print(f"           {label}: {path}")
        elif os.path.isfile(a.video[0]):
            print("[split]   writing clips (stream-copy)...")
            for o in run_split(a.video[0], t0, t1, t2, outdir, cfg["cut"].get("mode", "copy")):
                print("          ", o)

    lead = f"t0 = {fmt(t0)}   " if t0info["trimmed"] else ""
    print(f"\n{lead}t1 = {fmt(t1)}   t2 = {fmt(t2)}   confidence = {conf:.2f}"
          f"   length-plausibility = {res.get('length_plausibility', float('nan')):.2f}"
          f"{'   (refined)' if refined else ''}")


if __name__ == "__main__":
    main()
