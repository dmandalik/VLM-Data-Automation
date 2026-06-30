# sim_cut — demo→discussion cut detection

Automatically find where a medical simulation video switches from the active
mock scenario (**demo**) to the seated group debrief (**discussion**), and
optionally split the video there — with **no training data** and **no per-video
manual setup**, accurate to ~1–2 s, generalizing across rooms and cameras.

## Why this generalizes across rooms

The physical event is "most people go from standing to sitting." But the signal
that *robustly marks* it, validated on real footage, turned out to be **motion** —
not posture or spatial layout (those overlap too much between the two states).
The one room-invariant marker is the **walk-over motion spike**: the burst of
movement as people get up and cross to their seats.

So the boundary stage anchors on that spike — **`t1`** = its onset (demo end),
**`t2`** = its offset (discussion start). Motion comes straight from frame
differences, so there's no ROI to tune and nothing camera-specific. On a real
29-min sample it found t1 = 7:28 / t2 = 8:32 (conf 0.87), inside the owner's
confirmed 7–8 min cut.

## Install

Needs **Python 3.9+** and **ffmpeg** on your PATH (`brew install ffmpeg`).

```bash
cd sim_cut
python3 -m venv .venv && source .venv/bin/activate   # or your env of choice
pip install -r requirements.txt
```

The cut is decided from **motion alone**, so for the default path you only need
OpenCV + ffmpeg + the small UI deps — **no torch**. The detector (`ultralytics`,
listed in `requirements.txt`) is optional and only adds posture/spatial
diagnostics; skip it if you just want the cut.

## Run the review app (native macOS window)

The main way to use it. Takes a **folder of clips**, glues the whole folder into
one session (oldest→newest), auto-runs detection in the background, and lets you
review/edit the cut against the video, then splits.

```bash
cd sim_cut
pip install fastapi uvicorn pywebview opencv-python numpy scipy pyyaml   # motion-only UI, no torch
python -m ui.app                           # or double-click launch_sim_cut.command
```

Browse to a folder → it processes in the background → pick a session → scrub the
timeline, drag a clip's end-handles to crop **both** ends (or type mm:ss, or
"set = playhead"), add/delete segments → **Confirm & export**. Flagged sessions
(no cut, too short, segment gaps, low confidence) show a ⚠ badge. Exported clips
land in a **`<folder>_output`** folder next to your clips, labeled
`<folder>_Demo.mp4` / `<folder>_Discussion.mp4`.

The preview is a server-rendered **frame scrubber** (it shows decoded JPEG frames,
so it works regardless of the source codec — HEVC clips that browsers won't play
inline still scrub fine). Exports are full-quality stream-copy of the originals.

## Run from the CLI

```bash
# motion only — no detector/torch (needs just OpenCV + ffmpeg)
python -m src.cli --video session.mp4 --motion-only --split

# a segmented session (camera cut every ~30 min) — pass the parts in order
python -m src.cli --video seg1.mp4 seg2.mp4 seg3.mp4 --motion-only --split

# with the detector for posture/spatial diagnostics (needs ultralytics + torch, av)
python -m src.cli --video session.mp4 --split          # add --device mps on Apple GPU
```

Each run gets a labeled folder `outputs/<name>/` with `<name>_Demo.mp4`,
`<name>_Discussion.mp4` (with `--split`), `<name>_results.json` (t0/t1/t2,
confidence, length-plausibility) and a `<name>_diagnostic.png`. t1/t2 are also
printed to the terminal.

**Setup-trim (soft).** If a recording opens with a quiet setup lead-in before the
crowd floods in, the Demo clip auto-starts at that flood (`t0`) — but only when
the quiet→busy jump is clear; otherwise nothing is trimmed. Turn it off with
`--keep-setup`.

**Segmented sessions.** A session is often several ~30-min files and the cut can
fall anywhere. We glue the per-segment *features* onto one global clock (never the
raw video — that would inject a false spike at every seam), find the cut there,
and stitch the demo/discussion clips across segments via trim+concat
(stream-copy). Segments are ordered by filename (override with `--order`).

## Layout

```
sim_cut/
├── config.yaml            # every tunable, all stages
├── requirements.txt
├── launch_sim_cut.command # double-click launcher for the app
├── src/
│   ├── motion.py          # frame-diff motion signal (the primary cut signal)
│   ├── boundary.py        # walk-over-spike anchor → t1/t2 + confidence
│   ├── refine.py          # higher-fps motion refine near the boundary (no torch)
│   ├── fuse.py            # smoothing + score fusion
│   ├── cut.py             # ffmpeg split / concat across segments
│   ├── session.py         # glue ~30-min segments onto one global clock
│   ├── extract.py         # features + motion over a video → cached features.json
│   ├── frames.py          # seek-based frame sampling + caching
│   ├── analyze.py         # local runner + diagnostics
│   ├── features.py        # ROI-free, self-normalized per-frame features
│   ├── detector.py        # optional pose backends (YOLO / RTMPose) — diagnostics only
│   ├── inspect.py         # CLI: compare features on still frames
│   └── cli.py             # python -m src.cli --video ... [--split]
└── ui/
    ├── app.py             # native macOS window (pywebview) launcher
    ├── server.py          # FastAPI backend: scan, process, frame preview, export
    ├── sessions.py        # session scan / glue helpers
    ├── anomalies.py       # flags: no cut, too short, gaps, low confidence
    └── static/index.html  # the review UI (frame scrubber + segment editor)
```

`cache/` (cached features per file signature) and `outputs/` (results + clips) are
created on first run and are git-ignored.
