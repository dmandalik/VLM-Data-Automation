"""
frames.py -- seek-based frame sampling (handoff Stage A).

Decodes ONLY the sampled timestamps via PyAV seeking, so a 55 GB file samples
almost as fast as a small one (it reads a few percent of the bytes rather than
decoding the whole stream). Yields (timestamp_seconds, frame_bgr).
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np


class FrameSource:
    """Iterable of (timestamp_seconds, frame_bgr) sampled at `fps` via seeking.

    Open once, seek many times. `start_s`/`end_s` restrict the range (used by the
    coarse->fine refinement pass). `downscale_long_edge` shrinks frames before
    they leave the source (0 = full resolution).
    """

    def __init__(self, video_path: str, fps: float = 1.0,
                 start_s: float = 0.0, end_s: Optional[float] = None,
                 downscale_long_edge: int = 0):
        self.video_path = video_path
        self.fps = float(fps)
        self.start_s = float(start_s)
        self.end_s = end_s
        self.downscale_long_edge = int(downscale_long_edge or 0)

        import av  # lazy: keep the dependency optional until sampling is used
        self._av = av
        with av.open(video_path) as c:
            vs = c.streams.video[0]
            if vs.duration is not None and vs.time_base is not None:
                self.duration_s: Optional[float] = float(vs.duration * vs.time_base)
            elif c.duration is not None:
                self.duration_s = float(c.duration / av.time_base)
            else:
                self.duration_s = None

    def _maybe_downscale(self, img: np.ndarray) -> np.ndarray:
        le = self.downscale_long_edge
        if le > 0:
            import cv2
            h, w = img.shape[:2]
            scale = le / max(h, w)
            if scale < 1.0:
                img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                                 interpolation=cv2.INTER_AREA)
        return img

    def __iter__(self) -> Iterator[Tuple[float, np.ndarray]]:
        av = self._av
        end = self.end_s if self.end_s is not None else (self.duration_s or 1e9)

        targets = []
        t, step = self.start_s, 1.0 / self.fps
        while t < end:
            targets.append(t)
            t += step

        with av.open(self.video_path) as container:
            stream = container.streams.video[0]
            tb = stream.time_base
            for tt in targets:
                try:
                    container.seek(int(tt / tb), stream=stream, backward=True, any_frame=False)
                except Exception:
                    continue
                got = None
                for frame in container.decode(stream):
                    ftime = (frame.time if frame.time is not None
                             else (float(frame.pts * tb) if frame.pts is not None else tt))
                    if ftime + 1e-6 >= tt:
                        got = (ftime, frame)
                        break
                if got is None:
                    continue
                ftime, frame = got
                yield ftime, self._maybe_downscale(frame.to_ndarray(format="bgr24"))
