"""
detector.py -- pluggable person/pose detection.

Swapping the backend is a one-class change; feature code only ever sees the
`Person` dataclass. Two backends ship:

  - YoloPoseDetector : YOLO11-pose via ultralytics (AGPL-3.0; default, easiest
                       to stand up).
  - RtmPoseDetector  : RTMPose via rtmlib (Apache-2.0; prefer if this ever ships
                       in a product). RTMO (one-stage) keeps cost flat as the
                       huddle grows -- good for the dense demo crowd.

Use `build_detector(cfg)` to construct the configured backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Person:
    bbox: np.ndarray            # [x1, y1, x2, y2]
    conf: float
    kps: Optional[np.ndarray]   # [17, 3] COCO (x, y, score) or None


class Detector:
    """Implement detect(frame_bgr) -> List[Person]. Keep features model-agnostic."""

    def detect(self, frame_bgr: np.ndarray) -> List[Person]:
        raise NotImplementedError


class YoloPoseDetector(Detector):
    """Default detector. AGPL-3.0 -- fine for internal research; for commercial
    use prefer RTMPose/RTMO via rtmlib (Apache-2.0)."""

    def __init__(self, weights: str = "yolo11m-pose.pt", conf: float = 0.25,
                 device: Optional[str] = None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device

    def detect(self, frame_bgr: np.ndarray) -> List[Person]:
        r = self.model(frame_bgr, verbose=False, conf=self.conf,
                       device=self.device)[0]
        if r.boxes is None:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        kps = (r.keypoints.data.cpu().numpy()
               if r.keypoints is not None else [None] * len(xyxy))
        return [Person(bbox=b, conf=float(c), kps=k)
                for b, c, k in zip(xyxy, cf, kps)]


class RtmPoseDetector(Detector):
    """Apache-2.0 alternative via rtmlib. Best-effort: rtmlib's `Body` wrapper
    returns keypoints + scores, so the bbox is synthesized from the keypoint
    extent (slightly tighter than a true detection box -- only the posture
    aspect proxy is affected; centroid-based dispersion/density are unchanged).
    """

    def __init__(self, conf: float = 0.25, device: Optional[str] = None,
                 mode: str = "balanced", backend: str = "onnxruntime"):
        try:
            from rtmlib import Body
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "RTMPose backend needs: pip install rtmlib onnxruntime"
            ) from e
        self.model = Body(mode=mode, to_openpose=False, backend=backend,
                          device=device or "cpu")
        self.conf = conf

    def detect(self, frame_bgr: np.ndarray) -> List[Person]:
        keypoints, scores = self.model(frame_bgr)   # [N,17,2], [N,17]
        out: List[Person] = []
        for kp, sc in zip(keypoints, scores):
            person_conf = float(np.mean(sc))
            if person_conf < self.conf:
                continue
            x1, y1 = float(kp[:, 0].min()), float(kp[:, 1].min())
            x2, y2 = float(kp[:, 0].max()), float(kp[:, 1].max())
            kps = np.concatenate([kp, sc[:, None]], axis=1)   # [17,3]
            out.append(Person(bbox=np.array([x1, y1, x2, y2], float),
                              conf=person_conf, kps=kps))
        return out


def build_detector(cfg: dict) -> Detector:
    """Construct the detector named in cfg['backend'] ('yolo' | 'rtmpose')."""
    backend = (cfg.get("backend") or "yolo").lower()
    if backend == "yolo":
        return YoloPoseDetector(
            weights=cfg.get("weights", "yolo11m-pose.pt"),
            conf=cfg.get("conf", 0.25),
            device=cfg.get("device"),
        )
    if backend in ("rtmpose", "rtmo", "rtm"):
        return RtmPoseDetector(conf=cfg.get("conf", 0.25), device=cfg.get("device"))
    raise ValueError(f"unknown detector backend: {backend!r} (use 'yolo' or 'rtmpose')")
