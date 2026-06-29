"""
features.py -- per-frame, ROI-free, self-normalized scene-state features.

Ported from the handoff seed (starter_features.py). The math is unchanged -- it
is the proven separator between the two states -- but the tunables are now
parameters, so the pipeline can drive them from config.yaml.

  DEMO       : tight standing huddle around a stretcher
               -> LOW dispersion, HIGH crowding / local density
  DISCUSSION : participants seated in a dispersed ring
               -> HIGH dispersion, LOW crowding / local density

Everything is normalized by frame diagonal / area and by person count, so values
are comparable across resolutions, rooms, and camera placements.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

from .detector import Person

# --------------------------------------------------------------------------- #
# Defaults (overridable per-call; the pipeline passes values from config.yaml)
# --------------------------------------------------------------------------- #
EPS_PACK = 0.06     # NN dist (fraction of frame diagonal) below which two
                    # people count as "tightly packed" (huddle cue)
TALL_ASPECT = 1.6   # bbox height/width above which a person box is "tall"
                    # (standing people project as taller boxes in an oblique view)
KP_VIS = 0.30       # keypoint score below which a joint is not trusted

# COCO-17 keypoint indices
L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE = 5, 6, 11, 12, 13, 14


def filter_people(persons: List[Person], H: int, W: int) -> List[Person]:
    """Drop the infant manikin (and other non-people) before featurizing.

    Conservative: only removes very small, low-in-frame, horizontal boxes, so
    real (possibly seated/occluded) people are kept.
    """
    out: List[Person] = []
    for p in persons:
        x1, y1, x2, y2 = p.bbox
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        area_frac = (bw * bh) / (H * W)
        horizontal = (bh / bw) < 0.6
        tiny = area_frac < 0.002
        # a tiny, horizontal box low in the frame is likely the manikin
        if tiny and horizontal and (y1 / H) > 0.45:
            continue
        out.append(p)
    return out


def extract_features(persons: List[Person], H: int, W: int, *,
                     eps_pack: float = EPS_PACK,
                     tall_aspect: float = TALL_ASPECT,
                     kp_vis: float = KP_VIS) -> dict:
    """Compute the ROI-free feature dict for one frame. See module docstring."""
    diag = (H * H + W * W) ** 0.5
    area = H * W
    n = len(persons)
    f = {"person_count": n}
    if n == 0:
        return f

    boxes = np.array([p.bbox for p in persons], dtype=float)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    cents = np.stack([cx, cy], axis=1)
    bw = np.clip(boxes[:, 2] - boxes[:, 0], 1, None)
    bh = boxes[:, 3] - boxes[:, 1]
    aspect = bh / bw

    # --- DISPERSION: low in the demo huddle, high in the discussion ring ------
    if n >= 3:
        try:
            f["hull_area_frac"] = float(ConvexHull(cents).volume / area)
        except Exception:
            f["hull_area_frac"] = None
    if n >= 2:
        f["mean_pairwise_norm"] = float(pdist(cents).mean() / diag)
        sq = squareform(pdist(cents))
        np.fill_diagonal(sq, np.inf)
        nn = sq.min(axis=1) / diag
        f["nn_median_norm"] = float(np.median(nn))
        f["nn_min_norm"] = float(nn.min())
        f["frac_tightly_packed"] = float(np.mean(nn < eps_pack))   # HIGH in demo

    # --- LOCAL DENSITY peak: high when a tight cluster exists (demo) ----------
    gh, gw = max(H // 20, 1), max(W // 20, 1)
    grid = np.zeros((gh, gw), np.float32)
    for x, y in cents:
        grid[min(int(y // 20), gh - 1), min(int(x // 20), gw - 1)] += 1
    grid = cv2.GaussianBlur(grid, (0, 0), 1.5)
    f["density_peak"] = float(grid.max())
    f["density_peak_per_person"] = float(grid.max() / n)

    # --- POSTURE proxies (supplementary; viewpoint-sensitive) ----------------
    f["aspect_median"] = float(np.median(aspect))
    f["frac_tall_boxes"] = float(np.mean(aspect > tall_aspect))    # standing taller

    stand = sit = usable = 0
    for p in persons:
        if p.kps is None:
            continue
        sh, hip, kn = p.kps[[L_SH, R_SH]], p.kps[[L_HIP, R_HIP]], p.kps[[L_KNEE, R_KNEE]]
        if sh[:, 2].min() < kp_vis or hip[:, 2].min() < kp_vis or kn[:, 2].min() < kp_vis:
            continue
        usable += 1
        torso = abs(hip[:, 1].mean() - sh[:, 1].mean())
        thigh = abs(kn[:, 1].mean() - hip[:, 1].mean())
        # standing: knees sit well below hips relative to torso length
        if thigh > 0.55 * max(torso, 1):
            stand += 1
        else:
            sit += 1
    f["kp_usable"], f["kp_stand"], f["kp_sit"] = usable, stand, sit

    # --- POSTURE composite: robust fraction-seated estimate (PRIMARY signal) ---
    # The Phase-1 gate showed this is the clean demo/discussion separator on real
    # footage (and it is room-geometry invariant, unlike spatial dispersion).
    # Blend keypoint sit/stand with the keypoint-free box-aspect proxy, trusting
    # keypoints in proportion to how many people have usable ones (low in the
    # occluded huddle, so the box proxy carries those frames). See FINDINGS_phase1.md.
    n_post = stand + sit
    sit_box = 1.0 - f["frac_tall_boxes"]              # taller boxes => standing
    if n_post == 0:
        f["sit_fraction"] = float(sit_box)
    else:
        sit_kp = sit / n_post
        w_kp = min(n_post / n, 1.0)                   # confidence in keypoint evidence
        f["sit_fraction"] = float(w_kp * sit_kp + (1.0 - w_kp) * sit_box)
    return f
