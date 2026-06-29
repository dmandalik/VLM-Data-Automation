"""
src/inspect.py -- sanity-check the feature signal on still frames.

This is the FIRST validation step (handoff §7a). Run it on the two reference
frames before building anything temporal:

    python -m src.inspect tests/fixtures/demo.png tests/fixtures/discussion.png

Expect the demo frame to show LOWER dispersion (hull_area_frac,
mean_pairwise_norm, nn_median_norm) and HIGHER crowding (frac_tightly_packed,
density_peak_per_person) than the discussion frame. If that ordering does not
hold, stop and inspect detections before proceeding.
"""
from __future__ import annotations

import argparse
import json

import cv2

from .detector import build_detector
from .features import extract_features, filter_people

SIDE_BY_SIDE_KEYS = [
    "person_count", "sit_fraction", "frac_tall_boxes", "kp_usable", "kp_stand",
    "kp_sit", "hull_area_frac", "mean_pairwise_norm", "nn_median_norm",
    "frac_tightly_packed", "density_peak_per_person",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="image paths (pass 2 to compare)")
    ap.add_argument("--backend", default="yolo", help="yolo | rtmpose")
    ap.add_argument("--weights", default="yolo11m-pose.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-drop-manikin", action="store_true")
    ap.add_argument("--out", default="features_out.json")
    args = ap.parse_args()

    det = build_detector({"backend": args.backend, "weights": args.weights,
                          "conf": args.conf, "device": None})
    res = {}
    for path in args.images:
        im = cv2.imread(path)
        if im is None:
            raise FileNotFoundError(path)
        H, W = im.shape[:2]
        people = det.detect(im)
        if not args.no_drop_manikin:
            people = filter_people(people, H, W)
        feats = extract_features(people, H, W)
        res[path] = feats
        print(f"\n=== {path}  ({W}x{H}) -- {feats['person_count']} people ===")
        for k, v in feats.items():
            print(f"  {k:26s}: {v}")

    if len(res) == 2:
        print("\n=== SIDE-BY-SIDE  (expect DEMO: lower dispersion, higher packing/density) ===")
        (n1, f1), (n2, f2) = res.items()
        g = lambda x: (f"{x:.4f}" if isinstance(x, float) else str(x))
        print(f"{'feature':26s}{'img1':>14s}{'img2':>14s}")
        for k in SIDE_BY_SIDE_KEYS:
            print(f"{k:26s}{g(f1.get(k)):>14s}{g(f2.get(k)):>14s}")

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
