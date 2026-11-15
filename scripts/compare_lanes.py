"""Compare classical OpenCV lane detection with the segmentation model.

  python scripts/compare_lanes.py --synthetic 40                       # exact synthetic ground truth (sanity check)
  python scripts/compare_lanes.py --source my_road_video.mp4 --max-frames 100   # no GT: detection rate, agreement, speed
The segmentation column needs trained weights (models/lane_seg/lane_unet.pt); otherwise only classical is reported.
Writes outputs/metrics/lane_comparison.json, .md and a side-by-side image.
"""
import _bootstrap  # noqa: F401
import argparse
import json

import cv2
import numpy as np

from src.evaluation.lane_metrics import bottom_x_error, tolerant_iou
from src.lane_detection.classical import ClassicalLaneDetector
from src.perception.render import draw_lanes
from src.utils.config import load_config, resolve_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", type=int, default=0); ap.add_argument("--source"); ap.add_argument("--max-frames", type=int, default=100)
    a = ap.parse_args()
    cfg = load_config()
    classical = ClassicalLaneDetector(cfg["lanes"]["classical"])
    seg = None
    try:
        from src.lane_detection.segmentation import SegLaneDetector
        s = cfg["lanes"]["segmentation"]
        seg = SegLaneDetector(s["weights"], s["device"], s["threshold"])
    except Exception as e:
        print(f"[note] segmentation model not available -> classical only ({e})")
    frames = []   # (img, gt_mask or None)
    if a.synthetic:
        import random
        from src.data.synthetic import SyntheticScene, build_scenario
        rng = random.Random(123)
        for _ in range(a.synthetic):
            sc = SyntheticScene(seed=rng.randint(0, 10**6), brightness=rng.uniform(0.7, 1.15), noise=rng.uniform(0, 6), curvature=rng.uniform(-0.0004, 0.0004))
            sc.travelled = rng.uniform(0, 7); build_scenario(sc, "empty")
            for _k in range(rng.randint(0, 2)):
                sc.add(rng.choice(["car", "truck"]), rng.choice([3.5, -3.5]), rng.uniform(15, 60))
            img, gt = sc.render(); frames.append((img, gt["lane_mask"]))
    elif a.source:
        from src.app.session import iter_frames
        frames = [(f, None) for _, f in iter_frames(a.source, 1, a.max_frames)]
    else:
        raise SystemExit("Use --synthetic N or --source FILE")
    stats = {n: {"found": 0, "iou": [], "x_err": [], "ms": []} for n in (["classical"] + (["segmentation"] if seg else []))}
    agree, sample = [], None
    for img, gt in frames:
        outs = {"classical": classical.detect(img)}
        if seg: outs["segmentation"] = seg.detect(img)
        for n, r in outs.items():
            st = stats[n]; st["found"] += int(r.found); st["ms"].append(r.time_ms)
            if gt is not None:
                st["iou"].append(tolerant_iou(r.mask, gt))
                for ln, side in ((r.left, "left"), (r.right, "right")):
                    e = bottom_x_error(ln, gt, side)
                    if e is not None: st["x_err"].append(e)
        if seg: agree.append(tolerant_iou(outs["classical"].mask, outs["segmentation"].mask))
        if sample is None and len(frames) > 1:
            sample = np.hstack([draw_lanes(img, outs["classical"])] + ([draw_lanes(img, outs["segmentation"], (255, 160, 0))] if seg else []))
    n = len(frames)
    res = {"frames": n, "ground_truth": "synthetic (exact)" if a.synthetic else "none",
           "methods": {k: {"ego_lane_found_pct": 100 * v["found"] / n, "mean_ms": float(np.mean(v["ms"])),
                           "mean_tolerant_iou": float(np.mean(v["iou"])) if v["iou"] else None,
                           "mean_bottom_x_error_px": float(np.mean(v["x_err"])) if v["x_err"] else None} for k, v in stats.items()},
           "classical_vs_segmentation_agreement_iou": float(np.mean(agree)) if agree else None,
           "caveat": "Synthetic cartoon roads: use only to check the code. Real-road accuracy must be measured on TuSimple/CULane or your own labelled frames."}
    m = resolve_path("outputs/metrics"); m.mkdir(parents=True, exist_ok=True)
    (m / "lane_comparison.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    lines = ["| method | ego lane found % | mean ms | tolerant IoU vs GT | bottom-x error px |", "|---|---:|---:|---:|---:|"]
    for k, v in res["methods"].items():
        f = lambda x: "n/a" if x is None else f"{x:.2f}"
        lines.append(f"| {k} | {v['ego_lane_found_pct']:.0f} | {v['mean_ms']:.1f} | {f(v['mean_tolerant_iou'])} | {f(v['mean_bottom_x_error_px'])} |")
    (m / "lane_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if sample is not None:
        cv2.imwrite(str(resolve_path("outputs/visualizations") / "lane_comparison.png"), sample)
    print(json.dumps(res, indent=2)); print("\n".join(lines))


if __name__ == "__main__":
    main()
