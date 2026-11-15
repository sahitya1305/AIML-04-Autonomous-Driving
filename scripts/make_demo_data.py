"""Generate synthetic demo data (no download needed): an open-loop driving video + exact ground truth,
a mini YOLO-format dataset for training smoke tests, and a mini BDD100K-style label file for the analysis demo.

Everything produced here is SYNTHETIC cartoon data. Do not report it as real-world performance.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import random

import cv2
import numpy as np

from src.data.synthetic import SyntheticScene, build_scenario
from src.utils.config import resolve_path

NAMES = ["pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle"]
SYN2UNI = {"person": 0, "car": 2, "truck": 3, "bus": 4}


def make_video(out_dir, frames, scenario, fps=20):
    scene = build_scenario(SyntheticScene(seed=1), scenario)
    path = out_dir / "synthetic_drive.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (scene.w, scene.h))
    if not vw.isOpened():
        raise RuntimeError("OpenCV could not open an mp4 writer here. Try installing opencv-python (not -headless).")
    gt = []
    for i in range(frames):
        img, g = scene.render()
        vw.write(img)
        gt.append(g["boxes"])
        if i in (0, frames // 2, frames - 1):
            cv2.imwrite(str(out_dir / f"synthetic_frame_{i:04d}.png"), img)
        scene.step(ego_speed=12.0, dt=1.0 / fps)   # open loop: constant 12 m/s, the video cannot react
    vw.release()
    with open(out_dir / "synthetic_drive_gt.json", "w", encoding="utf-8") as f:
        json.dump({"synthetic": True, "fps": fps, "frames": gt}, f)
    print(f"video: {path}  ({frames} frames)")


def make_yolo_mini(out_dir, n, seed=0):
    rng = random.Random(seed)
    root = out_dir / "synthetic_yolo"
    for split, count in (("train", int(n * 0.8)), ("val", n - int(n * 0.8))):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            sc = SyntheticScene(seed=rng.randint(0, 10**6), brightness=rng.uniform(0.7, 1.15), noise=rng.uniform(0, 6),
                                curvature=rng.uniform(-0.0004, 0.0004))
            for _ in range(rng.randint(1, 4)):
                kind = rng.choice(["car", "car", "truck", "person"])
                sc.add(kind, rng.choice([-3.5, 0.0, 3.5]) + rng.uniform(-0.5, 0.5), rng.uniform(8, 60))
            img, gt = sc.render()
            stem = f"syn_{split}_{i:04d}"
            cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            lines = []
            for b in gt["boxes"]:
                x1, y1, x2, y2 = b["xyxy"]
                if x2 - x1 < 3 or y2 - y1 < 3:
                    continue
                lines.append(f"{SYN2UNI[b['label']]} {(x1+x2)/2/sc.w:.6f} {(y1+y2)/2/sc.h:.6f} {(x2-x1)/sc.w:.6f} {(y2-y1)/sc.h:.6f}")
            (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    yaml_text = f"path: {root.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    (root / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"mini YOLO dataset: {root}  ({n} images, SYNTHETIC)")


def make_mini_bdd_labels(out_dir, n, seed=0):
    """Fake BDD100K-format labels with a plausible car-heavy skew, ONLY to demo scripts/analyze_dataset.py."""
    rng = np.random.default_rng(seed)
    cats = ["car", "traffic sign", "traffic light", "pedestrian", "truck", "bus", "bicycle", "rider", "motorcycle", "train"]
    p = np.array([0.55, 0.17, 0.13, 0.07, 0.025, 0.01, 0.007, 0.006, 0.004, 0.0005]); p /= p.sum()
    items = []
    for i in range(n):
        labels = []
        for _ in range(int(rng.integers(3, 25))):
            c = rng.choice(cats, p=p)
            w, h = rng.uniform(10, 300), rng.uniform(10, 250)
            x1, y1 = rng.uniform(0, 1280 - w), rng.uniform(0, 720 - h)
            labels.append({"category": str(c), "attributes": {}, "box2d": {"x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h}})
        items.append({"name": f"synthetic_{i:05d}.jpg",
                      "attributes": {"weather": str(rng.choice(["clear", "overcast", "rainy", "snowy", "foggy"], p=[.55, .2, .15, .05, .05])),
                                     "scene": str(rng.choice(["city street", "highway", "residential"], p=[.6, .25, .15])),
                                     "timeofday": str(rng.choice(["daytime", "night", "dawn/dusk"], p=[.55, .35, .1]))},
                      "labels": labels})
    p_out = out_dir / "mini_bdd_labels_SYNTHETIC.json"
    p_out.write_text(json.dumps(items), encoding="utf-8")
    print(f"mini label file: {p_out}  (SYNTHETIC - demo of the analysis script only)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="datasets/samples")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--scenario", default="default", choices=["default", "stopped", "pedestrian", "empty"])
    ap.add_argument("--yolo-mini", type=int, default=100, help="images for the tiny synthetic YOLO dataset (0 = skip)")
    ap.add_argument("--bdd-mini", type=int, default=400, help="images in the fake BDD label file (0 = skip)")
    a = ap.parse_args()
    out = resolve_path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    make_video(out, a.frames, a.scenario)
    if a.yolo_mini:
        make_yolo_mini(out, a.yolo_mini)
    if a.bdd_mini:
        make_mini_bdd_labels(out, a.bdd_mini)
