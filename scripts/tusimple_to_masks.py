"""Convert TuSimple lane annotations (label_data_*.json) into binary lane masks for train_lane_seg.py.

  python scripts/tusimple_to_masks.py --root D:/tusimple/train_set --json label_data_0313.json --out datasets/processed/tusimple
Creates <out>/images/*.jpg (copies) and <out>/masks/*.png. TuSimple format: {raw_file, lanes:[[x per h_sample (-2 = none)]], h_samples:[y]}.
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import resolve_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True); ap.add_argument("--json", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--thickness", type=int, default=7); ap.add_argument("--max-images", type=int)
    a = ap.parse_args()
    root, out = Path(a.root), resolve_path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True); (out / "masks").mkdir(parents=True, exist_ok=True)
    n = 0
    for line in open(root / a.json, "r", encoding="utf-8"):
        item = json.loads(line)
        img = cv2.imread(str(root / item["raw_file"]))
        if img is None:
            continue
        mask = np.zeros(img.shape[:2], np.uint8)
        for lane in item["lanes"]:
            pts = [(x, y) for x, y in zip(lane, item["h_samples"]) if x >= 0]
            if len(pts) > 1:
                cv2.polylines(mask, [np.array(pts, np.int32)], False, 255, a.thickness)
        stem = f"{n:06d}"
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), img); cv2.imwrite(str(out / "masks" / f"{stem}.png"), mask)
        n += 1
        if a.max_images and n >= a.max_images:
            break
    print(f"wrote {n} image/mask pairs to {out}")


if __name__ == "__main__":
    main()
