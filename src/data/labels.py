"""Parsers for BDD100K (detection JSON) and KITTI (object txt) labels, and the unified class mapping."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

UNIFIED_CLASSES = ["pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle"]
BDD_TO_UNIFIED = {"pedestrian": 0, "rider": 1, "car": 2, "truck": 3, "bus": 4, "motorcycle": 5, "bicycle": 6}
# KITTI: Van->car, Person_sitting->pedestrian, Cyclist->rider. Tram/Misc/DontCare are dropped. KITTI has no bus/motorcycle.
KITTI_TO_UNIFIED = {"pedestrian": 0, "person_sitting": 0, "cyclist": 1, "car": 2, "van": 2, "truck": 3}


@dataclass
class ImageRecord:
    name: str
    width: Optional[int] = None
    height: Optional[int] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    objects: List[Tuple[str, Tuple[float, float, float, float]]] = field(default_factory=list)  # (raw category, xyxy)


def parse_bdd100k(path) -> List[ImageRecord]:
    """BDD100K det labels: a JSON list of {name, attributes, labels:[{category, box2d{x1,y1,x2,y2}}]}."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"BDD100K label file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list (BDD100K detection labels, e.g. det_train.json)")
    out = []
    for item in data:
        rec = ImageRecord(item["name"], 1280, 720, item.get("attributes", {}) or {})
        for lab in item.get("labels", []) or []:
            b = lab.get("box2d")
            if b:
                rec.objects.append((lab["category"], (b["x1"], b["y1"], b["x2"], b["y2"])))
        out.append(rec)
    return out


def parse_kitti(label_dir, image_dir=None) -> List[ImageRecord]:
    """KITTI object labels: one txt per image, rows: type trunc occl alpha x1 y1 x2 y2 h w l x y z ry."""
    d = Path(label_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"KITTI label folder not found: {d}")
    out = []
    for f in sorted(d.glob("*.txt")):
        rec = ImageRecord(f.stem + ".png", 1242, 375)
        if image_dir is not None:
            for ext in (".png", ".jpg"):
                ip = Path(image_dir) / (f.stem + ext)
                if ip.exists():
                    rec.name = ip.name
                    try:
                        import cv2
                        im = cv2.imread(str(ip))
                        if im is not None:
                            rec.height, rec.width = im.shape[:2]
                    except ImportError:
                        pass
                    break
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 8:
                continue
            rec.objects.append((parts[0], tuple(float(v) for v in parts[4:8])))
        out.append(rec)
    return out
