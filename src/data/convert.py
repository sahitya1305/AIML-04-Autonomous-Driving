"""Convert BDD100K / KITTI annotations to YOLO txt format with our 7 unified classes."""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from src.data.labels import (BDD_TO_UNIFIED, KITTI_TO_UNIFIED, UNIFIED_CLASSES, ImageRecord)


def to_yolo_lines(rec: ImageRecord, mapping: Dict[str, int]) -> List[str]:
    lines = []
    W, H = rec.width, rec.height
    for cat, (x1, y1, x2, y2) in rec.objects:
        cid = mapping.get(cat.lower())
        if cid is None:
            continue
        x1, x2 = sorted((min(max(x1, 0), W), min(max(x2, 0), W)))
        y1, y2 = sorted((min(max(y1, 0), H), min(max(y2, 0), H)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"{cid} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines


def write_data_yaml(root: Path) -> Path:
    text = f"path: {root.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
    text += "".join(f"  {i}: {n}\n" for i, n in enumerate(UNIFIED_CLASSES))
    p = root / "data.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def convert_records(records: List[ImageRecord], image_dir: Optional[Path], out_root: Path, dataset: str,
                    max_images: Optional[int] = None, val_fraction: float = 0.2, seed: int = 0,
                    require_image: bool = True) -> Dict[str, int]:
    """Writes out_root/{images,labels}/{train,val}. Images are copied when found in image_dir."""
    mapping = BDD_TO_UNIFIED if dataset == "bdd100k" else KITTI_TO_UNIFIED
    usable = [r for r in records if to_yolo_lines(r, mapping)]
    rng = random.Random(seed)
    rng.shuffle(usable)
    if max_images:
        usable = usable[:max_images]
    n_val = max(1, int(len(usable) * val_fraction)) if usable else 0
    stats = {"train": 0, "val": 0, "skipped_missing_image": 0}
    for i, rec in enumerate(usable):
        split = "val" if i < n_val else "train"
        src = None
        if image_dir is not None:
            cand = Path(image_dir) / rec.name
            src = cand if cand.exists() else None
        if src is None and require_image:
            stats["skipped_missing_image"] += 1
            continue
        for sub in ("images", "labels"):
            (out_root / sub / split).mkdir(parents=True, exist_ok=True)
        if src is not None:
            shutil.copy2(src, out_root / "images" / split / rec.name)
        (out_root / "labels" / split / (Path(rec.name).stem + ".txt")).write_text("\n".join(to_yolo_lines(rec, mapping)), encoding="utf-8")
        stats[split] += 1
    write_data_yaml(out_root)
    return stats
