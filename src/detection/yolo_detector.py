"""Ultralytics YOLO detector (PyTorch). Works with COCO-pretrained weights or your fine-tuned weights."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from src.perception.types import Detection
from src.utils.config import resolve_path


class YoloDetector:
    def __init__(self, weights: str = "models/detection/yolo11n.pt", imgsz: int = 640, conf: float = 0.15,
                 iou: float = 0.5, device: str = "cpu", max_det: int = 100,
                 classes: Optional[Iterable[str]] = None):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise ImportError("ultralytics is not installed. Run: pip install -r requirements-ml.txt") from e
        wp = resolve_path(weights)
        if wp.exists():
            model_arg = str(wp)
        elif Path(str(weights)).suffix == ".pt" and not Path(str(weights)).parent.name:
            model_arg = str(weights)  # bare name like yolo11n.pt -> ultralytics will download it
        else:
            raise FileNotFoundError(
                f"Detector weights not found: {wp}\nRun `python scripts/download_weights.py` first "
                f"(needs internet once), or point detector.weights to your own .pt file.")
        self.model = YOLO(model_arg)
        self.names = dict(self.model.names)
        self.imgsz, self.conf, self.iou, self.device, self.max_det = imgsz, conf, iou, device, max_det
        self.allowed = {c.lower() for c in classes} if classes else None
        self.name = f"yolo({Path(model_arg).name})"

    def detect(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> List[Detection]:
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device,
                                 max_det=self.max_det, verbose=False)[0]
        out: List[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for b, c, k in zip(xyxy, conf, cls):
            label = str(self.names.get(int(k), k))
            if self.allowed is not None and label.lower() not in self.allowed:
                continue
            out.append(Detection(tuple(float(v) for v in b), float(c), int(k), label))
        return out
