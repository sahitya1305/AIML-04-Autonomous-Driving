"""'Detector' that returns exact ground-truth boxes. TEST DOUBLE - not a real detector.

Used so tracking, obstacle logic, closed-loop demo and tests run without a neural network.
Sources: a SyntheticScene object (live) or a JSON file written by scripts/make_demo_data.py.
"""
from __future__ import annotations

import json
from typing import List, Optional

import numpy as np

from src.perception.types import Detection
from src.utils.config import resolve_path


class GroundTruthDetector:
    def __init__(self, source, jitter_px: float = 0.0, seed: int = 0):
        self.scene = None
        self.frames = None
        if hasattr(source, "ground_truth_boxes"):
            self.scene = source
        else:
            p = resolve_path(source)
            if not p.exists():
                raise FileNotFoundError(f"Ground-truth file not found: {p}. Run scripts/make_demo_data.py first.")
            with open(p, "r", encoding="utf-8") as f:
                self.frames = json.load(f)["frames"]
        self.jitter = jitter_px
        self.rng = np.random.default_rng(seed)
        self.name = "ground-truth(test double)"

    def detect(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> List[Detection]:
        boxes = self.scene.ground_truth_boxes() if self.scene is not None else self.frames[min(frame_idx or 0, len(self.frames) - 1)]
        dets = []
        for b in boxes:
            xyxy = np.array(b["xyxy"], dtype=np.float64)
            if self.jitter > 0:
                xyxy += self.rng.normal(0, self.jitter, 4)
            dets.append(Detection(tuple(float(v) for v in xyxy), 0.95, int(b["cls_id"]), b["label"]))
        return dets
