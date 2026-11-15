"""Segmentation-model lane detection: a small U-Net that predicts a lane-marking mask.

IMPORTANT: no pretrained lane weights are shipped. You must train it (scripts/train_lane_seg.py) on TuSimple/CULane
masks, or on the synthetic generator for a pipeline sanity check only. Until then `SegLaneDetector` refuses to load.
Requires PyTorch (requirements-ml.txt).
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.perception.types import LaneResult
from src.utils.config import resolve_path

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:  # pragma: no cover
    TORCH_OK = False

INPUT_W, INPUT_H = 320, 160   # both divisible by 8 (three 2x downsamplings)


def lanes_from_mask(mask: np.ndarray, roi_top: float = 0.58, name: str = "segmentation") -> LaneResult:
    """Fit left/right ego-lane lines from a binary mask so both approaches share the same output format."""
    h, w = mask.shape[:2]
    res = LaneResult(name=name, mask=mask)
    ys = np.arange(int(h * 0.65), h, 4)
    lx, ly, rx, ry = [], [], [], []
    def inner_run_center(xs, take_last):
        """Midpoint of the contiguous run of mask pixels closest to the image centre."""
        breaks = np.flatnonzero(np.diff(xs) > 3)
        run = np.split(xs, breaks + 1)[-1 if take_last else 0]
        return float(run.mean())

    for y in ys:
        xs = np.flatnonzero(mask[y] > 0)
        if xs.size == 0:
            continue
        l, r = xs[xs < w / 2], xs[xs >= w / 2]
        if l.size:
            lx.append(inner_run_center(l, True)); ly.append(y)
        if r.size:
            rx.append(inner_run_center(r, False)); ry.append(y)
    y_top = roi_top * h
    def fit(xs_, ys_):
        if len(xs_) < 5 or np.ptp(ys_) < 10:
            return None
        a, b = np.polyfit(ys_, xs_, 1)
        return ((float(a * h + b), float(h)), (float(a * y_top + b), float(y_top)))
    res.left, res.right = fit(lx, ly), fit(rx, ry)
    return res


if TORCH_OK:
    def _block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class TinyUNet(nn.Module):
        """~0.5M parameters; runs on CPU."""

        def __init__(self, base: int = 16):
            super().__init__()
            self.e1, self.e2, self.e3 = _block(3, base), _block(base, base * 2), _block(base * 2, base * 4)
            self.mid = _block(base * 4, base * 8)
            self.pool = nn.MaxPool2d(2)
            self.u3, self.d3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2), _block(base * 8, base * 4)
            self.u2, self.d2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2), _block(base * 4, base * 2)
            self.u1, self.d1 = nn.ConvTranspose2d(base * 2, base, 2, 2), _block(base * 2, base)
            self.out = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
            m = self.mid(self.pool(e3))
            d3 = self.d3(torch.cat([self.u3(m), e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.out(d1)   # logits


def preprocess_bgr(frame: np.ndarray) -> np.ndarray:
    img = cv2.resize(frame, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)


class SegLaneDetector:
    def __init__(self, weights: str = "models/lane_seg/lane_unet.pt", device: str = "cpu", threshold: float = 0.5,
                 roi_top: float = 0.58):
        if not TORCH_OK:
            raise ImportError("PyTorch is required for segmentation lane detection (pip install -r requirements-ml.txt)")
        wp = resolve_path(weights)
        if not wp.exists():
            raise FileNotFoundError(
                f"Lane segmentation weights not found: {wp}\nTrain them first: python scripts/train_lane_seg.py --synthetic 400 "
                f"(sanity check only) or train on TuSimple/CULane masks (see docs/DATASETS.md).")
        self.device, self.threshold, self.roi_top = device, threshold, roi_top
        self.model = TinyUNet().to(device)
        ckpt = torch.load(str(wp), map_location=device)
        self.model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        self.model.eval()
        self.meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}

    @torch.no_grad() if TORCH_OK else (lambda f: f)
    def detect(self, frame: np.ndarray) -> LaneResult:
        t0 = time.perf_counter()
        x = torch.from_numpy(preprocess_bgr(frame))[None].to(self.device)
        prob = torch.sigmoid(self.model(x))[0, 0].cpu().numpy()
        h, w = frame.shape[:2]
        mask = (cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR) > self.threshold).astype(np.uint8) * 255
        mask[: int(self.roi_top * h)] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        res = lanes_from_mask(mask, self.roi_top, "segmentation")
        res.time_ms = (time.perf_counter() - t0) * 1000.0
        return res
