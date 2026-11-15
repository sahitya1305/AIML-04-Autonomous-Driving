"""Monocular depth with MiDaS (via torch.hub) plus numpy helpers that work without torch.

MiDaS predicts RELATIVE INVERSE depth (bigger = closer) with an unknown scale per image. It is not metric.
We normalise it per frame to 0..1 (1 = nearest) and use it as a 'relative proximity' cue. Metric-ish distance in this
project comes from the pinhole model in src/perception/obstacle.py (approximate, needs camera calibration).
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.config import resolve_path


def normalize_depth(raw: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """Robustly rescale a raw MiDaS inverse-depth map to float32 0..1 (1 = near)."""
    raw = np.asarray(raw, dtype=np.float32)
    lo, hi = np.percentile(raw, [lo_pct, hi_pct])
    if hi - lo < 1e-6:
        return np.zeros_like(raw, dtype=np.float32)
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def colorize_depth(norm: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    return cv2.applyColorMap((np.clip(norm, 0, 1) * 255).astype(np.uint8), colormap)


def box_proximity(norm: np.ndarray, xyxy, lower_frac: float = 0.5) -> Optional[float]:
    """Median relative proximity (0..1) in the lower part of a box (closest to the road surface)."""
    h, w = norm.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1, x2 = max(0, x1), min(w, x2)
    y2 = min(h, y2)
    y1 = max(0, int(y2 - (y2 - y1) * lower_frac))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return float(np.median(norm[y1:y2, x1:x2]))


class MidasDepth:
    def __init__(self, model_type: str = "MiDaS_small", device: str = "cpu", cache_dir: str = "models/depth"):
        try:
            import torch
        except ImportError as e:  # pragma: no cover
            raise ImportError("PyTorch is required for MiDaS (pip install -r requirements-ml.txt)") from e
        self.torch = torch
        self.device = torch.device(device)
        cache = resolve_path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache))
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        except Exception as e:
            raise RuntimeError(
                "Could not load MiDaS from torch.hub. This needs internet access the first time (and `pip install timm`). "
                f"After one successful download the files are cached in {cache}. Original error: {e}") from e
        self.model.to(self.device).eval()
        self.transform = transforms.small_transform if model_type == "MiDaS_small" else (
            transforms.dpt_transform if model_type.startswith("DPT") else transforms.default_transform)
        self.model_type = model_type
        self.last_ms = 0.0

    def estimate_raw(self, frame_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            pred = self.model(self.transform(rgb).to(self.device))
            pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic",
                                                   align_corners=False).squeeze()
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return pred.cpu().numpy()

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Normalised relative proximity map, float32 0..1 (1 = near)."""
        return normalize_depth(self.estimate_raw(frame_bgr))
