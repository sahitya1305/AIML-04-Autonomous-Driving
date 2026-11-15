"""Classical OpenCV lane detection: colour+edge -> ROI -> Hough -> ego-lane line fit -> temporal smoothing."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.perception.types import LaneResult


class ClassicalLaneDetector:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        c = dict(blur=5, canny_low=50, canny_high=150, roi_top=0.58, roi_top_half_width=0.12,
                 hough_threshold=25, min_line_len=25, max_line_gap=40, min_abs_slope=0.35,
                 cluster_frac=0.06, smoothing=0.7, use_color=True)
        c.update(cfg or {})
        self.c = c
        self.prev_left = None
        self.prev_right = None

    def reset(self):
        self.prev_left = self.prev_right = None

    # -- helpers -------------------------------------------------------------------------
    def _roi(self, h: int, w: int) -> np.ndarray:
        top, hw = self.c["roi_top"] * h, self.c["roi_top_half_width"] * w
        poly = np.array([[0, h], [w / 2 - hw, top], [w / 2 + hw, top], [w, h]], np.int32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [poly], 255)
        return m

    def _edges(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        k = self.c["blur"] | 1
        edges = cv2.Canny(cv2.GaussianBlur(gray, (k, k), 0), self.c["canny_low"], self.c["canny_high"])
        if self.c["use_color"]:
            hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            white = cv2.inRange(hls, (0, 190, 0), (255, 255, 255))
            yellow = cv2.inRange(hls, (15, 80, 100), (35, 255, 255))
            color = cv2.bitwise_or(white, yellow)
            color_edges = cv2.Canny(color, 50, 150)
            edges = cv2.bitwise_or(edges, color_edges)
        return edges

    def _fit_side(self, segs: List[Tuple[float, float, float, float]], w: int, h: int, y_top: float):
        """Choose the segment cluster closest to image centre, then least-squares fit x = a*y + b."""
        if not segs:
            return None
        arr = np.array(segs, dtype=np.float64)
        slopes_inv = (arr[:, 2] - arr[:, 0]) / (arr[:, 3] - arr[:, 1] + 1e-9)    # dx/dy
        xb = arr[:, 0] + slopes_inv * (h - arr[:, 1])                           # x at bottom row
        seed = xb[np.argmin(np.abs(xb - w / 2))]
        sel = np.abs(xb - seed) < self.c["cluster_frac"] * w
        if not sel.any():
            return None
        pts = arr[sel]
        ys = np.r_[pts[:, 1], pts[:, 3]]
        xs = np.r_[pts[:, 0], pts[:, 2]]
        if np.ptp(ys) < 5:
            return None
        a, b = np.polyfit(ys, xs, 1)
        return np.array([a, b])

    def detect(self, frame: np.ndarray) -> LaneResult:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        edges = cv2.bitwise_and(self._edges(frame), self._roi(h, w))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, self.c["hough_threshold"],
                                minLineLength=self.c["min_line_len"], maxLineGap=self.c["max_line_gap"])
        left, right = [], []
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < self.c["min_abs_slope"]:
                    continue
                mid = (x1 + x2) / 2
                if slope < 0 and mid < w / 2:
                    left.append((x1, y1, x2, y2))
                elif slope > 0 and mid > w / 2:
                    right.append((x1, y1, x2, y2))
        y_top = self.c["roi_top"] * h
        fl, fr = self._fit_side(left, w, h, y_top), self._fit_side(right, w, h, y_top)
        s = self.c["smoothing"]
        fl = self._smooth("prev_left", fl, s)
        fr = self._smooth("prev_right", fr, s)

        def to_line(f):
            if f is None:
                return None
            a, b = f
            return ((float(a * h + b), float(h)), (float(a * y_top + b), float(y_top)))

        res = LaneResult(name="classical", left=to_line(fl), right=to_line(fr))
        mask = np.zeros((h, w), np.uint8)
        for ln in (res.left, res.right):
            if ln is not None:
                cv2.line(mask, tuple(int(round(v)) for v in ln[0]), tuple(int(round(v)) for v in ln[1]), 255, 5)
        res.mask = mask
        res.time_ms = (time.perf_counter() - t0) * 1000.0
        return res

    def _smooth(self, attr: str, new, s: float):
        prev = getattr(self, attr)
        if new is None:
            return prev  # hold last estimate for a missed frame
        out = new if prev is None else s * prev + (1 - s) * new
        setattr(self, attr, out)
        return out
