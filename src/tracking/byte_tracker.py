"""Compact ByteTrack-style multi-object tracker (numpy + optional scipy).

This is our own implementation of the ByteTrack idea (Zhang et al., 2022):
  1. Kalman-predict all tracks.
  2. Associate HIGH-confidence detections to tracks by IoU (Hungarian).
  3. Associate remaining tracks to LOW-confidence detections (recovers occluded / blurred objects).
  4. Unmatched tracks become 'lost' (kept for `track_buffer` frames); unmatched high-score detections start new tracks.
It is NOT the reference code from the ByteTrack repository. Ultralytics' built-in ByteTrack is an alternative.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.perception.types import Detection, Track


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU for boxes in xyxy format. Shapes (N,4),(M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a[:, None, :]
    b = b[None, :, :]
    ix1, iy1 = np.maximum(a[..., 0], b[..., 0]), np.maximum(a[..., 1], b[..., 1])
    ix2, iy2 = np.minimum(a[..., 2], b[..., 2]), np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def linear_assignment(cost: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Min-cost matching with a max-cost threshold. Returns (matches, unmatched_rows, unmatched_cols)."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost)
        pairs = list(zip(rows.tolist(), cols.tolist()))
    except ImportError:  # greedy fallback
        pairs, used_r, used_c = [], set(), set()
        for flat in np.argsort(cost, axis=None):
            r, c = divmod(int(flat), m)
            if r not in used_r and c not in used_c:
                pairs.append((r, c)); used_r.add(r); used_c.add(c)
    matches = [(r, c) for r, c in pairs if cost[r, c] <= thresh]
    mr, mc = {r for r, _ in matches}, {c for _, c in matches}
    return matches, [i for i in range(n) if i not in mr], [j for j in range(m) if j not in mc]


class KalmanXYAH:
    """Constant-velocity Kalman filter on (cx, cy, aspect, height) like the ByteTrack/DeepSORT filters."""

    def __init__(self):
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.H = np.eye(4, 8)
        self.wp, self.wv = 1.0 / 20, 1.0 / 160

    def initiate(self, z: np.ndarray):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self.wp * h, 2 * self.wp * h, 1e-2, 2 * self.wp * h, 10 * self.wv * h, 10 * self.wv * h, 1e-5, 10 * self.wv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        q = np.diag(np.square([self.wp * h, self.wp * h, 1e-2, self.wp * h, self.wv * h, self.wv * h, 1e-5, self.wv * h]))
        return self.F @ mean, self.F @ cov @ self.F.T + q

    def update(self, mean, cov, z):
        h = mean[3]
        r = np.diag(np.square([self.wp * h, self.wp * h, 1e-1, self.wp * h]))
        s = self.H @ cov @ self.H.T + r
        k = cov @ self.H.T @ np.linalg.inv(s)
        return mean + k @ (z - self.H @ mean), cov - k @ s @ k.T


def xyxy_to_xyah(b: Sequence[float]) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(z: np.ndarray) -> np.ndarray:
    h = z[3]
    w = z[2] * h
    return np.array([z[0] - w / 2, z[1] - h / 2, z[0] + w / 2, z[1] + h / 2])


class _STrack:
    _next_id = 1
    kf = KalmanXYAH()

    def __init__(self, det: Detection):
        self.mean, self.cov = self.kf.initiate(xyxy_to_xyah(det.xyxy))
        self.det = det
        self.track_id = 0
        self.hits = 1
        self.confirmed = False
        self.lost_for = 0
        self.updated_now = True

    @property
    def xyxy(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])

    def predict(self):
        mean = self.mean.copy()
        if self.lost_for > 0:
            mean[7] = 0.0  # no height velocity for lost tracks
        self.mean, self.cov = self.kf.predict(mean, self.cov)
        self.updated_now = False

    def update(self, det: Detection):
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(det.xyxy))
        self.det = det
        self.hits += 1
        self.lost_for = 0
        self.updated_now = True

    def activate(self):
        self.track_id = _STrack._next_id
        _STrack._next_id += 1
        self.confirmed = True


class ByteTracker:
    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.1, new_track_thresh: float = 0.6,
                 match_thresh: float = 0.8, low_match_thresh: float = 0.5, track_buffer: int = 30, min_hits: int = 2):
        self.high, self.low, self.new_thr = high_thresh, low_thresh, new_track_thresh
        self.match_thresh, self.low_match_thresh = match_thresh, low_match_thresh
        self.buffer, self.min_hits = track_buffer, min_hits
        self.tracks: List[_STrack] = []
        self.frame = 0

    def reset(self):
        self.tracks, self.frame = [], 0
        _STrack._next_id = 1

    def update(self, dets: Sequence[Detection]) -> List[Track]:
        self.frame += 1
        high = [d for d in dets if d.conf >= self.high]
        low = [d for d in dets if self.low <= d.conf < self.high]
        for t in self.tracks:
            t.predict()
        confirmed_active = [t for t in self.tracks if t.lost_for == 0 and t.confirmed]
        tentative = [t for t in self.tracks if not t.confirmed]
        lost = [t for t in self.tracks if t.lost_for > 0]

        # 1) high-score detections vs (active + lost + tentative) tracks
        pool = confirmed_active + lost + tentative
        cost = 1.0 - iou_matrix(np.array([t.xyxy for t in pool]), np.array([d.xyxy for d in high]))
        matches, um_t, um_d = linear_assignment(cost, self.match_thresh)
        for ti, di in matches:
            t = pool[ti]
            t.update(high[di])
            if not t.confirmed and (t.hits >= self.min_hits or self.frame == 1):
                t.activate()
        # 2) low-score detections vs still-unmatched ACTIVE tracks (ByteTrack's key idea)
        rem = [pool[i] for i in um_t if pool[i].confirmed and pool[i].lost_for == 0]
        cost2 = 1.0 - iou_matrix(np.array([t.xyxy for t in rem]), np.array([d.xyxy for d in low]))
        m2, um_t2, _ = linear_assignment(cost2, self.low_match_thresh)
        for ti, di in m2:
            rem[ti].update(low[di])
        still_unmatched = {id(rem[i]) for i in um_t2}
        # 3) bookkeeping
        for i in um_t:
            t = pool[i]
            if not t.confirmed:
                t.lost_for = self.buffer + 1          # tentative track that never matched: drop
            elif id(t) in still_unmatched or t.lost_for > 0:
                t.lost_for += 1
        # 4) new tracks
        for i in um_d:
            d = high[i]
            if d.conf >= self.new_thr:
                t = _STrack(d)
                if self.frame == 1 or self.min_hits <= 1:
                    t.activate()
                self.tracks.append(t)
        self.tracks = [t for t in self.tracks if t.lost_for <= self.buffer]

        out: List[Track] = []
        for t in self.tracks:
            if t.confirmed and t.updated_now:
                d = t.det
                out.append(Track(t.track_id, tuple(float(v) for v in d.xyxy), d.conf, d.cls_id, d.label))
        return out
