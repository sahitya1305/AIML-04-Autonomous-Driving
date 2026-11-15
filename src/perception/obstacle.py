"""Obstacle-response logic: distance estimate, ego-corridor test, time-to-collision, brake decision, vehicle model.

Distance is APPROXIMATE: pinhole model with assumed real-world object heights and an assumed camera field of view.
Calibrate `camera.hfov_deg` for your camera for meaningful numbers.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.perception.types import Decision, LaneResult, Track

DEFAULT_HEIGHTS = {"person": 1.7, "pedestrian": 1.7, "rider": 1.7, "bicycle": 1.7, "motorcycle": 1.5, "car": 1.5,
                   "bus": 3.2, "truck": 3.2}
VULNERABLE = {"person", "pedestrian", "rider", "bicycle", "motorcycle"}
SEVERITY = {"CLEAR": 0, "CAUTION": 1, "BRAKE": 2}


def focal_from_fov(width: int, hfov_deg: float) -> float:
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def pinhole_distance(box_h_px: float, label: str, focal_px: float, heights: Optional[Dict[str, float]] = None) -> Optional[float]:
    H = (heights or DEFAULT_HEIGHTS).get(label)
    if H is None or box_h_px < 2:
        return None
    return focal_px * H / box_h_px


def ground_distance(y_bottom: float, horizon_y: float, focal_px: float, cam_height: float) -> Optional[float]:
    dy = y_bottom - horizon_y
    return None if dy < 2 else focal_px * cam_height / dy


class ObstacleController:
    def __init__(self, cfg: Dict[str, Any], frame_size: Tuple[int, int], fps: float = 20.0):
        """frame_size = (width, height)."""
        self.c = cfg
        self.w, self.h = frame_size
        self.fps = fps
        cam = cfg.get("camera", {})
        self.focal = float(cam.get("focal_px") or focal_from_fov(self.w, cam.get("hfov_deg", 90.0)))
        self.cam_height = cam.get("height_m", 1.4)
        self.horizon_y = cam.get("horizon_y_ratio", 0.45) * self.h
        self.heights = {**DEFAULT_HEIGHTS, **cfg.get("object_heights", {})}
        self.hist: Dict[int, Deque[Tuple[float, float]]] = {}
        self.prev_state = "CLEAR"
        self.release_count = 0
        self.t = 0.0

    # -------------------------------------------------------------- geometry
    def default_corridor(self) -> np.ndarray:
        c = self.c["corridor"]
        w, h = self.w, self.h
        return np.array([[c["bottom_left"] * w, h], [c["top_left"] * w, c["top_y"] * h],
                         [c["top_right"] * w, c["top_y"] * h], [c["bottom_right"] * w, h]], np.float32)

    def corridor(self, lanes: Optional[LaneResult]) -> np.ndarray:
        """Ego corridor polygon: fitted lane lines extended up to the horizon, or the configured default trapezoid."""
        poly = lanes.ego_polygon if (lanes is not None and lanes.found) else None
        if poly is None:
            return self.default_corridor()
        bottom_w = poly[3][0] - poly[0][0]
        if bottom_w < 0.15 * self.w or bottom_w > 1.2 * self.w:   # implausible fit -> fall back
            return self.default_corridor()
        y_top = self.horizon_y + 2.0
        out = poly.copy()
        for i_top, i_bot in ((1, 0), (2, 3)):                      # extend each side line to the horizon
            (xb, yb), (xt, yt) = poly[i_bot], poly[i_top]
            if abs(yb - yt) > 1e-3:
                k = (yb - y_top) / (yb - yt)
                out[i_top] = (xb + (xt - xb) * k, y_top)
        min_half = 0.02 * self.w                                   # keep a minimum width at the far end
        cx = (out[1][0] + out[2][0]) / 2
        out[1][0], out[2][0] = min(out[1][0], cx - min_half), max(out[2][0], cx + min_half)
        return out

    @staticmethod
    def _inflate(poly: np.ndarray, frac: float) -> np.ndarray:
        cx = poly[:, 0].mean()
        out = poly.copy()
        out[:, 0] = cx + (poly[:, 0] - cx) * (1.0 + frac)
        return out

    def in_path(self, track: Track, corridor: np.ndarray) -> bool:
        poly = self._inflate(corridor, self.c.get("vulnerable_margin", 0.35)) if track.label in VULNERABLE else corridor
        x, y = track.bottom_center
        return cv2.pointPolygonTest(poly.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0

    def distance(self, tr: Track) -> Optional[float]:
        method = self.c.get("distance_method", "height")
        d_h = pinhole_distance(tr.height, tr.label, self.focal, self.heights)
        if method == "ground":
            return ground_distance(tr.xyxy[3], self.horizon_y, self.focal, self.cam_height) or d_h
        return d_h

    # -------------------------------------------------------------- decision
    def update(self, tracks: List[Track], lanes: Optional[LaneResult], dt: Optional[float] = None) -> Tuple[Decision, np.ndarray]:
        dt = dt or 1.0 / self.fps
        self.t += dt
        corridor = self.corridor(lanes)
        cruise = self.c["cruise_speed"]
        brake_d, caution_d = self.c["brake_distance"], self.c["caution_distance"]
        ttc_b, ttc_c = self.c["ttc_brake"], self.c["ttc_caution"]
        seen = set()
        cands: List[Track] = []
        for tr in tracks:
            tr.distance_m = self.distance(tr)
            tr.in_path = self.in_path(tr, corridor)
            if tr.distance_m is None:
                continue
            h = self.hist.setdefault(tr.track_id, deque(maxlen=int(self.fps * 1.5)))
            h.append((self.t, tr.distance_m))
            seen.add(tr.track_id)
            if len(h) >= 4:   # closing speed from ~distance history (m/s, positive = approaching)
                t0, d0 = h[0]
                tr.closing_speed = (d0 - tr.distance_m) / max(self.t - t0, 1e-3)
                if tr.closing_speed > 0.3:
                    tr.ttc = tr.distance_m / tr.closing_speed
            if tr.in_path:
                cands.append(tr)
        for tid in list(self.hist):
            if tid not in seen:
                del self.hist[tid]
        if not cands:
            new = Decision("CLEAR", cruise, 0.0, reason="no obstacle in path")
        else:
            near = min(cands, key=lambda t: t.distance_m)
            ttc = min((t.ttc for t in cands if t.ttc is not None), default=None)
            d = near.distance_m
            if d <= brake_d or (ttc is not None and ttc <= ttc_b):
                state, tgt, brake = "BRAKE", 0.0, 1.0
            elif d <= caution_d or (ttc is not None and ttc <= ttc_c):
                state, tgt, brake = "CAUTION", cruise * self.c.get("caution_speed_factor", 0.5), 0.3
            else:
                state, tgt, brake = "CLEAR", cruise, 0.0
            new = Decision(state, tgt, brake, near.track_id, near.label, d, ttc,
                           reason=f"{near.label} #{near.track_id} in path at ~{d:.1f} m" + (f", TTC {ttc:.1f}s" if ttc else ""))
        return self._hysteresis(new, cruise), corridor

    def _hysteresis(self, new: Decision, cruise: float) -> Decision:
        """Don't release a BRAKE/CAUTION instantly: require a few consecutive calmer frames (avoids flicker)."""
        need = int(self.c.get("release_frames", 8))
        if SEVERITY[new.state] < SEVERITY[self.prev_state]:
            self.release_count += 1
            if self.release_count < need:
                held = Decision(self.prev_state, 0.0 if self.prev_state == "BRAKE" else cruise * 0.5,
                                1.0 if self.prev_state == "BRAKE" else 0.3, new.nearest_id, new.nearest_label,
                                new.nearest_distance, new.ttc, reason="holding (release delay)")
                return held
        self.release_count = 0
        self.prev_state = new.state
        return new


class VehicleSim:
    """Point-mass longitudinal model: shows how a decision would change ego speed (simulated response)."""

    def __init__(self, cruise: float = 12.0, speed: Optional[float] = None, max_decel: float = 6.0,
                 comfort_decel: float = 2.5, accel: float = 1.5):
        self.speed = cruise if speed is None else speed
        self.cruise, self.max_decel, self.comfort, self.accel = cruise, max_decel, comfort_decel, accel

    def step(self, decision: Decision, dt: float) -> float:
        if decision.state == "BRAKE":
            self.speed = max(0.0, self.speed - self.max_decel * dt)
        elif self.speed > decision.target_speed:
            self.speed = max(decision.target_speed, self.speed - self.comfort * dt)
        else:
            self.speed = min(decision.target_speed, self.speed + self.accel * dt)
        return self.speed
