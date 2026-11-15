"""Drawing helpers: every panel the demo/GUI shows is produced here."""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from src.depth_estimation.midas import colorize_depth
from src.perception.types import Decision, FrameResult, LaneResult, Track

STATE_COLOR = {"CLEAR": (60, 180, 60), "CAUTION": (0, 190, 255), "BRAKE": (40, 40, 230)}


def track_color(tid: int):
    rng = np.random.default_rng(tid * 9973 + 7)
    return tuple(int(v) for v in rng.integers(60, 255, 3))


def _text(img, txt, org, color=(255, 255, 255), scale=0.5, thick=1, bg=None):
    (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x, y - th - base), (x + tw + 4, y + base), bg, -1)
    cv2.putText(img, txt, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_tracks(frame: np.ndarray, tracks, show_distance: bool = True, show_depth: bool = True) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = [int(round(v)) for v in t.xyxy]
        col = track_color(t.track_id)
        thick = 3 if t.in_path else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), col, thick)
        txt = f"{t.label} #{t.track_id} {t.conf:.2f}"
        if show_distance and t.distance_m is not None:
            txt += f" {t.distance_m:.0f}m"
        if show_depth and t.rel_depth is not None:
            txt += f" d={t.rel_depth:.2f}"
        _text(out, txt, (x1, max(y1 - 4, 12)), (0, 0, 0), 0.45, 1, bg=col)
    return out


def draw_lanes(frame: np.ndarray, lanes: Optional[LaneResult], color=(0, 255, 0), fill_alpha: float = 0.25) -> np.ndarray:
    out = frame.copy()
    if lanes is None:
        return out
    if lanes.mask is not None and lanes.name == "segmentation":
        overlay = out.copy()
        overlay[lanes.mask > 0] = (255, 120, 0)
        out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)
    if lanes.found:
        poly = lanes.ego_polygon.astype(np.int32)
        ov = out.copy()
        cv2.fillPoly(ov, [poly], color)
        out = cv2.addWeighted(ov, fill_alpha, out, 1 - fill_alpha, 0)
    for ln in (lanes.left, lanes.right):
        if ln is not None:
            cv2.line(out, tuple(int(v) for v in ln[0]), tuple(int(v) for v in ln[1]), (0, 0, 255) if lanes.name == "classical" else (255, 0, 0), 4, cv2.LINE_AA)
    label = f"{lanes.name}: {'ego lane found' if lanes.found else 'lane NOT found'} ({lanes.time_ms:.0f} ms)"
    _text(out, label, (8, 20), (255, 255, 255), 0.5, 1, bg=(0, 0, 0))
    return out


def draw_corridor(frame: np.ndarray, corridor: Optional[np.ndarray], color=(255, 255, 0)) -> np.ndarray:
    if corridor is None:
        return frame
    out = frame.copy()
    cv2.polylines(out, [corridor.astype(np.int32)], True, color, 1, cv2.LINE_AA)
    return out


def draw_hud(frame: np.ndarray, decision: Decision, speed: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    col = STATE_COLOR.get(decision.state, (200, 200, 200))
    bar = 26
    cv2.rectangle(out, (0, h - bar), (w, h), col, -1)
    txt = f"{decision.state}  speed {speed * 3.6:4.0f} km/h"
    if decision.nearest_distance is not None:
        txt += f"  | nearest in path: {decision.nearest_label} #{decision.nearest_id} ~{decision.nearest_distance:.0f} m"
    if decision.ttc is not None:
        txt += f"  TTC {decision.ttc:.1f}s"
    _text(out, txt, (8, h - 8), (0, 0, 0) if decision.state == "CAUTION" else (255, 255, 255), 0.5, 1)
    return out


def depth_panel(frame: np.ndarray, depth_norm: Optional[np.ndarray]) -> np.ndarray:
    if depth_norm is None:
        out = np.zeros_like(frame)
        _text(out, "depth disabled / MiDaS not loaded", (10, 30), (200, 200, 200), 0.55)
        return out
    out = colorize_depth(depth_norm)
    _text(out, "MiDaS relative proximity (bright = near)", (8, 20), (255, 255, 255), 0.5, 1, bg=(0, 0, 0))
    return out


def render_panels(frame: np.ndarray, res: FrameResult) -> Dict[str, np.ndarray]:
    det = draw_tracks(frame, res.tracks)
    lanes = draw_lanes(frame, res.lanes)
    if res.lanes_seg is not None:
        lanes = draw_lanes(lanes, res.lanes_seg, fill_alpha=0.0) if res.lanes is not None else draw_lanes(frame, res.lanes_seg)
    depth = depth_panel(frame, res.depth_norm)
    comb = draw_lanes(frame, res.lanes) if res.lanes is not None else frame.copy()
    comb = draw_corridor(comb, res.corridor)
    comb = draw_tracks(comb, res.tracks)
    comb = draw_hud(comb, res.decision, res.ego_speed)
    _text(det, f"tracks: {len(res.tracks)}", (8, 20), (255, 255, 255), 0.5, 1, bg=(0, 0, 0))
    return {"detections": det, "lanes": lanes, "depth": depth, "combined": comb}


def mosaic(panels: Dict[str, np.ndarray]) -> np.ndarray:
    h, w = panels["combined"].shape[:2]
    def r(p):
        return cv2.resize(p, (w, h)) if p.shape[:2] != (h, w) else p
    top = np.hstack([r(panels["detections"]), r(panels["lanes"])])
    bot = np.hstack([r(panels["depth"]), r(panels["combined"])])
    return np.vstack([top, bot])
