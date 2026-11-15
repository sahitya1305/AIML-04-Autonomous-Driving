"""Synthetic road-scene generator with exact ground truth.

Why it exists: it lets every downstream module (tracking, lanes, obstacle logic, pipeline, closed-loop
demo, tests) run without any dataset download, and gives exact ground truth for sanity checks.
The images are CARTOON renderings. They are NOT a substitute for real data and say nothing about
real-world accuracy.

Geometry: pinhole camera, flat ground, camera height `cam_h`, horizon at `horizon` * height.
A world point (X lateral, Z forward) projects to  u = cx + f*X/Z,  v = hy + f*cam_h/Z.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Real-world sizes in metres (width, height) used both to render and to estimate distance.
KIND_SIZE = {"car": (1.8, 1.5), "truck": (2.5, 3.2), "person": (0.55, 1.75), "bus": (2.5, 3.2)}
KIND_CLS = {"person": 0, "car": 2, "bus": 5, "truck": 7}  # COCO ids


@dataclass
class SceneObject:
    oid: int
    kind: str
    X: float          # lateral offset (m) from ego centre line, +right
    Z: float          # distance ahead (m)
    vz: float = 0.0   # forward speed of the object in the world (m/s)
    vx: float = 0.0   # lateral speed (m/s)
    color: Tuple[int, int, int] = (60, 60, 200)


class SyntheticScene:
    def __init__(self, width: int = 640, height: int = 360, hfov_deg: float = 90.0,
                 horizon: float = 0.45, cam_h: float = 1.4, curvature: float = 0.0,
                 seed: int = 0, brightness: float = 1.0, noise: float = 0.0):
        self.w, self.h = int(width), int(height)
        self.f = (self.w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.cx = self.w / 2.0
        self.hy = self.h * horizon
        self.cam_h = cam_h
        self.curv = curvature
        self.rng = np.random.default_rng(seed)
        self.brightness = brightness
        self.noise = noise
        self.objects: List[SceneObject] = []
        self.travelled = 0.0
        self.time = 0.0
        self.lane_half = 1.75

    # ------------------------------------------------------------------ geometry
    def _lat(self, X: float, Z: float) -> float:
        return X + self.curv * Z * Z

    def project(self, X: float, Z: float) -> Tuple[float, float]:
        Z = max(Z, 0.5)
        return self.cx + self.f * self._lat(X, Z) / Z, self.hy + self.f * self.cam_h / Z

    def object_box(self, o: SceneObject) -> Optional[Tuple[float, float, float, float]]:
        if o.Z < 3.0 or o.Z > 120.0:
            return None
        wm, hm = KIND_SIZE[o.kind]
        u, vb = self.project(o.X, o.Z)
        wp, hp = self.f * wm / o.Z, self.f * hm / o.Z
        box = (u - wp / 2, vb - hp, u + wp / 2, vb)
        if box[2] < 0 or box[0] > self.w or box[3] < 0 or box[1] > self.h:
            return None
        return box

    # ------------------------------------------------------------------ dynamics
    def add(self, kind: str, X: float, Z: float, vz: float = 0.0, vx: float = 0.0, color=None) -> SceneObject:
        color = color or tuple(int(c) for c in self.rng.integers(40, 230, 3))
        o = SceneObject(len(self.objects) + 1, kind, X, Z, vz, vx, color)
        self.objects.append(o)
        return o

    def step(self, ego_speed: float, dt: float) -> None:
        self.travelled += ego_speed * dt
        self.time += dt
        for o in self.objects:
            o.Z += (o.vz - ego_speed) * dt
            o.X += o.vx * dt

    # ------------------------------------------------------------------ rendering
    def _quad(self, img, X1, Z1, X2, Z2, color):
        pts = np.array([self.project(X1, Z1), self.project(X2, Z1), self.project(X2, Z2), self.project(X1, Z2)], np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)

    def _draw_line_strip(self, img, X, Z0, Z1, half_w, color, dash: Optional[Tuple[float, float]] = None):
        if dash is None:
            segs = [(Z0, Z1)]
        else:
            on, off = dash
            segs, period = [], on + off
            phase = self.travelled % period
            z = Z0 - phase
            while z < Z1:
                a, b = max(z, Z0), min(z + on, Z1)
                if b > a:
                    segs.append((a, b))
                z += period
        for a, b in segs:
            self._quad(img, X - half_w, a, X + half_w, b, color)

    def render(self) -> Tuple[np.ndarray, Dict]:
        img = np.zeros((self.h, self.w, 3), np.uint8)
        hy = int(self.hy)
        for y in range(hy):  # sky gradient
            t = y / max(hy, 1)
            img[y, :] = (int(235 - 40 * (1 - t)), int(206 - 30 * (1 - t)), int(160 + 20 * t))
        img[hy:, :] = (70, 120, 80)  # grass
        z_near = self.f * self.cam_h / max(self.h - self.hy, 1.0) * 1.02
        z_far = 150.0
        self._quad(img, -7.0, z_near, 7.0, z_far, (88, 88, 92))             # asphalt
        lw = 0.09
        for X in (-5.25, 5.25):
            self._draw_line_strip(img, X, z_near, z_far, lw, (230, 230, 230))          # outer solid
        for X in (-self.lane_half, self.lane_half):
            self._draw_line_strip(img, X, z_near, z_far, lw, (240, 240, 240), dash=(3.0, 4.0))  # ego lane
        # objects, far to near
        boxes = []
        for o in sorted(self.objects, key=lambda q: -q.Z):
            box = self.object_box(o)
            if box is None:
                continue
            self._draw_object(img, o, box)
            clipped = (max(0, box[0]), max(0, box[1]), min(self.w - 1, box[2]), min(self.h - 1, box[3]))
            boxes.append({"id": o.oid, "label": o.kind, "cls_id": KIND_CLS[o.kind], "xyxy": [float(v) for v in clipped],
                          "distance_m": float(o.Z), "X": float(o.X)})
        img = np.clip(img.astype(np.float32) * self.brightness, 0, 255)
        if self.noise > 0:
            img += self.rng.normal(0, self.noise, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        gt = {"boxes": boxes, "lane_mask": self.lane_mask()}
        return img, gt

    def lane_mask(self, half_w: float = 0.14) -> np.ndarray:
        """Exact mask of the two ego-lane boundary lines (continuous, ignoring dashes)."""
        m = np.zeros((self.h, self.w), np.uint8)
        z_near = self.f * self.cam_h / max(self.h - self.hy, 1.0) * 1.02
        for X in (-self.lane_half, self.lane_half):
            self._draw_line_strip(m, X, z_near, 150.0, half_w, 255)
        return m

    def _draw_object(self, img, o: SceneObject, box):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        w, h = max(x2 - x1, 1), max(y2 - y1, 1)
        col = o.color
        if o.kind == "person":
            cv2.rectangle(img, (x1 + w // 6, y1 + h // 4), (x2 - w // 6, y2 - h // 4), col, -1)
            cv2.circle(img, ((x1 + x2) // 2, y1 + h // 8), max(h // 9, 1), (170, 200, 235), -1)
            cv2.rectangle(img, (x1 + w // 5, y2 - h // 4), (x2 - w // 5, y2), (40, 40, 40), -1)
        else:
            body_top = y1 + (0 if o.kind in ("truck", "bus") else h // 4)
            cv2.rectangle(img, (x1, body_top), (x2, y2 - h // 8), col, -1)
            if o.kind == "car":
                cv2.rectangle(img, (x1 + w // 6, y1), (x2 - w // 6, y1 + h // 2), tuple(int(c * 0.8) for c in col), -1)
                cv2.rectangle(img, (x1 + w // 5, y1 + h // 16), (x2 - w // 5, y1 + h // 2 - h // 16), (50, 40, 30), -1)
            else:
                cv2.rectangle(img, (x1 + w // 10, y1 + h // 10), (x2 - w // 10, y1 + h // 3), (200, 200, 200), -1)
            lh = max(h // 10, 1)
            cv2.rectangle(img, (x1 + 1, y2 - h // 3), (x1 + w // 6, y2 - h // 3 + lh), (30, 30, 230), -1)
            cv2.rectangle(img, (x2 - w // 6, y2 - h // 3), (x2 - 1, y2 - h // 3 + lh), (30, 30, 230), -1)
            ww = max(w // 6, 2)
            cv2.rectangle(img, (x1 + w // 12, y2 - h // 6), (x1 + w // 12 + ww, y2), (20, 20, 20), -1)
            cv2.rectangle(img, (x2 - w // 12 - ww, y2 - h // 6), (x2 - w // 12, y2), (20, 20, 20), -1)

    def ground_truth_boxes(self) -> List[dict]:
        return self.render()[1]["boxes"]


def build_scenario(scene: SyntheticScene, name: str = "default") -> SyntheticScene:
    """Populate a scene. All distances are metres ahead of the ego vehicle."""
    if name == "stopped":
        scene.add("car", 0.0, 60.0, 0.0, color=(40, 40, 190))                 # stopped car in our lane
        scene.add("car", 3.5, 35.0, 12.0, color=(190, 120, 40))               # adjacent lane, ignored
    elif name == "pedestrian":
        scene.add("person", 4.5, 38.0, 0.0, vx=-1.4, color=(30, 160, 220))    # steps into our lane
        scene.add("truck", -3.5, 55.0, 9.0, color=(200, 200, 200))
    elif name == "empty":
        pass
    else:  # default: slow lead car, passing car in the next lane, crossing pedestrian later
        scene.add("car", 0.0, 70.0, 3.0, color=(40, 40, 190))
        scene.add("car", 3.5, 25.0, 15.0, color=(190, 120, 40))
        scene.add("truck", -3.5, 60.0, 9.0, color=(200, 200, 200))
        scene.add("person", 6.0, 90.0, 0.0, vx=-1.2, color=(30, 160, 220))
    return scene
