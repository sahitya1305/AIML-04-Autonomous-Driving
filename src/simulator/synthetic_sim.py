"""Closed-loop obstacle-reaction demo on the synthetic scene (no CARLA, no neural network needed).

Loop: render frame -> perception pipeline -> decision -> VehicleSim speed -> scene moves with that speed.
Because the ego speed feeds back into the scene, braking really prevents the collision (unlike an open-loop video).
Detections come from the ground-truth 'detector' (test double). The tracking, lane, obstacle and vehicle logic are real.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from src.data.synthetic import SyntheticScene, build_scenario
from src.detection.gt_detector import GroundTruthDetector
from src.perception.pipeline import PerceptionPipeline
from src.perception.render import mosaic, render_panels
from src.utils.config import resolve_path


def run_closed_loop(cfg: Dict[str, Any], scenario: str = "stopped", seconds: float = 25.0, fps: int = 20,
                    out_video: Optional[str] = None, out_csv: Optional[str] = None, panel: str = "combined") -> Dict[str, Any]:
    scene = build_scenario(SyntheticScene(seed=3), scenario)
    cfg = {**cfg, "detector": {**cfg["detector"], "backend": "groundtruth"}}
    pipe = PerceptionPipeline(cfg, frame_size=(scene.w, scene.h), fps=fps, detector=GroundTruthDetector(scene), build_models=False)
    writer = None
    if out_video:
        path = resolve_path(out_video)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (scene.w * (2 if panel == "mosaic" else 1), scene.h * (2 if panel == "mosaic" else 1)))
        if not writer.isOpened():
            raise RuntimeError("Could not open mp4 writer")
    rows = []
    dt = 1.0 / fps
    min_gap = float("inf")
    collided = False
    stopped_at = None
    for i in range(int(seconds * fps)):
        frame, _ = scene.render()
        res = pipe.process(frame, frame_idx=i, dt=dt)
        in_lane = [o for o in scene.objects if abs(o.X) < 1.6 and o.Z > 0]
        gap = min((o.Z for o in in_lane), default=float("inf"))
        min_gap = min(min_gap, gap)
        if gap < 2.5:
            collided = True
        if res.ego_speed < 0.05 and stopped_at is None and res.decision.state == "BRAKE":
            stopped_at = i * dt
        rows.append({"t": round(i * dt, 2), "speed_mps": round(res.ego_speed, 3), "state": res.decision.state,
                     "est_dist_m": None if res.decision.nearest_distance is None else round(res.decision.nearest_distance, 2),
                     "true_gap_m": None if gap == float("inf") else round(gap, 2), "ttc_s": res.decision.ttc})
        if writer is not None:
            panels = render_panels(frame, res)
            img = mosaic(panels) if panel == "mosaic" else panels[panel]
            writer.write(img)
        scene.step(res.ego_speed, dt)
    if writer is not None:
        writer.release()
    if out_csv:
        p = resolve_path(out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return {"scenario": scenario, "min_true_gap_m": min_gap, "collision": collided, "stopped_at_s": stopped_at,
            "max_speed_mps": max(r["speed_mps"] for r in rows), "frames": len(rows), "rows": rows}
