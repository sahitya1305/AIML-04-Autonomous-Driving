"""Main perception pipeline: detect -> track -> lanes -> depth -> obstacle decision."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.depth_estimation.midas import box_proximity
from src.lane_detection.classical import ClassicalLaneDetector
from src.perception.obstacle import ObstacleController, VehicleSim
from src.perception.types import FrameResult
from src.tracking.byte_tracker import ByteTracker
from src.utils.logger import get_logger
from src.utils.timing import Timer

log = get_logger("pipeline")


class PerceptionPipeline:
    """Owns all modules. Heavy modules (YOLO, MiDaS, lane U-Net) are injected or built lazily from the config."""

    def __init__(self, cfg: Dict[str, Any], frame_size=(640, 360), fps: float = 20.0, detector=None, depth=None,
                 lane_seg=None, build_models: bool = True, gt_source=None):
        self.cfg = cfg
        self.frame_size, self.fps = frame_size, fps
        pc = cfg.get("pipeline", {})
        self.enable_lanes = pc.get("lanes", True)
        self.enable_depth = pc.get("depth", True)
        self.enable_lane_seg = pc.get("lane_seg", False)
        self.depth_interval = max(1, int(cfg.get("depth", {}).get("interval", 3)))
        self.detector = detector
        self.depth = depth
        self.lane_seg = lane_seg
        if build_models:
            if self.detector is None:
                from src.detection.factory import build_detector
                self.detector = build_detector(cfg, gt_source)
            if self.enable_depth and self.depth is None and cfg.get("depth", {}).get("backend", "midas") == "midas":
                self.depth = self._try(self._build_depth, "MiDaS depth")
            if self.enable_lane_seg and self.lane_seg is None:
                self.lane_seg = self._try(self._build_seg, "lane segmentation")
        t = cfg.get("tracker", {})
        self.tracker = ByteTracker(t.get("high_thresh", 0.5), t.get("low_thresh", 0.1), t.get("new_track_thresh", 0.6),
                                   t.get("match_thresh", 0.8), t.get("low_match_thresh", 0.5), t.get("track_buffer", 30),
                                   t.get("min_hits", 2))
        self.lane_cls = ClassicalLaneDetector(cfg.get("lanes", {}).get("classical", {}))
        self.obstacle = ObstacleController(cfg["obstacle"], frame_size, fps)
        self.vehicle = VehicleSim(cfg["obstacle"]["cruise_speed"], max_decel=cfg["obstacle"].get("max_decel", 6.0))
        self._depth_cache: Optional[np.ndarray] = None
        self.frame_idx = 0

    # -- optional heavy modules: failures degrade gracefully with a clear warning --------
    @staticmethod
    def _try(fn, what):
        try:
            return fn()
        except Exception as e:
            log.warning("%s disabled: %s", what, e)
            return None

    def _build_depth(self):
        from src.depth_estimation.midas import MidasDepth
        d = self.cfg["depth"]
        return MidasDepth(d.get("model_type", "MiDaS_small"), d.get("device", "cpu"), d.get("cache_dir", "models/depth"))

    def _build_seg(self):
        from src.lane_detection.segmentation import SegLaneDetector
        s = self.cfg["lanes"]["segmentation"]
        return SegLaneDetector(s["weights"], s.get("device", "cpu"), s.get("threshold", 0.5))

    def reset(self):
        self.tracker.reset()
        self.lane_cls.reset()
        self.obstacle = ObstacleController(self.cfg["obstacle"], self.frame_size, self.fps)
        self.vehicle = VehicleSim(self.cfg["obstacle"]["cruise_speed"], max_decel=self.cfg["obstacle"].get("max_decel", 6.0))
        self._depth_cache, self.frame_idx = None, 0

    def process(self, frame: np.ndarray, frame_idx: Optional[int] = None, dt: Optional[float] = None,
                ego_speed: Optional[float] = None) -> FrameResult:
        """Run all stages on one BGR frame. If `ego_speed` is None the internal VehicleSim provides it."""
        idx = self.frame_idx if frame_idx is None else frame_idx
        self.frame_idx = idx + 1
        res = FrameResult(frame_idx=idx)
        T = res.timings_ms
        dt = dt or 1.0 / self.fps
        with Timer(T, "detect"):
            dets = self.detector.detect(frame, frame_idx=idx) if self.detector is not None else []
        with Timer(T, "track"):
            res.tracks = self.tracker.update(dets)
        if self.enable_lanes:
            with Timer(T, "lanes_classical"):
                res.lanes = self.lane_cls.detect(frame)
        if self.lane_seg is not None:
            with Timer(T, "lanes_seg"):
                res.lanes_seg = self.lane_seg.detect(frame)
        if self.enable_depth and self.depth is not None:
            with Timer(T, "depth"):
                if idx % self.depth_interval == 0 or self._depth_cache is None:
                    self._depth_cache = self.depth.estimate(frame)
            res.depth_norm = self._depth_cache
            if res.depth_norm is not None:
                for tr in res.tracks:
                    tr.rel_depth = box_proximity(res.depth_norm, tr.xyxy)
        with Timer(T, "obstacle"):
            lanes_for_path = res.lanes if res.lanes is not None else res.lanes_seg
            res.decision, res.corridor = self.obstacle.update(res.tracks, lanes_for_path, dt)
            res.ego_speed = self.vehicle.step(res.decision, dt) if ego_speed is None else ego_speed
        T["total"] = sum(v for k, v in T.items() if k != "total")
        return res
