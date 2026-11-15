"""Plain data containers shared by all modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

XYXY = Tuple[float, float, float, float]


@dataclass
class Detection:
    xyxy: XYXY
    conf: float
    cls_id: int
    label: str

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]


@dataclass
class Track:
    track_id: int
    xyxy: XYXY
    conf: float
    cls_id: int
    label: str
    # Filled in by later stages (obstacle logic / depth):
    distance_m: Optional[float] = None
    rel_depth: Optional[float] = None   # 0 (far) .. 1 (near), MiDaS relative inverse depth
    closing_speed: Optional[float] = None
    ttc: Optional[float] = None
    in_path: bool = False

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return ((self.xyxy[0] + self.xyxy[2]) / 2.0, self.xyxy[3])

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    def to_dict(self) -> dict:
        return {
            "id": int(self.track_id), "label": self.label, "conf": round(float(self.conf), 3),
            "xyxy": [round(float(v), 1) for v in self.xyxy],
            "distance_m": None if self.distance_m is None else round(float(self.distance_m), 2),
            "rel_depth": None if self.rel_depth is None else round(float(self.rel_depth), 3),
            "ttc": None if self.ttc is None else round(float(self.ttc), 2),
            "in_path": bool(self.in_path),
        }


@dataclass
class LaneResult:
    """Output of a lane detector. Lines are ((x_bottom, y_bottom), (x_top, y_top)) in pixels."""
    name: str = "lanes"
    left: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    right: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    mask: Optional[np.ndarray] = None   # uint8 HxW, 255 on lane pixels
    time_ms: float = 0.0

    @property
    def found(self) -> bool:
        return self.left is not None and self.right is not None

    @property
    def ego_polygon(self) -> Optional[np.ndarray]:
        if not self.found:
            return None
        (lb, lt), (rb, rt) = self.left, self.right
        return np.array([lb, lt, rt, rb], dtype=np.float32)


@dataclass
class Decision:
    state: str = "CLEAR"            # CLEAR | CAUTION | BRAKE
    target_speed: float = 0.0       # m/s
    brake: float = 0.0              # 0..1
    nearest_id: Optional[int] = None
    nearest_label: Optional[str] = None
    nearest_distance: Optional[float] = None
    ttc: Optional[float] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class FrameResult:
    frame_idx: int
    tracks: List[Track] = field(default_factory=list)
    lanes: Optional[LaneResult] = None          # classical
    lanes_seg: Optional[LaneResult] = None      # segmentation model
    depth_norm: Optional[np.ndarray] = None     # float32 HxW in 0..1 (1 = near)
    decision: Decision = field(default_factory=Decision)
    ego_speed: float = 0.0
    timings_ms: Dict[str, float] = field(default_factory=dict)
    corridor: Optional[np.ndarray] = None
