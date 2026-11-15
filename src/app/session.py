"""Backend shared by the CLI (scripts/run_pipeline.py), the Tk GUI (src/app/demo_gui.py) and main.py."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from src.perception.pipeline import PerceptionPipeline
from src.perception.render import mosaic, render_panels
from src.perception.types import FrameResult
from src.utils.config import resolve_path
from src.utils.logger import get_logger

log = get_logger("session")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def resize_keep_aspect(frame: np.ndarray, max_width: Optional[int]) -> np.ndarray:
    if not max_width or frame.shape[1] <= max_width:
        return frame
    s = max_width / frame.shape[1]
    return cv2.resize(frame, (max_width, int(round(frame.shape[0] * s))), interpolation=cv2.INTER_AREA)


def probe_source(source: str) -> Tuple[str, float, Tuple[int, int], int]:
    """Return (kind, fps, (w,h), n_frames). kind in image|video|folder|camera."""
    if str(source).isdigit():
        cap = cv2.VideoCapture(int(source))
        w, h = int(cap.get(3)), int(cap.get(4)); cap.release()
        return "camera", 20.0, (w or 640, h or 480), -1
    p = resolve_path(source)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT)
        if not files:
            raise ValueError(f"No images in folder {p}")
        im = cv2.imread(str(files[0]))
        return "folder", 10.0, (im.shape[1], im.shape[0]), len(files)
    if p.suffix.lower() in IMAGE_EXT:
        im = cv2.imread(str(p))
        if im is None:
            raise ValueError(f"Could not read image {p}")
        return "image", 1.0, (im.shape[1], im.shape[0]), 1
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ValueError(f"Could not open video {p}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    size = (int(cap.get(3)), int(cap.get(4)))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return "video", float(fps), size, n


def iter_frames(source: str, stride: int = 1, max_frames: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
    kind, *_ = probe_source(source)
    count = 0
    if kind == "image":
        yield 0, cv2.imread(str(resolve_path(source)))
        return
    if kind == "folder":
        files = sorted(f for f in resolve_path(source).iterdir() if f.suffix.lower() in IMAGE_EXT)
        for i, f in enumerate(files):
            if i % stride == 0:
                yield i, cv2.imread(str(f))
                count += 1
                if max_frames and count >= max_frames:
                    return
        return
    cap = cv2.VideoCapture(int(source) if kind == "camera" else str(resolve_path(source)))
    i = 0
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield i, fr
                count += 1
                if max_frames and count >= max_frames:
                    break
            i += 1
    finally:
        cap.release()


class OutputWriter:
    """Writes the chosen panel to an .mp4 (video input) or PNG files (image input)."""

    def __init__(self, out_dir: str, stem: str, kind: str, fps: float):
        self.dir = resolve_path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stem, self.kind, self.fps = stem, kind, fps
        self.writer = None
        self.path: Optional[Path] = None

    def write(self, panels: Dict[str, np.ndarray], which: str = "mosaic", idx: int = 0):
        img = mosaic(panels) if which == "mosaic" else panels[which]
        if self.kind == "image":
            for name, p in {**panels, "mosaic": mosaic(panels)}.items():
                cv2.imwrite(str(self.dir / f"{self.stem}_{name}.png"), p)
            self.path = self.dir / f"{self.stem}_{which}.png"
            return
        if self.writer is None:
            self.path = self.dir / f"{self.stem}_{which}.mp4"
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (img.shape[1], img.shape[0]))
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {self.path} (codec 'mp4v' unavailable?)")
        self.writer.write(img)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def build_pipeline_for(source: str, cfg: Dict[str, Any], gt_source=None) -> Tuple[PerceptionPipeline, str, float, Tuple[int, int]]:
    kind, fps, (w, h), _ = probe_source(source)
    mw = cfg.get("run", {}).get("max_width")
    if mw and w > mw:
        h, w = int(round(h * mw / w)), int(mw)
    pipe = PerceptionPipeline(cfg, frame_size=(w, h), fps=fps, gt_source=gt_source)
    return pipe, kind, fps, (w, h)


def run_session(source: str, cfg: Dict[str, Any], gt_source=None, pipeline: Optional[PerceptionPipeline] = None
                ) -> Iterator[Tuple[int, np.ndarray, FrameResult, Dict[str, np.ndarray]]]:
    """Generator yielding (index, resized_frame, result, panels) for each processed frame."""
    run = cfg.get("run", {})
    if pipeline is None:
        pipeline, *_ = build_pipeline_for(source, cfg, gt_source)
    for idx, frame in iter_frames(source, run.get("stride", 1), run.get("max_frames")):
        frame = resize_keep_aspect(frame, run.get("max_width"))
        res = pipeline.process(frame, frame_idx=idx)
        yield idx, frame, res, render_panels(frame, res)
