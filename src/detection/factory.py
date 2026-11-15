from __future__ import annotations

from typing import Any, Dict

from src.utils.config import resolve_path


def build_detector(cfg: Dict[str, Any], gt_source=None):
    """Create the detector selected by cfg['detector']['backend'] (yolo | onnx | groundtruth)."""
    d = cfg["detector"]
    backend = d.get("backend", "yolo")
    classes = d.get("classes")
    if backend == "yolo":
        from src.detection.yolo_detector import YoloDetector
        return YoloDetector(d["weights"], d.get("imgsz", 640), d.get("conf", 0.15), d.get("iou", 0.5),
                            d.get("device", "cpu"), d.get("max_det", 100), classes)
    if backend == "onnx":
        from src.detection.onnx_detector import OnnxDetector
        return OnnxDetector(d["onnx_path"], d.get("imgsz", 640), d.get("conf", 0.15), d.get("iou", 0.5), classes,
                            d.get("max_det", 100), opt_level=d.get("onnx_opt_level", "all"), threads=d.get("threads", 0))
    if backend == "groundtruth":
        from src.detection.gt_detector import GroundTruthDetector
        src = gt_source
        if src is None:
            raise FileNotFoundError("Ground-truth file not found: the 'groundtruth' detector only works for inputs that have a "
                                    "matching <name>_gt.json next to them (e.g. datasets/samples/synthetic_drive.mp4). "
                                    "For real footage use --detector yolo or onnx.")
        return GroundTruthDetector(src, d.get("gt_jitter_px", 0.0))
    raise ValueError(f"Unknown detector backend: {backend}")
