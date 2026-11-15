"""ONNX Runtime detector for YOLOv8/YOLO11-style exports (CPU friendly, no torch needed).

Supported output layouts
  * (1, 4+nc, N)  : raw head output, boxes are xywh in letterboxed input pixels  -> we run NMS
  * (1, N, 6)     : end-to-end export, rows are x1,y1,x2,y2,conf,cls              -> no NMS needed
The decode code is unit-tested on synthetic tensors. Real exported models: not verified in the build sandbox.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from src.detection.coco import COCO_NAMES
from src.perception.types import Detection
from src.utils.config import resolve_path


def letterbox(img: np.ndarray, size: int = 640, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else img
    top, left = (size - nh) / 2.0, (size - nw) / 2.0
    out = cv2.copyMakeBorder(resized, int(round(top - 0.1)), int(round(top + 0.1)), int(round(left - 0.1)),
                             int(round(left + 0.1)), cv2.BORDER_CONSTANT, value=color)
    if out.shape[:2] != (size, size):
        out = cv2.resize(out, (size, size))
    return out, r, (left, top)


def decode_yolo_output(out: np.ndarray, ratio: float, pad: Tuple[float, float], orig_hw: Tuple[int, int],
                       names: Dict[int, str], conf_thres: float = 0.25, iou_thres: float = 0.5,
                       allowed: Optional[Iterable[str]] = None, max_det: int = 100) -> List[Detection]:
    """Turn raw model output into Detections in ORIGINAL image pixel coordinates."""
    out = np.asarray(out)
    if out.ndim == 3:
        out = out[0]
    H, W = orig_hw
    allowed_set = {a.lower() for a in allowed} if allowed else None
    # end-to-end layout (N, 6)
    if out.ndim == 2 and out.shape[1] == 6 and out.shape[0] != 6:
        boxes, scores, cls = out[:, :4].copy(), out[:, 4], out[:, 5].astype(int)
        keep = scores >= conf_thres
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
    else:
        pred = out.T if out.shape[0] < out.shape[1] else out        # -> (N, 4+nc)
        xywh, cls_scores = pred[:, :4], pred[:, 4:]
        cls = cls_scores.argmax(1)
        scores = cls_scores[np.arange(len(cls)), cls]
        keep = scores >= conf_thres
        xywh, scores, cls = xywh[keep], scores[keep], cls[keep]
        boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                          xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1) if len(xywh) else np.zeros((0, 4))
        if len(boxes):
            tlwh = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
            tlwh[:, 0] += cls * 10000.0          # class-offset trick = per-class NMS using the long-standing NMSBoxes API
            idx = cv2.dnn.NMSBoxes(tlwh.tolist(), scores.astype(float).tolist(), conf_thres, iou_thres)
            idx = np.array(idx).reshape(-1)
            boxes, scores, cls = boxes[idx], scores[idx], cls[idx]
    dets: List[Detection] = []
    for b, s, k in zip(boxes, scores, cls):
        x1 = float(np.clip((b[0] - pad[0]) / ratio, 0, W - 1)); y1 = float(np.clip((b[1] - pad[1]) / ratio, 0, H - 1))
        x2 = float(np.clip((b[2] - pad[0]) / ratio, 0, W - 1)); y2 = float(np.clip((b[3] - pad[1]) / ratio, 0, H - 1))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        label = str(names.get(int(k), k))
        if allowed_set is not None and label.lower() not in allowed_set:
            continue
        dets.append(Detection((x1, y1, x2, y2), float(s), int(k), label))
    dets.sort(key=lambda d: -d.conf)
    return dets[:max_det]


class OnnxDetector:
    def __init__(self, onnx_path: str = "models/detection/yolo11n.onnx", imgsz: int = 640, conf: float = 0.15,
                 iou: float = 0.5, classes: Optional[Iterable[str]] = None, max_det: int = 100,
                 providers: Optional[List[str]] = None, opt_level: str = "all", threads: int = 0):
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover
            raise ImportError("onnxruntime is not installed. Run: pip install onnxruntime") from e
        p = resolve_path(onnx_path)
        if not p.exists():
            raise FileNotFoundError(f"ONNX model not found: {p}\nCreate it with: python scripts/export_onnx.py")
        so = ort.SessionOptions()
        so.graph_optimization_level = {
            "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }[opt_level]
        if threads:
            so.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(str(p), so, providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.imgsz = int(shape[2]) if isinstance(shape[2], int) else imgsz
        self.names = self._read_names()
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.classes = classes
        self.name = f"onnx({p.name}, opt={opt_level})"

    def _read_names(self) -> Dict[int, str]:
        meta = self.session.get_modelmeta().custom_metadata_map
        if "names" in meta:
            try:
                return {int(k): v for k, v in ast.literal_eval(meta["names"]).items()}
            except Exception:
                pass
        return dict(enumerate(COCO_NAMES))

    def preprocess(self, frame: np.ndarray):
        img, r, pad = letterbox(frame, self.imgsz)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), r, pad

    def detect(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> List[Detection]:
        x, r, pad = self.preprocess(frame)
        out = self.session.run(None, {self.input_name: x})[0]
        return decode_yolo_output(out, r, pad, frame.shape[:2], self.names, self.conf, self.iou, self.classes, self.max_det)
