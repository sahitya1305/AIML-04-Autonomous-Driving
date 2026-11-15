import numpy as np
import pytest

from src.detection.onnx_detector import OnnxDetector, decode_yolo_output, letterbox
from src.detection.yolo_detector import YoloDetector
from src.detection.coco import COCO_NAMES
from src.utils.config import resolve_path

NAMES = dict(enumerate(COCO_NAMES))


def test_letterbox_geometry():
    img = np.zeros((360, 640, 3), np.uint8)
    out, r, (px, py) = letterbox(img, 640)
    assert out.shape == (640, 640, 3) and abs(r - 1.0) < 1e-6 and abs(py - 140) < 1


def test_decode_raw_head_layout_maps_back_to_original_pixels():
    # one confident 'car' (id 2) at centre (320,320) size 100x60 in a 640 letterboxed input of a 1280x720 image
    nc = 80
    out = np.zeros((1, 4 + nc, 8400), np.float32)
    out[0, :4, 0] = [320, 320, 100, 60]
    out[0, 4 + 2, 0] = 0.9
    out[0, :4, 1] = [322, 321, 100, 60]     # near-duplicate -> removed by NMS
    out[0, 4 + 2, 1] = 0.8
    ratio, pad = 0.5, (0.0, 140.0)           # 1280x720 -> 640x360, vertical padding 140
    dets = decode_yolo_output(out, ratio, pad, (720, 1280), NAMES, 0.25, 0.5)
    assert len(dets) == 1 and dets[0].label == "car"
    x1, y1, x2, y2 = dets[0].xyxy
    assert abs(x1 - 540) < 1 and abs(x2 - 740) < 1 and abs(y1 - (290 - 140) * 2) < 1 and abs(y2 - (350 - 140) * 2) < 1


def test_decode_end_to_end_layout_and_class_filter():
    out = np.zeros((1, 300, 6), np.float32)
    out[0, 0] = [100, 100, 200, 200, 0.9, 2]
    out[0, 1] = [300, 100, 340, 200, 0.9, 15]       # 'cat' -> filtered by allowed classes
    dets = decode_yolo_output(out, 1.0, (0, 0), (480, 640), NAMES, 0.25, 0.5, allowed=["car"])
    assert [d.label for d in dets] == ["car"]


def test_decode_nothing_above_threshold():
    out = np.zeros((1, 84, 8400), np.float32)
    assert decode_yolo_output(out, 1.0, (0, 0), (640, 640), NAMES, 0.25, 0.5) == []


def test_missing_onnx_file_gives_clear_error():
    with pytest.raises(FileNotFoundError, match="export_onnx"):
        OnnxDetector("models/detection/does_not_exist.onnx")


def test_missing_yolo_weights_gives_clear_error():
    pytest.importorskip("ultralytics")
    with pytest.raises(FileNotFoundError, match="download_weights"):
        YoloDetector("models/detection/does_not_exist.pt")


@pytest.mark.needs_torch
def test_real_onnx_model_if_present():
    p = resolve_path("models/detection/yolo11n.onnx")
    if not p.exists():
        pytest.skip("yolo11n.onnx not present - run scripts/export_onnx.py (needs ultralytics)")
    det = OnnxDetector(str(p))
    assert isinstance(det.detect(np.zeros((360, 640, 3), np.uint8)), list)


@pytest.mark.needs_torch
def test_pytorch_and_onnx_agree_on_sample_image():
    pytest.importorskip("ultralytics")
    w, o, img = resolve_path("models/detection/yolo11n.pt"), resolve_path("models/detection/yolo11n.onnx"), resolve_path("datasets/samples/bus.jpg")
    if not (w.exists() and o.exists() and img.exists()):
        pytest.skip("needs yolo11n.pt, yolo11n.onnx and datasets/samples/bus.jpg (download_weights / export_onnx / download_samples)")
    import cv2
    im = cv2.imread(str(img))
    a = YoloDetector(str(w), conf=0.4).detect(im)
    b = OnnxDetector(str(o), conf=0.4).detect(im)
    assert abs(len(a) - len(b)) <= 1
