import numpy as np

from src.data.synthetic import SyntheticScene, build_scenario
from src.evaluation.lane_metrics import bottom_x_error, tolerant_iou
from src.lane_detection.classical import ClassicalLaneDetector
from src.lane_detection.segmentation import lanes_from_mask


def test_classical_finds_both_ego_lanes_on_clean_road(empty_scene):
    img, gt = empty_scene.render()
    res = ClassicalLaneDetector().detect(img)
    assert res.found and res.left[0][0] < img.shape[1] / 2 < res.right[0][0]
    assert bottom_x_error(res.left, gt["lane_mask"], "left") < 25
    assert bottom_x_error(res.right, gt["lane_mask"], "right") < 25


def test_classical_robust_to_noise_and_brightness():
    sc = SyntheticScene(seed=5, brightness=0.8, noise=6.0)
    build_scenario(sc, "empty")
    img, gt = sc.render()
    assert ClassicalLaneDetector().detect(img).found


def test_classical_blank_image_finds_nothing():
    res = ClassicalLaneDetector().detect(np.full((360, 640, 3), 100, np.uint8))
    assert not res.found


def test_temporal_smoothing_holds_last_estimate(empty_scene):
    d = ClassicalLaneDetector()
    img, _ = empty_scene.render()
    d.detect(img)
    held = d.detect(np.full_like(img, 100))
    assert held.found    # missed frame -> previous lane kept


def test_mask_to_lines_roundtrip(empty_scene):
    _, gt = empty_scene.render()
    res = lanes_from_mask(gt["lane_mask"])
    assert res.found
    assert bottom_x_error(res.left, gt["lane_mask"], "left") < 15


def test_tolerant_iou_properties():
    m = np.zeros((50, 50), np.uint8); m[:, 20] = 255
    assert tolerant_iou(m, m) == 1.0
    shifted = np.roll(m, 3, axis=1)
    assert tolerant_iou(m, shifted, tol_px=6) > 0.5   # 3px shift, 13px dilated bands -> 10/16
    assert tolerant_iou(m, np.roll(m, 30, axis=1), tol_px=3) < 0.1
