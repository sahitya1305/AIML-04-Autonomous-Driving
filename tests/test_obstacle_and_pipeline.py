import numpy as np
import pytest

from src.data.synthetic import SyntheticScene, build_scenario
from src.detection.gt_detector import GroundTruthDetector
from src.perception.obstacle import ObstacleController, VehicleSim, pinhole_distance, focal_from_fov
from src.perception.pipeline import PerceptionPipeline
from src.perception.types import Decision, Track
from src.simulator.synthetic_sim import run_closed_loop


def test_pinhole_distance_matches_synthetic_geometry():
    sc = SyntheticScene()
    sc.add("car", 0.0, 40.0)
    box = sc.ground_truth_boxes()[0]
    h = box["xyxy"][3] - box["xyxy"][1]
    d = pinhole_distance(h, "car", focal_from_fov(sc.w, 90.0))
    assert abs(d - 40.0) < 1.0


def test_in_corridor_logic(cfg):
    oc = ObstacleController(cfg["obstacle"], (640, 360))
    corridor = oc.default_corridor()
    inside = Track(1, (300, 150, 340, 250), 0.9, 2, "car")
    outside = Track(2, (20, 150, 60, 250), 0.9, 2, "car")
    assert oc.in_path(inside, corridor) and not oc.in_path(outside, corridor)


def test_decision_states_and_hysteresis(cfg):
    oc = ObstacleController(cfg["obstacle"], (640, 360), fps=20)
    def tr(dist_h):  # box height controls estimated distance
        return Track(1, (300, 360 - dist_h, 340, 360), 0.9, 2, "car")
    far = oc.update([tr(12)], None)[0]       # tiny box -> far
    assert far.state == "CLEAR"
    near = oc.update([tr(150)], None)[0]     # big box -> close
    assert near.state == "BRAKE" and near.brake == 1.0
    held = oc.update([tr(12)], None)[0]      # obstacle gone far: must hold for release_frames
    assert held.state == "BRAKE"
    for _ in range(12):
        d = oc.update([tr(12)], None)[0]
    assert d.state == "CLEAR"


def test_vehicle_sim_brakes_to_zero_and_resumes():
    v = VehicleSim(cruise=12.0)
    for _ in range(200):
        v.step(Decision("BRAKE", 0.0, 1.0), 0.05)
    assert v.speed == 0.0
    for _ in range(400):
        v.step(Decision("CLEAR", 12.0, 0.0), 0.05)
    assert abs(v.speed - 12.0) < 1e-6


def test_pipeline_runs_on_synthetic_frame(cfg):
    cfg = {**cfg, "pipeline": {**cfg["pipeline"], "depth": False}}
    sc = build_scenario(SyntheticScene(), "default")
    pipe = PerceptionPipeline(cfg, (sc.w, sc.h), 20, detector=GroundTruthDetector(sc), build_models=False)
    for i in range(6):
        frame, _ = sc.render()
        res = pipe.process(frame, i)
        sc.step(12, 0.05)
    assert len(res.tracks) >= 3 and res.lanes is not None and res.lanes.found
    assert {"detect", "track", "lanes_classical", "obstacle", "total"} <= set(res.timings_ms)


@pytest.mark.parametrize("scenario", ["stopped", "pedestrian", "default"])
def test_closed_loop_avoids_collision(cfg, scenario):
    out = run_closed_loop(cfg, scenario, seconds=25)
    assert not out["collision"], out
    assert out["min_true_gap_m"] > 3.0
    assert out["stopped_at_s"] is not None or out["min_true_gap_m"] > 5


def test_closed_loop_empty_road_never_brakes(cfg):
    out = run_closed_loop(cfg, "empty", seconds=10)
    assert all(r["state"] == "CLEAR" for r in out["rows"]) and out["max_speed_mps"] == 12.0


def test_adjacent_lane_traffic_is_ignored(cfg):
    sc = SyntheticScene(); sc.add("car", 3.5, 12.0, 12.0)      # close, but in the next lane, same speed as ego
    pipe = PerceptionPipeline(cfg, (sc.w, sc.h), 20, detector=GroundTruthDetector(sc), build_models=False)
    pipe.enable_depth = False
    for i in range(10):
        res = pipe.process(sc.render()[0], i)
    assert res.decision.state == "CLEAR"
