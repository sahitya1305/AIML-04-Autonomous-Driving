# API reference (main entry points)

```python
from src.utils.config import load_config
from src.perception.pipeline import PerceptionPipeline
cfg = load_config()                       # configs/default.yaml
cfg["detector"]["backend"] = "onnx"       # yolo | onnx | groundtruth
pipe = PerceptionPipeline(cfg, frame_size=(960, 540), fps=20)
res = pipe.process(frame_bgr)             # -> FrameResult
```

`FrameResult`: `tracks: List[Track]`, `lanes` (classical `LaneResult`), `lanes_seg`, `depth_norm` (HxW float32, 1=near), `decision: Decision`, `ego_speed`, `timings_ms`, `corridor`.
`Track`: `track_id, xyxy, conf, label, distance_m, rel_depth, ttc, in_path`. `Decision`: `state (CLEAR|CAUTION|BRAKE), target_speed, brake, nearest_*, ttc, reason`.

| function / class | signature |
|---|---|
| `PerceptionPipeline.process` | `(frame, frame_idx=None, dt=None, ego_speed=None) -> FrameResult` |
| `ByteTracker.update` | `(detections) -> List[Track]` |
| `ClassicalLaneDetector.detect` | `(frame) -> LaneResult` |
| `SegLaneDetector.detect` | `(frame) -> LaneResult` (needs weights) |
| `MidasDepth.estimate` | `(frame) -> float32 map 0..1` |
| `ObstacleController.update` | `(tracks, lanes, dt) -> (Decision, corridor_polygon)` |
| `run_closed_loop` | `(cfg, scenario, seconds, fps, out_video, out_csv, panel) -> dict` |
| `WandbLogger` | `.log(dict, step) .summary(dict) .log_video(key, path) .finish()` |

## CLI
`python main.py gui | run | check | closed-loop`; scripts in `scripts/` all support `--help`.

## ROS 2 parameters
`camera_publisher`: `source, topic, fps, loop, project_root`. `perception_node`: `config, input_topic, detector, record_path, project_root`. `vehicle_sim_node`: `cruise_speed, rate_hz`. `carla_bridge_node`: `host, port`.
