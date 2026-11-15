# Module documentation

| module | file | what it does | key limitation |
|---|---|---|---|
| Config | `src/utils/config.py` | YAML loading (default + layered), `--set a.b=c` overrides, paths resolved from project root | - |
| Synthetic world | `src/data/synthetic.py` | pinhole-camera road scene, exact boxes + lane mask, scenarios `default/stopped/pedestrian/empty` | cartoon; says nothing about real accuracy |
| Detector (YOLO) | `src/detection/yolo_detector.py` | Ultralytics inference, class filter | needs torch |
| Detector (ONNX) | `src/detection/onnx_detector.py` | letterbox, ONNX Runtime, decode for `(1,4+nc,N)` and `(1,N,6)` outputs, per-class NMS | real exported model not verified |
| GT detector | `src/detection/gt_detector.py` | returns ground-truth boxes | **test double**, not a detector |
| Tracker | `src/tracking/byte_tracker.py` | ByteTrack-style (own code) | no re-ID appearance model; no camera-motion compensation |
| Classical lanes | `src/lane_detection/classical.py` | colour+edges -> Hough -> straight-line ego lane fit | straight lines only; weak on curves, shadows, night, worn paint |
| Segmentation lanes | `src/lane_detection/segmentation.py` | 0.5M-param U-Net, mask -> lines | **no shipped weights**; you must train it |
| Depth | `src/depth_estimation/midas.py` | MiDaS_small via torch.hub, normalisation, box proximity | relative, not metric |
| Obstacle logic | `src/perception/obstacle.py` | distance, corridor, TTC, decision, vehicle model | approximate distances; flat-road assumption in `ground` mode |
| Pipeline | `src/perception/pipeline.py` | wires everything; heavy modules degrade gracefully with a warning | |
| Rendering | `src/perception/render.py` | panels, HUD | |
| Session/IO | `src/app/session.py` | read image/video/folder/webcam, write mp4/png | |
| GUI | `src/app/demo_gui.py` | Tk window: open, run, tabs, save | not opened in the sandbox |
| W&B | `src/utils/wandb_utils.py` | wrapper; disabled by default; online->offline fallback | |
| CARLA | `src/simulator/carla_client.py` | ego + RGB camera + parked obstacle, decision -> VehicleControl | never run |
| ROS 2 | `ros2_ws/src/perception_ros2` | 4 nodes + 2 launch files | never run |

## Lane approaches compared
| | classical OpenCV | segmentation U-Net |
|---|---|---|
| principle | hand-designed edges/colour + Hough lines | learned pixel-wise lane mask, then line fit |
| needs training data | no | yes (TuSimple/CULane or own masks) |
| speed on CPU | very fast (ms) | slower (tens of ms for this tiny net; not measured on your laptop) |
| curves | poor (straight-line model) | better in principle (mask follows curve), our line fit is still straight |
| robustness to shadows/night/wear | poor | better *if* trained on such data |
| failure mode | no lines / wrong lines, silently | missed pixels where training data lacked variety |
These are design expectations. `scripts/compare_lanes.py` measures both on your data; the only comparison run by the author was classical-only on synthetic frames.

## Class mapping used for training (7 classes)
pedestrian, rider, car, truck, bus, motorcycle, bicycle. KITTI: Van->car, Person_sitting->pedestrian, Cyclist->rider, Tram/Misc/DontCare dropped; KITTI has no bus/motorcycle.
The default **pretrained** COCO model uses COCO ids but is filtered to person/bicycle/car/motorcycle/bus/truck by name, so it works without fine-tuning.
