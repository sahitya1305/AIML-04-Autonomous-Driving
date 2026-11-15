# Test results and evaluation methodology

Run: `python -m pytest -q` (project root, venv active).

## Last run by the author (sandbox, Python 3.13, Linux, no torch/ROS): 72 passed, 5 skipped
Skipped tests need torch/ultralytics/ROS 2/model files and run automatically once those exist:
real MiDaS inference, real ONNX model, PyTorch-vs-ONNX agreement, ROS 2 pub/sub round-trip, YOLO missing-weights message.
**This is not a result from your laptop.** Run it yourself and record it here: `____ passed, ____ skipped, ____ failed`.

| test file | covers |
|---|---|
| test_imports_and_config | dependency/import checks, config layering, overrides, no secrets in repo |
| test_tracking | IoU, ID stability, occlusion recovery, low-score association, track expiry |
| test_lanes | classical lanes (clean/noisy/blank), smoothing, mask->line, IoU metric |
| test_depth | depth normalisation/colourisation/box proximity (+ optional real MiDaS) |
| test_obstacle_and_pipeline | distance model, corridor, decisions + hysteresis, vehicle model, pipeline run, closed-loop collision avoidance in 3 scenarios, no false braking, adjacent-lane traffic ignored |
| test_video_smoke | generates demo data, runs the CLI on a video, checks the output mp4 and metrics JSON |
| test_detection_onnx | letterbox, output decoding (both layouts), class filtering, clear errors for missing files, optional real ONNX/PyTorch agreement |
| test_data_scripts | BDD/KITTI parsing, statistics, YOLO conversion |
| test_ros_and_wandb | ROS Image conversion without ROS, W&B disabled/offline logic, optional ROS 2 round-trip |
| test_benchmark_and_notebooks | timing statistics, notebooks valid and free of outputs/keys |

## Evaluation methodology for your report
* Detection: mAP50 / mAP50-95 / precision / recall on the held-out split (`scripts/evaluate_detector.py`) - compare COCO-pretrained vs fine-tuned **on the same classes**.
* Tracking: no MOT metrics are implemented. If you need HOTA/MOTA use a labelled sequence and `py-motmetrics` (not included). The unit tests only check ID stability on synthetic motion.
* Lanes: `scripts/compare_lanes.py` (tolerant IoU, bottom-x error, detection rate, ms).
* Obstacle response: closed-loop scenarios in `outputs/metrics/closed_loop_*.json` (collision flag, minimum true gap, time to stop). Synthetic world + ground-truth detector -> validates logic, not perception.

## Caveat on the closed-loop plots
In `outputs/visualizations/closed_loop_*.png` the "estimated distance" line matches the "true gap" almost perfectly. That is **expected and not evidence of real-world distance accuracy**: the synthetic world is rendered with the same pinhole geometry and object sizes that the distance estimator assumes, and the detector is the ground-truth test double. On real footage the distance error will be much larger and must be measured (e.g. against KITTI LiDAR depth).
