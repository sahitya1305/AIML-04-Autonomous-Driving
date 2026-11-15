# ROS 2 (WSL2) and CARLA instructions

## A. ROS 2 demo without CARLA (recommended path - everything inside Ubuntu WSL2)
ROS 2 Jazzy is already installed per your message. In the **Ubuntu terminal**:
```bash
cd /mnt/c/Users/vatsa/Projects/AIML-04-Autonomous-Driving
bash scripts/ros2_setup_wsl.sh          # once: apt packages, venv with system site-packages, colcon build in ~/aiml04_ws
bash scripts/ros2_run_demo.sh           # camera_publisher -> perception_node -> vehicle_sim_node (ground-truth detector on the synthetic video)
```
In a **second Ubuntu terminal**: `bash scripts/ros2_smoke_test.sh` (lists topics, rate, one decision, speed). View images: `source /opt/ros/jazzy/setup.bash && ros2 run rqt_image_view rqt_image_view` (select `/perception/annotated`).
Real detector: export the ONNX model on Windows (`scripts/export_onnx.py`), then `bash scripts/ros2_run_demo.sh onnx` (needs `models/detection/yolo11n.onnx`; and set `AIML04` video source if desired via `ros2 launch ... source:=<path>`).
If a path differs from `C:\Users\vatsa\Projects\AIML-04-Autonomous-Driving`, edit `PROJECT_WIN` at the top of the scripts or `export PROJECT_WIN=...`.
Troubleshooting: `ModuleNotFoundError: src` -> `AIML04_ROOT` not set (use the script). `ModuleNotFoundError: yaml/scipy/cv2` -> re-run the setup script (apt packages). Slow file access -> the colcon workspace is intentionally in `~`, not `/mnt/c`.

## B. CARLA
CARLA's server realistically needs a discrete NVIDIA GPU; your Radeon iGPU is unlikely to run it. Options:
1. **Recorded CARLA frames** (compatible environment): record on any machine with CARLA using `scripts/carla_record.py`, copy the folder/mp4 to `datasets/raw/carla_recordings/`, then run `scripts/run_pipeline.py --source <mp4>` or `ros2 launch perception_ros2 video_demo.launch.py source:=<path>`.
2. **Cloud/other GPU PC** running CARLA; connect with `--host <ip>` (`scripts/carla_run.py`, or `ros2 launch perception_ros2 carla_demo.launch.py host:=<ip>`). The `carla` pip wheel version must equal the server version.
3. If neither is possible: use the synthetic closed-loop demo and state in the report that **CARLA integration is implemented but untested**.
The CARLA code in this repository has **not** been executed by the author.
