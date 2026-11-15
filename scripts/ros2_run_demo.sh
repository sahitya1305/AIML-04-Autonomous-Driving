#!/usr/bin/env bash
# Run INSIDE Ubuntu (WSL2). Launches camera -> perception -> simulated vehicle on recorded/synthetic data (no CARLA).
# Usage: bash scripts/ros2_run_demo.sh [detector]   (detector: groundtruth (default) | onnx)
PROJECT_WIN="${PROJECT_WIN:-/mnt/c/Users/vatsa/Projects/AIML-04-Autonomous-Driving}"
DETECTOR="${1:-groundtruth}"
source /opt/ros/jazzy/setup.bash
source ~/venvs/aiml04/bin/activate
source ~/aiml04_ws/install/setup.bash
export AIML04_ROOT="$PROJECT_WIN"
cd "$PROJECT_WIN"
ros2 launch perception_ros2 video_demo.launch.py detector:="$DETECTOR" record_path:=outputs/videos/ros2_demo.mp4
