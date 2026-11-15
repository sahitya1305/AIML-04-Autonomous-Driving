#!/usr/bin/env bash
# Run INSIDE Ubuntu (WSL2). One-time setup of a Python venv that can see ROS 2 Jazzy packages, then builds the ROS 2 package.
# Usage:  bash scripts/ros2_setup_wsl.sh
set -e
PROJECT_WIN="${1:-/mnt/c/Users/vatsa/Projects/AIML-04-Autonomous-Driving}"
source /opt/ros/jazzy/setup.bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-yaml python3-scipy python3-opencv python3-numpy ros-jazzy-rqt-image-view ros-jazzy-demo-nodes-py
python3 -m venv --system-site-packages ~/venvs/aiml04
source ~/venvs/aiml04/bin/activate
pip install --upgrade pip
pip install onnxruntime
# build the colcon workspace OUTSIDE /mnt/c (Windows drives are slow and break symlinks)
mkdir -p ~/aiml04_ws/src
rm -rf ~/aiml04_ws/src/perception_ros2
cp -r "$PROJECT_WIN/ros2_ws/src/perception_ros2" ~/aiml04_ws/src/
cd ~/aiml04_ws
colcon build --packages-select perception_ros2
echo
echo "Setup done. Next:  bash scripts/ros2_run_demo.sh"
