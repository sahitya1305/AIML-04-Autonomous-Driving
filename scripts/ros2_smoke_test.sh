#!/usr/bin/env bash
# Run INSIDE Ubuntu (WSL2) while the demo is running in another terminal: checks the topics carry data.
source /opt/ros/jazzy/setup.bash
echo "--- topics ---"; ros2 topic list
echo "--- hz of /perception/annotated (5 s) ---"; timeout 5 ros2 topic hz /perception/annotated
echo "--- one decision message ---"; timeout 5 ros2 topic echo --once /perception/decision
echo "--- vehicle speed ---"; timeout 5 ros2 topic echo --once /vehicle/speed
