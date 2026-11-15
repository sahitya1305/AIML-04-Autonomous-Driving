"""CARLA demo (UNTESTED): carla_bridge_node -> perception_node. Needs a CARLA server.
  ros2 launch perception_ros2 carla_demo.launch.py host:=<carla-host-ip> detector:=onnx
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    root = os.environ.get("AIML04_ROOT", "")
    return LaunchDescription([
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("detector", default_value="onnx"),
        DeclareLaunchArgument("record_path", default_value="outputs/videos/ros2_carla_demo.mp4"),
        Node(package="perception_ros2", executable="carla_bridge_node", name="carla_bridge", output="screen",
             parameters=[{"project_root": root, "host": LaunchConfiguration("host")}]),
        Node(package="perception_ros2", executable="perception_node", name="perception_node", output="screen",
             parameters=[{"project_root": root, "config": "configs/ros2.yaml", "detector": LaunchConfiguration("detector"),
                          "record_path": LaunchConfiguration("record_path")}]),
    ])
