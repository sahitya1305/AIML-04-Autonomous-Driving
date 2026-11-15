"""Video/recorded-data demo (no CARLA): camera_publisher -> perception_node -> vehicle_sim_node.
  ros2 launch perception_ros2 video_demo.launch.py source:=datasets/samples/synthetic_drive.mp4 detector:=groundtruth
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    root = os.environ.get("AIML04_ROOT", "")
    args = [DeclareLaunchArgument("source", default_value="datasets/samples/synthetic_drive.mp4"),
            DeclareLaunchArgument("detector", default_value="groundtruth"),
            DeclareLaunchArgument("config", default_value="configs/ros2.yaml"),
            DeclareLaunchArgument("record_path", default_value="")]
    common = {"project_root": root}
    return LaunchDescription(args + [
        Node(package="perception_ros2", executable="camera_publisher", name="camera_publisher", output="screen",
             parameters=[{**common, "source": LaunchConfiguration("source")}]),
        Node(package="perception_ros2", executable="perception_node", name="perception_node", output="screen",
             parameters=[{**common, "config": LaunchConfiguration("config"), "detector": LaunchConfiguration("detector"),
                          "record_path": LaunchConfiguration("record_path")}]),
        Node(package="perception_ros2", executable="vehicle_sim_node", name="vehicle_sim_node", output="screen", parameters=[common]),
    ])
