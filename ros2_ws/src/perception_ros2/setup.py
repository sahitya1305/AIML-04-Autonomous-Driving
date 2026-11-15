from glob import glob
from setuptools import find_packages, setup

package_name = "perception_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="student",
    maintainer_email="student@example.com",
    description="ROS 2 nodes for the AIML-04 perception stack",
    license="MIT",
    entry_points={
        "console_scripts": [
            "camera_publisher = perception_ros2.camera_publisher:main",
            "perception_node = perception_ros2.perception_node:main",
            "vehicle_sim_node = perception_ros2.vehicle_sim_node:main",
            "carla_bridge_node = perception_ros2.carla_bridge_node:main",
        ],
    },
)
