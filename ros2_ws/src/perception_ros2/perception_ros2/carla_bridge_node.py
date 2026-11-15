"""CARLA <-> ROS 2 bridge (UNTESTED: needs a CARLA server): publishes the CARLA RGB camera as sensor_msgs/Image on
/camera/image_raw and applies /perception/decision (BRAKE/CAUTION/CLEAR) to the ego vehicle."""
import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from perception_ros2.common import add_project_to_path


class CarlaBridge(Node):
    def __init__(self):
        super().__init__("carla_bridge")
        self.declare_parameter("project_root", "")
        self.declare_parameter("host", "")
        self.declare_parameter("port", 2000)
        root = add_project_to_path(self.get_parameter("project_root").value)
        from src.simulator.carla_client import CarlaCameraRig
        from src.utils.config import load_config
        from src.utils.ros_image import fill_imgmsg
        self.fill = fill_imgmsg
        cfg = load_config("configs/carla.yaml")
        if self.get_parameter("host").value:
            cfg["carla"]["host"] = self.get_parameter("host").value
        cfg["carla"]["port"] = int(self.get_parameter("port").value)
        self.rig = CarlaCameraRig(cfg)
        self.rig.start()
        self.pub = self.create_publisher(Image, "/camera/image_raw", 5)
        self.pub_speed = self.create_publisher(Float32, "/vehicle/speed", 10)
        self.create_subscription(String, "/perception/decision", self.on_decision, 10)
        self.create_timer(float(cfg["carla"]["fixed_delta_s"]), self.tick)

    def on_decision(self, msg):
        d = json.loads(msg.data)
        self.rig.apply_decision(d["state"], float(d.get("brake", 0.0)))

    def tick(self):
        frame = self.rig.get_frame()
        m = self.fill(Image(), frame, "carla_camera")
        m.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(m)
        self.pub_speed.publish(Float32(data=self.rig.speed_mps()))

    def destroy_node(self):
        self.rig.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = CarlaBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
