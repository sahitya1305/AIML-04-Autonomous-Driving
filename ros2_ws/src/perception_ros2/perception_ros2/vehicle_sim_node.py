"""Simulated vehicle: reacts to /perception/decision (std_msgs/String JSON) and publishes
/vehicle/speed (std_msgs/Float32) and /cmd_vel (geometry_msgs/Twist). This is the 'simulated response' to obstacles."""
import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32, String

from perception_ros2.common import add_project_to_path


class VehicleSimNode(Node):
    def __init__(self):
        super().__init__("vehicle_sim_node")
        self.declare_parameter("project_root", "")
        self.declare_parameter("cruise_speed", 12.0)
        self.declare_parameter("rate_hz", 20.0)
        add_project_to_path(self.get_parameter("project_root").value)
        from src.perception.obstacle import VehicleSim
        from src.perception.types import Decision
        self.Decision = Decision
        self.sim = VehicleSim(float(self.get_parameter("cruise_speed").value))
        self.decision = Decision("CLEAR", self.sim.cruise, 0.0)
        self.dt = 1.0 / float(self.get_parameter("rate_hz").value)
        self.create_subscription(String, "/perception/decision", self.on_decision, 10)
        self.pub_speed = self.create_publisher(Float32, "/vehicle/speed", 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_timer(self.dt, self.tick)
        self.last_state = "CLEAR"

    def on_decision(self, msg):
        try:
            d = json.loads(msg.data)
            self.decision = self.Decision(**{k: d.get(k) for k in ("state", "target_speed", "brake", "nearest_id", "nearest_label", "nearest_distance", "ttc", "reason")})
        except Exception as e:
            self.get_logger().warning(f"bad decision message: {e}")

    def tick(self):
        v = self.sim.step(self.decision, self.dt)
        self.pub_speed.publish(Float32(data=float(v)))
        cmd = Twist(); cmd.linear.x = float(v)
        self.pub_cmd.publish(cmd)
        if self.decision.state != self.last_state:
            self.get_logger().info(f"state -> {self.decision.state}: {self.decision.reason} (speed {v * 3.6:.0f} km/h)")
            self.last_state = self.decision.state


def main():
    rclpy.init()
    node = VehicleSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
