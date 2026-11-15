"""Subscribes to sensor_msgs/Image, runs the perception pipeline, publishes:
  /perception/annotated   sensor_msgs/Image   combined annotated frame
  /perception/tracks      std_msgs/String     JSON list of tracks (id, label, conf, box, distance)
  /perception/decision    std_msgs/String     JSON decision {state, target_speed, brake, ...}
Subscribes /vehicle/speed (std_msgs/Float32) so the HUD shows the speed of the simulated vehicle node.
"""
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from perception_ros2.common import add_project_to_path


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        self.declare_parameter("config", "configs/ros2.yaml")
        self.declare_parameter("input_topic", "/camera/image_raw")
        self.declare_parameter("detector", "")            # '' = use config, else yolo|onnx|groundtruth
        self.declare_parameter("record_path", "")         # e.g. outputs/videos/ros2_demo.mp4
        self.declare_parameter("project_root", "")
        root = add_project_to_path(self.get_parameter("project_root").value)
        from src.perception.pipeline import PerceptionPipeline
        from src.perception.render import render_panels
        from src.utils.config import load_config
        from src.utils.ros_image import fill_imgmsg, imgmsg_to_bgr
        self.to_bgr, self.fill, self.render, self.Pipeline = imgmsg_to_bgr, fill_imgmsg, render_panels, PerceptionPipeline
        self.cfg = load_config(self.get_parameter("config").value)
        det = self.get_parameter("detector").value
        if det:
            self.cfg["detector"]["backend"] = det
        self.pipe = None            # built on first frame (needs the frame size)
        self.speed = None
        self.writer = None
        self.last_t = None
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, self.get_parameter("input_topic").value, self.on_image, qos)
        self.create_subscription(Float32, "/vehicle/speed", self.on_speed, 10)
        self.pub_img = self.create_publisher(Image, "/perception/annotated", 5)
        self.pub_tracks = self.create_publisher(String, "/perception/tracks", 10)
        self.pub_dec = self.create_publisher(String, "/perception/decision", 10)
        self.get_logger().info(f"perception_node ready (detector backend: {self.cfg['detector']['backend']})")

    def on_speed(self, msg):
        self.speed = float(msg.data)

    def on_image(self, msg):
        try:
            frame = self.to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e))
            return
        now = time.perf_counter()
        dt = None if self.last_t is None else max(now - self.last_t, 1e-3)
        self.last_t = now
        if self.pipe is None:
            self.pipe = self.Pipeline(self.cfg, (frame.shape[1], frame.shape[0]), 20.0)
        res = self.pipe.process(frame, dt=dt, ego_speed=self.speed)
        out = self.fill(Image(), self.render(frame, res)["combined"], msg.header.frame_id)
        out.header.stamp = msg.header.stamp
        self.pub_img.publish(out)
        self.pub_tracks.publish(String(data=json.dumps([t.to_dict() for t in res.tracks])))
        self.pub_dec.publish(String(data=json.dumps(res.decision.to_dict())))
        rec = self.get_parameter("record_path").value
        if rec:
            import cv2
            from src.utils.config import resolve_path
            img = self.render(frame, res)["combined"]
            if self.writer is None:
                p = resolve_path(rec); p.parent.mkdir(parents=True, exist_ok=True)
                self.writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 20, (img.shape[1], img.shape[0]))
            self.writer.write(img)

    def destroy_node(self):
        if self.writer is not None:
            self.writer.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
