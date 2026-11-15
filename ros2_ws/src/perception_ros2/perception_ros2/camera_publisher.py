"""Publishes frames as sensor_msgs/Image on /camera/image_raw from a video file, image folder or the synthetic scene."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from perception_ros2.common import add_project_to_path


class CameraPublisher(Node):
    def __init__(self):
        super().__init__("camera_publisher")
        self.declare_parameter("source", "datasets/samples/synthetic_drive.mp4")   # file/folder, or 'synthetic'
        self.declare_parameter("topic", "/camera/image_raw")
        self.declare_parameter("fps", 20.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("project_root", "")
        add_project_to_path(self.get_parameter("project_root").value)
        from src.utils.ros_image import fill_imgmsg
        self.fill = fill_imgmsg
        self.loop = self.get_parameter("loop").value
        src = self.get_parameter("source").value
        self.pub = self.create_publisher(Image, self.get_parameter("topic").value, 5)
        self.scene = None
        self.frames = None
        if src == "synthetic":
            from src.data.synthetic import SyntheticScene, build_scenario
            self.scene = build_scenario(SyntheticScene(seed=1), "default")
        else:
            from src.app.session import iter_frames
            self.src = src
            self.frames = iter_frames(src)
        fps = float(self.get_parameter("fps").value)
        self.dt = 1.0 / fps
        self.timer = self.create_timer(self.dt, self.tick)
        self.get_logger().info(f"Publishing '{src}' on {self.get_parameter('topic').value} at {fps} fps")

    def next_frame(self):
        if self.scene is not None:
            img = self.scene.render()[0]
            self.scene.step(12.0, self.dt)
            return img
        try:
            return next(self.frames)[1]
        except StopIteration:
            if not self.loop:
                return None
            from src.app.session import iter_frames
            self.frames = iter_frames(self.src)
            return next(self.frames)[1]

    def tick(self):
        frame = self.next_frame()
        if frame is None:
            self.get_logger().info("End of input")
            self.timer.cancel()
            return
        msg = self.fill(Image(), frame, "camera")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
