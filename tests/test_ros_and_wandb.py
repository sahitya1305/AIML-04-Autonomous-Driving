import types

import numpy as np
import pytest

from src.utils.ros_image import fill_imgmsg, imgmsg_to_bgr
from src.utils.wandb_utils import WandbLogger, resolve_mode


def fake_msg():
    return types.SimpleNamespace(height=0, width=0, encoding="", is_bigendian=0, step=0, data=b"", header=types.SimpleNamespace(frame_id="", stamp=None))


def test_image_message_roundtrip_without_ros():
    frame = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)
    msg = fill_imgmsg(fake_msg(), frame)
    assert msg.encoding == "bgr8" and msg.step == 64 * 3 and len(msg.data) == 48 * 64 * 3
    assert np.array_equal(imgmsg_to_bgr(msg), frame)


def test_rgb_and_mono_decoding():
    m = fake_msg(); m.height, m.width, m.encoding, m.step = 2, 2, "rgb8", 6
    m.data = bytes([255, 0, 0] * 4)
    assert tuple(imgmsg_to_bgr(m)[0, 0]) == (0, 0, 255)
    m.encoding, m.step, m.data = "mono8", 2, bytes([7] * 4)
    assert imgmsg_to_bgr(m).shape == (2, 2, 3)
    m.encoding = "16UC1"
    with pytest.raises(ValueError):
        imgmsg_to_bgr(m)


def test_fill_rejects_bad_frame():
    with pytest.raises(ValueError):
        fill_imgmsg(fake_msg(), np.zeros((4, 4), np.uint8))


def test_wandb_disabled_is_a_noop():
    w = WandbLogger(enabled=False)
    w.log({"a": 1}); w.summary({"a": 1}); w.finish()
    assert not w.active


def test_wandb_mode_falls_back_offline_without_key():
    assert resolve_mode("online", creds=False) == "offline"
    assert resolve_mode("online", creds=True) == "online"
    assert resolve_mode("offline", creds=False) == "offline"


@pytest.mark.needs_ros
def test_ros2_pubsub_roundtrip():
    rclpy = pytest.importorskip("rclpy")
    from sensor_msgs.msg import Image
    rclpy.init()
    try:
        node = rclpy.create_node("pytest_node")
        got = []
        node.create_subscription(Image, "/pytest_img", got.append, 10)
        pub = node.create_publisher(Image, "/pytest_img", 10)
        msg = fill_imgmsg(Image(), np.full((8, 8, 3), 77, np.uint8))
        import time
        t0 = time.time()
        while not got and time.time() - t0 < 5:
            pub.publish(msg); rclpy.spin_once(node, timeout_sec=0.1)
        assert got and np.all(imgmsg_to_bgr(got[0]) == 77)
    finally:
        rclpy.shutdown()
