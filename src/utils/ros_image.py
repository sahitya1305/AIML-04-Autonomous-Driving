"""sensor_msgs/Image <-> numpy without cv_bridge.

Why not cv_bridge: the apt cv_bridge in ROS 2 Jazzy is compiled against NumPy 1.x and crashes when a NumPy 2 wheel is
installed in the Python environment. These helpers only need numpy, so they work in both cases.
Functions accept any object with the Image fields, so they are unit-testable without ROS.
"""
from __future__ import annotations

import numpy as np

_CHANNELS = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1}


def imgmsg_to_bgr(msg) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc not in _CHANNELS:
        raise ValueError(f"Unsupported image encoding '{msg.encoding}' (supported: {sorted(_CHANNELS)})")
    ch = _CHANNELS[enc]
    buf = np.frombuffer(bytes(msg.data) if not isinstance(msg.data, (bytes, bytearray, memoryview)) else msg.data, dtype=np.uint8)
    arr = buf.reshape(msg.height, msg.step)[:, : msg.width * ch].reshape(msg.height, msg.width, ch)
    if enc == "bgr8":
        return arr.copy()
    if enc == "rgb8":
        return arr[:, :, ::-1].copy()
    if enc == "bgra8":
        return arr[:, :, :3].copy()
    if enc == "rgba8":
        return arr[:, :, 2::-1].copy()
    return np.repeat(arr, 3, axis=2)  # mono8


def fill_imgmsg(msg, frame_bgr: np.ndarray, frame_id: str = "camera"):
    """Fill an (empty) sensor_msgs.msg.Image with a bgr8 frame and return it. Header stamp is set by the caller."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
        raise ValueError("Expected an HxWx3 uint8 BGR frame")
    msg.height, msg.width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(frame_bgr.shape[1] * 3)
    msg.data = np.ascontiguousarray(frame_bgr).tobytes()
    msg.header.frame_id = frame_id
    return msg
