"""CARLA camera rig: spawns an ego vehicle + RGB camera (+ optional parked obstacle) and yields BGR frames.

STATUS: written against the CARLA 0.9.x Python API but NOT run in the build sandbox (no CARLA there, and your laptop's
integrated Radeon GPU is unlikely to run the CARLA server). Treat as untested until you run it. The `carla` Python wheel
version must match the server version. No hard dependency: `import carla` happens lazily.
"""
from __future__ import annotations

import queue
from typing import Any, Dict, Optional

import numpy as np


class CarlaCameraRig:
    def __init__(self, cfg: Dict[str, Any]):
        try:
            import carla
        except ImportError as e:
            raise ImportError("The 'carla' Python package is not installed (it must match your CARLA server version). "
                              "No CARLA? Use recorded frames instead: see docs/ROS2_CARLA.md.") from e
        self.carla = carla
        c = cfg["carla"]
        self.cfg = c
        self.client = carla.Client(c["host"], int(c["port"]))
        self.client.set_timeout(float(c.get("timeout_s", 20)))
        self.world = self.client.load_world(c["town"]) if c.get("town") else self.client.get_world()
        self.actors = []
        self.q: "queue.Queue" = queue.Queue(maxsize=4)
        self._orig_settings = self.world.get_settings()
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = float(c.get("fixed_delta_s", 0.05))
        self.world.apply_settings(s)
        self.vehicle = None
        self.camera = None

    def start(self, spawn_obstacle: bool = True):
        carla, w, c = self.carla, self.world, self.cfg
        lib = w.get_blueprint_library()
        spawns = w.get_map().get_spawn_points()
        tf = spawns[int(c.get("spawn_index", 0)) % len(spawns)]
        self.vehicle = w.spawn_actor(lib.filter("vehicle.tesla.model3")[0], tf)
        self.actors.append(self.vehicle)
        cam_bp = lib.find("sensor.camera.rgb")
        for k, v in (("image_size_x", c["camera"]["width"]), ("image_size_y", c["camera"]["height"]), ("fov", c["camera"]["fov"])):
            cam_bp.set_attribute(k, str(v))
        cam_tf = carla.Transform(carla.Location(x=c["camera"]["x"], z=c["camera"]["z"]))
        self.camera = w.spawn_actor(cam_bp, cam_tf, attach_to=self.vehicle)
        self.actors.append(self.camera)
        self.camera.listen(self._on_image)
        if spawn_obstacle:
            wp = w.get_map().get_waypoint(self.vehicle.get_location()).next(float(c.get("obstacle_distance_m", 45)))[0]
            obs_tf = wp.transform
            obs_tf.location.z += 0.3
            obs = w.try_spawn_actor(lib.filter("vehicle.audi.a2")[0], obs_tf)
            if obs is not None:
                obs.apply_control(carla.VehicleControl(hand_brake=True))
                self.actors.append(obs)
        self.vehicle.set_autopilot(True)
        self.world.tick()

    def _on_image(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, :3].copy()  # BGRA -> BGR
        try:
            self.q.put_nowait(arr)
        except queue.Full:
            try:
                self.q.get_nowait(); self.q.put_nowait(arr)
            except queue.Empty:
                pass

    def get_frame(self, timeout: float = 5.0) -> np.ndarray:
        self.world.tick()
        return self.q.get(timeout=timeout)

    def speed_mps(self) -> float:
        v = self.vehicle.get_velocity()
        return float((v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5)

    def apply_decision(self, state: str, brake: float):
        """BRAKE -> autopilot off + full brake. CAUTION -> autopilot off + light brake. CLEAR -> autopilot back on."""
        carla = self.carla
        if state == "CLEAR":
            self.vehicle.set_autopilot(True)
        else:
            self.vehicle.set_autopilot(False)
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=float(brake), steer=0.0))

    def close(self):
        try:
            if self.camera is not None:
                self.camera.stop()
            for a in self.actors:
                a.destroy()
            self.world.apply_settings(self._orig_settings)
        except Exception:
            pass
