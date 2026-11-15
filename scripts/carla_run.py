"""Live perception loop on CARLA (no ROS): camera -> pipeline -> brake decision applied to the ego vehicle.

  python scripts/carla_run.py --seconds 40 --host 127.0.0.1 --save outputs/videos/carla_demo.mp4
Status: not run in the build sandbox. Needs a CARLA server (GPU machine) + matching `carla` package.
"""
import _bootstrap  # noqa: F401
import argparse

import cv2

from src.perception.pipeline import PerceptionPipeline
from src.perception.render import render_panels
from src.simulator.carla_client import CarlaCameraRig
from src.utils.config import load_config, resolve_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=40); ap.add_argument("--host"); ap.add_argument("--port", type=int)
    ap.add_argument("--save", default="outputs/videos/carla_demo.mp4"); ap.add_argument("--detector", choices=["yolo", "onnx"])
    ap.add_argument("--no-depth", action="store_true")
    a = ap.parse_args()
    cfg = load_config("configs/carla.yaml")
    if a.host: cfg["carla"]["host"] = a.host
    if a.port: cfg["carla"]["port"] = a.port
    if a.detector: cfg["detector"]["backend"] = a.detector
    if a.no_depth: cfg["pipeline"]["depth"] = False
    fps = 1.0 / cfg["carla"]["fixed_delta_s"]
    rig = CarlaCameraRig(cfg)
    vw, pipe = None, None
    try:
        rig.start()
        for i in range(int(a.seconds * fps)):
            frame = rig.get_frame()
            if pipe is None:
                pipe = PerceptionPipeline(cfg, (frame.shape[1], frame.shape[0]), fps)
            res = pipe.process(frame, i, ego_speed=rig.speed_mps())
            rig.apply_decision(res.decision.state, res.decision.brake)
            img = render_panels(frame, res)["combined"]
            if vw is None:
                p = resolve_path(a.save); p.parent.mkdir(parents=True, exist_ok=True)
                vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
            vw.write(img)
            if i % 20 == 0:
                print(f"t={i / fps:5.1f}s speed={rig.speed_mps() * 3.6:5.1f} km/h {res.decision.state} {res.decision.reason}")
    finally:
        if vw: vw.release()
        rig.close()


if __name__ == "__main__":
    main()
