"""Record camera frames from a running CARLA server into a folder + mp4 (so you can later run the pipeline OFFLINE).

  python scripts/carla_record.py --frames 400 --host 127.0.0.1
Status: not run in the build sandbox. Needs a CARLA server and the matching `carla` Python package.
"""
import _bootstrap  # noqa: F401
import argparse
import time
from pathlib import Path

import cv2

from src.simulator.carla_client import CarlaCameraRig
from src.utils.config import apply_set_overrides, load_config, resolve_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=400); ap.add_argument("--host"); ap.add_argument("--port", type=int)
    ap.add_argument("--no-obstacle", action="store_true")
    a = ap.parse_args()
    cfg = load_config("configs/carla.yaml")
    if a.host: cfg["carla"]["host"] = a.host
    if a.port: cfg["carla"]["port"] = a.port
    out = resolve_path("datasets/raw/carla_recordings") / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rig = CarlaCameraRig(cfg)
    vw = None
    try:
        rig.start(spawn_obstacle=not a.no_obstacle)
        for i in range(a.frames):
            f = rig.get_frame()
            cv2.imwrite(str(out / f"frame_{i:06d}.png"), f)
            if vw is None:
                vw = cv2.VideoWriter(str(out / "recording.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (f.shape[1], f.shape[0]))
            vw.write(f)
            if i % 50 == 0:
                print("frame", i)
    finally:
        if vw: vw.release()
        rig.close()
    print("Saved to", out, "\nRun the pipeline on it:  python scripts/run_pipeline.py --source", out / "recording.mp4")


if __name__ == "__main__":
    main()
