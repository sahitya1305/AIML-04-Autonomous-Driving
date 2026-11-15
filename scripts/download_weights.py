"""Download COCO-pretrained YOLO weights into models/detection/ (needs internet once).

  python scripts/download_weights.py                 # yolo11n.pt (default, ~5 MB)
  python scripts/download_weights.py --name yolov8n.pt
"""
import _bootstrap  # noqa: F401
import argparse
import os
import shutil
from pathlib import Path

from src.utils.config import resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="yolo11n.pt")
    a = ap.parse_args()
    dest_dir = resolve_path("models/detection")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / a.name
    if dest.exists():
        print(f"Already present: {dest}")
        return
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics is not installed. Run: pip install -r requirements-ml.txt")
    cwd = os.getcwd()
    os.chdir(dest_dir)  # ultralytics downloads bare model names into the current directory
    try:
        model = YOLO(a.name)
    finally:
        os.chdir(cwd)
    found = Path(getattr(model, "ckpt_path", "") or "")
    for cand in (dest, found, Path(cwd) / a.name, dest_dir / a.name):
        if cand and cand.exists() and cand.resolve() != dest.resolve():
            shutil.copy2(cand, dest)
            break
    if not dest.exists():
        raise SystemExit(f"Download finished but {dest} was not found. Look for {a.name} in the project/current folder and move it to models/detection/.")
    print(f"Saved weights: {dest}")
    print("MiDaS weights are downloaded automatically on first use (torch.hub) into models/depth/.")


if __name__ == "__main__":
    main()
