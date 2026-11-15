"""Prints what is installed / available. Run this first when something does not work."""
import _bootstrap  # noqa: F401
import importlib
import platform
import sys

from src.utils.config import resolve_path

print(f"Python {sys.version.split()[0]} on {platform.platform()}")
core = ["numpy", "cv2", "yaml", "scipy", "matplotlib", "pandas", "pytest", "onnxruntime", "tkinter"]
ml = ["torch", "torchvision", "ultralytics", "timm", "onnx", "wandb"]
extra = ["rclpy", "carla"]
for title, mods in (("CORE (requirements.txt)", core), ("ML (requirements-ml.txt)", ml), ("OPTIONAL (ROS 2 / CARLA)", extra)):
    print(f"\n{title}")
    for m in mods:
        try:
            mod = importlib.import_module(m)
            print(f"  [ok]      {m:<12} {getattr(mod, '__version__', '')}")
        except Exception as e:
            print(f"  [missing] {m:<12} ({type(e).__name__})")
try:
    import torch
    print(f"\ntorch CUDA available: {torch.cuda.is_available()}  (expected False on your laptop)")
except Exception:
    pass
try:
    import onnxruntime as ort
    print("ONNX Runtime providers:", ort.get_available_providers())
except Exception:
    pass
print("\nFiles")
for p in ["models/detection/yolo11n.pt", "models/detection/yolo11n.onnx", "models/lane_seg/lane_unet.pt",
          "datasets/samples/synthetic_drive.mp4", "datasets/samples/bus.jpg"]:
    print(f"  [{'ok' if resolve_path(p).exists() else '--'}] {p}")
