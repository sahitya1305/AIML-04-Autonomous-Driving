"""Export a YOLO .pt model to ONNX (CPU-friendly) and validate that the file loads and runs.

  python scripts/export_onnx.py                              # models/detection/yolo11n.pt -> yolo11n.onnx
  python scripts/export_onnx.py --weights experiments/.../best.pt --imgsz 640
This is ONNX only. TensorRT needs an NVIDIA GPU -> use notebooks/01_tensorrt_benchmark_colab.ipynb.
"""
import _bootstrap  # noqa: F401
import argparse
import shutil
from pathlib import Path

import numpy as np

from src.utils.config import resolve_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/detection/yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out", default=None, help="destination .onnx (default: next to the weights)")
    a = ap.parse_args()
    w = resolve_path(a.weights)
    if not w.exists():
        raise SystemExit(f"Weights not found: {w}. Run scripts/download_weights.py first.")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed: pip install -r requirements-ml.txt")
    produced = Path(YOLO(str(w)).export(format="onnx", imgsz=a.imgsz, opset=a.opset, simplify=True, dynamic=False))
    dest = resolve_path(a.out) if a.out else w.with_suffix(".onnx")
    if produced.resolve() != dest.resolve():
        shutil.copy2(produced, dest)
    print("ONNX file:", dest, f"({dest.stat().st_size / 1e6:.1f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.run(None, {inp.name: np.random.rand(1, 3, a.imgsz, a.imgsz).astype(np.float32)})
    print(f"Validation OK: input {inp.name} {inp.shape} -> output shapes {[o.shape for o in out]}")
    print("Next: python scripts/benchmark_inference.py")


if __name__ == "__main__":
    main()
