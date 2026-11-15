"""Benchmark detector latency BEFORE (PyTorch) and AFTER (ONNX Runtime, several optimisation levels) on this machine.

  python scripts/benchmark_inference.py --image datasets/samples/bus.jpg --runs 50
Writes outputs/metrics/benchmark_onnx.json and docs/BENCHMARK_REPORT_generated.md. Numbers are whatever YOUR machine measures.
TensorRT is NOT benchmarked here (needs NVIDIA GPU): see notebooks/01_tensorrt_benchmark_colab.ipynb.
"""
import _bootstrap  # noqa: F401
import argparse
import json

import cv2
import numpy as np

from src.evaluation.benchmark import machine_info, speedup, time_callable
from src.utils.config import resolve_path


def load_image(path):
    p = resolve_path(path)
    if p.exists():
        img = cv2.imread(str(p))
        if img is not None:
            return img, str(p)
    print(f"[note] image {p} not found, using the synthetic frame (speed benchmark only).")
    from src.data.synthetic import SyntheticScene, build_scenario
    return build_scenario(SyntheticScene(width=1280, height=720), "default").render()[0], "synthetic"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="datasets/samples/bus.jpg")
    ap.add_argument("--weights", default="models/detection/yolo11n.pt")
    ap.add_argument("--onnx", default="models/detection/yolo11n.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", type=int, default=50); ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    img, img_src = load_image(a.image)
    results = {}
    wp = resolve_path(a.weights)
    if wp.exists():
        try:
            from src.detection.yolo_detector import YoloDetector
            d = YoloDetector(str(wp), a.imgsz, conf=0.25, device="cpu")
            results["pytorch_cpu"] = time_callable(lambda: d.detect(img), a.warmup, a.runs)
        except ImportError as e:
            print("[skip] PyTorch baseline:", e)
    else:
        print(f"[skip] PyTorch baseline: {wp} missing (run scripts/download_weights.py)")
    op = resolve_path(a.onnx)
    if op.exists():
        from src.detection.onnx_detector import OnnxDetector
        for level in ("none", "all"):
            od = OnnxDetector(str(op), a.imgsz, conf=0.25, opt_level=level, threads=a.threads)
            results[f"onnxruntime_cpu_opt_{level}"] = time_callable(lambda: od.detect(img), a.warmup, a.runs)
    else:
        print(f"[skip] ONNX: {op} missing (run scripts/export_onnx.py)")
    if not results:
        raise SystemExit("Nothing to benchmark. Download weights and export ONNX first.")
    base = results.get("pytorch_cpu")
    for k, v in results.items():
        v["speedup_vs_pytorch"] = speedup(base["mean_ms"], v["mean_ms"]) if base else None
    out = {"image": img_src, "imgsz": a.imgsz, "runs": a.runs, "warmup": a.warmup, "machine": machine_info(), "results": results,
           "note": "End-to-end detector latency (pre-process + inference + post-process), CPU. TensorRT not included."}
    m = resolve_path("outputs/metrics"); m.mkdir(parents=True, exist_ok=True)
    (m / "benchmark_onnx.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = ["# ONNX benchmark (generated on this machine)", "", f"Machine: `{out['machine']}`  ", f"Image: {img_src}, imgsz {a.imgsz}, runs {a.runs} (warm-up {a.warmup})", "",
             "| backend | mean ms | median ms | p95 ms | FPS (mean) | speed-up vs PyTorch |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in results.items():
        sp = f"{v['speedup_vs_pytorch']:.2f}x" if v["speedup_vs_pytorch"] else "n/a"
        lines.append(f"| {k} | {v['mean_ms']:.1f} | {v['median_ms']:.1f} | {v['p95_ms']:.1f} | {v['fps_mean']:.1f} | {sp} |")
    (resolve_path("docs") / "BENCHMARK_REPORT_generated.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines)); print("\nsaved outputs/metrics/benchmark_onnx.json and docs/BENCHMARK_REPORT_generated.md")


if __name__ == "__main__":
    main()
