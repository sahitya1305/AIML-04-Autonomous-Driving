# Feature status matrix (honest)

"Sandbox" = the Linux container where this project was generated (Python 3.13, OpenCV 5.0, onnxruntime, numpy, scipy, pytest; **no** PyTorch, ultralytics, tkinter, ROS 2, CARLA, GPU, internet for model downloads).
Nothing below was run on your Windows laptop, WSL2 or any GPU. Re-run on your machines and update the "Your result" column.

Legend: **Implemented** (code exists) · **Tested (sandbox)** (an automated test or run passed in the sandbox) · **Ext. download** (needs dataset/weights) · **GPU/cloud** · **CARLA** · **Not verified** (never executed anywhere by the author)

| # | Feature | Implemented | Tested (sandbox) | Needs | Your result |
|---|---|:-:|:-:|---|---|
| 1 | Synthetic data generator + exact GT (`src/data/synthetic.py`) | yes | yes (tests, rendered frames inspected) | - | |
| 1 | BDD100K / KITTI parsers, YOLO conversion | yes | yes, on tiny hand-made label files only | real dataset for real use (**Ext. download**) | |
| 1 | Dataset analysis + class-balance plots | yes | yes, on **fake** labels (`--demo`) | real labels (**Ext. download**) | |
| 2 | YOLO detector (Ultralytics, COCO-pretrained) | yes | **Not verified** (no torch in sandbox) | `pip install -r requirements-ml.txt`, weights download | |
| 2 | ONNX Runtime detector + output decoding | yes | decoding tested on synthetic tensors; **real model not verified** | export first | |
| 2 | ByteTrack-style tracker (own implementation) | yes | yes (7 unit tests + pipeline runs) | - | |
| 2 | Fine-tuning script + W&B logging | yes | **Not verified** (never trained) | dataset + (GPU recommended) | |
| 2 | Boxes, labels, confidence, track IDs drawn | yes | yes (frames inspected, GT-detector input) | - | |
| 3 | Classical OpenCV lane detection | yes | yes (synthetic roads only; **not** tested on real roads) | - | |
| 3 | Segmentation lane model (tiny U-Net) + training + comparison script | yes | mask->line logic tested; **model training not verified** (no torch) | torch; TuSimple/CULane for real training (**Ext. download**) | |
| 4 | MiDaS depth module | yes | normalise/colourise/box-proximity helpers tested; **model loading not verified** | torch, timm, internet once | |
| 4 | Depth wired into pipeline (per-track proximity, panel) | yes | wiring covered when depth disabled/fake; **with real MiDaS not verified** | | |
| 5 | CARLA client / recorder / live loop | yes | **Not verified** | **CARLA** server + matching `carla` wheel | |
| 5 | ROS 2 package (3 nodes + bridge, launch files) | yes | Python compiles; image conversion tested without ROS; **nodes never run** | WSL2 ROS 2 Jazzy | |
| 5 | Obstacle-response (BRAKE/CAUTION/CLEAR, TTC, vehicle model) | yes | yes: closed-loop sim avoids collision in 3 scenarios, never brakes on empty road | - | |
| 5 | Fallback when no CARLA: synthetic closed-loop + recorded frames via ROS camera_publisher | yes | closed loop yes; ROS part **not verified** | | |
| 6 | ONNX export + validation script | yes | **Not verified** (needs ultralytics/onnx) | torch stack | |
| 6 | CPU benchmark PyTorch vs ONNX Runtime | yes | **Not verified** (no model in sandbox) | weights + ONNX file | |
| 6 | TensorRT notebook (Colab/Kaggle) | yes | JSON validity only; **never executed** | **GPU/cloud** | |
| 7 | W&B wrapper (online/offline fallback, no hard-coded keys) | yes | disabled no-op + mode fallback tested; **no real W&B run** | account/API key via env var | |
| 7 | Closed-loop obstacle demo + mp4 recording | yes | yes (mp4 + plot + CSV written) | - | |
| - | Tk demo UI | yes | `py_compile` only; **never opened** (no tkinter in sandbox) | tkinter (bundled with python.org installer) | |
| - | CLI (`run_pipeline.py`) | yes | yes on synthetic video with GT-detector; real-detector paths not verified | | |

## Measured numbers in this repository
None for YOLO accuracy, ONNX/TensorRT speed-ups, MiDaS accuracy or real-road lane accuracy. The only numbers that exist are:
* closed-loop results in `outputs/metrics/closed_loop_*.json` (synthetic world, ground-truth detector)
* synthetic lane comparison in `outputs/metrics/lane_comparison.*` (classical only, cartoon roads)
They validate the code, not real-world performance.
