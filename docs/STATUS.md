# Feature status matrix (honest)

This document distinguishes code that is implemented from features that were actually tested on the developer's Windows 11 laptop and WSL2 Ubuntu environment.

**Testing environment:**
- Windows 11
- Python 3.12.10
- AMD Ryzen 5 7530U with integrated AMD Radeon Graphics
- 16 GB RAM
- WSL2 Ubuntu 24.04.5 LTS
- ROS 2 Jazzy
- PyTorch CPU (CUDA unavailable)

Legend: **Implemented** = code exists · **Tested locally** = executed successfully on the laptop or WSL2 · **Synthetic only** = tested using generated data, not real-world datasets · **Not completed** = implementation or execution remains outstanding.

| # | Feature | Implementation | Actual result |
|---|---|---|---|
| 1 | Synthetic data generator + exact ground truth | Implemented | Tested locally; generated 300-frame synthetic driving video and sample images |
| 1 | BDD100K / KITTI parsers and YOLO conversion | Implemented | Tests passed on small hand-made label files; no complete real dataset evaluation |
| 1 | Dataset analysis and class-balance plots | Implemented | Tested on synthetic demo data: 400 images and 5,725 objects; not real BDD100K/KITTI statistics |
| 2 | YOLO detector (Ultralytics, COCO-pretrained) | Implemented | Successfully tested CPU inference on a real bus image using pretrained YOLO11n; no fine-tuning |
| 2 | ONNX Runtime detector and output decoding | Implemented | Successfully tested with exported YOLO11n ONNX model on a real bus image |
| 2 | ByteTrack-style tracker (own implementation) | Implemented | Unit tests and synthetic pipeline tests passed |
| 2 | YOLO fine-tuning script and W&B logging | Implemented | Script exists; actual model fine-tuning and online W&B experiment not completed |
| 2 | Bounding boxes, labels, confidence and track IDs | Implemented | Detection visualization tested; tracking functionality tested separately |
| 3 | Classical OpenCV lane detection | Implemented | Tested on synthetic roads only; no real-road evaluation |
| 3 | Lane segmentation model (tiny U-Net), training and comparison | Implemented | Classical lane comparison tested on synthetic roads; segmentation model training not completed |
| 4 | MiDaS monocular depth estimation | Implemented | Module exists; real MiDaS model inference not completed |
| 4 | Depth integration into perception pipeline | Implemented | Pipeline tested with depth disabled; real MiDaS integration not verified |
| 5 | CARLA client, recorder and live simulation | Implemented | Not tested; live CARLA integration not completed |
| 5 | ROS 2 package and nodes | Implemented | Successfully launched and tested in WSL2 ROS 2 Jazzy using recorded video; topics and messages verified |
| 5 | Obstacle response (BRAKE / CAUTION / CLEAR, TTC and vehicle model) | Implemented | Tested in three synthetic closed-loop scenarios; no collision reported in those simulations |
| 5 | Fallback using synthetic closed-loop simulation and ROS camera publisher | Implemented | Synthetic closed-loop simulation and ROS 2 recorded-video pipeline tested |
| 6 | ONNX export and validation | Implemented | Successfully exported YOLO11n to ONNX and validated the model structure |
| 6 | PyTorch CPU vs ONNX Runtime CPU benchmark | Implemented | Successfully benchmarked on the laptop; optimized ONNX Runtime was marginally faster |
| 6 | TensorRT optimization notebook | Implemented | Notebook created but not executed; requires a compatible NVIDIA GPU environment |
| 7 | W&B wrapper with online/offline fallback | Implemented | Wrapper exists; no actual online W&B experiment completed |
| 7 | Closed-loop obstacle demo and MP4 recording | Implemented | Synthetic scenarios executed and output files generated |
| - | Tkinter demo UI | Implemented | Tkinter availability checked; full UI interaction not verified |
| - | CLI (`run_pipeline.py`) | Implemented | Tested using synthetic video and ground-truth detector, and using YOLO/ONNX on a real bus image |

## Automated testing

The local automated test suite completed with:

- **73 passed**
- **4 skipped**
- **0 failed**

## Measured performance results

### YOLO11n CPU vs ONNX Runtime CPU

Benchmark configuration: `bus.jpg`, image size 640, 50 measured runs and 10 warm-up runs.

| Backend | Mean latency | Median latency | P95 latency | Approx. FPS |
|---|---:|---:|---:|---:|
| PyTorch CPU | 51.5 ms | 49.1 ms | 65.9 ms | 19.4 |
| ONNX Runtime CPU (no optimization) | 62.9 ms | 62.2 ms | 71.1 ms | 15.9 |
| ONNX Runtime CPU (all optimizations) | 50.2 ms | 50.0 ms | 59.6 ms | 19.9 |

Optimized ONNX Runtime achieved approximately **1.03x speedup** over PyTorch CPU in this benchmark. This is a modest improvement, not a major performance gain.

### Synthetic lane comparison

Using 40 synthetic road frames:

- Ego lane found: 100%
- Mean processing time: approximately 2.82 ms
- Tolerant IoU: approximately 0.521
- Bottom-x error: approximately 1.45 pixels

These measurements apply only to synthetic roads.

### Synthetic closed-loop obstacle scenarios

Three scenarios were tested:

- Stopped obstacle: no collision; vehicle stopped at approximately 6.4 seconds.
- Pedestrian: no collision; vehicle stopped at approximately 3.05 seconds.
- Empty road: no collision; vehicle did not need to stop.

These are simulated results and do not establish real-world driving safety.

## Important limitations

The following assignment components remain incomplete or unverified:

1. Fine-tuning YOLO on a real BDD100K or KITTI dataset.
2. Training and evaluating the lane segmentation model.
3. Running actual MiDaS depth inference.
4. Live CARLA simulation integration.
5. TensorRT optimization and GPU benchmarking.
6. Actual online Weights & Biases experiment logging.
7. Real-road accuracy evaluation.

The ROS 2 demonstration used recorded video, not a live CARLA camera.

The dataset analysis and closed-loop simulation used synthetic data. Their results must not be presented as real-world accuracy or safety measurements.

## Output and report files

Relevant generated results include:

- `outputs/metrics/benchmark_onnx.json`
- `outputs/metrics/lane_comparison.*`
- `outputs/metrics/closed_loop_*.json`
- `docs/BENCHMARK_REPORT_generated.md`

Generated videos and model weights are excluded from Git where specified by `.gitignore`. They should be provided separately when required for submission.

**Final note:** The project demonstrates a locally tested perception pipeline with pretrained YOLO inference, ONNX export and inference, synthetic obstacle-response simulation, and ROS 2 integration using recorded video. It is not a fully validated autonomous-driving system.