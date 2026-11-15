# Performance benchmarking report

**Status: no benchmark numbers have been measured yet.** The author had no PyTorch/ONNX model and no GPU. Never copy numbers into this report that you did not measure.

## Methodology (already implemented)
* Metric: end-to-end detector latency (pre-process + inference + post-process), batch 1, same image, 10 warm-up + 50 timed runs, reported as mean / median / p95 and FPS = 1000 / mean.
* ONNX part (your laptop, CPU): `python scripts/benchmark_inference.py` -> compares PyTorch CPU vs ONNX Runtime with graph optimisation `none` and `all`. Writes `outputs/metrics/benchmark_onnx.json` and `docs/BENCHMARK_REPORT_generated.md`.
* TensorRT part (cloud NVIDIA GPU): `notebooks/01_tensorrt_benchmark_colab.ipynb` -> PyTorch FP32 GPU vs ONNX vs TensorRT FP16. Writes `benchmark_gpu.json`.
* ONNX results and TensorRT results are **different experiments on different hardware**; do not compare them to each other.
* Also report full-pipeline FPS from `outputs/metrics/run_<name>.json` (per-stage mean ms are included).

## Results - fill in after you run
### ONNX (CPU, Ryzen 5 7530U)
| backend | mean ms | p95 ms | FPS | speed-up vs PyTorch |
|---|---|---|---|---|
| PyTorch CPU | | | | 1.00 |
| ONNX Runtime (opt none) | | | | |
| ONNX Runtime (opt all) | | | | |

### TensorRT (Colab/Kaggle GPU: ______)
| backend | mean ms | p95 ms | FPS | speed-up vs PyTorch |
|---|---|---|---|---|
| PyTorch FP32 | | | | 1.00 |
| ONNX Runtime | | | | |
| TensorRT FP16 | | | | |

### Accuracy impact
Run `scripts/evaluate_detector.py` for the PyTorch and ONNX weights (and compare detection counts printed by the notebook) before claiming "no accuracy loss".
