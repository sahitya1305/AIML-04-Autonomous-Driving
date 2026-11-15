# ONNX benchmark (generated on this machine)

Machine: `{'platform': 'Windows-11-10.0.26200-SP0', 'processor': 'AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD', 'python': '3.12.10', 'onnxruntime': '1.31.1', 'torch': '2.15.0+cpu', 'cuda_available': 'False'}`
Image: C:\Users\vatsa\Downloads\AIML-04-Autonomous-Driving\AIML-04-Autonomous-Driving\datasets\samples\bus.jpg, imgsz 640, runs 50 (warm-up 10)

| backend | mean ms | median ms | p95 ms | FPS (mean) | speed-up vs PyTorch |
|---|---:|---:|---:|---:|---:|
| pytorch_cpu | 51.5 | 49.1 | 65.9 | 19.4 | 1.00x |
| onnxruntime_cpu_opt_none | 62.9 | 62.2 | 71.1 | 15.9 | 0.82x |
| onnxruntime_cpu_opt_all | 50.2 | 50.0 | 59.6 | 19.9 | 1.03x |
