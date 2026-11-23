# Local Validation Report

Date: 15 November 2026

This report records tests actually performed on the local Windows and WSL2
environment. It supplements STATUS.md, which describes the original
project-generation environment.

## Successfully tested locally

- Synthetic driving dataset and video generation.
- Automated test suite: 73 passed, 4 skipped.
- YOLO11n pretrained inference on the sample bus image.
- YOLO annotated video inference on synthetic driving footage; video generated, but some detections were incorrect.
- ONNX export and structural validation.
- ONNX Runtime inference on the sample bus image.
- ByteTrack-style custom tracker and pipeline execution.
- Synthetic closed-loop scenarios: empty road, stopped vehicle, pedestrian.
- Classical OpenCV lane detection on synthetic roads.
- ROS2 Jazzy package build in WSL2.
- ROS2 camera-to-perception-to-vehicle pipeline using synthetic video.
- ROS2 topics and messages verified:
  /camera/image_raw
  /perception/annotated
  /perception/decision
  /perception/tracks
  /vehicle/speed
  /cmd_vel
- ROS2 demo recording saved as outputs/videos/ros2_demo.mp4.

## ONNX CPU benchmark

Test input: bus.jpg
Image size: 640 x 640
Runs: 50 (10 warm-up runs)

| Backend | Mean latency | Median latency | p95 latency | FPS |
|---|---:|---:|---:|---:|
| PyTorch CPU | 51.5 ms | 49.1 ms | 65.9 ms | 19.4 |
| ONNX Runtime CPU, no optimization | 62.9 ms | 62.2 ms | 71.1 ms | 15.9 |
| ONNX Runtime CPU, all optimizations | 50.2 ms | 50.0 ms | 59.6 ms | 19.9 |

Optimized ONNX Runtime achieved approximately 1.03x the PyTorch CPU
throughput in this benchmark. This is a modest improvement, not a
significant speed-up.

## Not yet verified

- YOLO fine-tuning on BDD100K or KITTI.
- Real-world dataset class distribution and analysis.
- Trained lane segmentation model and comparison.
- Real MiDaS model inference.
- CARLA simulator integration.
- TensorRT execution on NVIDIA GPU.
- Online Weights & Biases experiment logging.
- Real-road accuracy and real-world safety performance.

## Important limitations

The closed-loop scenarios and synthetic lane tests use generated data.
The real-image YOLO/ONNX test uses a single sample image. These results
do not establish model accuracy or autonomous-driving safety.

ROS2 was tested with a recorded synthetic driving video, not a live CARLA
vehicle.
