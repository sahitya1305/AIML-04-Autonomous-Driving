# Recording the demo video

**Easiest, fully offline (works now):** `python scripts/run_closed_loop_demo.py --scenario stopped --panel mosaic` writes `outputs/videos/closed_loop_stopped.mp4` showing detections + IDs + lanes + the HUD turning CAUTION -> BRAKE and the speed dropping. Also saves a speed/distance plot.

**Real footage:** `python scripts/run_pipeline.py --source <your_dashcam.mp4>` -> `outputs/videos/<name>_perception_mosaic.mp4` (4 panels: detections, lanes, depth, combined). With real YOLO there is no closed loop (the footage cannot react); say so in the presentation.

**Screen recording of the GUI:** Windows `Win+G` (Xbox Game Bar) -> record, or OBS Studio.

**ROS 2:** `bash scripts/ros2_run_demo.sh` records `outputs/videos/ros2_demo.mp4` from the `/perception/annotated` stream; add `rqt_image_view` for a live window (needs WSLg on Windows 11).

**CARLA (only if you have a suitable GPU/cloud machine):** `python scripts/carla_run.py --save outputs/videos/carla_demo.mp4`.

Suggested 90-second demo: GUI on a real clip (30 s) -> closed-loop `stopped` and `pedestrian` scenarios (40 s) -> ROS 2 topics with `ros2 topic echo /perception/decision` (20 s). State clearly which parts are synthetic.
