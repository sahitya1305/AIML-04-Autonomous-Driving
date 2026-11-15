# 10-minute presentation (12 slides, ~50 s each)

Fill the **[YOUR RESULT]** boxes only with numbers you measured. Say clearly what is synthetic and what is unverified.

1. **Title** - AIML-04 Autonomous Driving Perception Stack; names, date.
2. **Problem** - a car must see objects, lanes and distance in real time on modest hardware. Goal: integrated, demonstrable perception stack.
3. **Requirements -> modules** - 7 assignment items mapped to modules (use the table in README).
4. **Architecture** - show `docs/architecture.png`; data flow detect -> track -> lanes -> depth -> obstacle logic.
5. **Data** - BDD100K/KITTI exploration, class imbalance plot **[YOUR plot from analyze_dataset.py]**, synthetic data for testing and why.
6. **Detection + tracking** - YOLO (pretrained / fine-tuned **[YOUR mAP]**), ByteTrack idea (low-score second association), live clip with IDs.
7. **Lanes** - classical vs U-Net; table from `compare_lanes.py` **[YOUR numbers]**; failure cases (curves, shadows).
8. **Depth** - MiDaS relative depth, why distance uses geometry instead; demo frame.
9. **Obstacle response + simulator** - decision states, TTC, closed-loop video (synthetic) and ROS 2 graph; CARLA status (honest).
10. **Optimisation** - ONNX CPU benchmark **[YOUR table]**; TensorRT on Colab GPU **[YOUR table]**; W&B dashboard screenshot.
11. **Challenges, limitations, lessons** - CPU-only laptop, no CARLA GPU, cv_bridge/NumPy conflict, distance approximation, synthetic-vs-real gap, hyper-parameters tuned on synthetic data only.
12. **Next steps** - fine-tune on more BDD100K, curved-lane model, calibrate camera, real CARLA test, MOT metrics, TensorRT deployment on Jetson.

Live demo order (3 min inside slide 6-9): GUI on a real clip -> `closed-loop stopped` video -> `ros2 topic echo /perception/decision`.
