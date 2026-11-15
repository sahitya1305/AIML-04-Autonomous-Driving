# GitHub organisation and suggested commit sequence

Branches: `main` (stable) and short-lived `feat/*`. Tag `v0.1-demo` when the demo runs, `v1.0` at submission. Never commit `wandb/`, weights, datasets or keys (`.gitignore` covers them).

Suggested commits (use the message style `type: summary`):
1. `chore: project skeleton, configs, requirements, gitignore`
2. `feat(data): synthetic scene generator and demo data script`
3. `feat(detect): YOLO + ONNX Runtime detectors, ground-truth test double`
4. `feat(track): ByteTrack-style tracker with unit tests`
5. `feat(lanes): classical OpenCV lane detection + tests`
6. `feat(lanes): U-Net segmentation, training and comparison script`
7. `feat(depth): MiDaS depth module`
8. `feat(perception): obstacle logic, pipeline, renderer`
9. `feat(app): CLI session, Tk demo UI, main.py`
10. `feat(sim): closed-loop synthetic obstacle demo`
11. `feat(data): BDD100K/KITTI parsing, analysis, YOLO conversion`
12. `feat(train): fine-tuning, evaluation, W&B wrapper`
13. `feat(opt): ONNX export + CPU benchmark`
14. `feat(ros2): perception_ros2 package, launch files, WSL scripts`
15. `feat(carla): CARLA camera rig, recorder, live loop`
16. `docs(notebooks): TensorRT benchmark and fine-tuning notebooks`
17. `test: full test suite`
18. `docs: README, architecture, status matrix, report templates`
19. After you run things yourself: `docs: add measured benchmark results` (only with your numbers)

Commands (PowerShell, project root; your identity is already configured):
```powershell
git add -A; git status
git commit -m "chore: project skeleton, configs, requirements, gitignore"
git remote add origin https://github.com/<you>/AIML-04-Autonomous-Driving.git
git push -u origin main
```
To get a real history, commit in the groups above as you read/understand each part - it also helps you explain the code in the presentation.
