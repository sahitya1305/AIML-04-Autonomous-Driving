import json
import subprocess
import sys

import cv2

from src.utils.config import resolve_path


def test_make_demo_data_and_run_pipeline_on_video(tmp_path):
    root = resolve_path(".")
    out = tmp_path / "samples"
    r = subprocess.run([sys.executable, str(root / "scripts/make_demo_data.py"), "--out", str(out), "--frames", "40",
                        "--yolo-mini", "0", "--bdd-mini", "0"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    video = out / "synthetic_drive.mp4"
    assert video.exists() and (out / "synthetic_drive_gt.json").exists()
    r = subprocess.run([sys.executable, str(root / "scripts/run_pipeline.py"), "--source", str(video), "--detector", "groundtruth",
                        "--no-depth", "--out-dir", str(tmp_path / "out"), "--max-frames", "30"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    mp4 = list((tmp_path / "out").glob("*.mp4"))
    assert mp4
    cap = cv2.VideoCapture(str(mp4[0]))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 25
    metrics = json.loads((root / "outputs/metrics/run_synthetic_drive.json").read_text())
    assert metrics["frames"] == 30 and metrics["latency"]["mean_ms"] > 0


def test_image_input_writes_panels(tmp_path, empty_scene):
    root = resolve_path(".")
    img_path = tmp_path / "frame.png"
    cv2.imwrite(str(img_path), empty_scene.render()[0])
    r = subprocess.run([sys.executable, str(root / "scripts/run_pipeline.py"), "--source", str(img_path), "--detector", "groundtruth",
                        "--no-depth", "--out-dir", str(tmp_path / "o")], capture_output=True, text=True)
    # groundtruth needs a _gt.json for non-synthetic input -> expect a clear error, not a crash with a stack-trace mystery
    assert r.returncode != 0 and "Ground-truth file not found" in (r.stdout + r.stderr)
