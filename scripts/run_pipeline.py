"""Run the full perception pipeline on an image, video, image folder or webcam index.

Examples (PowerShell, project root, venv active):
  python scripts/run_pipeline.py --source datasets/samples/synthetic_drive.mp4 --detector groundtruth --no-depth
  python scripts/run_pipeline.py --source my_dashcam.mp4                      # YOLO + ByteTrack + lanes + MiDaS
  python scripts/run_pipeline.py --source my_dashcam.mp4 --detector onnx --no-depth --wandb
"""
import _bootstrap  # noqa: F401
import argparse
import json
import time
from pathlib import Path

from src.app.session import OutputWriter, build_pipeline_for, probe_source, run_session
from src.utils.config import apply_set_overrides, load_config, resolve_path
from src.utils.logger import get_logger
from src.utils.timing import latency_stats
from src.utils.wandb_utils import WandbLogger

log = get_logger("run")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--detector", choices=["yolo", "onnx", "groundtruth"])
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--no-lanes", action="store_true")
    ap.add_argument("--lane-seg", action="store_true", help="also run the segmentation lane model (needs trained weights)")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--panel", choices=["mosaic", "combined", "detections", "lanes", "depth"])
    ap.add_argument("--out-dir")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. detector.conf=0.3")
    a = ap.parse_args()

    cfg = apply_set_overrides(load_config(a.config), a.set)
    if a.detector: cfg["detector"]["backend"] = a.detector
    if a.no_depth: cfg["pipeline"]["depth"] = False
    if a.no_lanes: cfg["pipeline"]["lanes"] = False
    if a.lane_seg: cfg["pipeline"]["lane_seg"] = True
    if a.max_frames: cfg["run"]["max_frames"] = a.max_frames
    if a.wandb: cfg["wandb"]["enabled"] = True
    panel = a.panel or cfg["run"]["panel"]
    gt = None
    if cfg["detector"]["backend"] == "groundtruth":
        stem_gt = Path(str(a.source)).with_name(Path(str(a.source)).stem + "_gt.json")
        gt = str(stem_gt) if resolve_path(stem_gt).exists() else None

    pipe, kind, fps, size = build_pipeline_for(a.source, cfg, gt)
    stem = Path(str(a.source)).stem if not str(a.source).isdigit() else "camera"
    writer = OutputWriter(a.out_dir or cfg["run"]["output_dir"], f"{stem}_perception", kind, fps)
    wb = WandbLogger.from_config(cfg, job_type="inference", run_name=f"run-{stem}", extra_config={"source": str(a.source)})
    log.info("detector=%s | source=%s (%s, %dx%d, %.1f fps)", getattr(pipe.detector, "name", "?"), a.source, kind, *size, fps)

    totals, decisions, n = [], {"CLEAR": 0, "CAUTION": 0, "BRAKE": 0}, 0
    stage = {}
    t0 = time.perf_counter()
    try:
        for idx, frame, res, panels in run_session(a.source, cfg, pipeline=pipe):
            writer.write(panels, panel, idx)
            totals.append(res.timings_ms["total"])
            decisions[res.decision.state] += 1
            for k, v in res.timings_ms.items():
                stage.setdefault(k, []).append(v)
            n += 1
            if n % 20 == 0:
                log.info("frame %d | %d tracks | %s | %.0f ms", idx, len(res.tracks), res.decision.state, res.timings_ms["total"])
            wb.log({"latency_ms": res.timings_ms["total"], "n_tracks": len(res.tracks), "ego_speed_kmh": res.ego_speed * 3.6,
                    "brake": res.decision.brake}, step=idx)
    finally:
        writer.close()
    wall = time.perf_counter() - t0
    if n == 0:
        raise SystemExit("No frames were processed.")
    summary = {"source": str(a.source), "detector": getattr(pipe.detector, "name", "?"), "frames": n,
               "wall_fps": n / wall, "latency": latency_stats(totals), "stage_mean_ms": {k: sum(v) / len(v) for k, v in stage.items()},
               "decision_frames": decisions, "output": str(writer.path)}
    out_json = resolve_path("outputs/metrics") / f"run_{stem}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("done: %d frames, %.2f fps wall | output: %s | metrics: %s", n, n / wall, writer.path, out_json)
    wb.summary(summary["latency"])
    if writer.path and str(writer.path).endswith(".mp4"):
        wb.log_video("demo_video", writer.path, fps=int(fps))
    wb.finish()


if __name__ == "__main__":
    main()
