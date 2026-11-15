"""Obstacle-detection + reaction demo (closed loop, synthetic world). Writes a video and a speed/distance plot.

  python scripts/run_closed_loop_demo.py --scenario stopped
  python scripts/run_closed_loop_demo.py --scenario pedestrian --panel mosaic --wandb
Scenarios: default | stopped | pedestrian | empty
"""
import _bootstrap  # noqa: F401
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.simulator.synthetic_sim import run_closed_loop
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.utils.wandb_utils import WandbLogger

log = get_logger("closed-loop")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="stopped", choices=["default", "stopped", "pedestrian", "empty"])
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--panel", default="combined", choices=["combined", "mosaic", "detections", "lanes"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.wandb:
        cfg["wandb"]["enabled"] = True
    vid = f"outputs/videos/closed_loop_{a.scenario}.mp4"
    csv_path = f"outputs/metrics/closed_loop_{a.scenario}.csv"
    wb = WandbLogger.from_config(cfg, "closed_loop_demo", f"closed-loop-{a.scenario}")
    out = run_closed_loop(cfg, a.scenario, a.seconds, 20, vid, csv_path, a.panel)
    rows = out.pop("rows")
    t = [r["t"] for r in rows]
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(t, [r["speed_mps"] * 3.6 for r in rows]); ax[0].set_ylabel("ego speed (km/h)")
    ax[1].plot(t, [r["true_gap_m"] if r["true_gap_m"] is not None else float("nan") for r in rows], label="true gap")
    ax[1].plot(t, [r["est_dist_m"] if r["est_dist_m"] is not None else float("nan") for r in rows], "--", label="estimated distance")
    ax[1].set_ylabel("distance to obstacle in lane (m)"); ax[1].set_xlabel("time (s)"); ax[1].legend()
    ax[0].set_title(f"Closed-loop obstacle response - scenario '{a.scenario}' (synthetic world)")
    plot = resolve_path(f"outputs/visualizations/closed_loop_{a.scenario}.png")
    plot.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(plot, dpi=120); plt.close(fig)
    resolve_path(f"outputs/metrics/closed_loop_{a.scenario}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("result: %s", out)
    log.info("video: %s | plot: %s | csv: %s", vid, plot, csv_path)
    for r in rows:
        wb.log({"speed_kmh": r["speed_mps"] * 3.6, "true_gap_m": r["true_gap_m"] or 0.0}, step=int(r["t"] * 20))
    wb.summary({k: v for k, v in out.items() if isinstance(v, (int, float, bool, str))})
    wb.log_video("closed_loop_video", resolve_path(vid), fps=20)
    wb.finish()
    if out["collision"]:
        raise SystemExit("COLLISION occurred in the simulation (see CSV).")


if __name__ == "__main__":
    main()
