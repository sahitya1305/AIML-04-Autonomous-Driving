"""Evaluate a detector on a YOLO-format validation set (mAP, precision, recall) and log to W&B.

  python scripts/evaluate_detector.py --data datasets/processed/bdd100k_yolo/data.yaml --weights experiments/yolo11n_bdd100k/weights/best.pt
Comparing the pretrained COCO model with the fine-tuned model: run twice with different --weights and --tag.
Note: a COCO-pretrained model has COCO class names/ids; evaluating it on our 7-class dataset is only meaningful after
fine-tuning (class ids differ), so use this mainly for your fine-tuned weights.
"""
import _bootstrap  # noqa: F401
import argparse
import json

from src.utils.config import load_config, resolve_path
from src.utils.wandb_utils import WandbLogger


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="val"); ap.add_argument("--tag", default="eval"); ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed: pip install -r requirements-ml.txt")
    cfg = load_config(); cfg["wandb"]["enabled"] = a.wandb or cfg["wandb"]["enabled"]
    wb = WandbLogger.from_config(cfg, "evaluate", f"{a.tag}")
    model = YOLO(str(resolve_path(a.weights)))
    r = model.val(data=str(resolve_path(a.data)), imgsz=a.imgsz, device=a.device, split=a.split, plots=True)
    out = {"weights": a.weights, "data": a.data, "split": a.split, "mAP50-95": float(r.box.map), "mAP50": float(r.box.map50),
           "precision": float(r.box.mp), "recall": float(r.box.mr),
           "per_class_mAP50-95": {str(model.names[i]): float(v) for i, v in zip(r.box.ap_class_index, r.box.maps[r.box.ap_class_index])}}
    resolve_path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    p = resolve_path("outputs/metrics") / f"eval_{a.tag}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    wb.log({k: v for k, v in out.items() if isinstance(v, float)}); wb.summary({k: v for k, v in out.items() if isinstance(v, float)}); wb.finish()
    print(json.dumps(out, indent=2)); print("saved", p)


if __name__ == "__main__":
    main()
