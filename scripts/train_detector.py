"""Fine-tune YOLO on a YOLO-format dataset (output of scripts/prepare_dataset.py) and log to Weights & Biases.

  python scripts/train_detector.py --data datasets/processed/bdd100k_yolo/data.yaml --epochs 30 --device cpu
  python scripts/train_detector.py --data datasets/samples/synthetic_yolo/data.yaml --smoke      # 1-epoch plumbing check
On a laptop CPU use a small subset (--max-images in prepare_dataset) or train on Colab/Kaggle GPU (notebooks/02_*).
The result is only 'fine-tuned' if this script actually ran to completion; check experiments/<name>/weights/best.pt.
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.utils.wandb_utils import WandbLogger

import yaml

log = get_logger("train")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--data"); ap.add_argument("--model"); ap.add_argument("--epochs", type=int)
    ap.add_argument("--imgsz", type=int); ap.add_argument("--batch", type=int); ap.add_argument("--device")
    ap.add_argument("--name"); ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, tiny image size: verifies the workflow only")
    a = ap.parse_args()
    with open(resolve_path(a.config), "r", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    for k in ("data", "model", "epochs", "imgsz", "batch", "device", "name"):
        if getattr(a, k) is not None:
            c[k] = getattr(a, k)
    if a.wandb:
        c["wandb"]["enabled"] = True
    if a.smoke:
        c.update(epochs=1, imgsz=160, batch=8, name=c["name"] + "_smoke")
    data = resolve_path(c["data"])
    if not data.exists():
        raise SystemExit(f"Dataset yaml not found: {data}\nCreate it with scripts/prepare_dataset.py (see docs/DATASETS.md).")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed: pip install -r requirements-ml.txt")

    model_path = resolve_path(c["model"])
    model = YOLO(str(model_path) if model_path.exists() else c["model"])
    wb = WandbLogger.from_config({"wandb": c["wandb"], "train": {k: v for k, v in c.items() if k != "wandb"}}, "train", c["name"])

    def on_epoch_end(trainer):   # called by ultralytics after every epoch (train + val finished)
        try:
            m = {k.replace("(", "_").replace(")", ""): float(v) for k, v in trainer.metrics.items()}
            m.update({f"train/{k}": float(v) for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items()})
            m["lr"] = float(trainer.optimizer.param_groups[0]["lr"])
            wb.log(m, step=trainer.epoch)
        except Exception as e:
            log.warning("W&B epoch logging failed: %s", e)

    model.add_callback("on_fit_epoch_end", on_epoch_end)
    log.info("Training %s on %s for %s epochs (device=%s)", c["model"], data, c["epochs"], c["device"])
    model.train(data=str(data), epochs=c["epochs"], imgsz=c["imgsz"], batch=c["batch"], device=c["device"],
                workers=c.get("workers", 2), project=str(resolve_path(c["project"])), name=c["name"], patience=c.get("patience", 10),
                seed=c.get("seed", 0), exist_ok=True, plots=True)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    log.info("Done. Best weights: %s", best)
    summary = {"save_dir": str(save_dir), "best": str(best), "smoke": a.smoke, "final_metrics": {k: float(v) for k, v in model.trainer.metrics.items()}}
    resolve_path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    (resolve_path("outputs/metrics") / f"train_{c['name']}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    wb.summary(summary["final_metrics"])
    for plot in ("results.png", "confusion_matrix.png", "PR_curve.png"):
        wb.log_file(save_dir / plot)
    wb.finish()
    print(f"\nTo use the fine-tuned model:  python scripts/run_pipeline.py --source <video> --set detector.weights={best.as_posix()}")


if __name__ == "__main__":
    main()
