"""Train the tiny U-Net lane segmentation model.

  python scripts/train_lane_seg.py --synthetic 300 --epochs 8     # pipeline sanity check on CARTOON data (NOT real accuracy)
  python scripts/train_lane_seg.py --images D:/tusimple/images --masks D:/tusimple/masks --epochs 20   # real data
For TuSimple use scripts/tusimple_to_masks.py first. Output: models/lane_seg/lane_unet.pt
"""
import _bootstrap  # noqa: F401
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.utils.wandb_utils import WandbLogger

log = get_logger("lane-train")


def synthetic_pairs(n, seed=0):
    from src.data.synthetic import SyntheticScene, build_scenario
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        sc = SyntheticScene(seed=rng.randint(0, 10**6), brightness=rng.uniform(0.6, 1.2), noise=rng.uniform(0, 8),
                            curvature=rng.uniform(-0.0006, 0.0006), horizon=rng.uniform(0.42, 0.48))
        sc.travelled = rng.uniform(0, 7)
        for _ in range(rng.randint(0, 3)):
            sc.add(rng.choice(["car", "truck", "person"]), rng.choice([-3.5, 0.0, 3.5]), rng.uniform(8, 60))
        img, gt = sc.render()
        xs.append(img); ys.append(gt["lane_mask"])
    return xs, ys


def folder_pairs(img_dir, mask_dir):
    xs, ys = [], []
    for ip in sorted(Path(img_dir).glob("*")):
        mp = Path(mask_dir) / (ip.stem + ".png")
        if ip.suffix.lower() in (".jpg", ".jpeg", ".png") and mp.exists():
            xs.append(cv2.imread(str(ip))); ys.append(cv2.imread(str(mp), 0))
    return xs, ys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", type=int, default=0); ap.add_argument("--images"); ap.add_argument("--masks")
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--out", default="models/lane_seg/lane_unet.pt")
    a = ap.parse_args()
    try:
        import torch
        from src.lane_detection.segmentation import INPUT_H, INPUT_W, TinyUNet, preprocess_bgr
    except ImportError:
        raise SystemExit("PyTorch missing: pip install -r requirements-ml.txt")
    if a.synthetic:
        imgs, masks = synthetic_pairs(a.synthetic); source = f"synthetic({a.synthetic})"
        log.warning("Training on SYNTHETIC data: this only checks the workflow, not real-road accuracy.")
    elif a.images and a.masks:
        imgs, masks = folder_pairs(resolve_path(a.images), resolve_path(a.masks)); source = a.images
    else:
        raise SystemExit("Give --synthetic N or --images DIR --masks DIR")
    if len(imgs) < 4:
        raise SystemExit("Not enough training pairs found.")
    X = torch.from_numpy(np.stack([preprocess_bgr(i) for i in imgs]))
    Y = torch.from_numpy(np.stack([(cv2.resize(m, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA) > 40).astype(np.float32) for m in masks]))[:, None]
    n_val = max(1, len(X) // 10)
    Xv, Yv, Xt, Yt = X[:n_val], Y[:n_val], X[n_val:], Y[n_val:]
    model = TinyUNet(); opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    pos_w = torch.tensor([float((1 - Yt.mean()) / max(Yt.mean(), 1e-4))]).clamp(1, 60)   # lane pixels are rare
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

    def dice_loss(logit, y):
        p = torch.sigmoid(logit); inter = (p * y).sum((1, 2, 3)); return (1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).mean()

    def iou(logit, y):
        p = (torch.sigmoid(logit) > 0.5).float(); i = (p * y).sum(); u = ((p + y) > 0).float().sum(); return float(i / u.clamp(min=1))

    cfg = load_config(); cfg["wandb"]["enabled"] = a.wandb or cfg["wandb"]["enabled"]
    wb = WandbLogger.from_config(cfg, "train_lane_seg", "lane-unet", {"source": source, "epochs": a.epochs})
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(Xt)); tot = 0.0
        for i in range(0, len(Xt), a.batch):
            idx = perm[i:i + a.batch]; x, y = Xt[idx], Yt[idx]
            if random.random() < 0.5:
                x, y = x.flip(3), y.flip(3)
            logit = model(x); loss = bce(logit, y) + dice_loss(logit, y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            v_iou = iou(model(Xv), Yv)
        log.info("epoch %d/%d  loss %.4f  val IoU %.3f", ep + 1, a.epochs, tot / len(Xt), v_iou)
        wb.log({"lane/train_loss": tot / len(Xt), "lane/val_iou": v_iou}, step=ep)
    out = resolve_path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": {"source": source, "epochs": a.epochs, "val_iou": v_iou, "input": [INPUT_W, INPUT_H]}}, out)
    log.info("saved %s (val IoU %.3f on %s - NOT a real-world number if source is synthetic)", out, v_iou, source)
    wb.summary({"lane/final_val_iou": v_iou}); wb.finish()


if __name__ == "__main__":
    main()
