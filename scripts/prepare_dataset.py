"""Convert BDD100K or KITTI to YOLO format (7 classes) for fine-tuning.

  python scripts/prepare_dataset.py --dataset bdd100k --labels <det_train.json> --images <images/100k/train> --max-images 3000
  python scripts/prepare_dataset.py --dataset kitti --labels <training/label_2> --images <training/image_2>
Output: datasets/processed/<dataset>_yolo/{images,labels}/{train,val} + data.yaml
"""
import _bootstrap  # noqa: F401
import argparse

from src.data.convert import convert_records
from src.data.labels import parse_bdd100k, parse_kitti
from src.utils.config import resolve_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["bdd100k", "kitti"], required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-images", type=int, default=None, help="subsample (recommended on CPU / for quick tests)")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    a = ap.parse_args()
    imgs = resolve_path(a.images)
    recs = parse_bdd100k(resolve_path(a.labels)) if a.dataset == "bdd100k" else parse_kitti(resolve_path(a.labels), imgs)
    out = resolve_path(a.out or f"datasets/processed/{a.dataset}_yolo")
    stats = convert_records(recs, imgs, out, a.dataset, a.max_images, a.val_fraction)
    print(f"Wrote {out}: {stats}")
    if stats["train"] + stats["val"] == 0:
        raise SystemExit("No images converted - check that --images points at the folder containing the image files.")
    print(f"Use it with:  python scripts/train_detector.py --data {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
