"""Dataset exploration: class distribution, balance, box sizes (BDD100K or KITTI).

  python scripts/analyze_dataset.py --demo                                   # no download: fake labels, tests the script only
  python scripts/analyze_dataset.py --dataset bdd100k --labels datasets/raw/bdd100k/labels/det_20/det_train.json
  python scripts/analyze_dataset.py --dataset kitti --labels datasets/raw/kitti/training/label_2
Outputs: outputs/metrics/<name>_stats.json|.md and outputs/visualizations/<name>_*.png
"""
import _bootstrap  # noqa: F401
import argparse
import json

from src.data.analysis import compute_stats, plot_stats, stats_to_markdown
from src.data.labels import parse_bdd100k, parse_kitti
from src.utils.config import resolve_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["bdd100k", "kitti"], default="bdd100k")
    ap.add_argument("--labels", help="BDD100K det json, or KITTI label_2 folder")
    ap.add_argument("--images", help="(KITTI) image_2 folder, optional, used to read true image sizes")
    ap.add_argument("--demo", action="store_true", help="use the synthetic mini label file shipped with the project")
    ap.add_argument("--max-images", type=int)
    a = ap.parse_args()
    if a.demo:
        recs = parse_bdd100k(resolve_path("datasets/samples/mini_bdd_labels_SYNTHETIC.json"))
        name = "demo_SYNTHETIC"
        print("DEMO MODE: these labels are synthetic, only the script is being demonstrated.")
    else:
        if not a.labels:
            raise SystemExit("Provide --labels (or use --demo). See docs/DATASETS.md for download instructions.")
        lp = resolve_path(a.labels)
        recs = parse_bdd100k(lp) if a.dataset == "bdd100k" else parse_kitti(lp, resolve_path(a.images) if a.images else None)
        name = a.dataset
    if a.max_images:
        recs = recs[: a.max_images]
    s = compute_stats(recs, name)
    out_m = resolve_path("outputs/metrics"); out_m.mkdir(parents=True, exist_ok=True)
    (out_m / f"{name}_stats.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    md = stats_to_markdown(s)
    (out_m / f"{name}_stats.md").write_text(md, encoding="utf-8")
    plots = plot_stats(s, recs, resolve_path("outputs/visualizations"))
    print(md)
    print("plots:", *plots, sep="\n  ")


if __name__ == "__main__":
    main()
