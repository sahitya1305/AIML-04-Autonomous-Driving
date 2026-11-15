"""Dataset statistics + plots: class distribution, balance, box-size statistics, per-attribute counts."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np

from src.data.labels import ImageRecord


def compute_stats(records: List[ImageRecord], name: str = "dataset") -> Dict:
    raw = Counter()
    areas: Dict[str, List[float]] = {}
    heights: Dict[str, List[float]] = {}
    per_image = []
    attrs: Dict[str, Counter] = {}
    for r in records:
        per_image.append(len(r.objects))
        for k, v in r.attributes.items():
            if isinstance(v, str):
                attrs.setdefault(k, Counter())[v] += 1
        for cat, (x1, y1, x2, y2) in r.objects:
            raw[cat] += 1
            areas.setdefault(cat, []).append(max(x2 - x1, 0) * max(y2 - y1, 0))
            heights.setdefault(cat, []).append(max(y2 - y1, 0))
    total = sum(raw.values())
    classes = {}
    for cat, n in raw.most_common():
        a = np.array(areas[cat]); h = np.array(heights[cat])
        classes[cat] = {"count": n, "share_pct": round(100 * n / total, 2) if total else 0.0,
                        "median_box_h_px": float(np.median(h)), "median_area_px2": float(np.median(a)),
                        "small_pct(<32x32)": round(100 * float((a < 32 * 32).mean()), 1)}
    counts = [v["count"] for v in classes.values()]
    return {"name": name, "n_images": len(records), "n_objects": total,
            "objects_per_image": {"mean": float(np.mean(per_image)) if per_image else 0.0,
                                   "median": float(np.median(per_image)) if per_image else 0.0,
                                   "max": int(max(per_image)) if per_image else 0},
            "classes": classes,
            "imbalance_ratio_max_over_min": (max(counts) / max(min(counts), 1)) if counts else None,
            "attributes": {k: dict(v) for k, v in attrs.items()}}


def stats_to_markdown(s: Dict) -> str:
    lines = [f"# Dataset analysis: {s['name']}", "",
             f"- images: **{s['n_images']}**, objects: **{s['n_objects']}**, objects/image: mean {s['objects_per_image']['mean']:.1f}, max {s['objects_per_image']['max']}",
             f"- class imbalance (largest / smallest class): **{s['imbalance_ratio_max_over_min']:.0f}x**" if s["imbalance_ratio_max_over_min"] else "",
             "", "| class | count | share % | median box height (px) | small (<32x32) % |", "|---|---:|---:|---:|---:|"]
    for c, v in s["classes"].items():
        lines.append(f"| {c} | {v['count']} | {v['share_pct']} | {v['median_box_h_px']:.0f} | {v['small_pct(<32x32)']} |")
    for k, d in s["attributes"].items():
        lines += ["", f"**{k}**: " + ", ".join(f"{a}={b}" for a, b in sorted(d.items(), key=lambda x: -x[1]))]
    return "\n".join(lines) + "\n"


def plot_stats(s: Dict, records: List[ImageRecord], out_dir) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    names = list(s["classes"].keys()); counts = [s["classes"][c]["count"] for c in names]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(names, counts, color="#3b78c3"); ax[0].set_title("Objects per class"); ax[0].tick_params(axis="x", rotation=45)
    ax[1].bar(names, counts, color="#c3573b"); ax[1].set_yscale("log"); ax[1].set_title("Objects per class (log scale)"); ax[1].tick_params(axis="x", rotation=45)
    fig.suptitle(f"{s['name']}: class distribution / balance"); fig.tight_layout()
    p = out_dir / f"{s['name']}_class_distribution.png"; fig.savefig(p, dpi=120); plt.close(fig); paths.append(str(p))
    hs = {}
    for r in records:
        for cat, (x1, y1, x2, y2) in r.objects:
            hs.setdefault(cat, []).append(y2 - y1)
    top = [c for c in names[:6] if c in hs]
    if top:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot([hs[c] for c in top], showfliers=False); ax.set_xticks(range(1, len(top) + 1)); ax.set_xticklabels(top); ax.set_ylabel("box height (px)")
        ax.set_title(f"{s['name']}: object size by class"); fig.tight_layout()
        p = out_dir / f"{s['name']}_box_heights.png"; fig.savefig(p, dpi=120); plt.close(fig); paths.append(str(p))
    return paths
