# Datasets

**No real dataset is bundled.** The ZIP contains only synthetic cartoon data produced by `scripts/make_demo_data.py` (licence: ours, free to use) so everything runs offline.

## BDD100K (recommended, matches the assignment)
1. Register and download from https://bdd-data.berkeley.edu (the "100K Images" and "Labels" -> "Detection 2020" files). Full set is ~6 GB images; you do **not** need all of it.
2. Extract to `datasets/raw/bdd100k/` so you have `images/100k/train/*.jpg` and `labels/det_20/det_train.json`.
3. Explore: `python scripts/analyze_dataset.py --dataset bdd100k --labels datasets/raw/bdd100k/labels/det_20/det_train.json`
4. Convert a CPU-friendly subset: `python scripts/prepare_dataset.py --dataset bdd100k --labels <det_train.json> --images datasets/raw/bdd100k/images/100k/train --max-images 3000`
5. Train: Colab notebook `02_finetune_yolo_colab.ipynb` (recommended) or `python scripts/train_detector.py --data datasets/processed/bdd100k_yolo/data.yaml --device cpu --epochs 5` (slow).

Known BDD100K properties to discuss in your report (verify with your own analysis output): strong class imbalance (car dominates; rider, motorcycle, bicycle, bus are rare), many small objects, mixed weather/time-of-day attributes (provided in the labels and summarised by the analysis script).

## KITTI (alternative)
Download "left color images" and "training labels" of the 2D object benchmark from https://www.cvlibs.net/datasets/kitti . Then `analyze_dataset.py --dataset kitti --labels .../label_2 --images .../image_2` and `prepare_dataset.py --dataset kitti ...`. KITTI has no bus/motorcycle classes.

## Lane data for the segmentation model
TuSimple or CULane (not bundled). `scripts/tusimple_to_masks.py` converts TuSimple JSON labels to masks; then `scripts/train_lane_seg.py --images ... --masks ...`.

## Preprocessing done by the scripts
Class mapping to 7 unified classes, box clipping and normalisation to YOLO txt, random 80/20 split (seeded), optional subsampling. No augmentation beyond Ultralytics' defaults.

## Licensing
BDD100K and KITTI have their own licences (research/non-commercial terms) - read them before redistributing. This repository's `.gitignore` excludes raw/processed data.
