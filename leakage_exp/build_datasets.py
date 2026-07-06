"""Build paired YOLO datasets for the split-protocol experiment.

For each single-class detector (impact_crater, wrinkle_ridge) two datasets
are produced from the same tiles and the same box-extraction code:

  * ``<class>_random``  — random 80/20 tile-level split (train_test_split,
    seed 42), i.e. the protocol used in the YOLO preprocessing notebook.
    Tiles overlap (stride 128 = 50%), so adjacent train/val tiles share
    pixels.
  * ``<class>_spatial`` — the repo's leakage-free spatial block split
    (spatial_train_val_split, seed 42), identical box extraction.

Box extraction reproduces mask_to_yolo_boxes_multi from the YOLO
preprocessing notebook verbatim (connected components, size gate
lth < w,h < th with th=100 for craters, th=300 otherwise).
"""
import os
import sys
import csv
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "Moon-Recognition", "lunar_segmentation"))
from lunar_segmentation.data.splits import spatial_train_val_split  # noqa: E402

INDEX_CSV = os.path.join(ROOT, "data/MR/tiles/index.csv")
OUT_BASE = os.path.join(ROOT, "leakage_exp")

CLASSES = {
    "impact_crater": {"channel": 0, "th": 100},
    "wrinkle_ridge": {"channel": 2, "th": 300},
}
LTH = 5
SEED = 42


def mask_to_yolo_boxes_multi(single_mask, class_id, img_w, img_h,
                             connectivity=1, lth=LTH, th=300):
    labeled, num = label(single_mask,
                         structure=None if connectivity == 1 else np.ones((3, 3)))
    boxes = []
    for comp_id in range(1, num + 1):
        ys, xs = np.where(labeled == comp_id)
        if len(xs) == 0:
            continue
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        box_w = x_max - x_min + 1
        box_h = y_max - y_min + 1
        if (box_w < th and box_h < th) and (box_w > lth and box_h > lth):
            x_c = x_min + box_w / 2.0
            y_c = y_min + box_h / 2.0
            boxes.append([class_id, x_c / img_w, y_c / img_h,
                          box_w / img_w, box_h / img_h])
    return boxes


def extract_boxes(index_df):
    """One pass over all tiles; returns {class_name: {tile_path: [boxes]}}."""
    per_class = {c: {} for c in CLASSES}
    n = len(index_df)
    for i, row in enumerate(index_df.itertuples()):
        npz_path = os.path.join(ROOT, "data/MR", row.tile_path)
        with np.load(npz_path) as data:
            masks = data["mask"]
        H, W = masks.shape[1], masks.shape[2]
        for cname, cfg in CLASSES.items():
            m = masks[cfg["channel"]] > 0
            if not m.any():
                continue
            boxes = mask_to_yolo_boxes_multi(m, 0, W, H, th=cfg["th"])
            if boxes:
                per_class[cname][row.tile_path] = boxes
        if (i + 1) % 2000 == 0:
            print(f"  boxes: {i+1}/{n} tiles scanned", flush=True)
    return per_class


def write_dataset(name, cname, train_tiles, val_tiles, boxes_by_tile):
    out_root = os.path.join(OUT_BASE, name)
    for split in ["train", "val"]:
        os.makedirs(os.path.join(out_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_root, "labels", split), exist_ok=True)

    def save_one(tile_path, split):
        npz_path = os.path.join(ROOT, "data/MR", tile_path)
        with np.load(npz_path) as data:
            img = data["image"]
        img_hwc = np.transpose(img, (1, 2, 0))
        if img_hwc.dtype != np.uint8:
            if img_hwc.max() <= 1.0:
                img_hwc = (img_hwc * 255).clip(0, 255).astype(np.uint8)
            else:
                img_hwc = img_hwc.clip(0, 255).astype(np.uint8)
        base = os.path.splitext(os.path.basename(tile_path))[0]
        Image.fromarray(img_hwc).save(
            os.path.join(out_root, "images", split, base + ".png"))
        with open(os.path.join(out_root, "labels", split, base + ".txt"), "w") as f:
            for cls, xc, yc, w, h in boxes_by_tile[tile_path]:
                f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    for t in train_tiles:
        save_one(t, "train")
    for t in val_tiles:
        save_one(t, "val")

    with open(os.path.join(out_root, "data.yaml"), "w") as f:
        f.write(f"path: {out_root}\ntrain: images/train\nval: images/val\n"
                f"names:\n  0: {cname}\n")
    print(f"{name}: train={len(train_tiles)} val={len(val_tiles)}", flush=True)


def main():
    index_df = pd.read_csv(INDEX_CSV)
    print(f"{len(index_df)} tiles in index", flush=True)

    print("Extracting boxes (single pass over all npz)...", flush=True)
    per_class = extract_boxes(index_df)
    for cname, d in per_class.items():
        nb = sum(len(v) for v in d.values())
        print(f"  {cname}: {len(d)} tiles with boxes, {nb} boxes", flush=True)

    # Spatial membership (computed once on the full index, as in the
    # shared comparison protocol)
    train_sp, val_sp = spatial_train_val_split(
        index_df, val_fraction=0.2, tile_size=256, block_px=1024, seed=SEED)
    train_sp_set = set(train_sp["tile_path"])
    val_sp_set = set(val_sp["tile_path"])

    for cname, boxes_by_tile in per_class.items():
        tiles = sorted(boxes_by_tile.keys())

        # Arm 1: random tile-level 80/20 split (notebook protocol)
        tr, va = train_test_split(tiles, test_size=0.2, random_state=SEED)
        write_dataset(f"{cname}_random", cname, tr, va, boxes_by_tile)

        # Arm 2: spatial block split, same box extraction
        tr_s = [t for t in tiles if t in train_sp_set]
        va_s = [t for t in tiles if t in val_sp_set]
        write_dataset(f"{cname}_spatial", cname, tr_s, va_s, boxes_by_tile)


if __name__ == "__main__":
    main()
