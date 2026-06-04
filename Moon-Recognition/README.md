# Moon Feature Recognition — Mask R-CNN 

This branch is my contribution to the Moon Feature Recognition group project in
Computational Physics 2026. The project compares four different ways of segmenting
lunar features on Marius Hills tiles, one branch per student:

- Student 1 — thresholding + morphology (classical baseline)
- Student 2 — contour detection + watershed
- Student 3 — U-Net (semantic segmentation)
- **Student 4 — Mask R-CNN (instance segmentation, this branch)**

The point of my branch is to detect individual craters, pits and ridges as
separate objects with their own bounding box and mask, instead of one merged
semantic map.

## Files in this branch

- `amir_rcnn.ipynb` — the full notebook (preprocessing → model → training → evaluation → prediction).
- `metrics.json` — final validation metrics on the best checkpoint.
- `training_history.csv` / `.json` — train loss, val loss, val mAP and detection quality per epoch.
- `predictions_summary.json` — per-tile prediction counts and confidence scores for all 500 saved validation tiles.
- `figures/training_curves.png` — train/val loss, mAP@0.5, precision/recall/mask-IoU vs. epoch.
- `figures/best_model_10_prediction_samples_display.jpg` — ten validation tiles with the best model's predictions overlaid.
- `predictions/` — the per-tile prediction `.npz` files (boxes, labels, scores, masks).
- The trained weights (`best_model.pth`, ~200 MB) are attached to the GitHub Release of this branch, because GitHub doesn't accept files over 100 MB in a regular commit.

## What the notebook does

Starting from a ResNet-50 + FPN Mask R-CNN pretrained on COCO, I:

- Replaced the standard box and mask heads with **deeper custom heads** — a residual mask head with GroupNorm, and a 4-layer MLP box head. That's the architecture contribution.
- Scanned all 15,935 tiles and auto-selected the four mask channels that actually have enough data to train on: impact_crater, pit_skylight, wrinkle_ridge, lobate_scarp. The other three (irregular_mare_patch, apollo_site, candidate_rille) appear in fewer than 200 tiles each, so they get dropped automatically.
- Tuned anchors to small lunar scales (8 to 128 px) so the RPN can spot small craters as well as ridges.
- Used a discriminative learning rate (pretrained backbone at lr x 0.1, randomly-initialised custom heads at the full lr), a 1-epoch linear warmup + cosine schedule, gradient clipping, and AMP where the device supports it.
- Filtered the dense semantic crater channel per tile (max_channel_fraction = 0.20) so connected components only runs on tiles where craters are sparse enough to separate into individual instances.

The full run was 20 epochs over all tiles on a friend's CPU machine (around 13 hours).
Best checkpoint comes from **epoch 8**; after that the model overfits cleanly (val loss
climbs while train loss keeps falling).

## Final numbers

| metric | value |
|---|---|
| mAP@0.5 | 0.010 |
| precision | 0.10 |
| recall | 0.056 |
| F1 | 0.073 |
| mean mask IoU (matched objects) | 0.64 |
| mean mask Dice (matched objects) | 0.77 |
| count MAE | 6.3 |
| TP / FP / FN | 461 / 3,973 / 7,790 |

AP@0.5 per class:

| class | AP |
|---|---|
| impact_crater | 0.023 |
| wrinkle_ridge | 0.017 |
| pit_skylight | 0.0003 |
| lobate_scarp | 0.000 |

## What I learned from the result

The low mAP isn't a bug, and I want to explain why because it's actually the most
interesting part of this work:

- **The dataset is semantic, not instance.** Looking at the lunar_segmentation
  README and preprocessing.py, the masks were built for U-Net training:
  impact_crater is stored as filled circular polygons with no dilation, which
  in heavily cratered terrain fills ~82% of a tile on average. Connected
  components can only separate craters where they don't touch. This is exactly
  the limitation the lecture flags on p.54: "Mask R-CNN requires reliable
  instance targets, not only vague semantic layers."
- **Class imbalance hurts the rare classes.** Counting instances on 300 train
  tiles I got 1,160 craters and 1,547 wrinkle ridges, but only 28 pits and 21
  lobate scarps. The two rare classes never really got off the ground (AP near
  zero), which lines up with the "rare features" warning on p.27.
- **The mask shape is good when detection hits.** Mask IoU 0.64 and Dice 0.77
  on matched objects means the architecture works fine; the bottleneck is the
  detector finding the right candidates, not drawing the masks once it has them.
- **The training curves show clean overfitting starting around epoch 5**, so
  the early-stopped epoch-8 checkpoint is the right one to report.

So the comparison with Student 3's U-Net should be informative rather than
embarrassing: Mask R-CNN gives separated instances and good mask shapes; U-Net
should win on per-pixel coverage on the dense semantic channels.

## How to reproduce

1. Install dependencies:

       pip install -r requirements.txt

2. Point MR_ROOT_CANDIDATES in section 3 of the notebook at your local copy
   of the Marius Hills .npz tiles. The expected layout is:

       data/MR/tiles/index.csv
       data/MR/data/processed/tiles/marius_hills/*.npz

3. Download best_model.pth from the Release (link in the GitHub UI) and put
   it next to the notebook.
4. **Inference only** (recommended for everyone except me): run cells 0–9 to
   set things up, then the "Load best_model.pth" cell, then jump straight
   to the final evaluation, the 10-sample showcase, and the prediction export.
5. **Re-training from scratch** (~13 hours on a strong CPU, much less on GPU):
   skip the "Load best_model.pth" cell and run the training loop instead.
   It resumes from last_model.pth if one is present.

## Acknowledgements

The Marius Hills tiles, the lunar_segmentation preprocessing pipeline, and the
overall project structure were provided as part of the course. My contribution
is the Mask R-CNN side of the four-method comparison.

— *Amirmohammad Saiedi Saber*
