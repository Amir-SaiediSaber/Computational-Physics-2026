# Computational Physics 2026 — Group Project

Two recognition tasks tackled with several methods each, one branch per student,
merged into `main` for a combined comparison.

## Sub-projects

### 1. Astrophysical Object Recognition (`Astrophysical-Objects-Recognition/`)
Classification of SDSS objects into **GALAXY / QSO / STAR** using traditional ML
(SVC, Decision Tree, Random Forest, CatBoost, LightGBM), a voting ensemble, and a
PyTorch neural net, with SHAP/permutation-importance interpretability.
Packaged as the installable `stellar_classification` module.
- Data: `data/AOR/star_classification.csv`

### 2. Moon Feature Recognition (`Moon-Recognition/`)
Segmenting/detecting lunar features (craters, ridges, pits, …) on Marius Hills
tiles. Four approaches are compared on shared tiles:

| Method | Location | Type |
|---|---|---|
| Classical (thresholding + morphology) | `Moon-Recognition/notebooks/` | baseline |
| YOLOv8 detection | `Moon-Recognition/yolo/` | object detection |
| U-Net | `Moon-Recognition/lunar_segmentation/` | semantic segmentation |
| Mask R-CNN | `Moon-Recognition/notebooks/amir_rcnn.ipynb` | instance segmentation |

- Data: `data/MR/` (tiles, weights, results) — not tracked in git (large).
- Trained weights >100 MB are attached to the relevant GitHub Release, not committed.

## Environments
- `stellar` — AOR stellar classification
- `comp_lab` — Moon-Recognition (torch + torchvision + rasterio). YOLO additionally needs `pip install ultralytics`.

## Report
See `report/` for the written comparison.
