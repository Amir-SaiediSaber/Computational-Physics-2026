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

## Environment
The `stellar` conda env runs everything in this project:
`torch torchvision ultralytics opencv-python rasterio scikit-image scikit-learn imbalanced-learn shap lightgbm catboost`.
On macOS, set `KMP_DUPLICATE_LIB_OK=TRUE` to avoid the libomp double-init abort.

## Reproduce all models
```bash
KMP_DUPLICATE_LIB_OK=TRUE python verify_models.py
```
Smoke-tests every model (loads weights + runs a few samples). The Mask R-CNN
weights (194 MB) are not in git — fetch once with
`gh release download v1.0 -R Amir-SaiediSaber/Computational-Physics-2026 -p best_model.pth -D data/MR/weights/`.

## Report
See `report/` for the written comparison.
