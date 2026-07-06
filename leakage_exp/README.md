# Split-protocol study (YOLO ensemble members)

Quantifies how the train/validation split protocol affects reported detection
scores on the Marius Hills tiles, whose 128-px stride makes adjacent tiles
overlap by 50%. Companion analysis to `Moon-Recognition/comparison/`; the
results appear in the report's "Effect of the split protocol" section.

Two single-class YOLOv8n detectors (the crater and wrinkle-ridge members of
the decoupled ensemble) are trained twice each with identical hyperparameters
(pretrained `yolov8n.pt`, imgsz 256, batch 16, seed 0), changing only the
split used to build the dataset:

* **random** — random 80/20 tile-level split (`train_test_split`, seed 42),
  the protocol of the original YOLO preprocessing notebook;
* **spatial** — the repository's leakage-free spatial block split
  (`spatial_train_val_split`, block 1024 px, seed 42), the shared protocol of
  `Moon-Recognition/comparison/`.

Box extraction reproduces `preprocessing.ipynb` verbatim (connected
components, size gates 5 < w,h < 100 px for craters, < 300 px otherwise).

## Results

`pair_results.csv` (40 epochs, matched budget):

| class | random (own val) | spatial (own val) | random model on spatial val |
|---|---|---|---|
| impact crater | 0.122 | 0.106 | 0.129 |
| wrinkle ridge | 0.146 | 0.077 | 0.117 |

mAP50. The third column evaluates the random-split model on the spatial
validation tiles, isolating the training-side component of the leakage.

The extended-budget pair (`wrinkle_ridge_*_e150`, 150 epochs) shows the gap
growing with training: random 0.217 (still rising at epoch 150) vs.
spatial 0.082 (flat after ~40 epochs). `fig_split_divergence.png` plots the
two validation curves (regenerate with `make_divergence_figure.py`).

## Files

* `build_datasets.py` — builds the four YOLO datasets (`<class>_<protocol>/`)
  from `data/MR/tiles/index.csv` and the NPZ tiles.
* `train_pair.py` — trains the four 40-epoch models sequentially and writes
  `pair_results.csv`. Safe to re-run after an interruption: completed runs
  are skipped (`DONE` markers) and partial runs resume from `last.pt`.
* `train_long_ridge.py` — the 150-epoch extended-budget ridge pair.
* `make_divergence_figure.py` — renders `fig_split_divergence.png` from the
  two `runs/wrinkle_ridge_*_e150/results.csv` training logs.
* `southpole_fixed_loader_check.py` — re-runs the south-pole segmentation
  inference on a 300-tile sample through the package `Predictor` (corrected
  checkpoint loading + training preprocessing) and writes per-class
  aggregate statistics to `southpole_fixed_loader_stats.json`, for
  comparison against the reference south-pole table in the report.
* `runs/<name>/` — per-run training artifacts: `args.yaml` (full ultralytics
  config), `results.csv` (per-epoch metrics), `weights/best.pt`, and the
  standard curve/confusion plots. Datasets and intermediate images are not
  committed; rebuild them with `build_datasets.py`.

## Reproduce

From the repository root (env: `stellar`; the scripts auto-select
cuda / mps / cpu):

```bash
KMP_DUPLICATE_LIB_OK=TRUE python leakage_exp/build_datasets.py
KMP_DUPLICATE_LIB_OK=TRUE python leakage_exp/train_pair.py
KMP_DUPLICATE_LIB_OK=TRUE python leakage_exp/train_long_ridge.py   # optional, long
python leakage_exp/make_divergence_figure.py
KMP_DUPLICATE_LIB_OK=TRUE python leakage_exp/southpole_fixed_loader_check.py
```

`KMP_DUPLICATE_LIB_OK=TRUE` works around the duplicate-libomp issue on macOS.
To re-score a trained model without retraining, load
`runs/<name>/weights/best.pt` with `ultralytics.YOLO` and call `.val()` on
the corresponding `data.yaml`.
