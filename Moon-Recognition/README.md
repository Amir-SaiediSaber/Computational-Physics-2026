# Moon Recognition (MR) — U-Net lunar feature segmentation

Multi-label pixel-wise segmentation of seven lunar surface feature classes
from LRO WAC tiles (Marius Hills, 15,931 supplied 256×256 tiles), with a
configurable SmallUNet and a full ablation campaign. Report section:
`report/moon_recognition.tex`.

## Layout

```
Moon-Recognition/
├── notebooks/
│   ├── architecture_comparison.ipynb       Studies 1–3: loss / residual / dropout /
│   │                                       width / augmentation ablations
│   └── south_pole_complete_workflow.ipynb  out-of-distribution inference (3,639 tiles)
├── lunar_segmentation/                     package, configs, scripts (see its README)
└── IMPROVEMENTS.md                         audited fixes: issue → fix → justification
```

Data lives in the repo-level `data/MR/` (gitignored): `tiles/` + `index.csv`
(supplied dataset), `weights/`, `results/` (checkpoints, curves, logs).

## Reproducing the results

1. **Environment**: `pip install -r lunar_segmentation/requirements.txt`
   (conda env `stellar` on the development machine).
2. **Training / ablations**: run `notebooks/architecture_comparison.ipynb`.
   Top cell toggles: `MAX_TILES=500, MAX_EPOCHS=3` (smoke test) →
   `MAX_TILES=None, MAX_EPOCHS=30` (definitive CloudVeneto T4 protocol,
   see `lunar_segmentation/configs/unet_config.yaml`). The definitive runs
   (~24 h total) produced `data/MR/results/cloudveneto_train.log` and the
   checkpoints used by the report.
3. **Qualitative figures** (report Sec. "Qualitative Evaluation"):
   `python lunar_segmentation/scripts/qualitative_eval.py` (minutes on MPS;
   needs the full-run checkpoints in `data/MR/results/checkpoints/`).
4. **South-pole inference**: `notebooks/south_pole_complete_workflow.ipynb`.

## Known caveats (tracked in IMPROVEMENTS.md)

- The published ablations used a **random** tile split; tiles overlap 50%,
  so validation leaks (100% of val tiles overlap a train tile — measured).
  A leakage-free spatial block split is implemented in
  `lunar_segmentation/lunar_segmentation/data/splits.py`; Study 3
  (augmentation) must be re-validated with it (≈4–7 h on the T4).
- The `impact_crater` mask channel covers 82.1% of all pixels (near-coverage
  layer): crater AP ≈ 0.95 must be read against its 0.821 chance baseline.
  The strongest genuine result is `wrinkle_ridge` (AP 0.558 ≈ 90× chance).
