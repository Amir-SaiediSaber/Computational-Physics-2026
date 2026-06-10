# Computational Physics 2026 — Group 2614

Astrophysics machine-learning projects for the Laboratory of Computational
Physics (Physics of Data, University of Padova). Base pipeline by
prof. T. Zingales; model work by group members on per-model branches.

## Repository map

```
Computational-Physics-2026/
├── Astrophysical-Objects-Recognition/   AOR — stellar object classification (SDSS)
│   ├── notebooks/                       main analysis notebook
│   └── stellar_classification/         python package (see its README)
├── Moon-Recognition/                    MR — lunar surface feature segmentation (LRO WAC)
│   ├── notebooks/                       training/ablation + south-pole inference notebooks
│   ├── lunar_segmentation/             python package + configs + scripts (see its README)
│   └── IMPROVEMENTS.md                 audited fixes log: issue → fix → justification
├── report/                              LaTeX report sections + exported figures
│   ├── moon_recognition.tex            MR section (\input from the main document)
│   └── preview.tex                     standalone wrapper to compile the MR section
└── data/                                datasets, weights, results — NOT tracked (gitignored)
    ├── AOR/star_classification.csv
    └── MR/{tiles, weights, results, lunar_south_pole}
```

## Sub-projects

- **AOR — Astrophysical Objects Recognition**: 3-class stellar classification
  (dwarf / giant / white dwarf) with traditional ML, a voting ensemble, and a
  PyTorch network, plus SHAP interpretability.
- **MR — Moon Recognition**: multi-label pixel-wise segmentation of seven
  lunar surface feature classes from LRO WAC tiles of the Marius Hills region
  with a configurable SmallUNet, ablation studies (loss, residuals, dropout,
  width, augmentation), and out-of-distribution inference on the lunar south
  pole. See `Moon-Recognition/README.md` for reproduction instructions.

## Setup

```bash
pip install -r Moon-Recognition/lunar_segmentation/requirements.txt          # MR
pip install -r Astrophysical-Objects-Recognition/stellar_classification/requirements.txt  # AOR
```

Definitive MR training runs were executed on CloudVeneto (NVIDIA Tesla T4);
notebooks expose `MAX_TILES` / `MAX_EPOCHS` toggles for local prototyping
(Apple Silicon MPS or CPU).

## Git workflow

Per course guidance: one branch per model/member (`U-net`, Mask R-CNN, …),
pull requests into `main`, no large data files or checkpoints in the repo
(`data/` is gitignored). The MR/U-Net work lives on the `U-net` branch;
every fix is an individually justified commit, indexed in
`Moon-Recognition/IMPROVEMENTS.md`.

## Report

```bash
cd report && pdflatex preview.tex   # compiles the MR section standalone
```
