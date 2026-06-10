# Moon Recognition — Code-Review Improvements Log

Audit of the U-Net lunar-segmentation section (June 2026), after the full
CloudVeneto T4 runs and the report draft. Every change is one commit with the
full justification in its message (`git log`); this file is the index, plus
the verification evidence and the open items that still require a re-run.

Baseline pipeline by prof. Zingales (`tiziano1590`, commits up to 2026-04-21);
all subsequent work on the `U-net` branch.

## Changes

| Commit | File(s) | Issue | Fix & justification |
|---|---|---|---|
| `9539c1d` | `models/unet.py` | Docstrings contradicted the code: "bilinear upsample" on a `ConvTranspose2d`, diagram drawn for 128×128 while training uses 256×256, ambiguous "3-level" | Docs corrected; **zero functional change** (only comments touched) |
| `46c1bcc` | `inference/predictor.py` | (a) `load_state_dict(strict=False)` could silently skip mismatched layers and predict with random weights; (b) inference window default 128/64 ≠ training tiles 256/128 | (a) keep `strict=False` for legacy checkpoints but log a loud warning listing missing/unexpected keys; (b) defaults now match the training protocol. **Reproducibility note:** the south-pole figures in the report were generated with the old 128/64 window — pass them explicitly to reproduce exactly |
| `f7d565e` | `training/trainer.py` | (a) Evaluation materialised ~23 GB of full-set tensors (fine on the T4 node, OOM on a laptop); (b) a class with zero positive pixels in val got AP = 0, penalising the model for the split | (a) bool target storage + per-class sigmoid/threshold; **verified numerically identical** to the old code (per-class precision/recall/IoU/AP equal within 1e-12 on synthetic data, `stellar` env, torch 2.11); (b) absent classes now report NaN + new `support` column; pandas `.mean()` skips NaN. All 7 classes have support > 0 in the reported runs, so published numbers are unaffected |
| `402bb62` | `data/splits.py` (new), `.gitignore` | **Train/val leakage**: tiles overlap 50% (stride 128 / size 256) and the split is a random permutation. Measured on the real index: **100% of the 3 186 validation tiles overlap ≥ 1 training tile** | `spatial_train_val_split()` assigns contiguous 1024-px blocks to val and drops overlapping boundary train tiles → strictly disjoint pixels (11 289 train / 3 248 val, 1 394 tiles = 8.8% dropped). `assert_no_overlap()` sanity check passes. Also root-anchored the `.gitignore` `data/` pattern, which was silently ignoring new files in the package's `data/` source module |
| `5a6bc7d` | `data/resolver.py` | Manifest entries hardcoded to other machines (`/mnt/data/...pdf`, `/home/zingales/...csv`) — dead on any machine running this code | Paper kept as a provenance comment; crater CSV needs no entry (ships inside the USGS Robbins bundle, located by filename search in `label_loader`) |
| `50ab819` | `configs/unet_config.yaml` | Config described an early local protocol (batch 8, 20 epochs); every reported result uses the T4 protocol (batch 16, 30 epochs, cosine 1e-3→1e-5) | Config now records the definitive protocol + the local prototyping variant + best architecture/loss |
| `89b9e7d` | `report/moon_recognition.tex`, `preview.tex` | Report claims diverged from the code: wrong split description, wrong CLAHE params, "non-overlapping" tiles, phantom gradient clipping, dilation order, **wrong citation** (bib entry was the DeepLabv3+ paper, unrelated to augmentation), no mention of split leakage | All claims aligned with the code; new explicit "Limitation" paragraph with the 100%-overlap measurement; Study 3 analysis now separates the geological explanation from the leakage confound and marks the conclusion as pending re-validation; citation replaced with Shorten & Khoshgoftaar 2019 (augmentation survey); `amssymb` added (pre-existing compile error on `\checkmark`); compile verified clean |

## Verification performed

- `multilabel_metrics` refactor: asserted identical to the previous
  implementation (1e-12) on synthetic data, including the bool-target path,
  in the project env (`stellar`, torch 2.11).
- `spatial_train_val_split`: run on the real `data/MR/tiles/index.csv`;
  `assert_no_overlap` passes; leakage of the old random split measured at
  100% of val tiles.
- `resolver.py`: module imports; no `local_file` modes remain.
- Report: compiles with zero LaTeX errors via `preview.tex`.

## Open items (require a CloudVeneto re-run, not doable locally)

1. **Re-run Study 3 (augmentation) with `spatial_train_val_split`.** The
   "no augmentation is better" result is confounded by split leakage; the
   report now says so explicitly. This is the single most important re-run.
2. Optionally re-run Studies 1–2 on the spatial split for leakage-free
   absolute numbers (relative rankings are expected to be more stable).
3. South-pole inference re-run with the new 256/128 window defaults
   (current figures used 128/64).

## Known decisions NOT changed (and why)

- `strict=False` weight loading kept: legacy checkpoints
  (`best_trained_legacy.pth`) need it; the new warning makes silent
  mismatch impossible.
- Focal loss `alpha = 0.25` (down-weights positives) kept: standard
  RetinaNet pairing with `gamma = 2`; rare-class emphasis is handled by the
  per-class Dice weights `[1, 4, 5, 5, 5, 5, 5]`. Changing it would
  invalidate all completed ablations.
- The executed `architecture_comparison.ipynb` was committed **before** any
  fix (commit `4b4d4a6`) so the provenance of the published results is
  preserved; the notebook still uses the random split that produced them.
