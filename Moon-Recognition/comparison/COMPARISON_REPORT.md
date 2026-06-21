# Lunar crater detection — head-to-head comparison of the four methods

*Generated from the local integration branch. Reproduce with the commands at the bottom.*

## 1. The problem: four methods, three different tasks

The group has four crater-finding methods, but they output different things:

| Method | Author | Task type | Native output |
|---|---|---|---|
| U-Net | (yours) | semantic segmentation | per-pixel class probabilities |
| Mask R-CNN | Amir | instance segmentation | per-instance box + mask + score |
| YOLOv8 | Alireza | object detection | per-object box + score |
| Classical | *(newly implemented here)* | thresholding + morphology | binary mask + components |

You cannot compare an mAP against a pixel-IoU against a thresholded mask. So the comparison only means something if every method is reduced to **the same prediction on the same data, scored with the same metric**.

## 2. Shared protocol

- **Data & split:** the leakage-free **spatial** train/val split (`spatial_train_val_split`, seed=42) on the Marius Hills tiles — the same split your report uses. Evaluation is on held-out validation tiles only.
- **Common prediction:** every method is reduced to a **binary impact-crater pixel mask** (256×256):
  - U-Net → crater-channel probability, thresholded;
  - Mask R-CNN → union of crater-class instance masks with score ≥ τ;
  - YOLOv8 → union of predicted boxes with score ≥ τ (the model collapses to a single class);
  - Classical → threshold + morphology output.
- **Ground truth:** mask channel 0 (`impact_crater`) > 0.
- **Thresholds (τ)** for the score-based methods are tuned for best micro-F1 on a **calibration** subset, then reported on a **disjoint test** subset — no tuning on the test data.
- **Metrics:** pixel precision / recall / F1 / IoU, **MCC** (Matthews correlation — robust to class imbalance), crater **count MAE** (connected components, area 30–6000 px), and inference **ms/tile**.

## 3. The base-rate trap (why the obvious comparison is wrong)

The `impact_crater` channel is **extremely prevalent**: across the validation tiles its mean pixel coverage is **0.81** (median 0.96) — most tiles are almost entirely labelled crater. **2,489 of 3,248 val tiles are >70 % crater.**

On such data the pixel metrics are dominated by base rate. A trivial **"predict every pixel is crater"** baseline scores **F1 ≈ 0.90, IoU ≈ 0.82**. Full all-val results (prevalence 0.82):

| (all val, prevalence 0.82) | F1 | IoU | **MCC** | Precision | Recall | ms/tile |
|---|---|---|---|---|---|---|
| YOLOv8 | 0.916 | 0.846 | **0.470** | 0.893 | 0.940 | 9 |
| U-Net | 0.903 | 0.823 | **0.013** | 0.823 | 1.000 | 12 |
| *Predict-all baseline* | *0.903* | *0.823* | *0.000* | *0.823* | *1.000* | *0* |
| Classical | 0.374 | 0.230 | **0.078** | 0.878 | 0.238 | 1 |
| Mask R-CNN | 0.116 | 0.061 | **−0.166** | 0.616 | 0.064 | 918 |

Two things this exposes:
1. **The U-Net's headline 0.90 F1 is not skill** — F1/IoU match the predict-all baseline and MCC ≈ 0. It is exploiting the base rate, nothing more.
2. **The ranking is regime-dependent.** On these dense tiles YOLOv8 has genuine skill (MCC 0.47), whereas Mask R-CNN goes *negative* (−0.166): it is out-of-distribution here because its instance prep deliberately skips tiles with >20 % crater coverage. So "which model is best" depends entirely on crater density — which is itself a headline result, and a reason any single-number leaderboard is misleading.

## 4. The fair comparison (discriminative band)

To compare on data where the task is actually non-trivial — and where the instance methods are in-distribution (Mask R-CNN's instance prep **skips** tiles with >20 % crater coverage, `max_channel_fraction=0.20`) — evaluation is restricted to tiles with **crater coverage 1–20 %** (157 val tiles, test prevalence ≈ 0.09).

| Method | F1 | IoU | **MCC** | Precision | Recall | Count MAE | ms/tile |
|---|---|---|---|---|---|---|---|
| **Mask R-CNN** (instance) | 0.270 | 0.156 | **0.180** | 0.206 | 0.388 | 79 | 919 |
| **YOLOv8** (detection) | 0.249 | 0.142 | **0.159** | 0.162 | 0.543 | 102 | 10 |
| **Classical** (thr+morph) | 0.140 | 0.075 | **0.070** | 0.170 | 0.119 | 16 | 1 |
| **U-Net** (semantic) | 0.170 | 0.093 | **−0.001** | 0.093 | 0.999 | 24 | 12 |
| *Predict-all (baseline)* | *0.170* | *0.093* | *0.000* | *0.093* | *1.000* | *24* | *0* |

(See `fig_metrics.png` and `fig_qualitative.png`.)

## 5. What we actually learn

- **U-Net has no skill on sparse-crater tiles.** Its MCC is −0.001 and its F1 equals the predict-all baseline exactly; the qualitative panel shows it predicting crater over the *entire* tile. It learned the dataset's dominant base rate, not crater shape. This is the single most important finding and it directly explains the weak numbers everyone reported.
- **Mask R-CNN is the only method with clear skill** (MCC 0.180) — it genuinely localizes craters — **but it is ~90× slower** (919 ms/tile vs ~10) and it over-segments (count MAE 79).
- **YOLOv8** is a strong speed/accuracy compromise (MCC 0.159 at 10 ms/tile) but over-detects badly at the low confidence its best-F1 needs (count MAE 102).
- **The classical baseline, though weak (MCC 0.070), beats the U-Net** on the prevalence-robust metric while being 1 ms/tile and fully interpretable. That a 30-line OpenCV function out-discriminates the trained U-Net here is a genuine result, not a throwaway.
- **Best method depends on crater density (regime flip).** Sparse tiles (1–20 %): Mask R-CNN wins (MCC 0.18), U-Net ≈ chance. Dense tiles (all-val): YOLOv8 wins (MCC 0.47), Mask R-CNN goes *negative* (out-of-distribution). No method dominates everywhere — the right framing is a per-regime comparison, not one leaderboard.
- **Trade-off summary (fair band):** accuracy (MCC) → Mask R-CNN > YOLO > Classical > U-Net≈chance; speed → Classical ≈ YOLO ≈ U-Net ≫ Mask R-CNN (90×); interpretability → Classical > the rest.

## 6. Limitations (be honest about these)

- The shared metric is **pixel coverage of the crater class**; it structurally favours area-covering methods (semantic/box-fill) over instance methods. The count metric and MCC partly offset this, but no single number is perfectly fair across task types.
- **YOLO's classes are uninterpretable** (`class0…class6`, and it only ever predicts `class0`); all its boxes are treated as crater. If its training labels were not crater, its column is mislabelled — needs confirming with Alireza's real label map.
- Mask R-CNN GT "count" via connected components on a dense semantic mask is noisy; count MAE should be read as indicative, not definitive.
- Numbers are on a 117-tile test sample (fair band) / 90-tile sample (all val); they are stable enough for ranking but not final-report precision. Scale up `--test` for the manuscript.

## 7. Implications for the 9 July retake

1. The comparison framework here **is** the "organic comparison" the professor asked for: one split, one reduction, one metric set, with a trivial baseline to calibrate claims.
2. **Lead with MCC / skill-over-baseline, not F1/IoU** — otherwise the report repeats the base-rate mistake.
3. The U-Net section needs reframing: its real story is *base-rate collapse on an imbalanced label*, which motivates the other three methods. That is a much stronger narrative than "0.90 F1."
4. There is now a real **classical baseline** to anchor the comparison.

## 8. Reproduce
```bash
# fair discriminative band
KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/compare_models.py \
    --frac_lo 0.01 --frac_hi 0.20 --calib 40 --test 117 --tag fairband
# full-val base-rate view
KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/compare_models.py \
    --frac_lo 0.0 --frac_hi 1.01 --calib 30 --test 90 --tag allval
# figures
KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/make_figures.py
```
Files: `compare_models.py` (harness), `classical_baseline.py` (4th method), `maskrcnn_loader.py` (rebuilds Amir's net), `comparison_results_*.csv`, `fig_metrics.png`, `fig_qualitative.png`.
