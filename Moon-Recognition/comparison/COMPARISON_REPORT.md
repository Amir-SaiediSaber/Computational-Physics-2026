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
2. **The ranking is regime-dependent.** On these dense tiles YOLOv8 has genuine skill while Mask R-CNN goes *negative* (−0.166) — out-of-distribution, since its instance prep skips >20 % coverage tiles. "Which model is best" depends entirely on crater density. The calibrated, report-grade per-regime numbers are in §4 (this all-val snapshot uses a separate threshold calibration, so YOLO's MCC here is higher than its §4 dense figure — see the operating-point caveat in §6).

## 4. Report-grade per-regime comparison

The best method depends on crater density, so the headline result is **per density band**, on a larger stratified sample (**418 test tiles** across six bands; each method's threshold calibrated once on a disjoint 150-tile pool). See **`fig_regime.png`** (MCC and F1 vs crater coverage) and `fig_qualitative.png`.

**Sparse / discriminative regime (coverage 1–20 %, n = 148, prevalence ≈ 0.09)** — also the regime Mask R-CNN was trained for (`max_channel_fraction=0.20`):

| Method | F1 | IoU | **MCC** | Precision | Recall | Count MAE | ms/tile |
|---|---|---|---|---|---|---|---|
| **Mask R-CNN** (instance) | 0.243 | 0.139 | **0.147** | 0.161 | 0.495 | 172 | 923 |
| **YOLOv8** (detection) | 0.214 | 0.120 | **0.104** | 0.142 | 0.432 | 32 | 10 |
| **Classical** (thr+morph) | 0.134 | 0.072 | **0.064** | 0.166 | 0.112 | 15 | 1 |
| **U-Net** (semantic) | 0.168 | 0.092 | **−0.014** | 0.093 | 0.891 | 23 | 13 |

**Dense regime (coverage ≥ 20 %, n = 270)** — the bulk of the dataset (~77 % of tiles):

| Method | F1 | IoU | **MCC** | Precision | Recall | Count MAE | ms/tile |
|---|---|---|---|---|---|---|---|
| **YOLOv8** (detection) | 0.770 | 0.626 | **0.273** | 0.654 | 0.937 | 41 | 11 |
| **U-Net** (semantic) | 0.749 | 0.599 | **0.039** | 0.602 | 0.993 | 22 | 13 |
| **Classical** (thr+morph) | 0.343 | 0.207 | **0.064** | 0.661 | 0.231 | 34 | 1 |
| **Mask R-CNN** (instance) | 0.197 | 0.109 | **−0.178** | 0.417 | 0.129 | 79 | 922 |

The flip is unambiguous: **Mask R-CNN** leads on sparse tiles (the only method with real skill there) and goes to **negative MCC** on dense tiles (out-of-distribution); **YOLOv8** is best on dense tiles; **U-Net never beats chance** (MCC ≈ 0) in either regime — even where its F1 reaches 0.75, that is base rate, not skill (`fig_regime.png`, right panel).

## 5. What we actually learn

- **U-Net has no skill in either regime.** MCC is −0.014 (sparse) and 0.039 (dense) — flat along zero at every crater density (`fig_regime.png`). It predicts crater over almost the whole tile (recall ≈ 0.89–0.99, precision ≈ prevalence); the qualitative panel shows an all-crater column. It learned the dataset's dominant base rate, not crater shape. This is the single most important finding and it directly explains the weak numbers everyone reported.
- **Mask R-CNN is the only method with clear skill on sparse tiles** (MCC 0.147) — it genuinely localizes craters — but **~90× slower** (≈920 ms/tile vs ~10) and out-of-distribution on dense tiles (MCC −0.178), where its instance prep was never meant to operate.
- **YOLOv8** is the strongest on the dense majority (MCC 0.273) and a solid speed/accuracy compromise on sparse tiles (MCC 0.104 at 10 ms/tile), though it over-detects at low confidence (count MAE 32–41).
- **The classical baseline (MCC ≈ 0.064 in both regimes) beats the U-Net** on the prevalence-robust metric while being 1 ms/tile and fully interpretable. That a 30-line OpenCV function out-discriminates the trained U-Net is a genuine result, not a throwaway.
- **Best method depends on crater density (regime flip).** Sparse (1–20 %): Mask R-CNN > YOLO > Classical > U-Net ≈ chance. Dense (≥20 %): YOLO > U-Net ≈ Classical > Mask R-CNN (negative). No method dominates everywhere — the right framing is per-regime, not one leaderboard.
- **Trade-off summary:** accuracy (MCC) is regime-dependent (above); speed → Classical (1 ms) ≈ YOLO ≈ U-Net (~10 ms) ≫ Mask R-CNN (~920 ms, 90×); interpretability → Classical > the rest.

## 6. Limitations (be honest about these)

- The shared metric is **pixel coverage of the crater class**; it structurally favours area-covering methods (semantic/box-fill) over instance methods. The count metric and MCC partly offset this, but no single number is perfectly fair across task types.
- **YOLO's classes are uninterpretable** (`class0…class6`, and it only ever predicts `class0`); all its boxes are treated as crater. If its training labels were not crater, its column is mislabelled — needs confirming with Alireza's real label map.
- Mask R-CNN GT "count" via connected components on a dense semantic mask is noisy; count MAE should be read as indicative, not definitive.
- **Operating-point sensitivity:** MCC/F1 depend on the threshold τ. Here τ is calibrated once per method on a mixed-regime pool; a τ tuned per regime would raise some dense-tile numbers (e.g. YOLO). The *rankings and the regime flip are robust*, but absolute values shift with τ — report the calibration protocol alongside any number.
- Report-grade numbers are on **418 test tiles** (148 sparse / 270 dense) with a 150-tile calibration pool — stable for ranking; widen the per-band caps (`K_TEST`) for final-manuscript precision.

## 7. Implications for the 9 July retake

1. The comparison framework here **is** the "organic comparison" the professor asked for: one split, one reduction, one metric set, with a trivial baseline to calibrate claims.
2. **Lead with MCC / skill-over-baseline, not F1/IoU** — otherwise the report repeats the base-rate mistake.
3. The U-Net section needs reframing: its real story is *base-rate collapse on an imbalanced label*, which motivates the other three methods. That is a much stronger narrative than "0.90 F1."
4. There is now a real **classical baseline** to anchor the comparison.

## 8. Reproduce
```bash
# report-grade per-regime analysis (primary; writes regime_*.csv + fig_regime.png)
KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/regime_analysis.py
# single-band views (optional)
KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/compare_models.py \
    --frac_lo 0.01 --frac_hi 0.20 --calib 40 --test 117 --tag fairband
KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/compare_models.py \
    --frac_lo 0.0 --frac_hi 1.01 --calib 30 --test 90 --tag allval
# bar + qualitative figures
KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/make_figures.py
```
Files: `regime_analysis.py` (per-regime, report-grade), `compare_models.py` (single-band harness), `classical_baseline.py` (4th method), `maskrcnn_loader.py` (rebuilds Amir's net), `regime_per_band.csv` / `regime_aggregate.csv` / `comparison_results_*.csv`, `fig_regime.png`, `fig_metrics.png`, `fig_qualitative.png`.
