#!/usr/bin/env python
"""Per-regime crater comparison + report-grade aggregate numbers.

Single inference pass per tile (Mask R-CNN is ~90x slower than the rest, so we
never re-run it per band). Threshold per method is calibrated on a disjoint
calibration pool, then metrics are computed per crater-density band and as
fair/dense aggregates.

  KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/regime_analysis.py
"""
import os, sys, glob, time, warnings, logging, numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Moon-Recognition/lunar_segmentation"))
sys.path.insert(0, HERE)
import compare_models as C
from lunar_segmentation.data.splits import spatial_train_val_split
from lunar_segmentation.models.unet import SmallUNet
from lunar_segmentation.inference.predictor import Predictor
from lunar_segmentation.data.preprocessing import CLASS_NAMES
import maskrcnn_loader
from ultralytics import YOLO

BANDS = [(0.01, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 0.70), (0.70, 1.01)]
BAND_LBL = ["1-5%", "5-10%", "10-20%", "20-40%", "40-70%", ">70%"]
K_TEST, N_CALIB, SEED = 90, 150, 0
COLORS = {"U-Net (semantic)": "#102A43", "Mask R-CNN (instance)": "#E87A2B",
          "YOLOv8 (detection)": "#1F6FB2", "Classical (thr+morph)": "#9aa5b1"}


def load_tile(row):
    d = np.load(os.path.join(ROOT, "data/MR", row["tile_path"]))
    return d["image"].astype(np.float32), (d["mask"][0] > 0)


def main():
    df = pd.read_csv(os.path.join(ROOT, "data/MR/tiles/index.csv"))
    _, val = spatial_train_val_split(df, 0.2, 256, 1024, 42)
    val = val.assign(frac=val["positive_pixels"] / 65536.0)
    val = val[val["frac"] >= 0.01].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(val))
    calib_idx = perm[:N_CALIB]
    rest = val.iloc[perm[N_CALIB:]]
    # stratified test pool: up to K_TEST per band
    test_rows = []
    for lo, hi in BANDS:
        b = rest[(rest["frac"] >= lo) & (rest["frac"] < hi)]
        test_rows.append(b.iloc[: K_TEST])
    test_df = pd.concat(test_rows).reset_index(drop=True)
    calib_df = val.iloc[calib_idx]
    print(f"device={C.DEV} | calib={len(calib_df)} | test={len(test_df)} "
          f"(per-band: {[len(t) for t in test_rows]})")

    calib_imgs = [load_tile(r) for _, r in calib_df.iterrows()]
    test_imgs = [load_tile(r) for _, r in test_df.iterrows()]
    test_frac = test_df["frac"].values
    test_gtcnt = [C.count_components(gt) for _, gt in test_imgs]

    # models
    unet = Predictor(SmallUNet(3, len(CLASS_NAMES)),
                     weights_path=os.path.join(ROOT, "data/MR/weights/best_trained.pth"), device=C.DEV)
    mrcnn, names, _ = maskrcnn_loader.build_and_load(os.path.join(ROOT, "data/MR/weights/best_model.pth"), device=C.DEV)
    clab = names.index("impact_crater")
    yolo = YOLO(sorted(glob.glob(os.path.join(ROOT, "Moon-Recognition/yolo/YOLO/runs/detect/train*/weights/best.pt")))[-1])
    methods = {
        "U-Net (semantic)":      dict(fn=lambda im: C.unet_repr(unet, im),               grid=np.round(np.arange(0.30, 0.91, 0.05), 2)),
        "Mask R-CNN (instance)": dict(fn=lambda im: C.maskrcnn_repr(mrcnn, im, clab),     grid=np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])),
        "YOLOv8 (detection)":    dict(fn=lambda im: C.yolo_repr(yolo, im),                grid=np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])),
        "Classical (thr+morph)": dict(fn=lambda im: C.classical_repr(im),                 grid=np.array([0.5])),
    }

    per_band, agg = [], []
    for name, spec in methods.items():
        print(f"  {name} ...", flush=True)
        creps = [spec["fn"](im)[0] for im, _ in calib_imgs]
        treps, ttimes = [], []
        for im, _ in test_imgs:
            rep, dt = spec["fn"](im); treps.append(rep); ttimes.append(dt)
        tau = C.best_tau(creps, [gt for _, gt in calib_imgs], spec["grid"])

        def eval_subset(mask):
            idx = np.where(mask)[0]
            if len(idx) == 0:
                return None
            return C.evaluate([treps[i] for i in idx], [test_imgs[i][1] for i in idx],
                              [test_gtcnt[i] for i in idx], tau, [ttimes[i] for i in idx])

        for (lo, hi), lbl in zip(BANDS, BAND_LBL):
            m = eval_subset((test_frac >= lo) & (test_frac < hi))
            if m: per_band.append(dict(method=name, band=lbl, tau=float(tau), n=int(((test_frac >= lo) & (test_frac < hi)).sum()), **m))
        for tag, mask in [("fair(1-20%)", test_frac < 0.20), ("dense(>=20%)", test_frac >= 0.20)]:
            m = eval_subset(mask)
            if m: agg.append(dict(method=name, regime=tag, tau=float(tau), n=int(mask.sum()), **m))

    pb = pd.DataFrame(per_band); pb.to_csv(os.path.join(HERE, "regime_per_band.csv"), index=False)
    ag = pd.DataFrame(agg); ag.to_csv(os.path.join(HERE, "regime_aggregate.csv"), index=False)
    print("\n=== AGGREGATE (report-grade) ===")
    print(ag[["method", "regime", "n", "tau", "f1", "iou", "mcc", "count_mae", "ms_per_tile"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # figure: MCC and F1 vs density band
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for metric, ax, ttl in zip(["mcc", "f1"], axes, ["MCC vs crater density (skill)", "Pixel F1 vs crater density"]):
        for name in methods:
            d = pb[pb.method == name].set_index("band").reindex(BAND_LBL)
            ax.plot(BAND_LBL, d[metric], "o-", color=COLORS[name], label=name, lw=2, ms=5)
        ax.set_xlabel("crater coverage band"); ax.set_title(ttl); ax.grid(alpha=0.3)
        if metric == "mcc":
            ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("MCC (0 = no skill)")
        else:
            ax.set_ylabel("F1")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Per-regime crater detection (leakage-free spatial val, seed=42)", fontsize=12)
    fig.tight_layout()
    p = os.path.join(HERE, "fig_regime.png"); fig.savefig(p, dpi=140); print("\nwrote", p)


if __name__ == "__main__":
    main()
