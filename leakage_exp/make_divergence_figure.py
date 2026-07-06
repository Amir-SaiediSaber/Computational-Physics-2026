"""Figure: validation mAP50 vs epoch for the extended-budget ridge pair.

Random-split vs spatial-split training curves (150 epochs). Raw per-epoch
values are drawn faint; a 7-epoch rolling mean carries the trend. A marker at
epoch 40 ties the figure to the matched-budget table.
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig_split_divergence.png")

SERIES = [
    ("wrinkle_ridge_random_e150", "random tile split", "#2a78d6", "-"),
    ("wrinkle_ridge_spatial_e150", "spatial block split", "#d6662a", "--"),
]


def load(run):
    rows = list(csv.DictReader(open(os.path.join(HERE, "runs", run, "results.csv"))))
    ep = np.array([int(float(r["epoch"])) for r in rows])
    m = np.array([float(r["metrics/mAP50(B)"]) for r in rows])
    return ep, m


def roll(x, w=7):
    out = np.convolve(x, np.ones(w) / w, mode="valid")
    pad = np.full(w - 1, np.nan)
    return np.concatenate([pad, out])


fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)

for run, label, color, ls in SERIES:
    ep, m = load(run)
    ax.plot(ep, m, color=color, lw=0.7, alpha=0.28)
    sm = roll(m)
    ax.plot(ep, sm, color=color, lw=2.0, ls=ls)
    # direct label at line end
    ax.annotate(label, xy=(ep[-1], sm[-1]), xytext=(4, 0),
                textcoords="offset points", color=color,
                fontsize=8.5, va="center")

ax.axvline(40, color="#888888", lw=0.8, ls=":")
ax.annotate("matched-budget\ncomparison (40 ep)", xy=(40, ax.get_ylim()[1]),
            xytext=(-6, -4), textcoords="offset points", ha="right", va="top",
            fontsize=7.5, color="#555555")

ax.set_xlabel("training epoch", fontsize=9)
ax.set_ylabel("validation mAP50 (wrinkle ridge)", fontsize=9)
ax.set_xlim(0, 195)
ax.set_ylim(0, None)
ax.tick_params(labelsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", lw=0.4, alpha=0.35)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
