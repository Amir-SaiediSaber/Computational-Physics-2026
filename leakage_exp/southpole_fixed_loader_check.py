"""South-pole inference with the fixed checkpoint loader.

Re-runs the scaled inference workflow (package Predictor, training
preprocessing) on a random sample of south pole tiles and aggregates the
same statistics as the reference table (avg max prob per class), to verify
that the pipeline with the corrected checkpoint loading reproduces the
reference behaviour (crater-dominant, not uniform ~0.5).
"""
import os
import sys
import glob
import random
import json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG = os.path.join(ROOT, "Moon-Recognition", "lunar_segmentation")
sys.path.insert(0, PKG)

from PIL import Image  # noqa: E402
from lunar_segmentation.inference.predictor import Predictor  # noqa: E402
from lunar_segmentation.data.preprocessing import build_three_channel_input  # noqa: E402

CLASSES = ["impact_crater", "pit_skylight", "wrinkle_ridge", "lobate_scarp",
           "irregular_mare_patch", "apollo_site", "candidate_rille"]

TILES = sorted(glob.glob(os.path.join(
    ROOT, "data/MR/lunar_south_pole/tiles/*.png")))
print(f"{len(TILES)} south pole tiles found")
random.seed(42)
sample = random.sample(TILES, 300)

from lunar_segmentation.models.unet import SmallUNet  # noqa: E402

# best_trained.pth is the published non-residual w=32 checkpoint
model = SmallUNet(in_channels=3, num_classes=7, base_width=32,
                  use_residual=False)
pred = Predictor(model,
                 weights_path=os.path.join(ROOT, "data/MR/weights/best_trained.pth"),
                 device="cpu")

max_probs = []
for i, p in enumerate(sample):
    gray = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    chw = build_three_channel_input(gray)
    probs = pred.predict(chw)             # (7, H, W) probabilities
    max_probs.append(probs.reshape(7, -1).max(axis=1))
    if (i + 1) % 50 == 0:
        print(f"{i+1}/300", flush=True)

mp = np.array(max_probs)                   # (300, 7)
stats = {c: {"avg_max_prob": float(mp[:, k].mean()),
             "global_max": float(mp[:, k].max())}
         for k, c in enumerate(CLASSES)}
print(json.dumps(stats, indent=1))
with open(os.path.join(os.path.dirname(__file__),
                       "southpole_fixed_loader_stats.json"), "w") as f:
    json.dump(stats, f, indent=1)
