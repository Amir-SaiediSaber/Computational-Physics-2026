# South Pole Complete Workflow - Code Explanation

## Overview

This notebook implements a **complete workflow for lunar surface feature segmentation** on south pole imagery. It processes individual PNG tiles from the lunar south pole, runs inference using a pre-trained UNet model, generates visualizations, aggregates results, and produces quality assessment reports.

**Goal:** Detect and classify lunar surface features (craters, boulders, etc.) in south pole tile images using deep learning segmentation.

---

## 1. Environment Setup

```python
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from IPython.display import Image as IPImage, display
```

**Purpose:** Import all required libraries for:
- `torch` - PyTorch for deep learning inference
- `numpy` - Numerical operations on probability arrays
- `matplotlib` - Visualization of results
- `PIL` - Loading PNG tile images
- `Path` - Cross-platform path handling

```python
PROJECT_ROOT = Path('/home/ubuntu/moon/Moon-Recognition/lunar_segmentation')
sys.path.insert(0, str(PROJECT_ROOT))
```

**Purpose:** Add the `lunar_segmentation` project to Python's import path so custom modules can be imported.

```python
from lunar_segmentation.models.unet import SmallUNet
from lunar_segmentation.data.preprocessing import build_three_channel_input, CLASS_NAMES
from lunar_segmentation.inference.predictor import Predictor
from lunar_segmentation.training.trainer import Trainer, BCEDiceLoss
from lunar_segmentation.data.datasets import MoonTileDataset
```

**Custom modules imported:**
- `SmallUNet` - A lightweight UNet architecture for segmentation
- `build_three_channel_input` - Converts grayscale tile to 3-channel CHW format
- `CLASS_NAMES` - List of feature classes (e.g., crater, boulder, etc.)
- `Predictor` - Wrapper for running model inference
- `Trainer`, `BCEDiceLoss` - Training utilities (used if model needs training)
- `MoonTileDataset` - PyTorch Dataset for loading training tiles

---

## 2. Data Loading and Configuration

```python
BASE_DIR = Path('/home/ubuntu/moon/MR')
TILES_DIR = BASE_DIR / 'lunar_south_pole/tiles'
OUTPUT_DIR = BASE_DIR / 'results/south_pole/inference'
MODEL_WEIGHTS = BASE_DIR / 'weights/best_trained.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_INFERENCE_TILES = 20
```

**Configuration variables:**
- `BASE_DIR` - Root directory for all data
- `TILES_DIR` - Directory containing input PNG tiles (`tile_*.png`)
- `OUTPUT_DIR` - Where inference results are saved
- `MODEL_WEIGHTS` - Path to pre-trained model checkpoint
- `DEVICE` - Use GPU if available, otherwise CPU
- `MAX_INFERENCE_TILES` - Limit inference to 20 tiles (set to `None` for all 3447 tiles)

```python
PROBS_DIR = OUTPUT_DIR / 'probabilities'
VIZ_DIR = OUTPUT_DIR / 'visualizations'
```

**Output directories:**
- `probabilities/` - Stores `.npz` files with probability cubes (C, H, W) per tile
- `visualizations/` - Stores `_viz.png` visualization images

**Workflow decision logic:**
```python
SKIP_INFERENCE = len(existing_probs) > 0 and len(tile_files) == 0
```
If pre-computed probabilities exist and no source tiles are found, skip inference and use existing results.

---

## 3. Model Initialization

```python
model = SmallUNet(in_channels=3, num_classes=len(CLASS_NAMES))
```

**SmallUNet Architecture:**
- Input: 3-channel image (C=3, H, W)
- Output: Multi-class probability map (C=num_classes, H, W)
- A lightweight UNet variant for efficient lunar feature segmentation

```python
if MODEL_WEIGHTS.exists():
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True))
    print("Loaded pre-trained weights.")
```

Load pre-trained weights if available. Uses `weights_only=True` for safe loading (PyTorch security best practice).

```python
model.to(DEVICE)
model.eval()
predictor = Predictor(model=model, device=DEVICE)
```

- Move model to GPU/CPU
- Set to evaluation mode (`model.eval()` disables dropout, uses running stats for batch norm)
- Create `Predictor` wrapper for easy inference calls

---

## 3.1 Optional Model Training

This section **only runs if `best_trained.pth` is missing** and training data is available.

```python
if not MODEL_WEIGHTS.exists():
    data_index = BASE_DIR / 'tiles/index.csv'
    df = pd.read_csv(data_index)
    df['tile_path'] = df['tile_path'].apply(lambda p: str(BASE_DIR / p))
```

- Load training data index CSV
- Convert relative paths to absolute paths

```python
df['exists'] = df['tile_path'].apply(lambda p: Path(p).exists())
df_valid = df[df['exists']].copy()
```

Filter to only tiles that actually exist on disk.

```python
if not use_cuda and len(df_valid) > 500:
    df_valid = df_valid.sample(n=500, random_state=42)
```

On CPU, subsample to 500 tiles for faster training. Use all tiles if GPU is available.

```python
dataset = MoonTileDataset(df_valid, augment=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
```

- `MoonTileDataset` - Custom PyTorch Dataset that loads tiles and their masks
- `augment=True` - Apply random flips/rotations for data augmentation
- `batch_size=4` - Process 4 tiles at a time

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = BCEDiceLoss()
trainer = Trainer(model, optimizer, criterion, device=DEVICE)
```

- **Optimizer:** Adam with learning rate 1e-4
- **Loss:** `BCEDiceLoss` - Combination of Binary Cross Entropy and Dice loss (common for segmentation)
- **Trainer:** Custom training loop wrapper

```python
for epoch in range(num_epochs):
    loss = trainer.train_one_epoch(loader)
    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), MODEL_WEIGHTS)
```

Train for 3 epochs (CPU) or 5 epochs (GPU). Save best weights based on training loss.

---

## 4. Inference Functions

### `load_tile_and_predict(tile_path)`

```python
def load_tile_and_predict(tile_path):
    tile = Image.open(tile_path).convert('L')
    tile_gray = np.array(tile).astype(np.float32)
    tile_chw = build_three_channel_input(tile_gray)
    prob_cube = predictor.predict(tile_chw)
```

**Steps:**
1. Load PNG tile as grayscale (`convert('L')`)
2. Convert to numpy float32 array
3. `build_three_channel_input()` - Converts single-channel grayscale to 3-channel CHW tensor (likely by duplicating the channel or applying different filters)
4. `predictor.predict()` - Runs model inference, returns probability cube of shape (C, H, W)

**Returns:** Dictionary with `path`, `probabilities`, `image`, `success`, `error`

### `generate_tile_viz(display_img, prob_cube, class_names, output_path)`

Generates a 2×4 visualization grid for each tile:

**Top row (columns 0-3):**
- Column 0: Original grayscale image
- Columns 1-3: Probability overlay for classes 0-2 (normalized, viridis colormap, overlaid on grayscale)

**Bottom row (columns 0-3):**
- Columns 0-3: Binary mask for classes 0-3 using threshold 0.1
  - If no pixels above threshold: show normalized probability (plasma colormap)
  - If features detected: show binary mask (Reds for class 0, hot for others)

```python
adjusted_threshold = 0.1
mask = (prob_img > adjusted_threshold).astype(np.uint8)
```

Features are considered "detected" if probability > 0.1.

### `run_batch_inference(tiles_dir, predictor, output_dir, max_tiles=None)`

```python
tile_files = sorted(tiles_dir.glob('tile_*.png'))
if max_tiles:
    tile_files = tile_files[:max_tiles]
```

Find all `tile_*.png` files and optionally limit to `max_tiles`.

For each tile:
1. Call `load_tile_and_predict()`
2. Save probability cube as `.npz` file: `probabilities/{tile_stem}.npz`
3. Generate visualization: `visualizations/{tile_stem}_viz.png`

---

## 5. Run Inference

```python
if SKIP_INFERENCE:
    print("SKIPPING INFERENCE - using pre-computed results.")
```

If pre-computed results exist, skip inference.

```python
already_done = 0
tiles_to_process = []
for tf in tile_files:
    prob_file = PROBS_DIR / f"{tf.stem}.npz"
    if prob_file.exists():
        already_done += 1
    else:
        tiles_to_process.append(tf)
```

Check which tiles have already been processed (to avoid re-computation).

```python
results = run_batch_inference(TILES_DIR, predictor, OUTPUT_DIR, max_tiles=MAX_INFERENCE_TILES)
success_count = sum(1 for r in results if r['success'])
failure_count = len(results) - success_count
```

Run batch inference on tiles that haven't been processed yet.

---

## 6. Aggregation

### `aggregate_results(probs_dir, class_names)`

**Problem:** Loading all 3447 tiles' probability cubes into RAM simultaneously would require enormous memory.

**Solution:** Stream through `.npz` files one at a time, keeping only running statistics.

```python
sum_probs = first_cube.astype(np.float64)  # Running sum
max_probs = first_cube.copy()              # Element-wise max across tiles
global_max = np.array([float(np.max(first_cube[i])) for i in range(n_classes)])
```

**Accumulators:**
- `sum_probs` (float64) - Running sum of all probability cubes → divide by N to get mean
- `max_probs` (float32) - Element-wise maximum across all tiles
- `global_max` (float32) - Per-class scalar maximum probability

```python
for idx, f in enumerate(prob_files[1:]):
    cube = np.load(f)['probabilities']
    sum_probs += cube.astype(np.float64)
    np.maximum(max_probs, cube, out=max_probs)  # In-place element-wise max
    for i in range(n_classes):
        cm = float(np.max(cube[i]))
        if cm > global_max[i]:
            global_max[i] = cm
    del cube  # Free memory immediately
```

Stream through remaining files, updating accumulators in-place. Delete each cube after processing to free RAM.

```python
avg_probs = (sum_probs / n).astype(np.float32)
```

Compute mean probability across all tiles.

**Stats computed:**
```python
stats = {
    'tile_count': n,
    'class_names': class_names,
    'avg_max_per_class': {c: float(np.max(avg_probs[i])) for i, c in enumerate(class_names)},
    'global_max_per_class': {c: float(global_max[i]) for i, c in enumerate(class_names)},
    'features_detected': {c: int((max_probs[i] > 0.1).sum()) for i, c in enumerate(class_names)},
}
```

- `avg_max_per_class` - Maximum probability in the average probability map (per class)
- `global_max_per_class` - Highest probability ever seen across all tiles (per class)
- `features_detected` - Number of pixels above 0.1 threshold in the max-probability map (per class)

Save stats to `stats.json` (excluding private keys like `_avg_probs`).

---

## 7. Visualization

```python
viz_files = sorted(VIZ_DIR.glob('*_viz.png'))
for i, viz_file in enumerate(viz_files[:10]):
    display(IPImage(filename=str(viz_file)))
```

Display the first 10 visualization images inline in the notebook.

---

## 8. Quality Assessment

```python
for c in CLASS_NAMES:
    print(f"{c}:")
    print(f"  Max Prob: {stats['avg_max_per_class'][c]:.4f}")
    print(f"  Detections (>0.1): {stats['features_detected'][c]}")
```

Print per-class statistics: maximum probability and number of detected feature pixels.

**Summary bar chart:**
```python
fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, max_probs, width, label='Max Probability', color='steelblue')
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, detections, width, label='Detections', color='coral')
```

Create a dual-axis bar chart:
- Left axis: Maximum probability per class (steelblue bars)
- Right axis: Number of detections per class (coral bars)

Save chart to `results/south_pole/quality_summary.png`.

**Summary text file:**
```python
with open(summary_path, 'w') as f:
    f.write(f"South Pole Inference Summary\n")
    f.write(f"Total tiles processed: {stats['tile_count']}\n")
    for c in CLASS_NAMES:
        f.write(f"  {c}:\n")
        f.write(f"    Max probability: {stats['global_max_per_class'][c]:.4f}\n")
        f.write(f"    Avg max: {stats['avg_max_per_class'][c]:.4f}\n")
        f.write(f"    Detections (>0.1): {stats['features_detected'][c]}\n")
```

Save human-readable summary to `results/south_pole/summary.txt`.

---

## 9. Single Tile Demo

```python
demo_tile = tile_files[0]
result = load_tile_and_predict(demo_tile)
```

Pick the first tile and run inference on it.

```python
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes[0, 0].imshow(result['image'], cmap='gray')  # Original
for i, name in enumerate(CLASS_NAMES):
    prob_map = prob_cube[i]
    ax.imshow(prob_map, cmap='viridis')  # Probability map per class
```

Display a detailed 2×4 grid for the demo tile:
- Top-left: Original grayscale image
- Remaining 7 subplots: Probability maps for each class (viridis colormap)

---

## 10. Summary

The notebook demonstrates a complete ML workflow:

1. **Setup** - Configure paths and import modules
2. **Load Model** - Initialize SmallUNet with pre-trained weights (or train if missing)
3. **Inference** - Process PNG tiles through the model to get probability maps
4. **Visualize** - Create overlay visualizations showing detected features
5. **Aggregate** - Stream all results to compute global statistics without loading everything into RAM
6. **Assess Quality** - Generate charts and summary reports

**Key design decisions:**
- Memory-efficient streaming aggregation (process 3447 tiles without OOM)
- Threshold of 0.1 for feature detection (from visualization section)
- Optional training section for reproducibility
- Pre-computed results detection to skip redundant inference

**Output files:**
- `results/south_pole/inference/probabilities/*.npz` - Per-tile probability cubes
- `results/south_pole/inference/visualizations/*_viz.png` - Per-tile visualizations
- `results/south_pole/stats.json` - Aggregated statistics
- `results/south_pole/quality_summary.png` - Quality assessment chart
- `results/south_pole/summary.txt` - Human-readable summary
