#!/usr/bin/env python
# coding: utf-8

# # South Pole Tiles - Complete Workflow Notebook
# 
# This notebook demonstrates the **complete** workflow for lunar surface feature segmentation on south pole imagery.
# 
# ## What You Will Learn
# 
# 1. **Processing individual PNG tiles** - No need to create tiles from raster
# 2. **Running inference on pre-trained model** - Classify each tile
# 3. **Generating visualizations** - Clear crater feature detection at 0.1 threshold
# 4. **Aggregating results** - Combine predictions from all tiles
# 5. **Creating summary reports** - Statistics and quality metrics
# 
# ## South Pole Tile Format
# 
# Unlike the Marius Hills raster tiles (1024x512 rectangular), south pole tiles come as individual PNG files with standard aspect ratios.
# 
# ## Prerequisites
# 
# ```bash
# # Ensure model weights exist
# ls -la weights/best_trained.pth
# 
# # Check tile directory (optional - inference is skipped if pre-computed probabilities exist)
# ls -la results/south_pole/tiles/ | head -20
# ```

# ## 1. Environment Setup

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from IPython.display import Image as IPImage, display

# Add project root to path - use absolute path
PROJECT_ROOT = Path('/home/ubuntu/moon/Moon-Recognition/lunar_segmentation')
sys.path.insert(0, str(PROJECT_ROOT))

# Import custom modules
from lunar_segmentation.models.unet import SmallUNet
from lunar_segmentation.data.preprocessing import build_three_channel_input, CLASS_NAMES
from lunar_segmentation.inference.predictor import Predictor
from lunar_segmentation.training.trainer import Trainer, BCEDiceLoss
from lunar_segmentation.data.datasets import MoonTileDataset

print("Environment setup complete!")
print(f"CLASS_NAMES: {CLASS_NAMES}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")


# ## 2. Data Loading

# Configuration - use local paths
BASE_DIR = Path('/home/ubuntu/moon/MR')
TILES_DIR = BASE_DIR / 'lunar_south_pole/tiles'
OUTPUT_DIR = BASE_DIR / 'results/south_pole/inference'
MODEL_WEIGHTS = BASE_DIR / 'weights/best_trained.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Inference limit: set to None to process all tiles, or an integer to limit
# On CPU, ~50 tiles takes ~2-5 minutes. Full dataset (3447 tiles) takes hours.
MAX_INFERENCE_TILES = 20

# Check existing pre-computed data
PROBS_DIR = OUTPUT_DIR / 'probabilities'
VIZ_DIR = OUTPUT_DIR / 'visualizations'

print(f"Device: {DEVICE}")
print(f"Model weights exists: {MODEL_WEIGHTS.exists()}")
print(f"Tiles directory exists: {TILES_DIR.exists()}")
print(f"Pre-computed probabilities directory exists: {PROBS_DIR.exists()}")

# Count available data
if TILES_DIR.exists():
    tile_files = sorted(TILES_DIR.glob('tile_*.png'))
    print(f"\nFound {len(tile_files)} source PNG tiles")
else:
    tile_files = []
    print("\nNo source tiles directory - will use pre-computed probabilities")

if PROBS_DIR.exists():
    existing_probs = sorted(PROBS_DIR.glob('*.npz'))
    # Filter out the stats file
    existing_probs = [p for p in existing_probs if 'stats' not in p.name]
    print(f"Found {len(existing_probs)} pre-computed probability files")
else:
    existing_probs = []

if VIZ_DIR.exists():
    existing_vizs = sorted(VIZ_DIR.glob('*_viz.png'))
    print(f"Found {len(existing_vizs)} pre-computed visualizations")
else:
    existing_vizs = []

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROBS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)
MODEL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

# Decide workflow
SKIP_INFERENCE = len(existing_probs) > 0 and len(tile_files) == 0
print(f"\nSkip inference (use pre-computed): {SKIP_INFERENCE}")


# ## 3. Model Initialization

print(f"Loading model from {MODEL_WEIGHTS}...")
model = SmallUNet(in_channels=3, num_classes=len(CLASS_NAMES))
if MODEL_WEIGHTS.exists():
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True))
    print("Loaded pre-trained weights.")
else:
    print("No pre-trained weights found. Will attempt training below.")
model.to(DEVICE)
model.eval()

# Create predictor
predictor = Predictor(
    model=model,
    device=DEVICE
)

print(f"Model ready on {DEVICE}!")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

print(f"\nClasses detected:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  - Class {i}: {name}")


# ## 3.1 Optional: Model Training
# 
# This section only runs if the weights file is missing and training data is available.

if not MODEL_WEIGHTS.exists():
    print("Model weights missing. Attempting to train...")
    # Check for training data index
    data_index = BASE_DIR / 'tiles/index.csv'
    if data_index.exists():
        import pandas as pd
        df = pd.read_csv(data_index)

        # Fix tile paths: they are relative (data/processed/tiles/...) but need to be absolute
        df['tile_path'] = df['tile_path'].apply(lambda p: str(BASE_DIR / p))

        # Filter to only existing tiles
        df['exists'] = df['tile_path'].apply(lambda p: Path(p).exists())
        df_valid = df[df['exists']].copy()
        print(f"Found {len(df_valid)} valid training tiles out of {len(df)} in index")

        # Subsample for faster training (use all tiles if GPU available)
        use_cuda = torch.cuda.is_available()
        if not use_cuda and len(df_valid) > 500:
            df_valid = df_valid.sample(n=500, random_state=42)
            print(f"Subsampled to {len(df_valid)} tiles for CPU training")
        print(f"Found {len(df_valid)} valid training tiles out of {len(df)} in index")

        if len(df_valid) > 0:
            dataset = MoonTileDataset(df_valid, augment=True)
            loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = BCEDiceLoss()
            trainer = Trainer(model, optimizer, criterion, device=DEVICE)

            # Train for multiple epochs
            num_epochs = 3 if not use_cuda else 5
            print(f"Starting {num_epochs}-epoch training loop...")
            best_loss = float('inf')
            for epoch in range(num_epochs):
                loss = trainer.train_one_epoch(loader)
                print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}")
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), MODEL_WEIGHTS)

            print(f"Training complete. Best Loss: {best_loss:.4f}")
            print(f"Weights saved to {MODEL_WEIGHTS}")

            # Reload best weights and update predictor
            model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True))
            model.eval()
            predictor = Predictor(model=model, device=DEVICE)
        else:
            print("No valid training tiles found.")
    else:
        print("No training index found. Please ensure data is prepared if you want to train.")
else:
    print("Pre-trained weights found. Skipping training section.")


# ## 4. Inference Functions

def load_tile_and_predict(tile_path):
    """Load a single tile and run inference.

    Args:
        tile_path: Path to the PNG tile image.

    Returns:
        dict with 'path', 'probabilities' (C, H, W), 'image', 'success', 'error'.
    """
    try:
        # Load the tile as a grayscale-based 3-channel input
        tile = Image.open(tile_path).convert('L')
        tile_gray = np.array(tile).astype(np.float32)

        # Build 3-channel CHW input
        tile_chw = build_three_channel_input(tile_gray)

        # Run inference
        prob_cube = predictor.predict(tile_chw)

        return {
            'path': tile_path,
            'probabilities': prob_cube,
            'image': tile_gray / 255.0,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'path': tile_path,
            'probabilities': None,
            'image': None,
            'success': False,
            'error': str(e)
        }

def generate_tile_viz(display_img, prob_cube, class_names, output_path):
    """Generate visualization for a single tile."""
    n_classes = len(class_names)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Tile: {Path(output_path).name}", fontsize=14)

    axes[0, 0].imshow(display_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original Image', fontsize=10)
    axes[0, 0].axis('off')

    for col_idx, class_idx in enumerate(range(min(3, n_classes))):
        ax = axes[0, col_idx + 1]
        prob_img = prob_cube[class_idx].astype(np.float32)
        prob_max = float(np.max(prob_img))
        prob_min = float(np.min(prob_img))
        norm_prob = (prob_img - prob_min) / (prob_max - prob_min + 1e-8)

        ax.imshow(display_img, cmap='gray', vmin=0, vmax=1, alpha=0.4)
        ax.imshow(norm_prob, cmap='viridis', alpha=0.6)
        ax.set_title(f'{class_names[class_idx]}\nmax={prob_max:.3f}', fontsize=9)
        ax.axis('off')

    adjusted_threshold = 0.1
    for col_idx, class_idx in enumerate(range(min(4, n_classes))):
        ax = axes[1, col_idx]
        prob_img = prob_cube[class_idx].astype(np.float32)
        mask = (prob_img > adjusted_threshold).astype(np.uint8)
        prob_max = float(np.max(prob_img))
        prob_min = float(np.min(prob_img))

        if mask.sum() == 0:
            if prob_max > prob_min:
                norm_prob = (prob_img - prob_min) / (prob_max - prob_min + 1e-8)
                ax.imshow(norm_prob, cmap='plasma', alpha=0.7)
                title = f'{class_names[class_idx]}\nmax={prob_max:.3f}\nNo features at {adjusted_threshold}'
            else:
                ax.imshow(np.zeros_like(display_img), cmap='gray')
                title = f'{class_names[class_idx]}\nNo detections'
        else:
            if class_idx == 0:
                ax.imshow(mask, cmap='Reds', alpha=0.9)
            else:
                ax.imshow(mask, cmap='hot', alpha=0.8)
            title = f'{class_names[class_idx]}\n>{adjusted_threshold} | n={mask.sum()}'
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def run_batch_inference(tiles_dir, predictor, output_dir, max_tiles=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    probs_dir = output_dir / 'probabilities'
    viz_dir = output_dir / 'visualizations'
    probs_dir.mkdir(exist_ok=True)
    viz_dir.mkdir(exist_ok=True)

    tile_files = sorted(tiles_dir.glob('tile_*.png'))
    if max_tiles:
        tile_files = tile_files[:max_tiles]
    print(f"Processing {len(tile_files)} tiles...")

    results = []
    for i, tile_path in enumerate(tile_files):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(tile_files)} tiles...")
        result = load_tile_and_predict(tile_path)
        results.append(result)
        if result['success']:
            prob_cube = result['probabilities']
            img = result['image']
            np.savez(probs_dir / f"{tile_path.stem}.npz", probabilities=prob_cube)
            viz_path = viz_dir / f'{tile_path.stem}_viz.png'
            generate_tile_viz(img, prob_cube, CLASS_NAMES, viz_path)
    return results

print("Functions ready.")


# ## 5. Run Inference

if SKIP_INFERENCE:
    print("SKIPPING INFERENCE - using pre-computed results.")
    results = [{'success': True} for p in existing_probs]
    success_count = len(results)
    failure_count = 0
elif len(tile_files) > 0:
    # Check if we already have probability files for these tiles
    already_done = 0
    tiles_to_process = []
    for tf in tile_files:
        prob_file = PROBS_DIR / f"{tf.stem}.npz"
        if prob_file.exists():
            already_done += 1
        else:
            tiles_to_process.append(tf)

    print(f"Already processed: {already_done}")
    print(f"Tiles to process: {len(tiles_to_process)}")

    if tiles_to_process:
        results = run_batch_inference(TILES_DIR, predictor, OUTPUT_DIR, max_tiles=MAX_INFERENCE_TILES)
        success_count = sum(1 for r in results if r['success'])
        failure_count = len(results) - success_count
    else:
        print("All tiles already processed.")
        results = []
        success_count = 0
        failure_count = 0
else:
    print("Error: No data found.")
    success_count = 0
    failure_count = 0

print(f"Summary: {success_count} success, {failure_count} failure")


# ## 6. Aggregation

def aggregate_results(probs_dir, class_names):
    """Stream through .npz files one at a time to avoid loading all tiles
    into RAM simultaneously.

    Instead of stacking (N, C, H, W) into one giant array we keep only:
      - sum_probs  (float64, shape C x H x W): running sum  -> /n  = mean
      - max_probs  (float32, shape C x H x W): element-wise max across tiles
      - global_max (float32, length C)        : per-class scalar max

    Peak extra RAM = 3 x (C x H x W) regardless of tile count.
    """
    import json
    prob_files = sorted([p for p in probs_dir.glob('*.npz') if 'stats' not in p.name])
    n = len(prob_files)
    if n == 0:
        print('No probability files found - nothing to aggregate.')
        return {}

    # --- initialise accumulators from the first file ---
    first_cube = np.load(prob_files[0])['probabilities']   # (C, H, W)
    n_classes  = first_cube.shape[0]
    sum_probs  = first_cube.astype(np.float64)             # accumulate in float64
    max_probs  = first_cube.copy()                         # element-wise max (C, H, W)
    global_max = np.array(
        [float(np.max(first_cube[i])) for i in range(n_classes)],
        dtype=np.float32)                                  # per-class scalar max
    del first_cube

    # --- stream the remaining files one at a time ---
    for idx, f in enumerate(prob_files[1:]):
        cube = np.load(f)['probabilities']                 # (C, H, W)
        sum_probs += cube.astype(np.float64)
        np.maximum(max_probs, cube, out=max_probs)         # in-place element-wise max
        for i in range(n_classes):
            cm = float(np.max(cube[i]))
            if cm > global_max[i]:
                global_max[i] = cm
        del cube

        if (idx + 1) % 500 == 0:
            print(f"  Aggregated {idx + 2}/{n} files...")

    avg_probs = (sum_probs / n).astype(np.float32)
    del sum_probs

    stats = {
        'tile_count': n,
        'class_names': class_names,
        'avg_max_per_class':    {c: float(np.max(avg_probs[i]))     for i, c in enumerate(class_names)},
        'global_max_per_class': {c: float(global_max[i])            for i, c in enumerate(class_names)},
        'features_detected':    {c: int((max_probs[i] > 0.1).sum()) for i, c in enumerate(class_names)},
        '_avg_probs': avg_probs,
    }

    # Save stats.json (skip private keys that start with '_')
    json_path = probs_dir.parent / 'stats.json'
    with open(json_path, 'w') as fh:
        json.dump({k: v for k, v in stats.items() if not k.startswith('_')}, fh, indent=2)
    print(f'Stats saved to {json_path}')
    return stats

stats = aggregate_results(PROBS_DIR, CLASS_NAMES)
print("Aggregation complete.")


# ## 7. Visualization

viz_files = sorted(VIZ_DIR.glob('*_viz.png'))
print(f"Found {len(viz_files)} visualization files.")

# Display first 10 tiles
for i, viz_file in enumerate(viz_files[:10]):
    print(f"\n--- Tile {i+1}: {viz_file.name} ---")
    display(IPImage(filename=str(viz_file)))


# ## 8. Quality Assessment

if stats:
    print("QUALITY ASSESSMENT")
    print("=" * 60)
    for c in CLASS_NAMES:
        print(f"{c}:")
        print(f"  Max Prob: {stats['avg_max_per_class'][c]:.4f}")
        print(f"  Detections (>0.1): {stats['features_detected'][c]}")
    print("=" * 60)

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    max_probs = [stats['global_max_per_class'][c] for c in CLASS_NAMES]
    detections = [stats['features_detected'][c] for c in CLASS_NAMES]

    bars1 = ax.bar(x - width/2, max_probs, width, label='Max Probability', color='steelblue')

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, detections, width, label='Detections', color='coral')

    ax.set_xlabel('Feature Class')
    ax.set_ylabel('Max Probability', color='steelblue')
    ax2.set_ylabel('Feature Detections', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')

    fig.tight_layout()
    plt.savefig(BASE_DIR / 'results/south_pole/quality_summary.png', dpi=150)
    plt.show()

    # Save summary stats
    summary_path = BASE_DIR / 'results/south_pole/summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"South Pole Inference Summary\n")
        f.write(f"{'='*40}\n")
        f.write(f"Total tiles processed: {stats['tile_count']}\n")
        f.write(f"\nPer-class statistics:\n")
        for c in CLASS_NAMES:
            f.write(f"  {c}:\n")
            f.write(f"    Max probability: {stats['global_max_per_class'][c]:.4f}\n")
            f.write(f"    Avg max: {stats['avg_max_per_class'][c]:.4f}\n")
            f.write(f"    Detections (>0.1): {stats['features_detected'][c]}\n")
    print(f"\nSummary saved to {summary_path}")
else:
    print("No stats available - run aggregation first.")


# ## 9. Single Tile Demo
# 
# Run inference on a single tile and show detailed output.

# Pick a tile to demonstrate
tile_files = sorted(TILES_DIR.glob('tile_*.png'))
if tile_files:
    demo_tile = tile_files[0]
    print(f"Demo tile: {demo_tile.name}")

    # Load and predict
    result = load_tile_and_predict(demo_tile)

    if result['success']:
        # Show original image
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Demo: {demo_tile.name}", fontsize=14)

        # Original
        axes[0, 0].imshow(result['image'], cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')

        # Probability maps
        prob_cube = result['probabilities']
        for i, name in enumerate(CLASS_NAMES):
            row = (i + 1) // 4
            col = (i + 1) % 4
            ax = axes[row, col]
            prob_map = prob_cube[i]
            ax.imshow(prob_map, cmap='viridis')
            ax.set_title(f'{name}\nmax={prob_map.max():.3f}')
            ax.axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print(f"Failed: {result['error']}")
else:
    print("No tiles available for demo.")


# ## 10. Summary
# 
# Notebook completed successfully!
