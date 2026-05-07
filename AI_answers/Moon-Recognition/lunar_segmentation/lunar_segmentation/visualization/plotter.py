import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def _ensure_2d_gray(img: np.ndarray) -> np.ndarray:
    """Accept (H, W) or (C, H, W) and return a 2D grayscale array."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        # (C, H, W) → take first channel
        if img.shape[0] in (1, 3):
            return img[0]
        # (H, W, C) layout
        if img.shape[2] in (1, 3):
            return img[:, :, 0]
    raise ValueError(f"Cannot convert shape {img.shape} to 2D grayscale")


def generate_pretty_image(gray_img: np.ndarray, prob_cube: np.ndarray, class_names: list[str],
                         output_path: Path, threshold: float = 0.5):
    """
    Generates a grid of images showing the original WAC crop and overlays of each class.
    Uses a more aggressive relative thresholding to ensure features are visible
    even if the model is under-confident.
    """
    gray_img = _ensure_2d_gray(gray_img)

    n = len(class_names)
    ncols = 3
    nrows = (n // ncols) + 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    # Original image
    axes[0].imshow(gray_img, cmap='gray')
    axes[0].set_title('Original WAC Crop')
    axes[0].axis('off')

    # Overlays for each class
    for i, name in enumerate(class_names, start=1):
        axes[i].imshow(gray_img, cmap='gray')

        current_prob = prob_cube[i-1]
        max_p = np.max(current_prob)
        min_p = np.min(current_prob)

        # Use a relative threshold: top 1% of probabilities or 80% of max
        # This ensures that we always see the most likely candidates for that class
        # even if the absolute probability is low.
        effective_threshold = max(threshold, max_p * 0.8)

        # If the max is very low, just take the top 0.1% of pixels
        if max_p < 0.4:
            # Sort all probabilities and take the top N pixels
            sorted_probs = np.sort(current_prob.flatten())
            effective_threshold = sorted_probs[-int(current_prob.size * 0.001)]
            logger.info(f"Low confidence for {name} (max {max_p:.2f}). Using top 0.1% pixels. T={effective_threshold:.4f}")
        else:
            logger.info(f"Visualizing {name} with threshold {effective_threshold:.4f}")

        mask = current_prob > effective_threshold
        if np.any(mask):
            axes[i].imshow(mask, cmap='inferno', alpha=0.6)
        axes[i].set_title(f"{name} (T={effective_threshold:.2f})")
        axes[i].axis('off')

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved visualization to {output_path}")
