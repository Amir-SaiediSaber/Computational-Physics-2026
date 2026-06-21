"""Classical crater-detection baseline (thresholding + morphology).

This is the 4th method the comparison needs: a non-learned reference that the
deep models must beat to justify themselves. Pure OpenCV/NumPy.

Pipeline: CLAHE contrast → adaptive threshold (dark crater floors/shadows) →
morphological open/close → connected components filtered by area + elongation.
Returns a binary crater mask and an instance count, mirroring the area filters
(min 30 / max 6000 px) used by the Mask R-CNN instance preparation so the count
metric is comparable.
"""
import cv2
import numpy as np


def detect_craters_classical(gray_u8, min_area=30, max_area=6000,
                             block_size=31, c=5, max_aspect=3.0):
    """Detect craters in a grayscale uint8 tile.

    Returns (mask, count): mask is uint8 {0,1} of shape HxW, count is the number
    of accepted connected components.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_u8)
    th = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block_size, c)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    n, lab, stats, _ = cv2.connectedComponentsWithStats((th > 0).astype(np.uint8), connectivity=8)
    mask = np.zeros(gray_u8.shape, dtype=np.uint8)
    count = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if max(w, h) / (min(w, h) + 1e-9) > max_aspect:   # drop elongated (ridge-like) blobs
            continue
        mask[lab == i] = 1
        count += 1
    return mask, count
