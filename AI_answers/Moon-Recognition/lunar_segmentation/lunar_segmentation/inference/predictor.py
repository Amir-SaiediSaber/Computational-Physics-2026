import numpy as np
import torch
import logging
from pathlib import Path
from ..data.preprocessing import iter_tile_origins, CLASS_NAMES

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, model, weights_path: Path = None, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        if weights_path:
            # Check if weights are stored as flat dict or nested under 'model'
            weights = torch.load(str(weights_path), map_location=self.device)
            if isinstance(weights, dict) and 'model' in weights:
                # Nested format
                weights = weights['model']
            # Flatten any 'model.' prefix keys
            flat_weights = {}
            for k, v in weights.items():
                new_k = k.replace('model.', '', 1)
                flat_weights[new_k] = v
            self.model.load_state_dict(flat_weights, strict=False)
            logger.info(f"Loaded weights from {weights_path}")

    def predict(self, image_chw: np.ndarray, tile_size: int = 128, stride: int = 64):
        """
        Run sliding window prediction on a large raster.

        Args:
            image_chw: 3-channel input image (H, W)
            tile_size: Size of each tile (default 128)
            stride: Stride between tiles (default 64)
        """
        self.model.eval()
        c, h, w = image_chw.shape
        prob_sum = np.zeros((len(CLASS_NAMES), h, w), dtype=np.float32)
        count_sum = np.zeros((len(CLASS_NAMES), h, w), dtype=np.float32)

        with torch.no_grad():
            for r, c0 in iter_tile_origins(h, w, tile_size, stride):
                tile = image_chw[:, r:r+tile_size, c0:c0+tile_size]
                if tile.shape[1] != tile_size or tile.shape[2] != tile_size:
                    continue

                x = torch.from_numpy(tile[None].astype(np.float32)).to(self.device)
                # we assume model output is logits
                probs = torch.sigmoid(self.model(x)).cpu().numpy()[0]
                prob_sum[:, r:r+tile_size, c0:c0+tile_size] += probs
                count_sum[:, r:r+tile_size, c0:c0+tile_size] += 1.0

        return prob_sum / np.clip(count_sum, 1.0, None)
