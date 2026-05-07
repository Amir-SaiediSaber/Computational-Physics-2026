import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(0, 2, 3))
    den = probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
    dice = 1 - (num + eps) / (den + eps)
    return dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets) + dice_loss(logits, targets)

def multilabel_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    from ..data.preprocessing import CLASS_NAMES
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    per_class = []
    for i, name in enumerate(CLASS_NAMES):
        p = preds[:, i]
        t = targets[:, i]
        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        per_class.append({'class': name, 'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou})
    return pd.DataFrame(per_class)

class Trainer:
    def __init__(self, model, optimizer, criterion, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize trainer.

        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device string ('cuda' or 'cpu', auto-detected)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(self.device)

    def train_one_epoch(self, loader):
        self.model.train()
        losses = []
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses)) if losses else np.nan

    def evaluate(self, loader, criterion=None):
        self.model.eval()
        metrics_list = []
        criterion = criterion or self.criterion
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                if criterion:
                    metrics_list.append(multilabel_metrics(logits, y))
                else:
                    # If no criterion provided, we just need logits
                    pass

        if not metrics_list:
            return None

        # Combine metrics from all batches
        return pd.concat(metrics_list).groupby('class').mean()
