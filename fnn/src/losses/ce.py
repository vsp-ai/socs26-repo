from __future__ import annotations

import torch.nn.functional as F

from .base import ClassificationLoss


class CrossEntropyLoss(ClassificationLoss):
    """Standard cross-entropy objective for classification FNN training."""

    name = "ce"

    def __call__(self, y_pred, y_true, *, epoch: int | None = None):
        _ = epoch
        return F.cross_entropy(y_pred, y_true)
