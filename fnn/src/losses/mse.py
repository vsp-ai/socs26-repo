from __future__ import annotations

from .base import RegressionLoss


class MSERegressionLoss(RegressionLoss):
    name = "mse"

    def __call__(self, y_pred, y_true, *, epoch: int | None = None):
        _ = epoch
        sq_err = (y_pred - y_true) ** 2
        return sq_err.mean()
