from __future__ import annotations

from abc import ABC, abstractmethod


class RegressionLoss(ABC):
    """Interface for regression-style FNN training objectives."""

    name: str

    @abstractmethod
    def __call__(self, y_pred, y_true, *, epoch: int | None = None):
        raise NotImplementedError

    def rank_counts(self, y_pred, y_true):
        raise NotImplementedError(f"{self.name} does not define rank metrics")


class ClassificationLoss(ABC):
    """Interface for classification-style FNN training objectives."""

    name: str

    @abstractmethod
    def __call__(self, y_pred, y_true, *, epoch: int | None = None):
        raise NotImplementedError
