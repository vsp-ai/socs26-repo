from .base import ClassificationLoss, RegressionLoss
from .registry import (
    classification_loss_names,
    get_classification_loss,
    get_regression_loss,
    register_classification_loss,
    register_regression_loss,
    regression_loss_names,
)

__all__ = [
    "ClassificationLoss",
    "RegressionLoss",
    "classification_loss_names",
    "get_classification_loss",
    "get_regression_loss",
    "register_classification_loss",
    "register_regression_loss",
    "regression_loss_names",
]
