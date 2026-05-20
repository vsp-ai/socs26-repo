from __future__ import annotations

from .base import ClassificationLoss, RegressionLoss
from .ce import CrossEntropyLoss
from .mse import MSERegressionLoss
from .rank import PairwiseRankLoss


_REGRESSION_LOSSES: dict[str, type[RegressionLoss]] = {}
_CLASSIFICATION_LOSSES: dict[str, type[ClassificationLoss]] = {}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def register_regression_loss(loss_cls: type[RegressionLoss]) -> type[RegressionLoss]:
    name = _normalize_name(loss_cls.name)
    if not name:
        raise ValueError("Regression loss name cannot be empty")
    if name in _REGRESSION_LOSSES:
        raise ValueError(f"Duplicate regression loss: {name}")
    _REGRESSION_LOSSES[name] = loss_cls
    return loss_cls


def register_classification_loss(loss_cls: type[ClassificationLoss]) -> type[ClassificationLoss]:
    name = _normalize_name(loss_cls.name)
    if not name:
        raise ValueError("Classification loss name cannot be empty")
    if name in _CLASSIFICATION_LOSSES:
        raise ValueError(f"Duplicate classification loss: {name}")
    _CLASSIFICATION_LOSSES[name] = loss_cls
    return loss_cls


def get_regression_loss(name: str) -> RegressionLoss:
    normalized = _normalize_name(name)
    try:
        return _REGRESSION_LOSSES[normalized]()
    except KeyError as exc:
        expected = ", ".join(regression_loss_names())
        raise ValueError(f"Unknown regression_mode: {normalized}. Expected one of [{expected}]") from exc


def get_classification_loss(name: str) -> ClassificationLoss:
    normalized = _normalize_name(name)
    try:
        return _CLASSIFICATION_LOSSES[normalized]()
    except KeyError as exc:
        expected = ", ".join(classification_loss_names())
        raise ValueError(f"Unknown classification loss: {normalized}. Expected one of [{expected}]") from exc


def regression_loss_names() -> tuple[str, ...]:
    return tuple(sorted(_REGRESSION_LOSSES))


def classification_loss_names() -> tuple[str, ...]:
    return tuple(sorted(_CLASSIFICATION_LOSSES))


register_classification_loss(CrossEntropyLoss)
register_regression_loss(MSERegressionLoss)
register_regression_loss(PairwiseRankLoss)
