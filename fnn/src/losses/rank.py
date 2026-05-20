from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import RegressionLoss


class PairwiseRankLoss(RegressionLoss):
    name = "rank"

    @staticmethod
    def pairwise_terms(y_pred, y_true):
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        sign = torch.sign(true_diff)
        pair_valid = sign != 0
        upper = torch.triu(
            torch.ones(pair_valid.shape[-2:], device=pair_valid.device, dtype=torch.bool),
            diagonal=1,
        )
        pair_valid = pair_valid & upper
        return pred_diff, sign, pair_valid

    def __call__(self, y_pred, y_true, *, epoch: int | None = None):
        _ = epoch
        pred_diff, sign, pair_valid = self.pairwise_terms(y_pred, y_true)
        pair_loss = F.softplus(-sign * pred_diff)
        denom = pair_valid.sum().clamp(min=1).to(y_pred.dtype)
        return pair_loss.masked_select(pair_valid).sum() / denom

    def rank_counts(self, y_pred, y_true):
        pred_diff, sign, pair_valid = self.pairwise_terms(y_pred, y_true)
        correct = ((sign * pred_diff) > 0) & pair_valid
        return correct.sum().to(y_pred.dtype), pair_valid.sum().to(y_pred.dtype)
