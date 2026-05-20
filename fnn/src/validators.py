class BaseValidator:
    name = "base"

    def validate(self, epoch: int, fnn) -> dict:
        raise NotImplementedError

    def is_better(self, metrics: dict, best_metrics: dict) -> bool:
        raise NotImplementedError


class FidelityValidator(BaseValidator):
    name = "fidelity"

    def __init__(self, valid_loader, set_name: str = "Validation", output_log: bool = False):
        self.valid_loader = valid_loader
        self.set_name = set_name
        self.output_log = output_log

    def validate(self, epoch: int, fnn) -> dict:
        acc, f1 = fnn.test(
            test_loader=self.valid_loader,
            set_name=self.set_name,
            output_log=self.output_log,
        )
        extra = getattr(fnn, "last_test_metrics", {}) or {}
        return {
            "val_acc": float(acc),
            "val_f1": float(f1),
            "val_mse": extra.get("mse"),
            "val_rank_acc": extra.get("rank_acc"),
        }

    def is_better(self, metrics: dict, best_metrics: dict) -> bool:
        f1 = metrics["val_f1"]
        best_f1 = best_metrics.get("val_f1", -1.0)
        if f1 > best_f1:
            return True
        if abs(f1 - best_f1) < 1e-10:
            return metrics["train_loss"] < best_metrics.get("train_loss", float("inf"))
        return False
